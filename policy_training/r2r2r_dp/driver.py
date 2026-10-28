"""Optimizer-update-based TinyDP training driver (audit M2, M3, M7). TinyDP's sources are not modified.

Reused unchanged from TinyDP: the sim SequenceDataset (via ManifestSimDataset), VideoSampler,
CollateFunction, DiffusionPolicy, add_weight_decay, NativeScalerWithGradNormCount, ModelEMA and the
diffusers cosine scheduler. Replaced: the epoch loop of tinydp/util/engine.py and script/train.py.

Semantics (each differs from the released loop, see docs/policy_training/AUDIT_REPAIR_REPORT.md):
  * one optimizer update = exactly `accum` consecutive micro-batches of a continuous stream that runs
    across epoch boundaries, so no accumulated gradient is ever discarded;
  * the LR scheduler and the EMA advance once per *successful* optimizer update, i.e. only when
    optimizer.step() actually executed; training stops after `total_updates` successful updates;
    warm-up is counted in successful updates. An accumulation group whose step GradScaler skipped
    (non-finite gradients) consumes its micro-batches and is retried on the next group, without
    advancing scheduler, EMA, update counter or validation/checkpoint cadence; the run aborts after
    MAX_CONSECUTIVE_SKIPPED_UPDATES consecutive or MAX_SKIPPED_UPDATES total skipped groups;
  * logged losses are raw (not divided by `accum`); validation reports the raw mean loss per window;
  * sampler.set_epoch(epoch) is called for every epoch of the stream;
  * every item's randomness (proprio noise, colour jitter) is seeded from (seed, stream, epoch, index),
    so the data stream does not depend on worker count and can be resumed at any micro-batch;
  * validation uses EMA weights under a forked, fixed-seed RNG and does not perturb training RNG;
  * full checkpoints hold model, EMA, optimizer, scheduler, grad-scaler, RNG state, update and data
    position, plus the resolved run record and the action statistics (self-contained for inference);
    loading one validates every component (keys, shapes, dtypes, finiteness, counters, data position,
    sample order, per-parameter Adam state against the recorded optimizer history, optimizer and
    scheduler hyperparameters against the schedule, grad-scaler and RNG payloads) before any runtime
    state is changed, and restores the previous state if applying it fails; a checkpoint that fails
    the same validation is never written (docs/policy_training/F7_CHECKPOINT_REPAIR.md);
  * validation and EMA export restore the raw weights and the previous train/eval mode even when
    they fail, and non-finite or empty validation raises instead of being logged as a result.
"""
import copy
import glob
import hashlib
import json
import math
import os
import random
import shutil
import time
from typing import Dict, List, Optional

import numpy as np
import torch

CKPT_FORMAT = "r2r2r-dp-ckpt-v3"
INFERENCE_FORMATS = ("r2r2r-dp-ckpt-v2", CKPT_FORMAT)  # weights-only loading; continuation needs CKPT_FORMAT
TRAIN_STREAM, VAL_STREAM = 0, 1
# Bounded retry policy for accumulation groups whose optimizer step GradScaler skipped.
MAX_CONSECUTIVE_SKIPPED_UPDATES = 20  # GradScaler halves its scale per skip: 2**16 -> 2**-4
MAX_SKIPPED_UPDATES = 1000


class CheckpointError(ValueError):
    pass


class SkippedStepError(RuntimeError):
    pass


def _jsonable(x):
    """Strict-JSON number: non-finite floats become the strings 'nan', 'inf', '-inf'."""
    x = float(x)
    return x if math.isfinite(x) else str(x)


def sample_order_sha256(ds) -> str:
    """Digest of the dataset's window order as (episode uid, start, end): what the sampler indexes."""
    uid_of, rows = {}, []
    for path, s, e in ds.start_end:
        ep = os.path.dirname(os.path.dirname(path))
        if ep not in uid_of:
            with open(os.path.join(ep, "source.json")) as f:
                uid_of[ep] = json.load(f)["uid"]
        rows.append([uid_of[ep], int(s), int(e)])
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


class SeededItems(torch.utils.data.Dataset):
    """Items are requested as (stream, epoch, index); all item randomness is seeded from that key."""

    def __init__(self, base, seed: int):
        self.base, self.seed = base, seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, key):
        stream, epoch, idx = key
        a, b = np.random.SeedSequence([self.seed, stream, epoch, idx]).generate_state(2)
        # With num_workers=0 this runs in the training process: restore the caller's RNG afterwards so
        # diffusion noise never depends on where items were produced.
        py, npst = random.getstate(), np.random.get_state()
        with torch.random.fork_rng(devices=[]):
            random.seed(int(a))
            np.random.seed(int(a))
            torch.default_generator.manual_seed(int(b))
            try:
                return self.base[idx]
            finally:
                random.setstate(py)
                np.random.set_state(npst)


class TrainBatchStream:
    """Endless batch sampler over VideoSampler epochs, starting at (epoch, batch)."""

    def __init__(self, sampler, start_epoch: int = 0, start_batch: int = 0):
        self.sampler, self.start_epoch, self.start_batch = sampler, start_epoch, start_batch

    def __iter__(self):
        e, b = self.start_epoch, self.start_batch
        while True:
            self.sampler.set_epoch(e)
            for batch in list(iter(self.sampler))[b:]:
                yield [(TRAIN_STREAM, e, i) for i in batch]
            e, b = e + 1, 0


def window_batches(ds, seq_length: int, batch_size: int, stride: Optional[int]) -> List[List]:
    """Fixed validation batches: every window (stride None) or windows whose start % stride == 0."""
    n_windows = len(ds) // seq_length
    ks = [k for k in range(n_windows) if stride is None or ds.start_end[k * seq_length][1] % stride == 0]
    return [[(VAL_STREAM, 0, k * seq_length + j) for k in ks[i:i + batch_size] for j in range(seq_length)]
            for i in range(0, len(ks), batch_size)]


def _rng_state(device) -> Dict:
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if device.type == "cuda":
        st["cuda"] = torch.cuda.get_rng_state(device)
    return st


def _set_rng_state(st: Dict, device) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(st["cuda"], device)


def _save(path: str, obj) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


class Trainer:
    # successful updates are self.update; attempted = successful + skipped accumulation groups
    attempted_updates = 0
    skipped_updates = 0
    consecutive_skipped = 0
    optimizer_calls = 0  # counted by the wrapper installed in _count_optimizer_steps
    train_order_sha256 = None
    unusable = None  # set when a failed checkpoint load could not be rolled back; train() then refuses
    param_steps = None  # optimizer history, see __init__

    def __init__(self, run: Dict, device: str = "cuda", num_workers: Optional[int] = None):
        from diffusers.optimization import get_scheduler
        import tinydp.util.misc as misc
        from tinydp.dataset.image_dataset import CollateFunction, VideoSampler
        from tinydp.dataset.utils import aug_vision_transform, default_vision_transform
        from tinydp.policy.model import DiffusionPolicy
        from tinydp.util.ema import ModelEMA
        from .protocol import config_from_dict
        from . import tinydp_bridge as tb

        self.run, self.sched = run, run["protocol"]["schedule"]
        self.cfg = cfg = config_from_dict(run["tinydp"])
        self.device = torch.device(device)
        self.out = run["paths"]["output_dir"]
        self.amp = self.sched["amp_dtype"] == "bfloat16" and self.device.type == "cuda"
        seed = cfg.shared_cfg.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True  # as released train.py

        tb.install_ordered_glob()
        cls = tb.make_manifest_dataset_class(run["paths"]["train_root"], run["paths"]["eval_root"],
                                             run["paths"]["action_statistics"])
        res = cfg.shared_cfg.image_size
        self.ds_train = cls(cfg.dataset_cfg, cfg.shared_cfg, cfg.logging_cfg, aug_vision_transform(resolution=res), split="train")
        self.ds_val = cls(cfg.dataset_cfg, cfg.shared_cfg, cfg.logging_cfg, default_vision_transform(resolution=res), split="val")
        B, L = cfg.shared_cfg.batch_size, cfg.shared_cfg.seq_length
        self.sampler = VideoSampler(self.ds_train, batch_size=B, sequence_length=L, seed=self.sched["sampler_seed"])
        self.batches_per_epoch = len(self.sampler)
        self.train_order_sha256 = sample_order_sha256(self.ds_train)
        self.collate = CollateFunction(L, pred_left_only=False, pred_right_only=cfg.model_cfg.policy_cfg.pred_right_only)
        self.val_subset = window_batches(self.ds_val, L, B, self.sched["val_subset_window_stride"])
        self.val_full = window_batches(self.ds_val, L, B, None)
        self.num_workers = cfg.trainer_cfg.num_workers if num_workers is None else num_workers
        self.pin = cfg.trainer_cfg.pin_memory and self.device.type == "cuda"

        self.model = DiffusionPolicy(shared_cfg=cfg.shared_cfg, model_cfg=cfg.model_cfg).to(self.device)
        self.optimizer = torch.optim.AdamW(misc.add_weight_decay(self.model, cfg.optimizer_cfg.weight_decay),
                                           lr=cfg.optimizer_cfg.lr)
        self.scaler = misc.NativeScalerWithGradNormCount()
        self.scheduler = get_scheduler(name="cosine", optimizer=self.optimizer,
                                       num_warmup_steps=self.sched["warmup_updates"],
                                       num_training_steps=self.sched["total_updates"])
        self.ema = ModelEMA(self.model, decay=self.sched["ema_decay_per_update"], device=self.device)
        # Optimizer history: per optimizer parameter (group order), the number of executed optimizer steps in
        # which it had a gradient, i.e. the Adam step it must carry. Kept by the _count_optimizer_steps wrapper.
        names = {id(q): n for n, q in self.model.named_parameters()}
        self.optimizer_param_names = [[names[id(q)] for q in g["params"]] for g in self.optimizer.param_groups]
        self.param_steps = [[0] * len(g["params"]) for g in self.optimizer.param_groups]
        self.update, self.micro, self.epoch, self.batch_in_epoch = 0, 0, 0, 0
        self.stats_json = json.load(open(run["paths"]["action_statistics"]))

    # ---- data ----
    def _loader(self, ds, batch_sampler, persistent: bool):
        g = torch.Generator().manual_seed(0)  # DataLoader base seeds never touch the global RNG
        kw = dict(num_workers=self.num_workers, collate_fn=self.collate, pin_memory=self.pin, generator=g)
        if self.num_workers > 0:
            kw.update(prefetch_factor=2, persistent_workers=persistent)
        return torch.utils.data.DataLoader(SeededItems(ds, self.cfg.shared_cfg.seed), batch_sampler=batch_sampler, **kw)

    def _advance_position(self):
        self.micro += 1
        self.batch_in_epoch += 1
        if self.batch_in_epoch == self.batches_per_epoch:
            self.epoch, self.batch_in_epoch = self.epoch + 1, 0

    # ---- evaluation ----
    def evaluate(self, batches, weights: str = "ema") -> Dict:
        """Mean raw loss per window. Restores raw weights and train/eval mode on success and failure."""
        if not batches:
            raise ValueError("validation: no batches")
        t0 = time.time()
        was_training = self.model.training
        raw = {k: v.detach().clone() for k, v in self.model.state_dict().items()} if weights == "ema" else None
        total, n = 0.0, 0
        try:
            if raw is not None:
                self.ema.apply_ema(self.model)
            self.model.eval()
            devs = [self.device.index or 0] if self.device.type == "cuda" else []
            with torch.random.fork_rng(devices=devs):
                torch.manual_seed(self.sched["val_seed"])
                if self.device.type == "cuda":
                    torch.cuda.manual_seed(self.sched["val_seed"])
                it = iter(self._loader(self.ds_val, batches, persistent=False))
                try:
                    with torch.no_grad():
                        for i, batch in enumerate(it):
                            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
                                loss = float(self.model(batch).item())
                            if not math.isfinite(loss):
                                raise FloatingPointError(f"validation: non-finite loss {loss} in batch {i}")
                            w = batch["proprio"].shape[0]
                            total += loss * w
                            n += w
                finally:
                    if hasattr(it, "_shutdown_workers"):
                        it._shutdown_workers()
                    del it
        finally:
            if raw is not None:
                self.model.load_state_dict(raw)
            self.model.train(was_training)
        if n == 0:
            raise ValueError("validation: no windows evaluated")
        mean = total / n
        if not math.isfinite(mean):
            raise FloatingPointError(f"validation: non-finite mean loss {mean}")
        return {"loss": mean, "windows": n, "weights": weights, "seconds": round(time.time() - t0, 2)}

    # ---- checkpoints ----
    def _common(self) -> Dict:
        return {"format": CKPT_FORMAT, "update": self.update, "micro_batches": self.micro,
                "attempted_updates": self.attempted_updates, "skipped_updates": self.skipped_updates,
                "train_order_sha256": self.train_order_sha256,
                "run": {k: self.run[k] for k in ("tinydp", "config_sha256", "train_size", "protocol", "launch", "provenance")
                        if k in self.run},
                "action_statistics": self.stats_json}

    def save_full(self) -> str:
        state = self._common()
        state.update(kind="full", model=self.model.state_dict(), ema=self.ema.state_dict(),
                     optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict(),
                     scaler=self.scaler.state_dict(), rng=_rng_state(self.device),
                     data_position={"epoch": self.epoch, "batch": self.batch_in_epoch,
                                    "batches_per_epoch": self.batches_per_epoch},
                     optimizer_history={"param_names": [list(g) for g in self.optimizer_param_names],
                                        "steps": [list(g) for g in self.param_steps]})
        path = os.path.join(self.out, f"ckpt_full_u{self.update:07d}.pt")
        problems = self.checkpoint_problems(state)
        if problems:  # never publish (or rotate older checkpoints out for) a state that could not be resumed
            raise CheckpointError(f"refusing to write {path}: " + "; ".join(problems))
        _save(path, state)
        for old in sorted(glob.glob(os.path.join(self.out, "ckpt_full_u*.pt")))[:-self.sched["keep_full_checkpoints"]]:
            os.remove(old)
        return path

    def save_inference(self) -> str:
        raw = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        try:
            self.ema.apply_ema(self.model)
            weights = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        finally:
            self.model.load_state_dict(raw)
        bad = [k for k, v in weights.items() if v.is_floating_point() and not torch.isfinite(v).all()]
        if bad:
            raise FloatingPointError(f"EMA export: non-finite weights in {bad[:5]}")
        state = self._common()
        state.update(kind="inference", weights="ema", model=weights)
        path = os.path.join(self.out, f"policy_ema_u{self.update:07d}.pt")
        _save(path, state)
        return path

    def checkpoint_problems(self, st: Dict) -> List[str]:
        """Everything that must hold for st to continue this run exactly; [] if it may be loaded.
        Never changes runtime state; a malformed component is reported as a problem, not raised."""
        try:
            return self._checkpoint_problems(st)
        except Exception as e:  # noqa: BLE001 - any unexpected structure means the checkpoint cannot be resumed
            return [f"malformed checkpoint ({type(e).__name__}: {e})"]

    def _checkpoint_problems(self, st: Dict) -> List[str]:
        if not isinstance(st, dict):
            return [f"not a checkpoint dictionary ({type(st).__name__})"]
        if st.get("format") != CKPT_FORMAT or st.get("kind") != "full":
            return [f"not a {CKPT_FORMAT} full checkpoint (format={st.get('format')!r}, kind={st.get('kind')!r})"]
        need = {"update", "micro_batches", "attempted_updates", "skipped_updates", "train_order_sha256", "run",
                "action_statistics", "model", "ema", "optimizer", "scheduler", "scaler", "rng", "data_position"}
        if need - set(st):
            return [f"checkpoint lacks {sorted(need - set(st))}"]
        p = []
        if st["run"].get("config_sha256") != self.run["config_sha256"]:
            p.append("checkpoint was trained with a different resolved configuration")
        if st["action_statistics"] != self.stats_json:
            p.append("checkpoint action statistics differ from this run's statistics")
        if st["train_order_sha256"] != self.train_order_sha256:
            p.append("training sample order differs from the checkpoint (view order or membership changed)")
        p += _tensor_dict_problems("model", st["model"], self.model.state_dict())
        p += _tensor_dict_problems("ema", st["ema"], self.ema.ema)
        u, att, sk, mb = st["update"], st["attempted_updates"], st["skipped_updates"], st["micro_batches"]
        accum = self.sched["accum_micro_batches"]
        if not all(type(v) is int and v >= 0 for v in (u, att, sk, mb)):
            p.append("counters must be non-negative integers")
            return p
        if att != u + sk:
            p.append(f"attempted_updates {att} != update {u} + skipped_updates {sk}")
        if mb != accum * att:
            p.append(f"micro_batches {mb} != accum {accum} x attempted_updates {att}")
        dp = st["data_position"]
        if dp.get("batches_per_epoch") != self.batches_per_epoch:
            p.append("training data differs from the checkpoint (batches per epoch)")
        elif not (type(dp.get("epoch")) is int and type(dp.get("batch")) is int and 0 <= dp["batch"] < self.batches_per_epoch
                  and dp["epoch"] * self.batches_per_epoch + dp["batch"] == mb):
            p.append(f"data position {dp} is inconsistent with {mb} consumed micro-batches")
        p += self._scheduler_problems(st["scheduler"], u)
        p += self._optimizer_problems(st["optimizer"], st.get("optimizer_history"), u)
        p += self._scaler_problems(st["scaler"])
        p += _rng_problems(st["rng"], self.device)
        return p

    def _scheduled_lrs(self, u: int) -> List[float]:
        """Per parameter group, the learning rate the schedule sets after u successful updates."""
        return [base * lam(u) for lam, base in zip(self.scheduler.lr_lambdas, self.scheduler.base_lrs)]

    def _scheduler_problems(self, sc, u: int) -> List[str]:
        ref = self.scheduler.state_dict()
        if not isinstance(sc, dict) or set(sc) != set(ref):
            return [f"scheduler state keys {sorted(sc) if isinstance(sc, dict) else type(sc).__name__} "
                    f"!= {sorted(ref)}"]
        p = []
        if sc["last_epoch"] != u or type(sc["last_epoch"]) is not int:
            p.append(f"scheduler last_epoch {sc['last_epoch']!r} != update {u}")
        if sc["_step_count"] != u + 1 or type(sc["_step_count"]) is not int:
            p.append(f"scheduler _step_count {sc['_step_count']!r} != update {u} + 1")
        p += [f"scheduler {k} {sc[k]!r} != {ref[k]!r}" for k in sorted(ref)
              if k not in ("last_epoch", "_step_count", "_last_lr") and sc[k] != ref[k]]
        lrs = self._scheduled_lrs(u)
        last = sc["_last_lr"]
        if not (isinstance(last, list) and len(last) == len(lrs) and all(map(_same_lr, last, lrs))):
            p.append(f"scheduler last lr {last!r} is not the scheduled {lrs} at update {u}")
        return p

    def _expected_param_steps(self, hist, u: int):
        """(Adam step every optimizer parameter must carry, problems) from the checkpoint's optimizer history.

        Parameters need not share one history: a parameter without a gradient in some update is not stepped
        by AdamW. v3 checkpoints written before the history was recorded (986a3fa and earlier) carry none;
        for them every parameter must have been stepped in all u updates, which holds for this policy (every
        parameter receives a gradient in every update; see F7_CHECKPOINT_REPAIR.md)."""
        if hist is None:
            return [[u] * len(g) for g in self.optimizer_param_names], []
        if not isinstance(hist, dict) or set(hist) != {"param_names", "steps"}:
            return None, ["optimizer history is malformed"]
        if hist["param_names"] != self.optimizer_param_names:
            return None, ["optimizer parameter groups map to different model parameters than this run's"]
        steps = hist["steps"]
        if not (isinstance(steps, list) and all(isinstance(g, list) for g in steps)
                and [len(g) for g in steps] == [len(g) for g in self.optimizer_param_names]
                and all(type(k) is int and 0 <= k <= u for g in steps for k in g)):
            return None, [f"optimizer history steps must be one integer in [0, update {u}] per optimizer parameter"]
        top = max((k for g in steps for k in g), default=0)
        if top != u:  # an executed update steps at least one parameter; none before the first update
            return None, [f"optimizer history: the most-updated parameter has {top} steps, the checkpoint claims {u} "
                          "successful updates"]
        return steps, []

    def _optimizer_problems(self, opt, hist, u: int) -> List[str]:
        ref = self.optimizer.state_dict()
        if not (isinstance(opt, dict) and set(opt) == {"state", "param_groups"} and isinstance(opt["state"], dict)
                and isinstance(opt["param_groups"], list) and all(isinstance(g, dict) for g in opt["param_groups"])):
            return ["optimizer: not an optimizer state dictionary"]
        if [g.get("params") for g in opt["param_groups"]] != [g["params"] for g in ref["param_groups"]]:
            return ["optimizer parameter groups differ from this model"]
        p = []
        for i, (g, r, lr) in enumerate(zip(opt["param_groups"], ref["param_groups"], self._scheduled_lrs(u))):
            if set(g) != set(r):
                p.append(f"optimizer group {i}: keys {sorted(set(g) ^ set(r))} differ from this optimizer's")
                continue
            p += [f"optimizer group {i}: {k} {g[k]!r} != {r[k]!r}" for k in r if k not in ("params", "lr") and g[k] != r[k]]
            if not _same_lr(g["lr"], lr):
                p.append(f"optimizer group {i}: lr {g['lr']!r} is not the scheduled {lr!r} at update {u}")
        steps, q = self._expected_param_steps(hist, u)
        if q:
            return p + q
        unknown = sorted(set(opt["state"]) - {i for g in ref["param_groups"] for i in g["params"]}, key=str)
        if unknown:
            p.append(f"optimizer: state for {len(unknown)} unknown parameter ids, e.g. {unknown[:3]}")
        missing, unexpected, malformed = [], [], []
        for g, r, names, ks in zip(self.optimizer.param_groups, ref["param_groups"], self.optimizer_param_names, steps):
            need = {"step", "exp_avg", "exp_avg_sq"} | ({"max_exp_avg_sq"} if r["amsgrad"] else set())
            for param, pid, name, k in zip(g["params"], r["params"], names, ks):
                s = opt["state"].get(pid)
                if s is None:
                    if k:
                        missing.append(f"{name} ({k} recorded updates)")
                elif not k:
                    unexpected.append(name)
                else:
                    why = _adam_entry_problem(s, param, k, need)
                    if why:
                        malformed.append(f"{name}: {why}")
        if missing:
            p.append(f"optimizer: no Adam state for {len(missing)} parameters with recorded updates, e.g. {missing[:3]}")
        if unexpected:
            p.append(f"optimizer: Adam state for {len(unexpected)} parameters without recorded updates, "
                     f"e.g. {unexpected[:3]}")
        if malformed:
            p.append(f"optimizer: {len(malformed)} invalid Adam state entries, e.g. {malformed[:3]}")
        return p

    def _scaler_problems(self, sc) -> List[str]:
        ref = self.scaler.state_dict()  # {} when the grad scaler is disabled (CPU)
        if not isinstance(sc, dict):
            return ["grad-scaler state is not a dictionary"]
        if not sc:
            return ["grad-scaler state is empty, but this run's grad scaler is enabled"] if ref else []
        keys = {"scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"}
        if set(sc) != keys:
            return [f"grad-scaler state keys {sorted(sc)} != {sorted(keys)}"]
        p = []
        scale, gf, bf, gi, tr = (sc[k] for k in ("scale", "growth_factor", "backoff_factor", "growth_interval",
                                                 "_growth_tracker"))
        if not (_is_real(scale) and math.isfinite(scale) and scale > 0):
            p.append(f"grad-scaler scale {scale!r} is not a positive finite number")
        if not (_is_real(gf) and math.isfinite(gf) and gf > 1 and _is_real(bf) and 0 < bf < 1
                and type(gi) is int and gi >= 1 and type(tr) is int and 0 <= tr < gi):
            p.append(f"grad-scaler settings out of range (growth_factor={gf!r}, backoff_factor={bf!r}, "
                     f"growth_interval={gi!r}, _growth_tracker={tr!r})")
        elif ref and (gf, bf, gi) != (ref["growth_factor"], ref["backoff_factor"], ref["growth_interval"]):
            p.append("grad-scaler growth/backoff settings differ from this run's")
        return p

    def load_full(self, path: str) -> None:
        """Validate the whole checkpoint, then apply it; if applying fails the previous runtime state is restored."""
        try:
            st = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:  # noqa: BLE001
            raise CheckpointError(f"{path}: cannot be read ({type(e).__name__}: {e})") from e
        problems = self.checkpoint_problems(st)
        if problems:
            raise CheckpointError(f"{path}: " + "; ".join(problems))
        before = self._runtime_state()
        try:
            self._apply_checkpoint(st)
        except Exception as e:  # noqa: BLE001
            try:
                self._restore_runtime_state(before)
            except Exception as r:  # noqa: BLE001
                self.unusable = (f"{path}: could not be applied ({type(e).__name__}: {e}) and the previous state "
                                 f"could not be restored ({type(r).__name__}: {r})")
                raise CheckpointError(self.unusable) from e
            raise CheckpointError(f"{path}: could not be applied ({type(e).__name__}: {e}); "
                                  "previous state restored") from e

    _RUNTIME_ATTRS = ("update", "micro", "attempted_updates", "skipped_updates", "epoch", "batch_in_epoch",
                      "param_steps", "_resume_rng")

    def _runtime_state(self) -> Dict:
        return {"model": {k: v.detach().clone() for k, v in self.model.state_dict().items()},
                "ema": {k: v.detach().clone() for k, v in self.ema.ema.items()},
                "optimizer": copy.deepcopy(self.optimizer.state_dict()),
                "scheduler": copy.deepcopy(self.scheduler.state_dict()),
                "scaler": copy.deepcopy(self.scaler.state_dict()),
                "attrs": {k: copy.deepcopy(getattr(self, k, None)) for k in self._RUNTIME_ATTRS}}

    def _restore_runtime_state(self, s: Dict) -> None:
        self.model.load_state_dict(s["model"])
        self.ema.load_state_dict(s["ema"])
        self.optimizer.load_state_dict(s["optimizer"])
        self.scheduler.load_state_dict(s["scheduler"])
        self.scaler.load_state_dict(s["scaler"])
        for k, v in s["attrs"].items():
            setattr(self, k, v)

    def _apply_checkpoint(self, st: Dict) -> None:
        self.model.load_state_dict(st["model"])
        self.ema.load_state_dict({k: v.to(self.device) for k, v in st["ema"].items()})  # exact key set checked
        self.optimizer.load_state_dict(st["optimizer"])
        self.scheduler.load_state_dict(st["scheduler"])
        self.scaler.load_state_dict(st["scaler"])
        self.update, self.micro = st["update"], st["micro_batches"]
        self.attempted_updates, self.skipped_updates = st["attempted_updates"], st["skipped_updates"]
        self.epoch, self.batch_in_epoch = st["data_position"]["epoch"], st["data_position"]["batch"]
        self.param_steps = [list(g) for g in self._expected_param_steps(st.get("optimizer_history"), st["update"])[0]]
        self._resume_rng = st["rng"]

    # ---- training ----
    def _log(self, rec: Dict) -> None:
        with open(os.path.join(self.out, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(rec, allow_nan=False) + "\n")

    def _count_optimizer_steps(self) -> None:
        """Count calls that reach optimizer.step(). GradScaler.step() skips the call when gradients are
        non-finite; its return value cannot tell (AdamW.step() returns None either way). Installed as an
        instance attribute, so it counts each executed step exactly once, also for subclasses."""
        if self.param_steps is None:  # a Trainer assembled without __init__ (diagnostic drivers)
            self.param_steps = [[0] * len(g["params"]) for g in self.optimizer.param_groups]
        if getattr(self.optimizer.step, "_r2r2r_counter", False):
            return
        inner = self.optimizer.step

        def step(*args, **kwargs):
            # AdamW steps exactly the parameters that have a gradient: that is the optimizer history
            stepped = [[q.grad is not None for q in g["params"]] for g in self.optimizer.param_groups]
            out = inner(*args, **kwargs)
            self.optimizer_calls += 1
            for counts, flags in zip(self.param_steps, stepped):
                for j, f in enumerate(flags):
                    counts[j] += f
            return out
        step._r2r2r_counter = True
        if hasattr(inner, "_wrapped_by_lr_sched"):  # keep the LR scheduler's step-order bookkeeping intact
            step._wrapped_by_lr_sched = inner._wrapped_by_lr_sched
        self.optimizer.step = step

    def train(self, stop_after: Optional[int] = None) -> Dict:
        if self.unusable:
            raise CheckpointError(f"trainer state is inconsistent, refusing to train: {self.unusable}")
        s = self.sched
        total, accum = s["total_updates"], s["accum_micro_batches"]
        stop = total if stop_after is None else min(total, stop_after)
        self._count_optimizer_steps()
        stream = self._loader(self.ds_train, TrainBatchStream(self.sampler, self.epoch, self.batch_in_epoch), persistent=True)
        it = iter(stream)
        if getattr(self, "_resume_rng", None) is not None:
            _set_rng_state(self._resume_rng, self.device)
            self._resume_rng = None
        self.model.train()
        params = list(self.model.parameters())
        t_start, t_data, n_updates, n_skipped = time.time(), 0.0, 0, 0
        while self.update < stop:
            t0 = time.time()
            self.optimizer.zero_grad(set_to_none=True)
            raws, wait = [], 0.0
            calls_before = self.optimizer_calls
            for m in range(accum):
                tw = time.time()
                batch = next(it)
                wait += time.time() - tw
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
                    loss = self.model(batch)
                raw = float(loss.item())
                if not math.isfinite(raw):
                    raise FloatingPointError(f"non-finite loss {raw} at update {self.update + 1}, micro-batch {m}")
                raws.append(raw)
                grad_norm = self.scaler(loss / accum, self.optimizer, parameters=params,
                                        update_grad=(m == accum - 1), clip_grad=s["grad_clip"])
                self._advance_position()
            self.attempted_updates += 1
            t_data += wait
            lr_used = self.optimizer.param_groups[0]["lr"]
            steps = self.optimizer_calls - calls_before
            if steps == 0:  # GradScaler found non-finite gradients and did not call optimizer.step()
                self.skipped_updates += 1
                self.consecutive_skipped += 1
                n_skipped += 1
                self._log({"type": "train_skipped", "update": self.update + 1, "attempt": self.attempted_updates,
                           "micro_batches": self.micro, "epoch": self.epoch, "batch_in_epoch": self.batch_in_epoch,
                           "lr": lr_used, "loss_raw_micro": raws, "grad_norm_before_clip": _jsonable(grad_norm),
                           "grad_scale_after": _jsonable(self.scaler._scaler.get_scale()),
                           "consecutive_skipped": self.consecutive_skipped, "skipped_total": self.skipped_updates})
                if (self.consecutive_skipped >= MAX_CONSECUTIVE_SKIPPED_UPDATES
                        or self.skipped_updates >= MAX_SKIPPED_UPDATES):
                    raise SkippedStepError(
                        f"optimizer step skipped {self.consecutive_skipped} times in a row ({self.skipped_updates} "
                        f"in total) before update {self.update + 1}; aborting (last durable checkpoint is unchanged)")
                continue
            if steps != 1:
                raise RuntimeError(f"optimizer.step() ran {steps} times in one accumulation group")
            self.consecutive_skipped = 0
            self.scheduler.step()
            self.ema.update(self.model)
            self.update += 1
            n_updates += 1
            if self.update % s["log_every_updates"] == 0 or self.update == stop:
                self._log({"type": "train", "update": self.update, "attempt": self.attempted_updates,
                           "micro_batches": self.micro, "epoch": self.epoch,
                           "batch_in_epoch": self.batch_in_epoch, "lr_used": lr_used,
                           "loss_raw_mean": sum(raws) / len(raws), "loss_raw_micro": raws,
                           "grad_norm_before_clip": _jsonable(grad_norm), "skipped_total": self.skipped_updates,
                           "update_s": round(time.time() - t0, 4),
                           "data_wait_s": round(wait, 4)})
            if self.update % s["val_every_updates"] == 0 or self.update == total:
                self._log({"type": "val_subset", "update": self.update, **self.evaluate(self.val_subset)})
            if self.update % s["full_val_every_updates"] == 0 or self.update == total:
                self._log({"type": "val_full", "update": self.update, **self.evaluate(self.val_full)})
            if self.update % s["inference_export_every_updates"] == 0 or self.update == total:
                tc = time.time()
                self._log({"type": "export", "update": self.update, "path": self.save_inference(),
                           "seconds": round(time.time() - tc, 2)})
            if self.update % s["full_ckpt_every_updates"] == 0 or self.update == stop:
                tc = time.time()
                self._log({"type": "checkpoint", "update": self.update, "path": self.save_full(),
                           "seconds": round(time.time() - tc, 2)})
        elapsed = time.time() - t_start
        if hasattr(it, "_shutdown_workers"):
            it._shutdown_workers()
        del it, stream
        return {"updates_this_session": n_updates, "final_update": self.update, "elapsed_s": round(elapsed, 2),
                "data_wait_s": round(t_data, 2), "skipped_updates_this_session": n_skipped,
                "attempted_updates_total": self.attempted_updates, "skipped_updates_total": self.skipped_updates,
                "finished": self.update == total}


def _is_real(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _same_lr(x, expected: float) -> bool:
    return _is_real(x) and math.isfinite(x) and math.isclose(x, expected, rel_tol=1e-12, abs_tol=1e-30)


def _adam_entry_problem(s, param, k: int, need) -> Optional[str]:
    """Why the AdamW state s cannot be the state of `param` after k steps, or None."""
    if not isinstance(s, dict) or set(s) != need:
        return f"keys {sorted(s) if isinstance(s, dict) else type(s).__name__} != {sorted(need)}"
    step = s["step"]
    if torch.is_tensor(step):
        step = float(step) if step.numel() == 1 and not step.is_complex() else None
    if not (_is_real(step) and math.isfinite(step) and step == k):
        return f"step {s['step']!r} != {k} recorded updates"
    for m in sorted(need - {"step"}):
        t = s[m]
        if not torch.is_tensor(t) or t.shape != param.shape or t.dtype != param.dtype:
            return (f"{m}: expected {param.dtype} {tuple(param.shape)}, got "
                    f"{(t.dtype, tuple(t.shape)) if torch.is_tensor(t) else type(t).__name__}")
        if not torch.isfinite(t).all():
            return f"{m}: non-finite values"
        if m != "exp_avg" and (t < 0).any():
            return f"{m}: negative second-moment values"
    return None


def _check_byte_state(s, ref: torch.Tensor, gen: torch.Generator) -> None:
    if not torch.is_tensor(s) or s.dtype != torch.uint8 or s.shape != ref.shape:
        raise ValueError(f"expected uint8 {tuple(ref.shape)}, got "
                         f"{(s.dtype, tuple(s.shape)) if torch.is_tensor(s) else type(s).__name__}")
    gen.set_state(s)


def _rng_problems(rng, device) -> List[str]:
    """Each saved RNG payload must be accepted by a scratch generator of its kind; global RNG is untouched."""
    need = {"python", "numpy", "torch"} | ({"cuda"} if device.type == "cuda" else set())
    if not isinstance(rng, dict) or not need <= set(rng) or set(rng) - need - {"cuda"}:
        return [f"checkpoint RNG state incomplete or unexpected "
                f"({sorted(rng) if isinstance(rng, dict) else type(rng).__name__}; needs {sorted(need)})"]
    checks = {"python": lambda s: random.Random().setstate(s),
              "numpy": lambda s: np.random.RandomState().set_state(s),
              "torch": lambda s: _check_byte_state(s, torch.get_rng_state(), torch.Generator())}
    if device.type == "cuda":
        checks["cuda"] = lambda s: _check_byte_state(s, torch.cuda.get_rng_state(device), torch.Generator(device=device))
    p = []
    for k, check in checks.items():
        try:
            check(rng[k])
        except Exception as e:  # noqa: BLE001
            p.append(f"rng.{k} is not a valid state ({type(e).__name__}: {e})")
    return p


def _tensor_dict_problems(name: str, got: Dict, ref: Dict) -> List[str]:
    """Exact key set, shapes, dtypes and finite floating values of a checkpoint tensor dictionary."""
    if not isinstance(got, dict):
        return [f"{name}: not a dictionary"]
    p = []
    missing, extra = sorted(set(ref) - set(got)), sorted(set(got) - set(ref))
    if missing:
        p.append(f"{name}: {len(missing)} missing tensors, e.g. {missing[:3]}")
    if extra:
        p.append(f"{name}: {len(extra)} unexpected tensors, e.g. {extra[:3]}")
    for k in sorted(set(ref) & set(got)):
        v, r = got[k], ref[k]
        if not torch.is_tensor(v) or v.shape != r.shape or v.dtype != r.dtype:
            p.append(f"{name}.{k}: expected {r.dtype} {tuple(r.shape)}, got "
                     f"{(v.dtype, tuple(v.shape)) if torch.is_tensor(v) else type(v).__name__}")
        elif v.is_floating_point() and not torch.isfinite(v).all():
            p.append(f"{name}.{k}: non-finite values")
    return p


def write_run_files(run: Dict) -> None:
    """Self-contained run directory: resolved run record + exact copy of the action statistics."""
    out = run["paths"]["output_dir"]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "run_config.json"), "w") as f:
        json.dump(run, f, indent=1, sort_keys=True, allow_nan=False)
    shutil.copyfile(run["paths"]["action_statistics"], os.path.join(out, "action_statistics.json"))


def truncate_metrics(out: str, update: int) -> None:
    """On resume, keep log records up to the checkpoint's update; keep the original log as a backup."""
    p = os.path.join(out, "metrics.jsonl")
    if not os.path.exists(p):
        return
    lines = open(p).read().splitlines()
    k = 0
    while os.path.exists(f"{p}.before_resume_{k}"):
        k += 1
    shutil.copyfile(p, f"{p}.before_resume_{k}")
    keep = [l for l in lines if json.loads(l)["update"] <= update]
    with open(p, "w") as f:
        f.write("".join(l + "\n" for l in keep))
