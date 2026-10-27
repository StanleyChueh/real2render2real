"""Optimizer-update-based TinyDP training driver (audit M2, M3, M7). TinyDP's sources are not modified.

Reused unchanged from TinyDP: the sim SequenceDataset (via ManifestSimDataset), VideoSampler,
CollateFunction, DiffusionPolicy, add_weight_decay, NativeScalerWithGradNormCount, ModelEMA and the
diffusers cosine scheduler. Replaced: the epoch loop of tinydp/util/engine.py and script/train.py.

Semantics (each differs from the released loop, see docs/policy_training/AUDIT_REPAIR_REPORT.md):
  * one optimizer update = exactly `accum` consecutive micro-batches of a continuous stream that runs
    across epoch boundaries, so no accumulated gradient is ever discarded;
  * the LR scheduler and the EMA advance once per optimizer update; training stops after
    `total_updates` updates; warm-up is counted in updates;
  * logged losses are raw (not divided by `accum`); validation reports the raw mean loss per window;
  * sampler.set_epoch(epoch) is called for every epoch of the stream;
  * every item's randomness (proprio noise, colour jitter) is seeded from (seed, stream, epoch, index),
    so the data stream does not depend on worker count and can be resumed at any micro-batch;
  * validation uses EMA weights under a forked, fixed-seed RNG and does not perturb training RNG;
  * full checkpoints hold model, EMA, optimizer, scheduler, grad-scaler, RNG state, update and data
    position, plus the resolved run record and the action statistics (self-contained for inference).
"""
import glob
import json
import math
import os
import random
import shutil
import time
from typing import Dict, List, Optional

import numpy as np
import torch

CKPT_FORMAT = "r2r2r-dp-ckpt-v2"
TRAIN_STREAM, VAL_STREAM = 0, 1


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
        t0 = time.time()
        raw = None
        if weights == "ema":
            raw = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self.ema.apply_ema(self.model)
        self.model.eval()
        total, n = 0.0, 0
        devs = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devs):
            torch.manual_seed(self.sched["val_seed"])
            if self.device.type == "cuda":
                torch.cuda.manual_seed(self.sched["val_seed"])
            loader = self._loader(self.ds_val, batches, persistent=False)
            with torch.no_grad():
                for batch in loader:
                    batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                    with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
                        loss = self.model(batch)
                    w = batch["proprio"].shape[0]
                    total += float(loss.item()) * w
                    n += w
            del loader
        if raw is not None:
            self.model.load_state_dict(raw)
        self.model.train()
        return {"loss": total / n, "windows": n, "weights": weights, "seconds": round(time.time() - t0, 2)}

    # ---- checkpoints ----
    def _common(self) -> Dict:
        return {"format": CKPT_FORMAT, "update": self.update, "micro_batches": self.micro,
                "run": {k: self.run[k] for k in ("tinydp", "config_sha256", "train_size", "protocol", "launch", "provenance")
                        if k in self.run},
                "action_statistics": self.stats_json}

    def save_full(self) -> str:
        state = self._common()
        state.update(kind="full", model=self.model.state_dict(), ema=self.ema.state_dict(),
                     optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict(),
                     scaler=self.scaler.state_dict(), rng=_rng_state(self.device),
                     data_position={"epoch": self.epoch, "batch": self.batch_in_epoch,
                                    "batches_per_epoch": self.batches_per_epoch})
        path = os.path.join(self.out, f"ckpt_full_u{self.update:07d}.pt")
        _save(path, state)
        for old in sorted(glob.glob(os.path.join(self.out, "ckpt_full_u*.pt")))[:-self.sched["keep_full_checkpoints"]]:
            os.remove(old)
        return path

    def save_inference(self) -> str:
        raw = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.ema.apply_ema(self.model)
        state = self._common()
        state.update(kind="inference", weights="ema", model={k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()})
        self.model.load_state_dict(raw)
        path = os.path.join(self.out, f"policy_ema_u{self.update:07d}.pt")
        _save(path, state)
        return path

    def load_full(self, path: str) -> None:
        st = torch.load(path, map_location="cpu", weights_only=False)
        if st.get("format") != CKPT_FORMAT or st.get("kind") != "full":
            raise ValueError(f"{path} is not a {CKPT_FORMAT} full checkpoint")
        if st["run"]["config_sha256"] != self.run["config_sha256"]:
            raise ValueError("checkpoint was trained with a different resolved configuration")
        if st["action_statistics"] != self.stats_json:
            raise ValueError("checkpoint action statistics differ from this run's statistics")
        if st["data_position"]["batches_per_epoch"] != self.batches_per_epoch:
            raise ValueError("training data differs from the checkpoint (batches per epoch)")
        self.model.load_state_dict(st["model"])
        self.ema.load_state_dict({k: v.to(self.device) for k, v in st["ema"].items()})
        self.optimizer.load_state_dict(st["optimizer"])
        self.scheduler.load_state_dict(st["scheduler"])
        self.scaler.load_state_dict(st["scaler"])
        self.update, self.micro = st["update"], st["micro_batches"]
        self.epoch, self.batch_in_epoch = st["data_position"]["epoch"], st["data_position"]["batch"]
        self._resume_rng = st["rng"]

    # ---- training ----
    def _log(self, rec: Dict) -> None:
        with open(os.path.join(self.out, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")

    def train(self, stop_after: Optional[int] = None) -> Dict:
        s = self.sched
        total, accum = s["total_updates"], s["accum_micro_batches"]
        stop = total if stop_after is None else min(total, stop_after)
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
            scale_before = self.scaler._scaler.get_scale()
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
            lr_used = self.optimizer.param_groups[0]["lr"]
            skipped = self.scaler._scaler.get_scale() < scale_before  # GradScaler skipped a non-finite step
            n_skipped += int(skipped)
            self.scheduler.step()
            self.ema.update(self.model)
            self.update += 1
            n_updates += 1
            t_data += wait
            if self.update % s["log_every_updates"] == 0 or self.update == stop:
                self._log({"type": "train", "update": self.update, "micro_batches": self.micro, "epoch": self.epoch,
                           "batch_in_epoch": self.batch_in_epoch, "lr_used": lr_used,
                           "loss_raw_mean": sum(raws) / len(raws), "loss_raw_micro": raws,
                           "grad_norm_before_clip": float(grad_norm), "optimizer_step_skipped": bool(skipped),
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
                "data_wait_s": round(t_data, 2), "optimizer_steps_skipped": n_skipped, "finished": self.update == total}


def write_run_files(run: Dict) -> None:
    """Self-contained run directory: resolved run record + exact copy of the action statistics."""
    out = run["paths"]["output_dir"]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "run_config.json"), "w") as f:
        json.dump(run, f, indent=1, sort_keys=True)
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
