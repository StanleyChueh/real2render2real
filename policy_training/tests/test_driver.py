"""Audit M2/M3/M7: update-based schedule, EMA, accumulation, raw losses, validation, checkpoint/resume."""
import copy
import json
import os
import types

import pytest
import torch

import train_r2r2r_dp
from r2r2r_dp import driver
from r2r2r_dp.protocol import resolve
from run_fixtures import make_run_fixture


# ---------- released TinyDP loop: reproduce the audit's counterexample (inherited defect) ----------

class _Scalar(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, batch):
        return ((self.w - batch["x"]) ** 2).mean()


def test_released_engine_counterexample(monkeypatch):
    """6 micro-batches, accum 4: 1 optimizer step, 6 scheduler steps, 6 EMA updates, a dropped tail, val/4."""
    from tinydp.util import engine
    from tinydp.util.misc import NativeScalerWithGradNormCount
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    model = _Scalar()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    steps = {"opt": 0, "ema": 0}
    orig = opt.step

    def counted_step(*a, **k):
        steps["opt"] += 1
        return orig(*a, **k)
    opt.step = counted_step
    ema = types.SimpleNamespace(update=lambda m: steps.__setitem__("ema", steps["ema"] + 1))
    args = types.SimpleNamespace(trainer_cfg=types.SimpleNamespace(accum_iter=4))
    data = [{"x": torch.tensor([0.1 * i])} for i in range(6)]
    engine.train_one_epoch(model, data, opt, sched, torch.device("cpu"), 0, NativeScalerWithGradNormCount(),
                           args=args, ema=ema)
    assert steps == {"opt": 1, "ema": 6} and sched.last_epoch == 6
    assert model.w.grad is not None and model.w.grad.abs().item() > 0  # tail gradient left unapplied
    w1 = model.w.detach().clone()
    # next epoch: zero_grad at its start drops the tail; the step uses only this epoch's 4 micro-batches
    engine.train_one_epoch(model, [{"x": torch.tensor([0.95])}] * 4, opt, sched, torch.device("cpu"), 1,
                           NativeScalerWithGradNormCount(), args=args, ema=ema)
    assert model.w.detach().item() == pytest.approx((w1 - 0.1 * 2 * (w1 - 0.95)).item())
    raw = float(((model.w.detach() - 0.9) ** 2).mean())
    stats = engine.train_one_epoch(model, [{"x": torch.tensor([0.9])}], opt, sched, torch.device("cpu"), 0,
                                   NativeScalerWithGradNormCount(), validate=True, args=args)
    assert stats["loss"] == pytest.approx(raw / 4)  # validation loss reported divided by accum_iter


def test_released_multiepoch_loader_reshuffles_each_epoch():
    """Corrects 02_training_protocol.md and the audit: _RepeatSampler calls set_epoch, order changes per epoch."""
    from tinydp.dataset.image_dataset import VideoSampler
    from tinydp.util.misc import MultiEpochsDataLoader

    class D(torch.utils.data.Dataset):
        def __len__(self):
            return 64

        def __getitem__(self, i):
            return i
    dl = MultiEpochsDataLoader(D(), batch_sampler=VideoSampler(D(), batch_size=2, sequence_length=4), num_workers=0,
                               collate_fn=lambda b: b)
    orders = [[b[0] for b in dl] for _ in range(3)]
    assert orders[0] != orders[1] != orders[2]
    ref = VideoSampler(D(), batch_size=2, sequence_length=4)
    for e in range(3):
        ref.set_epoch(e)
        assert orders[e] == [b[0] for b in ref]


# ---------- v2 driver ----------

def _trainer(fx, tmp, n=3, **kw):
    run = resolve(fx["protocol"], n, os.path.join(str(tmp), "out"))
    driver.write_run_files(run)
    return driver.Trainer(run, device="cpu", **kw)


@pytest.fixture(scope="module")
def fx9(tmp_path_factory):
    # 30 micro-batches per epoch (not a multiple of 4); 9 updates = 36 micro-batches crosses an epoch boundary
    return make_run_fixture(tmp_path_factory.mktemp("d9"), total_updates=9, warmup_updates=3, val_every_updates=100,
                            full_val_every_updates=100, full_ckpt_every_updates=100, inference_export_every_updates=100)


def test_update_semantics(fx9, tmp_path, monkeypatch):
    from diffusers.optimization import get_scheduler
    from tinydp.dataset.image_dataset import VideoSampler
    t = _trainer(fx9, tmp_path)
    assert t.batches_per_epoch == 30
    keys, train_losses, ema_calls = [], [], []
    orig_get = driver.SeededItems.__getitem__
    monkeypatch.setattr(driver.SeededItems, "__getitem__", lambda self, k: (keys.append(k), orig_get(self, k))[1])
    orig_fwd = t.model.forward
    t.model.forward = lambda b: (lambda l: (train_losses.append(float(l)) if t.model.training else None, l)[1])(orig_fwd(b))
    orig_ema = t.ema.update
    t.ema.update = lambda m: (ema_calls.append(1), orig_ema(m))[1]
    summary = t.train()
    assert summary["final_update"] == 9 and summary["optimizer_steps_skipped"] == 0
    assert t.scheduler.last_epoch == 9 and len(ema_calls) == 9
    adam_steps = {int(s["step"]) for s in t.optimizer.state.values()}
    assert adam_steps == {9}
    assert (t.micro, t.epoch, t.batch_in_epoch) == (36, 1, 6)
    # micro-batch stream = epoch 0 (set_epoch(0)) batches 0..29, then epoch 1 (set_epoch(1)) batches 0..5
    ref = VideoSampler(t.ds_train, batch_size=2, sequence_length=4, seed=42)
    expected = []
    for e, nb in ((0, 30), (1, 6)):
        ref.set_epoch(e)
        expected += [(driver.TRAIN_STREAM, e, i) for b in list(iter(ref))[:nb] for i in b]
    assert keys[:len(expected)] == expected
    # LR counted in optimizer updates (warm-up 3 updates, cosine over 9)
    probe = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=2e-4)
    sch = get_scheduler("cosine", optimizer=probe, num_warmup_steps=3, num_training_steps=9)
    want = []
    for _ in range(9):
        want.append(probe.param_groups[0]["lr"])
        sch.step()
    recs = [json.loads(l) for l in open(os.path.join(t.out, "metrics.jsonl"))]
    tr = [r for r in recs if r["type"] == "train"]
    assert [r["update"] for r in tr] == list(range(1, 10))
    assert [r["lr_used"] for r in tr] == pytest.approx(want)
    # raw (not divided by accum) losses, 4 per update, exactly the model outputs
    assert [x for r in tr for x in r["loss_raw_micro"]] == train_losses[:36]
    assert all(len(r["loss_raw_micro"]) == 4 for r in tr)
    assert {r["type"] for r in recs} == {"train", "val_subset", "val_full", "export", "checkpoint"}


def test_validation_is_deterministic_raw_and_uses_ema(fx9, tmp_path):
    t = _trainer(fx9, tmp_path)
    t.train(stop_after=2)
    raw_params = {k: v.clone() for k, v in t.model.state_dict().items()}
    seen = []
    orig = t.model.forward

    def fwd(b):
        ema_now = all(torch.equal(p, t.ema.ema[n]) for n, p in t.model.named_parameters())
        loss = orig(b)
        seen.append((float(loss), b["proprio"].shape[0], ema_now))
        return loss
    t.model.forward = fwd
    rng = torch.get_rng_state()
    a = t.evaluate(t.val_subset)
    b = t.evaluate(t.val_subset)
    a.pop("seconds"), b.pop("seconds")
    assert a == b and a["weights"] == "ema"
    n = len(seen) // 2
    assert all(e for _, _, e in seen)
    assert a["loss"] == pytest.approx(sum(l * w for l, w, _ in seen[:n]) / sum(w for _, w, _ in seen[:n]))
    assert a["windows"] == sum(w for _, w, _ in seen[:n]) == 4
    assert torch.equal(rng, torch.get_rng_state())  # validation does not perturb the training RNG
    assert all(torch.equal(raw_params[k], v) for k, v in t.model.state_dict().items())  # raw weights restored
    assert len(t.val_full) == 20 and sum(len(b) for b in t.val_full) == 2 * 20 * 4


@pytest.fixture(scope="module")
def fx6(tmp_path_factory):
    return make_run_fixture(tmp_path_factory.mktemp("d6"))  # 6 updates, val every 2, ckpt/export every 3


def _cli(fx, out, *extra):
    return train_r2r2r_dp.main(["--protocol", fx["protocol_path"], "--train-size", "3", "--output-dir", out,
                                "--device", "cpu", *extra])


def _records(out):
    keep = ("train", "val_subset", "val_full")
    return [{k: v for k, v in json.loads(l).items() if k not in ("update_s", "data_wait_s", "seconds")}
            for l in open(os.path.join(out, "metrics.jsonl")) if json.loads(l)["type"] in keep]


def test_interrupted_resume_equals_uninterrupted(fx6, tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    sa = _cli(fx6, a)
    assert sa["finished"] and sa["final_update"] == 6
    # interrupted run: crashes after update 5 (its u5 checkpoint lost), resumes from u3 with 2 spawn workers
    sb = _cli(fx6, b, "--stop-after-updates", "5")
    assert sb["final_update"] == 5 and not sb["finished"]
    os.remove(os.path.join(b, "ckpt_full_u0000005.pt"))
    sr = _cli(fx6, b, "--resume", "--num-workers", "2")
    assert sr["updates_this_session"] == 3 and sr["final_update"] == 6 and sr["finished"]
    A = torch.load(os.path.join(a, "ckpt_full_u0000006.pt"), weights_only=False)
    B = torch.load(os.path.join(b, "ckpt_full_u0000006.pt"), weights_only=False)
    for k in ("model", "ema"):
        assert A[k].keys() == B[k].keys() and all(torch.equal(A[k][n], B[k][n]) for n in A[k])
    for (ka, va), (kb, vb) in zip(A["optimizer"]["state"].items(), B["optimizer"]["state"].items()):
        assert all(torch.equal(va[x], vb[x]) for x in va)
    assert A["scheduler"] == B["scheduler"] and A["data_position"] == B["data_position"]
    assert (A["update"], A["micro_batches"]) == (B["update"], B["micro_batches"]) == (6, 24)
    assert _records(a) == _records(b)  # logs of the crashed segment beyond u3 were truncated and re-run
    assert os.path.exists(os.path.join(b, "metrics.jsonl.before_resume_0"))
    hist = [json.loads(l) for l in open(os.path.join(b, "resume_history.jsonl"))]
    assert hist[0]["checkpoint"].endswith("ckpt_full_u0000003.pt") and hist[0]["runtime"]["num_workers_override"] == 2


def test_resume_rejects_changed_configuration(fx6, tmp_path):
    out = str(tmp_path / "r")
    _cli(fx6, out, "--stop-after-updates", "3")
    p = copy.deepcopy(fx6["protocol"])
    p["tinydp"]["optimizer_cfg.lr"] = 1e-4
    pp = tmp_path / "p.json"
    pp.write_text(json.dumps(p))
    with pytest.raises(SystemExit):
        train_r2r2r_dp.main(["--protocol", str(pp), "--train-size", "3", "--output-dir", out, "--device", "cpu", "--resume"])
    run = resolve(p, 3, out)
    t = driver.Trainer(run, device="cpu")
    with pytest.raises(ValueError, match="different resolved configuration"):
        t.load_full(os.path.join(out, "ckpt_full_u0000003.pt"))
    with pytest.raises(SystemExit):  # resume without any checkpoint
        _cli(fx6, str(tmp_path / "empty"), "--resume")


def test_checkpoint_contents_and_cadence(fx6, tmp_path):
    out = str(tmp_path / "c")
    _cli(fx6, out)
    st = torch.load(os.path.join(out, "ckpt_full_u0000006.pt"), weights_only=False)
    assert st["format"] == driver.CKPT_FORMAT and st["kind"] == "full"
    assert {"model", "ema", "optimizer", "scheduler", "scaler", "rng", "data_position", "update", "micro_batches",
            "run", "action_statistics"} <= set(st)
    assert st["action_statistics"] == json.load(open(os.path.join(out, "action_statistics.json")))
    rc = json.load(open(os.path.join(out, "run_config.json")))
    assert st["run"]["config_sha256"] == rc["config_sha256"]
    assert rc["runtime"] == {"device": "cpu", "num_workers_override": None, "stop_after_updates": None}
    recs = [json.loads(l) for l in open(os.path.join(out, "metrics.jsonl"))]
    at = lambda typ: [r["update"] for r in recs if r["type"] == typ]
    assert at("val_subset") == [2, 4, 6] and at("val_full") == [6]
    assert at("checkpoint") == [3, 6] and at("export") == [3, 6]


def test_item_seeding_is_keyed_and_leaves_caller_rng_untouched(fx6, tmp_path):
    """Regression: with num_workers=0 per-item seeding used to overwrite the training RNG (diffusion noise)."""
    import random
    import numpy as np
    t = _trainer(fx6, tmp_path)
    items = driver.SeededItems(t.ds_train, 0)
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    before = (torch.get_rng_state(), np.random.get_state()[1].copy(), random.getstate())
    x1 = items[(driver.TRAIN_STREAM, 0, 5)]
    after = (torch.get_rng_state(), np.random.get_state()[1].copy(), random.getstate())
    assert torch.equal(before[0], after[0]) and np.array_equal(before[1], after[1]) and before[2] == after[2]
    x2 = items[(driver.TRAIN_STREAM, 0, 5)]
    x3 = items[(driver.TRAIN_STREAM, 1, 5)]
    assert all(torch.equal(x1[k], x2[k]) for k in x1)  # same key -> same augmentation and proprio noise
    assert not torch.equal(x1["proprio"], x3["proprio"])  # another epoch -> fresh noise
