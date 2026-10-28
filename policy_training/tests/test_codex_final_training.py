"""Codex final audit F1, F6, F7: skipped AMP steps, transactional validation/export, strict checkpoint loading.

Each test failed on 43ab363 (see docs/policy_training/CODEX_FINDINGS_REPAIR.md) and uses the real
Trainer on the synthetic CPU fixture; GradScaler skips are produced by a real torch.amp.GradScaler('cpu')
seeing an Inf gradient, not by mocking its result.
"""
import json
import os

import pytest
import torch

from r2r2r_dp import driver
from r2r2r_dp.protocol import resolve
from run_fixtures import make_run_fixture


@pytest.fixture(scope="module")
def fx4(tmp_path_factory):
    # 30 micro-batches per epoch; validation/export only at the end so tests stay fast
    return make_run_fixture(tmp_path_factory.mktemp("cf4"), total_updates=4, warmup_updates=2, val_every_updates=100,
                            full_val_every_updates=100, full_ckpt_every_updates=2, inference_export_every_updates=100)


def _trainer(fx, out, bad_attempts=(), always_bad=False):
    run = resolve(fx["protocol"], 3, str(out))
    driver.write_run_files(run)
    t = driver.Trainer(run, device="cpu")
    t.scaler._scaler = torch.amp.GradScaler("cpu")  # enabled on CPU: real found-inf / skip logic
    p = next(t.model.parameters())
    p.register_hook(lambda g: torch.full_like(g, float("inf"))
                    if always_bad or t.attempted_updates + 1 in bad_attempts else g)
    return t


def _lrs(out, typ="train"):
    return [json.loads(l) for l in open(os.path.join(out, "metrics.jsonl")) if json.loads(l)["type"] == typ]


# ---------- F1: skipped optimizer steps ----------

def test_skipped_step_advances_nothing_but_data(fx4, tmp_path):
    ref = _trainer(fx4, tmp_path / "ref")
    ref.train()
    t = _trainer(fx4, tmp_path / "skip", bad_attempts=(2,))
    s = t.train()
    assert s["finished"] and s["final_update"] == 4
    assert (t.update, t.attempted_updates, t.skipped_updates, t.optimizer_calls) == (4, 5, 1, 4)
    assert t.scheduler.last_epoch == 4 and {int(v["step"]) for v in t.optimizer.state.values()} == {4}
    assert t.micro == 4 * 5 and t.epoch * t.batches_per_epoch + t.batch_in_epoch == t.micro
    assert t.scaler._scaler.get_scale() == 65536.0 / 2
    skipped = _lrs(t.out, "train_skipped")
    assert [(r["update"], r["attempt"], r["grad_norm_before_clip"]) for r in skipped] == [(2, 2, "inf")]
    # learning rates per *successful* update are the reference schedule: the skip did not consume one
    assert [r["lr_used"] for r in _lrs(t.out)] == [r["lr_used"] for r in _lrs(ref.out)]
    assert [r["update"] for r in _lrs(t.out)] == [1, 2, 3, 4] and [r["attempt"] for r in _lrs(t.out)] == [1, 3, 4, 5]
    ck = torch.load(os.path.join(t.out, "ckpt_full_u0000004.pt"), weights_only=False)
    assert (ck["update"], ck["attempted_updates"], ck["skipped_updates"], ck["micro_batches"]) == (4, 5, 1, 20)
    assert t.checkpoint_problems(ck) == []


def test_zero_learning_rate_step_is_a_successful_update(fx4, tmp_path):
    t = _trainer(fx4, tmp_path / "lr0")
    t.train(stop_after=1)
    assert _lrs(t.out)[0]["lr_used"] == 0.0  # diffusers warm-up, as released
    assert (t.update, t.optimizer_calls, t.skipped_updates, t.scheduler.last_epoch) == (1, 1, 0, 1)


def test_persistent_overflow_aborts_without_counting(fx4, tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "MAX_CONSECUTIVE_SKIPPED_UPDATES", 3)
    monkeypatch.setattr(driver, "MAX_SKIPPED_UPDATES", 10)  # keeps the test bounded even if the first limit regresses
    t = _trainer(fx4, tmp_path / "abort", always_bad=True)
    with pytest.raises(driver.SkippedStepError, match="3 times in a row"):
        t.train()
    assert (t.update, t.attempted_updates, t.skipped_updates, t.optimizer_calls) == (0, 3, 3, 0)
    assert t.scheduler.last_epoch == 0 and not t.optimizer.state
    assert not [f for f in os.listdir(t.out) if f.startswith("ckpt_full")]


def test_resume_after_skip_equals_uninterrupted(fx4, tmp_path):
    a = _trainer(fx4, tmp_path / "a", bad_attempts=(2,))
    a.train()
    b = _trainer(fx4, tmp_path / "b", bad_attempts=(2,))
    b.train(stop_after=2)  # attempts 1, 2 (skipped), 3 -> checkpoint u2 with 12 micro-batches consumed
    c = _trainer(fx4, tmp_path / "b")
    c.load_full(os.path.join(c.out, "ckpt_full_u0000002.pt"))
    assert (c.update, c.attempted_updates, c.skipped_updates, c.micro) == (2, 3, 1, 12)
    c.train()
    A = torch.load(os.path.join(a.out, "ckpt_full_u0000004.pt"), weights_only=False)
    C = torch.load(os.path.join(c.out, "ckpt_full_u0000004.pt"), weights_only=False)
    for k in ("model", "ema"):
        assert all(torch.equal(A[k][n], C[k][n]) for n in A[k])
    for va, vc in zip(A["optimizer"]["state"].values(), C["optimizer"]["state"].values()):
        assert all(torch.equal(va[x], vc[x]) for x in va)
    assert A["scheduler"] == C["scheduler"] and A["scaler"] == C["scaler"] and A["data_position"] == C["data_position"]


# ---------- F6: transactional validation and export ----------

@pytest.fixture(scope="module")
def trained(tmp_path_factory, fx4):
    t = _trainer(fx4, tmp_path_factory.mktemp("cf_trained") / "out")
    t.train(stop_after=2)
    return t


def _snapshot(t):
    return {k: v.clone() for k, v in t.model.state_dict().items()}


def _same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_validation_failure_restores_weights_and_mode(trained, mode):
    t = trained
    t.model.train(mode == "train")
    raw = _snapshot(t)
    orig = t.model.forward
    t.model.forward = lambda b: (_ for _ in ()).throw(RuntimeError("injected"))
    try:
        with pytest.raises(RuntimeError, match="injected"):
            t.evaluate(t.val_subset)
    finally:
        t.model.forward = orig
    assert _same(raw, _snapshot(t)) and t.model.training == (mode == "train")


def test_nonfinite_and_empty_validation_raise(trained):
    t = trained
    t.model.train()
    raw = _snapshot(t)
    orig = t.model.forward
    t.model.forward = lambda b: orig(b) * float("nan")
    try:
        with pytest.raises(FloatingPointError, match="non-finite loss"):
            t.evaluate(t.val_subset)
    finally:
        t.model.forward = orig
    assert _same(raw, _snapshot(t)) and t.model.training
    with pytest.raises(ValueError, match="no batches"):
        t.evaluate([])


def test_nonfinite_validation_aborts_training(fx4, tmp_path):
    t = _trainer(fx4, tmp_path / "nanval")
    orig = t.model.forward
    t.model.forward = lambda b: orig(b) if t.model.training else orig(b) * float("nan")
    with pytest.raises(FloatingPointError):
        t.train()  # validation at the last update
    assert not [r for r in map(json.loads, open(os.path.join(t.out, "metrics.jsonl"))) if r["type"].startswith("val")]


def test_export_failure_restores_raw_weights(trained, monkeypatch):
    t = trained
    raw = _snapshot(t)
    monkeypatch.setattr(t.ema, "apply_ema", lambda m: (_ for _ in ()).throw(RuntimeError("injected")))
    with pytest.raises(RuntimeError, match="injected"):
        t.save_inference()
    assert _same(raw, _snapshot(t))
    monkeypatch.undo()
    name = next(iter(t.ema.ema))
    saved = t.ema.ema[name].clone()
    t.ema.ema[name][...] = float("nan")
    try:
        with pytest.raises(FloatingPointError, match="non-finite weights"):
            t.save_inference()
    finally:
        t.ema.ema[name].copy_(saved)
    assert _same(raw, _snapshot(t))


# ---------- F7: strict checkpoint validation ----------

@pytest.fixture(scope="module")
def ckpt(trained):
    return torch.load(os.path.join(trained.out, "ckpt_full_u0000002.pt"), weights_only=False)


def _bad(c, which):
    import copy
    c = copy.deepcopy(c)
    k = next(iter(c["ema"]))
    if which == "ema_missing":
        del c["ema"][k]
    elif which == "ema_extra":
        c["ema"]["not.a.parameter"] = torch.zeros(1)
    elif which == "ema_shape":
        c["ema"][k] = c["ema"][k].flatten()[:1]
    elif which == "ema_dtype":
        c["ema"][k] = c["ema"][k].double()
    elif which == "ema_nan":
        c["ema"][k] = c["ema"][k].clone().fill_(float("nan"))
    elif which == "model_missing":
        del c["model"][next(iter(c["model"]))]
    elif which == "model_inf":
        k = next(n for n, v in c["model"].items() if v.is_floating_point())
        c["model"][k] = c["model"][k].clone().fill_(float("inf"))
    elif which == "micro":
        c["micro_batches"] += 1
    elif which == "micro_with_matching_position":  # only micro == accum x attempted can catch this
        c["micro_batches"] += 4
        c["data_position"]["batch"] += 4
    elif which == "attempted":
        c["attempted_updates"] += 1
    elif which == "position":
        c["data_position"]["batch"] += 1
    elif which == "scheduler":
        c["scheduler"]["last_epoch"] += 1
    elif which == "adam_step":
        st = next(iter(c["optimizer"]["state"].values()))
        st["step"] = st["step"] + 1
    elif which == "order":
        c["train_order_sha256"] = "0" * 64
    elif which == "scaler":
        c["scaler"]["scale"] = float("nan")
    elif which == "format":
        c["format"] = "r2r2r-dp-ckpt-v2"
    elif which == "counter_bool":
        c["update"] = True
    return c


@pytest.mark.parametrize("which", ["ema_missing", "ema_extra", "ema_shape", "ema_dtype", "ema_nan", "model_missing",
                                   "model_inf", "micro", "micro_with_matching_position", "attempted", "position", "scheduler", "adam_step", "order",
                                   "scaler", "format", "counter_bool"])
def test_corrupt_checkpoint_rejected_before_any_state_changes(fx4, ckpt, tmp_path, which):
    t = _trainer(fx4, tmp_path / "load")
    before = _snapshot(t), {k: v.clone() for k, v in t.ema.ema.items()}, t.optimizer.state_dict()["state"]
    path = tmp_path / "bad.pt"
    torch.save(_bad(ckpt, which), path)
    with pytest.raises(driver.CheckpointError):
        t.load_full(str(path))
    assert _same(before[0], _snapshot(t)) and _same(before[1], t.ema.ema) and not t.optimizer.state_dict()["state"]
    assert (t.update, t.micro, t.attempted_updates) == (0, 0, 0)


def test_valid_checkpoint_loads(fx4, ckpt, tmp_path):
    t = _trainer(fx4, tmp_path / "ok")
    assert t.checkpoint_problems(ckpt) == []
    path = tmp_path / "ok.pt"
    torch.save(ckpt, path)
    t.load_full(str(path))
    assert _same(ckpt["ema"], t.ema.ema) and t.update == 2
