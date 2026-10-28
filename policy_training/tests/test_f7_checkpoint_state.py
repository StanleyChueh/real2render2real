"""Codex re-audit N1 (F7 PARTIAL): strict optimizer / scheduler / grad-scaler / RNG validation on resume.

Each corruption case below was accepted by Trainer.checkpoint_problems at 986a3fa (see
docs/policy_training/F7_CHECKPOINT_REPAIR.md, reproduction_on_986a3fa/) or failed only after runtime state
had been changed. Real Trainer on the synthetic CPU fixture; no mocking of the optimizer or its state.
"""
import copy
import hashlib
import json
import os

import pytest
import torch

import train_r2r2r_dp
from r2r2r_dp import driver
from r2r2r_dp.protocol import resolve
from run_fixtures import make_run_fixture

TOTAL = 4


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    # full checkpoint after every update; validation/export only at the end
    return make_run_fixture(tmp_path_factory.mktemp("f7"), total_updates=TOTAL, warmup_updates=2,
                            val_every_updates=100, full_val_every_updates=100, full_ckpt_every_updates=1,
                            keep_full_checkpoints=5, inference_export_every_updates=100)


def _trainer(fx, out, enabled_scaler=True, drop_grad=None):
    """drop_grad=(param_name, attempts): that parameter has no gradient in those accumulation groups."""
    run = resolve(fx["protocol"], 3, str(out))
    driver.write_run_files(run)
    t = driver.Trainer(run, device="cpu")
    if enabled_scaler:
        t.scaler._scaler = torch.amp.GradScaler("cpu")  # real scaler state (scale, growth tracker) in checkpoints
    if drop_grad:
        name, attempts = drop_grad
        q = dict(t.model.named_parameters())[name]
        q.register_post_accumulate_grad_hook(
            lambda q: setattr(q, "grad", None) if t.attempted_updates + 1 in attempts else None)
    return t


def _load(path):
    return torch.load(path, weights_only=False)


def _ck(t, u):
    return os.path.join(t.out, f"ckpt_full_u{u:07d}.pt")


@pytest.fixture(scope="module")
def run3(fx, tmp_path_factory):
    """Trainer interrupted after u3 (checkpoints u1..u3) and the uninterrupted reference run to u4."""
    base = tmp_path_factory.mktemp("f7runs")
    ref = _trainer(fx, base / "ref")
    ref.train()
    t = _trainer(fx, base / "int")
    t.train(stop_after=3)
    return {"ref": _load(_ck(ref, TOTAL)), "u3": _load(_ck(t, 3)), "dir": t.out}


def _first_state(st):
    return st["optimizer"]["state"][next(iter(st["optimizer"]["state"]))]


def _corrupt(st, case):
    st = copy.deepcopy(st)
    o, s0 = st["optimizer"], _first_state(st)
    if case == "missing_one_adam_state":  # Codex N1 counterexample
        del o["state"][next(iter(o["state"]))]
    elif case == "missing_all_adam_state":  # claims 3 completed updates without any optimizer history
        o["state"] = {}
    elif case in ("missing_exp_avg", "missing_exp_avg_sq", "missing_step"):
        del s0[case[len("missing_"):]]
    elif case == "extra_state_key":
        s0["max_exp_avg_sq"] = s0["exp_avg_sq"].clone()
    elif case == "exp_avg_shape":
        s0["exp_avg"] = torch.zeros(2, 3)
    elif case == "exp_avg_sq_transposed":
        k = next(i for i, v in o["state"].items() if v["exp_avg_sq"].dim() == 2 and v["exp_avg_sq"].shape[0] != v["exp_avg_sq"].shape[1])
        o["state"][k]["exp_avg_sq"] = o["state"][k]["exp_avg_sq"].t().contiguous()
    elif case == "exp_avg_dtype":
        s0["exp_avg"] = s0["exp_avg"].double()
    elif case == "step_not_scalar":
        s0["step"] = torch.tensor([3.0, 3.0])
    elif case == "step_fractional":
        s0["step"] = torch.tensor(2.5)
    elif case == "step_behind":  # one parameter's Adam step disagrees with the recorded history
        s0["step"] = s0["step"] - 1
    elif case == "step_nan":
        s0["step"] = torch.tensor(float("nan"))
    elif case == "exp_avg_nan":
        s0["exp_avg"] = s0["exp_avg"].clone().fill_(float("nan"))
    elif case == "exp_avg_sq_inf":
        s0["exp_avg_sq"] = s0["exp_avg_sq"].clone().fill_(float("inf"))
    elif case == "exp_avg_sq_negative":
        s0["exp_avg_sq"] = -s0["exp_avg_sq"] - 1.0
    elif case == "unknown_param_id":
        o["state"][10 ** 6] = copy.deepcopy(s0)
    elif case == "group_moved_param":  # same sizes would be needed for the old check; ids moved between groups
        g0, g1 = o["param_groups"]
        g0["params"], g1["params"] = g0["params"][:-1], [g0["params"][-1]] + g1["params"]
    elif case == "group_count":
        o["param_groups"] = o["param_groups"][:1]
    elif case == "group_weight_decay":
        o["param_groups"][0]["weight_decay"], o["param_groups"][1]["weight_decay"] = \
            o["param_groups"][1]["weight_decay"], o["param_groups"][0]["weight_decay"]
    elif case == "group_betas":
        o["param_groups"][0]["betas"] = (0.5, 0.999)
    elif case == "group_amsgrad":
        o["param_groups"][0]["amsgrad"] = True
    elif case == "group_missing_key":
        del o["param_groups"][1]["eps"]
    elif case == "group_lr_nan":
        o["param_groups"][0]["lr"] = float("nan")
    elif case == "group_lr_off_schedule":
        o["param_groups"][1]["lr"] *= 1.5
    elif case == "group_initial_lr":
        o["param_groups"][0]["initial_lr"] = 1.0
    elif case == "not_an_optimizer":
        st["optimizer"] = {"state": []}
    elif case == "history_names_permuted":
        n = st["optimizer_history"]["param_names"][0]
        n[0], n[1] = n[1], n[0]
    elif case == "history_step_too_large":
        st["optimizer_history"]["steps"][0][0] = st["update"] + 1
    elif case == "history_bool":
        st["optimizer_history"]["steps"][0][0] = True
    elif case == "history_all_zero":  # history and state both say "no updates" while the checkpoint claims 3
        st["optimizer_history"]["steps"] = [[0] * len(g) for g in st["optimizer_history"]["steps"]]
        o["state"] = {}
    elif case == "history_missing_without_uniform_state":  # no history: every parameter must carry u steps
        del st["optimizer_history"]
        del o["state"][next(iter(o["state"]))]
    elif case == "scheduler_last_lr_nan":
        st["scheduler"]["_last_lr"] = [float("nan")] * len(st["scheduler"]["_last_lr"])
    elif case == "scheduler_step_count":
        st["scheduler"]["_step_count"] += 1
    elif case == "scheduler_base_lrs":
        st["scheduler"]["base_lrs"] = [1.0, 1.0]
    elif case == "scheduler_missing_key":
        del st["scheduler"]["_last_lr"]
    elif case == "scaler_missing_key":
        del st["scaler"]["_growth_tracker"]
    elif case == "scaler_tracker_range":
        st["scaler"]["_growth_tracker"] = st["scaler"]["growth_interval"]
    elif case == "scaler_empty":
        st["scaler"] = {}
    elif case == "scaler_backoff":
        st["scaler"]["backoff_factor"] = 2.0
    elif case == "rng_torch_length":
        st["rng"]["torch"] = torch.zeros(1, dtype=torch.uint8)
    elif case == "rng_torch_dtype":
        st["rng"]["torch"] = st["rng"]["torch"].float()
    elif case == "rng_python":
        st["rng"]["python"] = (3, (1, 2, 3), None)
    elif case == "rng_numpy":
        st["rng"]["numpy"] = ("PCG64",) + tuple(st["rng"]["numpy"][1:])
    elif case == "rng_unexpected_key":
        st["rng"]["mps"] = torch.zeros(1)
    else:
        raise ValueError(case)
    return st


CASES = ["missing_one_adam_state", "missing_all_adam_state", "missing_exp_avg", "missing_exp_avg_sq", "missing_step",
         "extra_state_key", "exp_avg_shape", "exp_avg_sq_transposed", "exp_avg_dtype", "step_not_scalar",
         "step_fractional", "step_behind", "step_nan", "exp_avg_nan", "exp_avg_sq_inf", "exp_avg_sq_negative",
         "unknown_param_id", "group_moved_param", "group_count", "group_weight_decay", "group_betas", "group_amsgrad",
         "group_missing_key", "group_lr_nan", "group_lr_off_schedule", "group_initial_lr", "not_an_optimizer",
         "history_names_permuted", "history_step_too_large", "history_bool", "history_all_zero",
         "history_missing_without_uniform_state", "scheduler_last_lr_nan", "scheduler_step_count",
         "scheduler_base_lrs", "scheduler_missing_key", "scaler_missing_key", "scaler_tracker_range", "scaler_empty",
         "scaler_backoff", "rng_torch_length", "rng_torch_dtype", "rng_python", "rng_numpy", "rng_unexpected_key"]

EXPECT = {  # the reported problem names the defect, not an unrelated check
    "missing_one_adam_state": "no Adam state for 1 parameters with recorded updates",
    "missing_all_adam_state": "no Adam state for",
    "missing_exp_avg": "invalid Adam state", "missing_step": "invalid Adam state", "exp_avg_shape": "exp_avg: expected",
    "step_behind": "!= 3 recorded updates", "exp_avg_sq_negative": "negative second-moment",
    "group_moved_param": "parameter groups differ", "group_lr_nan": "lr nan is not the scheduled",
    "history_all_zero": "most-updated parameter has 0 steps", "history_names_permuted": "different model parameters",
    "scheduler_last_lr_nan": "scheduler last lr", "scaler_tracker_range": "grad-scaler settings out of range",
    "rng_torch_length": "rng.torch is not a valid state", "rng_python": "rng.python", "rng_numpy": "rng.numpy",
}


def _snapshot(t):
    return ({k: v.clone() for k, v in t.model.state_dict().items()}, {k: v.clone() for k, v in t.ema.ema.items()},
            copy.deepcopy(t.scheduler.state_dict()), copy.deepcopy(t.scaler.state_dict()))


def _assert_fresh(t, snap):
    m, e, sc, sk = snap
    assert all(torch.equal(m[k], v) for k, v in t.model.state_dict().items())
    assert all(torch.equal(e[k], v) for k, v in t.ema.ema.items())
    assert not t.optimizer.state and t.scheduler.state_dict() == sc and t.scaler.state_dict() == sk
    assert (t.update, t.micro, t.attempted_updates, t.skipped_updates, t.optimizer_calls) == (0, 0, 0, 0, 0)
    assert all(k == 0 for g in t.param_steps for k in g) and getattr(t, "_resume_rng", None) is None


@pytest.mark.parametrize("case", CASES)
def test_corrupt_optimizer_state_rejected_before_resume(fx, run3, tmp_path, case):
    t = _trainer(fx, tmp_path / "load")
    snap = _snapshot(t)
    bad = _corrupt(run3["u3"], case)
    problems = t.checkpoint_problems(bad)
    assert problems, f"{case} accepted"
    if case in EXPECT:
        assert any(EXPECT[case] in q for q in problems), problems
    path = tmp_path / "bad.pt"
    torch.save(bad, path)
    with pytest.raises(driver.CheckpointError):
        t.load_full(str(path))
    _assert_fresh(t, snap)  # nothing applied, so no resumed optimization can start from it


def test_valid_checkpoint_has_full_history_and_resumes_bitwise(fx, run3, tmp_path):
    u3, ref = run3["u3"], run3["ref"]
    t = _trainer(fx, tmp_path / "resume")
    assert t.checkpoint_problems(u3) == []
    h = u3["optimizer_history"]
    assert h["param_names"] == t.optimizer_param_names and {k for g in h["steps"] for k in g} == {3}
    assert len(u3["optimizer"]["state"]) == sum(len(g) for g in h["steps"])  # every parameter has Adam state
    t.load_full(os.path.join(run3["dir"], "ckpt_full_u0000003.pt"))
    assert t.param_steps == h["steps"] and t.param_steps is not h["steps"]
    s = t.train()
    assert s["finished"] and s["updates_this_session"] == 1
    C = _load(_ck(t, TOTAL))
    assert t.checkpoint_problems(C) == []
    for k in ("model", "ema"):
        assert all(torch.equal(ref[k][n], C[k][n]) for n in ref[k])
    assert ref["optimizer"]["state"].keys() == C["optimizer"]["state"].keys()
    for i, v in ref["optimizer"]["state"].items():
        assert v.keys() == C["optimizer"]["state"][i].keys() and all(torch.equal(v[x], C["optimizer"]["state"][i][x]) for x in v)
    assert ref["optimizer"]["param_groups"] == C["optimizer"]["param_groups"]
    for k in ("scheduler", "scaler", "data_position", "optimizer_history", "update", "micro_batches", "attempted_updates"):
        assert ref[k] == C[k], k
    assert torch.equal(ref["rng"]["torch"], C["rng"]["torch"]) and ref["rng"]["python"] == C["rng"]["python"]


def test_fresh_zero_update_checkpoint_is_legitimate(fx, tmp_path):
    t = _trainer(fx, tmp_path / "u0")
    path = t.save_full()  # passes the same validation before it is written
    st = _load(path)
    assert st["update"] == 0 and st["optimizer"]["state"] == {} and t.checkpoint_problems(st) == []
    assert {k for g in st["optimizer_history"]["steps"] for k in g} == {0}
    r = _trainer(fx, tmp_path / "u0r")
    r.load_full(path)
    assert r.update == 0 and not r.optimizer.state
    r.train(stop_after=1)
    assert r.update == 1 and r.checkpoint_problems(_load(_ck(r, 1))) == []
    # the same zero-update checkpoint claiming optimizer history is rejected
    forged = copy.deepcopy(st)
    forged["optimizer"]["state"] = copy.deepcopy(_load(_ck(r, 1))["optimizer"]["state"])
    assert any("without recorded updates" in q for q in r.checkpoint_problems(forged))
    forged = copy.deepcopy(st)
    forged["optimizer_history"]["steps"][0][0] = 1
    assert any("most-updated parameter has 1 steps" in q or "[0, update 0]" in q for q in r.checkpoint_problems(forged))


def test_checkpoint_without_history_written_before_the_repair_still_loads(fx, run3, tmp_path):
    """v3 checkpoints of 986a3fa have no optimizer_history; with uniform full Adam state they stay valid."""
    legacy = copy.deepcopy(run3["u3"])
    del legacy["optimizer_history"]
    t = _trainer(fx, tmp_path / "legacy")
    assert t.checkpoint_problems(legacy) == []
    path = tmp_path / "legacy.pt"
    torch.save(legacy, path)
    t.load_full(str(path))
    assert {k for g in t.param_steps for k in g} == {3}
    t.train()
    C = _load(_ck(t, TOTAL))
    assert "optimizer_history" in C and t.checkpoint_problems(C) == []
    assert all(torch.equal(run3["ref"]["model"][n], C["model"][n]) for n in C["model"])


def test_nonuniform_parameter_histories_are_validated_per_parameter(fx, tmp_path):
    """A parameter without a gradient in some updates legitimately carries fewer Adam steps."""
    name = _trainer(fx, tmp_path / "probe").optimizer_param_names[1][-1]
    drop = (name, (1, 3))  # no gradient in attempts 1 and 3 -> 2 of 4 updates
    a = _trainer(fx, tmp_path / "a", drop_grad=drop)
    a.train()
    A = _load(_ck(a, TOTAL))
    g, j = 1, len(a.optimizer_param_names[1]) - 1
    assert A["optimizer_history"]["steps"][g][j] == 2 and {k for gg in A["optimizer_history"]["steps"] for k in gg} == {2, 4}
    pid = A["optimizer"]["param_groups"][g]["params"][j]
    assert int(A["optimizer"]["state"][pid]["step"]) == 2 and a.checkpoint_problems(A) == []
    # u1: the parameter has never been stepped and has no Adam state yet; that is valid too
    U1 = _load(_ck(a, 1))
    assert pid not in U1["optimizer"]["state"] and a.checkpoint_problems(U1) == []
    # interrupted at u2, resumed with the same gradient pattern: identical to the uninterrupted run
    b = _trainer(fx, tmp_path / "b", drop_grad=drop)
    b.train(stop_after=2)
    c = _trainer(fx, tmp_path / "b", drop_grad=drop)
    c.load_full(_ck(c, 2))
    c.train()
    C = _load(_ck(c, TOTAL))
    assert A["optimizer_history"] == C["optimizer_history"]
    assert all(torch.equal(A["model"][n], C["model"][n]) for n in A["model"])
    for i, v in A["optimizer"]["state"].items():
        assert all(torch.equal(v[x], C["optimizer"]["state"][i][x]) for x in v)
    # the history is checked per parameter: uniform steps, a lost history or a dropped entry are all rejected
    bad = copy.deepcopy(A)
    bad["optimizer"]["state"][pid]["step"] = torch.tensor(4.0)
    assert any("!= 2 recorded updates" in q for q in a.checkpoint_problems(bad))
    bad = copy.deepcopy(A)
    del bad["optimizer_history"]  # pre-repair rule (uniform history) cannot hold for this checkpoint
    assert any("!= 4 recorded updates" in q for q in a.checkpoint_problems(bad))
    bad = copy.deepcopy(A)
    del bad["optimizer"]["state"][pid]
    assert any("no Adam state for 1 parameters" in q for q in a.checkpoint_problems(bad))


def test_failure_while_applying_restores_previous_state(fx, run3, tmp_path, monkeypatch):
    t = _trainer(fx, tmp_path / "apply")
    snap = _snapshot(t)
    path = tmp_path / "u3.pt"
    torch.save(run3["u3"], path)

    real, calls = t.scaler.load_state_dict, []

    def fail_once(sd):  # fails while applying the checkpoint, works for the rollback
        calls.append(sd)
        if len(calls) == 1:
            raise RuntimeError("injected")
        return real(sd)
    monkeypatch.setattr(t.scaler, "load_state_dict", fail_once)  # model, EMA, optimizer, scheduler already applied
    with pytest.raises(driver.CheckpointError, match="injected.*previous state restored"):
        t.load_full(str(path))
    monkeypatch.undo()
    _assert_fresh(t, snap)
    assert t.unusable is None


def test_failed_rollback_makes_the_trainer_refuse_to_train(fx, run3, tmp_path, monkeypatch):
    t = _trainer(fx, tmp_path / "norollback")
    path = tmp_path / "u3.pt"
    torch.save(run3["u3"], path)

    def boom(_):
        raise RuntimeError("injected")
    monkeypatch.setattr(t.scaler, "load_state_dict", boom)  # also breaks the rollback
    with pytest.raises(driver.CheckpointError, match="could not be restored"):
        t.load_full(str(path))
    with pytest.raises(driver.CheckpointError, match="refusing to train"):
        t.train()
    assert t.optimizer_calls == 0 and not [f for f in os.listdir(t.out) if f.startswith("ckpt_full")]


def test_unreadable_or_non_dict_checkpoint_is_a_checkpoint_error(fx, tmp_path):
    t = _trainer(fx, tmp_path / "unreadable")
    p = tmp_path / "garbage.pt"
    p.write_bytes(b"not a torch file")
    with pytest.raises(driver.CheckpointError, match="cannot be read"):
        t.load_full(str(p))
    torch.save([1, 2, 3], p)
    with pytest.raises(driver.CheckpointError, match="not a checkpoint dictionary"):
        t.load_full(str(p))


def test_inconsistent_live_state_is_never_published(fx, tmp_path):
    """If the in-memory optimizer state loses a parameter's history mid-run, the next checkpoint is refused:
    nothing is written, older checkpoints are not rotated out and the session does not report completion."""
    t = _trainer(fx, tmp_path / "live")
    t.train(stop_after=2)
    before = {f: hashlib.sha256(open(os.path.join(t.out, f), "rb").read()).hexdigest()
              for f in os.listdir(t.out) if f.startswith("ckpt_full")}
    del t.optimizer.state[t.optimizer.param_groups[0]["params"][0]]  # AdamW silently re-initialises it
    with pytest.raises(driver.CheckpointError, match="refusing to write"):
        t.train()
    after = {f: hashlib.sha256(open(os.path.join(t.out, f), "rb").read()).hexdigest()
             for f in os.listdir(t.out) if f.startswith("ckpt_full")}
    assert after == before and not os.path.exists(f"{_ck(t, 3)}.tmp")
    logs = [json.loads(l) for l in open(os.path.join(t.out, "metrics.jsonl"))]
    assert max(r["update"] for r in logs if r["type"] == "checkpoint") == 2


@pytest.mark.parametrize("case", ["missing_one_adam_state", "missing_exp_avg", "group_lr_nan", "scheduler_last_lr_nan",
                                  "rng_torch_length"])
def test_entrypoint_resume_rejects_and_preserves_valid_checkpoints(fx, tmp_path, case):
    """train_r2r2r_dp.py --resume on a corrupted latest checkpoint: CheckpointError before any update; earlier
    valid checkpoints, the metrics log and the session summaries are untouched (986a3fa: Codex N1 finished
    the run, rotated out the valid checkpoint and wrote a checkpoint its own validator rejects)."""
    out = str(tmp_path / "cli")
    p = copy.deepcopy(fx["protocol"])
    p["schedule"]["keep_full_checkpoints"] = 2
    pp = tmp_path / "keep2.json"
    pp.write_text(json.dumps(p))
    cli = ["--protocol", str(pp), "--train-size", "3", "--device", "cpu", "--output-dir", out]
    train_r2r2r_dp.main(cli + ["--stop-after-updates", "3"])
    assert sorted(f for f in os.listdir(out) if f.startswith("ckpt_full")) == ["ckpt_full_u0000002.pt",
                                                                              "ckpt_full_u0000003.pt"]
    p3 = os.path.join(out, "ckpt_full_u0000003.pt")
    torch.save(_corrupt(_load(p3), case), p3)
    files = {f: hashlib.sha256(open(os.path.join(out, f), "rb").read()).hexdigest()
             for f in os.listdir(out) if f != "resume_history.jsonl"}
    with pytest.raises(driver.CheckpointError):
        train_r2r2r_dp.main(cli + ["--resume"])
    assert files == {f: hashlib.sha256(open(os.path.join(out, f), "rb").read()).hexdigest()
                     for f in os.listdir(out) if f != "resume_history.jsonl"}
