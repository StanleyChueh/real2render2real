"""Bounded CUDA check of Codex final audit F1 (usage: CUDA_VISIBLE_DEVICES=0 .venv-tinydp/bin/python
policy_training/tools/cuda_skip_probe.py OUT.json).

Synthetic fixture and the small CPU-test model, but on CUDA with the protocol's bf16 autocast and the
released NativeScaler (real torch.amp.GradScaler('cuda')). An Inf gradient is injected into one
accumulation group; the probe records what the driver counted, then resumes from the checkpoint
written after the skip and compares state with an uninterrupted run. Writes one JSON report; exit 1
if any expectation fails. Uses < 1 GB of GPU memory and a few seconds.
"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "tests")]

import torch  # noqa: E402

from r2r2r_dp import driver  # noqa: E402
from r2r2r_dp.protocol import resolve  # noqa: E402
from run_fixtures import make_run_fixture  # noqa: E402


def trainer(fx, out, bad=()):
    run = resolve(fx["protocol"], 3, out)
    driver.write_run_files(run)
    t = driver.Trainer(run, device="cuda")
    p = next(t.model.parameters())
    p.register_hook(lambda g: torch.full_like(g, float("inf")) if t.attempted_updates + 1 in bad else g)
    return t


def counters(t):
    return {"update": t.update, "attempted": t.attempted_updates, "skipped": t.skipped_updates,
            "optimizer_calls": t.optimizer_calls, "scheduler_last_epoch": t.scheduler.last_epoch,
            "adam_steps": sorted({int(v["step"]) for v in t.optimizer.state.values()}), "micro": t.micro,
            "scale": t.scaler._scaler.get_scale(), "amp": t.amp, "scaler_enabled": t.scaler._scaler.is_enabled()}


def main():
    assert torch.cuda.device_count() == 1, "set CUDA_VISIBLE_DEVICES to exactly one GPU"
    rep = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__}
    with tempfile.TemporaryDirectory() as d:
        fx = make_run_fixture(os.path.join(d, "fx"), total_updates=4, warmup_updates=2, val_every_updates=100,
                              full_val_every_updates=100, full_ckpt_every_updates=2, inference_export_every_updates=100)
        a = trainer(fx, os.path.join(d, "a"), bad=(2,))
        a.train()
        rep["skip_run"] = counters(a)
        rep["skip_records"] = [json.loads(l) for l in open(os.path.join(a.out, "metrics.jsonl"))
                               if json.loads(l)["type"] == "train_skipped"]
        b = trainer(fx, os.path.join(d, "b"), bad=(2,))
        b.train(stop_after=2)
        c = trainer(fx, os.path.join(d, "b"))
        c.load_full(os.path.join(c.out, "ckpt_full_u0000002.pt"))
        rep["resumed_from"] = counters(c)
        c.train()
        rep["resumed_run"] = counters(c)
        A = torch.load(os.path.join(a.out, "ckpt_full_u0000004.pt"), weights_only=False)
        C = torch.load(os.path.join(c.out, "ckpt_full_u0000004.pt"), weights_only=False)
        rep["resume_state_equal"] = {k: A[k] == C[k] for k in ("scheduler", "scaler", "data_position", "update",
                                                                "attempted_updates", "skipped_updates", "micro_batches")}
        rep["resume_max_abs_model_diff"] = max(float((A["model"][k].float() - C["model"][k].float()).abs().max())
                                               for k in A["model"] if A["model"][k].is_floating_point())
        rep["peak_gpu_mem_allocated_MiB"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    exp = {"update": 4, "attempted": 5, "skipped": 1, "optimizer_calls": 4, "scheduler_last_epoch": 4,
           "adam_steps": [4], "micro": 20, "scale": 32768.0, "amp": True, "scaler_enabled": True}
    checks = {"skip_run": rep["skip_run"] == exp,
              "resumed_from_u2": {k: rep["resumed_from"][k] for k in ("update", "attempted", "skipped", "micro")}
              == {"update": 2, "attempted": 3, "skipped": 1, "micro": 12},
              "resumed_counters": {k: v for k, v in rep["resumed_run"].items() if k != "optimizer_calls"}
              == {k: v for k, v in exp.items() if k != "optimizer_calls"},
              "resume_state_equal": all(rep["resume_state_equal"].values()),
              "one_skip_record": [r["update"] for r in rep["skip_records"]] == [2]}
    rep["checks"], rep["all_ok"] = checks, all(checks.values())
    with open(sys.argv[1], "w") as f:
        json.dump(rep, f, indent=1, allow_nan=False)
    print(json.dumps({"checks": checks, "skip_run": rep["skip_run"], "model_diff": rep["resume_max_abs_model_diff"]}, indent=1))
    sys.exit(0 if rep["all_ok"] else 1)


if __name__ == "__main__":
    main()
