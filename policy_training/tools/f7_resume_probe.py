"""Entrypoint-level check of Codex re-audit N1 / F7 (usage: CUDA_VISIBLE_DEVICES='' .venv-tinydp/bin/python
policy_training/tools/f7_resume_probe.py OUT.json).

Synthetic CPU fixture. For each case: train with the real entrypoint to u3 (full checkpoints every update,
keep 2, so u2 and u3 exist), corrupt the latest checkpoint (u3) in place, then run `train_r2r2r_dp.main
--resume` to u4. Records whether the resume was rejected, whether the session reported completion, which
checkpoint files were added or removed, whether the previous valid checkpoint u2 and the metrics log are
byte-identical afterwards, the problems of any newly written checkpoint and the largest model difference to
an intact continuation. Writes one JSON report; exit 1 if any expectation fails (valid: resume completes
and equals the intact run; every corruption: rejected with CheckpointError before training, nothing changed).
Runs on any commit that has train_r2r2r_dp.py and r2r2r_dp.driver, so it serves as before/after evidence.
"""
import copy
import glob
import hashlib
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "tests")]

import torch  # noqa: E402

import train_r2r2r_dp  # noqa: E402
from r2r2r_dp import driver  # noqa: E402
from r2r2r_dp.protocol import resolve  # noqa: E402
from run_fixtures import make_run_fixture  # noqa: E402


def _first(st):
    return st["optimizer"]["state"][next(iter(st["optimizer"]["state"]))]


def corrupt(st, case):
    if case == "missing_adam_state":  # Codex N1 counterexample: one parameter's entire Adam state deleted
        del st["optimizer"]["state"][next(iter(st["optimizer"]["state"]))]
    elif case == "missing_all_adam_state":
        st["optimizer"]["state"] = {}
    elif case in ("missing_exp_avg", "missing_exp_avg_sq", "missing_step"):
        del _first(st)[case[len("missing_"):]]
    elif case == "bad_moment_shape":
        _first(st)["exp_avg"] = torch.zeros(2, 3)
    elif case == "nan_moment":
        _first(st)["exp_avg_sq"] = _first(st)["exp_avg_sq"].clone().fill_(float("nan"))
    elif case == "optimizer_nan_lr":
        st["optimizer"]["param_groups"][0]["lr"] = float("nan")
    elif case == "scheduler_nan_lr":
        st["scheduler"]["_last_lr"] = [float("nan")] * len(st["scheduler"]["_last_lr"])
    elif case == "invalid_rng":
        st["rng"]["torch"] = torch.zeros(1, dtype=torch.uint8)
    elif case != "valid":
        raise ValueError(case)
    return st


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    cases = ["valid", "missing_adam_state", "missing_all_adam_state", "missing_exp_avg", "missing_exp_avg_sq",
             "missing_step", "bad_moment_shape", "nan_moment", "optimizer_nan_lr", "scheduler_nan_lr", "invalid_rng"]
    rep = {"torch": torch.__version__, "cases": {}}
    with tempfile.TemporaryDirectory() as d:
        fx = make_run_fixture(os.path.join(d, "fx"), total_updates=4, warmup_updates=0, val_every_updates=100,
                              full_val_every_updates=100, full_ckpt_every_updates=1, keep_full_checkpoints=2,
                              inference_export_every_updates=100)
        cli = ["--protocol", fx["protocol_path"], "--train-size", "3", "--device", "cpu"]
        ref = os.path.join(d, "intact")
        train_r2r2r_dp.main(cli + ["--output-dir", ref])
        R = torch.load(os.path.join(ref, "ckpt_full_u0000004.pt"), weights_only=False)
        base = os.path.join(d, "base")
        train_r2r2r_dp.main(cli + ["--output-dir", base, "--stop-after-updates", "3"])
        for case in cases:
            out = os.path.join(d, case)
            shutil.copytree(base, out)
            p3 = os.path.join(out, "ckpt_full_u0000003.pt")
            torch.save(corrupt(torch.load(p3, weights_only=False), case), p3)
            before = {os.path.basename(f): sha(f) for f in glob.glob(os.path.join(out, "*")) if os.path.isfile(f)}
            rec = {}
            try:
                s = train_r2r2r_dp.main(cli + ["--output-dir", out, "--resume"])
                rec["resume"] = "completed"
                rec["summary"] = {k: s[k] for k in ("final_update", "updates_this_session", "finished")}
            except driver.CheckpointError as e:
                rec["resume"] = "rejected: CheckpointError"
                rec["error"] = str(e).split(": ", 1)[1][:400]
            except BaseException as e:  # noqa: BLE001 - recorded as evidence, not handled
                rec["resume"] = f"failed: {type(e).__name__}"
                rec["error"] = str(e)[:400]
            after = {os.path.basename(f): sha(f) for f in glob.glob(os.path.join(out, "*")) if os.path.isfile(f)}
            rec["ckpt_added"] = sorted(f for f in after if f.startswith("ckpt_full") and f not in before)
            rec["ckpt_removed"] = sorted(f for f in before if f.startswith("ckpt_full") and f not in after)
            rec["u2_valid_ckpt_unchanged"] = before.get("ckpt_full_u0000002.pt") == after.get("ckpt_full_u0000002.pt")
            rec["metrics_unchanged"] = before["metrics.jsonl"] == after["metrics.jsonl"]
            rec["session_summaries_written"] = "session_summaries.jsonl" in after and \
                before.get("session_summaries.jsonl") != after["session_summaries.jsonl"]
            p4 = os.path.join(out, "ckpt_full_u0000004.pt")
            if os.path.exists(p4):
                C = torch.load(p4, weights_only=False)
                t = driver.Trainer(resolve(fx["protocol"], 3, os.path.join(d, "probe_" + case)), device="cpu")
                rec["u4_ckpt_problems"] = t.checkpoint_problems(C)
                rec["u4_max_abs_model_diff_vs_intact"] = max(
                    float((R["model"][k].float() - C["model"][k].float()).abs().max()) for k in R["model"])
            rep["cases"][case] = rec
            print(case, json.dumps(rec)[:300], flush=True)
    checks = {}
    for case, r in rep["cases"].items():
        if case == "valid":
            checks[case] = (r["resume"] == "completed" and r["summary"]["finished"] and r["u4_ckpt_problems"] == []
                            and r["u4_max_abs_model_diff_vs_intact"] == 0.0)
        else:
            checks[case] = (r["resume"] == "rejected: CheckpointError" and not r["ckpt_added"] and not r["ckpt_removed"]
                            and r["u2_valid_ckpt_unchanged"] and r["metrics_unchanged"]
                            and not r["session_summaries_written"])
    rep["checks"], rep["all_ok"] = checks, all(checks.values())
    with open(sys.argv[1], "w") as f:
        json.dump(rep, f, indent=1, allow_nan=False)
    print(json.dumps(checks, indent=1))
    sys.exit(0 if rep["all_ok"] else 1)


if __name__ == "__main__":
    main()
