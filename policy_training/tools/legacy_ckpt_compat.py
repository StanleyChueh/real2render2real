"""Read-only compatibility check of existing full checkpoints with the current validator (usage:
CUDA_VISIBLE_DEVICES='' .venv-tinydp/bin/python policy_training/tools/legacy_ckpt_compat.py OUT.json RUN_CONFIG CKPT...).

For each checkpoint: builds a CPU Trainer from RUN_CONFIG (the run record of the run that wrote them; output
directory redirected to a temporary directory, nothing is written next to the checkpoints), records whether
the checkpoint carries an optimizer history, its Adam-state coverage and step counts, the problems reported
by Trainer.checkpoint_problems, and whether load_full applies it. File hashes are taken before and after to
show the checkpoints were not modified. Exit 1 if any checkpoint is rejected or changed.
"""
import hashlib
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402

from r2r2r_dp import driver  # noqa: E402


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    out, run_config, ckpts = sys.argv[1], sys.argv[2], sys.argv[3:]
    with open(run_config) as f:
        run = json.load(f)
    rep, ok = {"run_config": run_config, "config_sha256": run["config_sha256"], "checkpoints": {}}, True
    with tempfile.TemporaryDirectory() as d:
        run["paths"]["output_dir"] = d
        for path in ckpts:
            before = sha(path)
            st = torch.load(path, map_location="cpu", weights_only=False)
            t = driver.Trainer(run, device="cpu")
            n_params = sum(len(g["params"]) for g in t.optimizer.param_groups)
            rec = {"sha256": before, "format": st["format"], "update": st["update"],
                   "has_optimizer_history": "optimizer_history" in st,
                   "optimizer_params": n_params, "adam_state_entries": len(st["optimizer"]["state"]),
                   "adam_steps": sorted({int(v["step"]) for v in st["optimizer"]["state"].values()}),
                   "problems": t.checkpoint_problems(st)}
            try:
                t.load_full(path)
                rec["load_full"] = "applied"
                rec["param_steps_after_load"] = sorted({k for g in t.param_steps for k in g})
            except driver.CheckpointError as e:
                rec["load_full"] = f"rejected: {e}"
            rec["unchanged"] = sha(path) == before
            ok &= rec["problems"] == [] and rec["load_full"] == "applied" and rec["unchanged"]
            rep["checkpoints"][path] = rec
            print(json.dumps({path: rec}), flush=True)
    rep["all_ok"] = ok
    with open(out, "w") as f:
        json.dump(rep, f, indent=1, allow_nan=False)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
