"""Run TinyDP's unmodified script/train.py on manifest views.

    CUDA_VISIBLE_DEVICES=0 python policy_training/train_tinydp.py \
        --train-root VIEWS/train_subsets/n1461 --eval-root VIEWS/splits/val \
        --action-statistics STATS/action_statistics_train_n1461.json \
        -- --dataset-cfg.is-sim-data --model-cfg.policy-cfg.pred-right-only \
           --logging-cfg.output-dir OUT --trainer-cfg.epochs 200

Everything after '--' is passed to TinyDP. The only change to TinyDP's behaviour is the dataset class:
tinydp.dataset.image_dataset_sim.SequenceDataset is replaced by ManifestSimDataset (train split reads
--train-root, val split reads --eval-root, both with the given train-only statistics) and its file glob
is sorted. WandB is disabled; --logging-cfg.log-name (which would start a WandB run) is rejected.
"""
import argparse
import hashlib
import json
import os
import runpy
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TRAIN_PY = os.path.join(REPO, "dependencies", "tinydp", "script", "train.py")
sys.path.insert(0, HERE)


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _git(*args, cwd=REPO):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True).stdout.strip()


def _value(argv, flag):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def prepare(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--action-statistics", required=True)
    if "--" not in argv:
        ap.error("separate TinyDP arguments with '--'")
    ours, tinydp_args = argv[: argv.index("--")], argv[argv.index("--") + 1:]
    a = ap.parse_args(ours)
    if "--dataset-cfg.is-sim-data" not in tinydp_args:
        ap.error("--dataset-cfg.is-sim-data is required (R2R2R sim schema)")
    for banned in ("--logging-cfg.log-name", "--dataset-cfg.dataset-root", "--dataset-cfg.action-statistics",
                   "--dataset-cfg.data-subsample-num-traj", "--dataset-cfg.data-subsample-ratio", "--dataset-cfg.balance-data"):
        if any(t == banned or t.startswith(banned + "=") for t in tinydp_args):
            ap.error(f"{banned} is controlled by this wrapper / the manifest")
    out = _value(tinydp_args, "--logging-cfg.output-dir")
    if out is None:
        ap.error("--logging-cfg.output-dir is required")
    for r in (a.train_root, a.eval_root):
        if not os.path.isdir(r):
            ap.error(f"{r} does not exist")
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_SILENT"] = "true"

    from r2r2r_dp import tinydp_bridge as tb
    import tinydp.dataset.image_dataset_sim as sim
    tb.install_ordered_glob()
    sim.SequenceDataset = tb.make_manifest_dataset_class(a.train_root, a.eval_root, a.action_statistics)

    os.makedirs(out, exist_ok=True)
    prov = {
        "train_root": os.path.realpath(a.train_root), "eval_root": os.path.realpath(a.eval_root),
        "action_statistics": os.path.realpath(a.action_statistics), "action_statistics_sha256": _sha256(a.action_statistics),
        "tinydp_args": tinydp_args, "repo_commit": _git("rev-parse", "HEAD"), "repo_dirty": bool(_git("status", "--porcelain")),
        "tinydp_commit": _git("rev-parse", "HEAD", cwd=os.path.join(REPO, "dependencies", "tinydp")),
        "train_py_sha256": _sha256(TRAIN_PY), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    with open(os.path.join(out, "r2r2r_provenance.json"), "w") as f:
        json.dump(prov, f, indent=1)
    return tinydp_args


if __name__ == "__main__":
    tinydp_args = prepare(sys.argv[1:])
    sys.argv = [TRAIN_PY, *tinydp_args]
    runpy.run_path(TRAIN_PY, run_name="__main__")
