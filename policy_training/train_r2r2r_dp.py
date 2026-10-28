"""Validated TinyDP training entry point (protocol v3). Replaces train_tinydp.py for scientific runs.

    CUDA_VISIBLE_DEVICES=0 .venv-tinydp/bin/python policy_training/train_r2r2r_dp.py \
        --protocol policy_training/configs/baseline_v3.json --train-size 50 \
        --output-dir ~/Stanley_ws/r2r2r_policy_runs/train/baseline_v3_n0050_seed0

    # continue an interrupted run (same protocol, train size and output dir):
    ... --resume

Order of operations: resolve the protocol -> validate data, splits, statistics and output dir
(launch_checks) -> only then build datasets and the model -> train for optimizer updates.
--stop-after-updates N ends the session after successful update N with a full checkpoint (bounded tests).
Exit status is non-zero for any failed check, non-finite loss/validation, aborted run (repeatedly
skipped optimizer steps) or a session that ends before its target update.
No WandB, TensorBoard upload or other remote logging is used.
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)


def _git(*args, cwd=REPO):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True).stdout.strip()


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def provenance():
    import torch
    files = ["train_r2r2r_dp.py"] + [os.path.join("r2r2r_dp", f) for f in sorted(os.listdir(os.path.join(HERE, "r2r2r_dp")))
                                     if f.endswith(".py")]
    return {"repo_commit": _git("rev-parse", "HEAD"), "repo_dirty": bool(_git("status", "--porcelain", "--", "policy_training")),
            "tinydp_commit": _git("rev-parse", "HEAD", cwd=os.path.join(REPO, "dependencies", "tinydp")),
            "tinydp_dirty": bool(_git("status", "--porcelain", cwd=os.path.join(REPO, "dependencies", "tinydp"))),
            "code_sha256": {f: _sha(os.path.join(HERE, f)) for f in files},
            "torch": torch.__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def build_run(protocol_path, train_size, output_dir, views_root=None, resume=False):
    from r2r2r_dp.launch_checks import check_launch
    from r2r2r_dp.protocol import load_protocol, resolve
    run = resolve(load_protocol(protocol_path), train_size, output_dir, views_root)
    run["launch"] = check_launch(run, allow_existing_output=resume)
    run["provenance"] = provenance()
    return run


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--train-size", type=int, required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--views-root", default=None, help="override the protocol's views_root (membership is still verified)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-after-updates", type=int, default=None)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--num-workers", type=int, default=None, help="override workers (data stream is worker-count invariant)")
    a = ap.parse_args(argv)
    if a.num_workers is not None and a.num_workers < 0:
        ap.error("--num-workers must be >= 0")
    if a.stop_after_updates is not None and a.stop_after_updates < 1:
        ap.error("--stop-after-updates must be >= 1")
    os.environ["WANDB_MODE"] = "disabled"

    import torch
    from r2r2r_dp import driver
    if a.device == "cuda" and torch.cuda.device_count() != 1:
        ap.error(f"exactly one visible CUDA device required (got {torch.cuda.device_count()}); set CUDA_VISIBLE_DEVICES")
    torch.multiprocessing.set_start_method("spawn", force=True)  # as released train.py

    run = build_run(a.protocol, a.train_size, a.output_dir, a.views_root, resume=a.resume)
    # not part of the configuration hash: the data stream is worker-count invariant (test_driver.py)
    run["runtime"] = {"device": a.device, "num_workers_override": a.num_workers, "stop_after_updates": a.stop_after_updates}
    out = run["paths"]["output_dir"]
    ckpt = None
    if a.resume:
        ckpts = sorted(glob.glob(os.path.join(out, "ckpt_full_u*.pt")))
        if not ckpts:
            ap.error(f"--resume: no full checkpoint in {out}")
        with open(os.path.join(out, "run_config.json")) as f:
            prev = json.load(f)
        if prev["config_sha256"] != run["config_sha256"]:
            ap.error("--resume: resolved configuration differs from the run being resumed")
        ckpt = ckpts[-1]
        with open(os.path.join(out, "resume_history.jsonl"), "a") as f:
            f.write(json.dumps({"checkpoint": ckpt, "launch": run["launch"], "provenance": run["provenance"],
                                "runtime": run["runtime"]}, allow_nan=False) + "\n")
    else:
        driver.write_run_files(run)

    trainer = driver.Trainer(run, device=a.device, num_workers=a.num_workers)
    if ckpt:
        trainer.load_full(ckpt)
        driver.truncate_metrics(out, trainer.update)
    print(json.dumps({"config_sha256": run["config_sha256"], "launch": run["launch"], "start_update": trainer.update,
                      "start_attempted_updates": trainer.attempted_updates, "train_order_sha256": trainer.train_order_sha256,
                      "batches_per_epoch": trainer.batches_per_epoch, "val_subset_batches": len(trainer.val_subset),
                      "val_full_batches": len(trainer.val_full)}, indent=1), flush=True)
    summary = trainer.train(stop_after=a.stop_after_updates)
    if a.device == "cuda":
        summary["peak_gpu_mem_allocated_GiB"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    with open(os.path.join(out, "session_summaries.jsonl"), "a") as f:
        f.write(json.dumps(summary, allow_nan=False) + "\n")
    print(json.dumps(summary), flush=True)
    target = run["protocol"]["schedule"]["total_updates"] if a.stop_after_updates is None else \
        min(a.stop_after_updates, run["protocol"]["schedule"]["total_updates"])
    if summary["final_update"] != target:
        raise SystemExit(f"session ended at update {summary['final_update']}, target {target}")
    return summary


if __name__ == "__main__":
    main()
