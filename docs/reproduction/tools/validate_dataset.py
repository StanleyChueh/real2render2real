"""Strict integrity check for R2R2R generated rollouts (read-only).

Enforces the per-episode schema in policy_training/r2r2r_dp/schema.py: exact HDF5 keys, dtypes and
shapes, finite values, unit quaternions, gripper semantics, joint-name order, exactly the expected
cameras with consecutive frame ids 0000..T-1, and JPEG/RGB/resolution of every frame. With
--decode all, every JPEG is fully decoded (otherwise headers only).

Exit status: 0 all episodes valid, 1 at least one invalid episode, 2 no episodes / missing directory.

usage: python docs/reproduction/tools/validate_dataset.py TASK_DIR [TASK_DIR ...] [--decode header|all]
                                                        [--workers N] [--json out.json]
  TASK_DIR e.g. output_data/yumi_coffee_maker (episodes under successes/)
"""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "policy_training"))
from r2r2r_dp.schema import H5_REL, YUMI_COFFEE_MAKER, validate_episode  # noqa: E402


def _validate(args):
    ep, decode = args
    rec = validate_episode(ep, YUMI_COFFEE_MAKER, decode=decode)
    if not rec["errors"]:
        import h5py
        with h5py.File(os.path.join(ep, H5_REL), "r") as f:
            ja = f["joint_angles"][:].astype(np.float64)
        rec["max_joint_step"] = float(np.abs(np.diff(ja[:, :14], axis=0)).max())
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dirs", nargs="+")
    ap.add_argument("--decode", choices=("header", "all"), default="header")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--json")
    a = ap.parse_args()
    eps = []
    for d in a.task_dirs:
        if not os.path.isdir(os.path.join(d, "successes")):
            print(f"ERROR: {d}/successes does not exist", file=sys.stderr)
            sys.exit(2)
        eps += sorted(glob.glob(os.path.join(d, "successes", "*")))
    if not eps:
        print("ERROR: no episodes found", file=sys.stderr)
        sys.exit(2)
    with ProcessPoolExecutor(a.workers) as ex:
        recs = list(ex.map(_validate, [(e, a.decode) for e in eps], chunksize=4))
    ok = [r for r in recs if not r["errors"]]
    jumps = np.array([r["max_joint_step"] for r in ok]) if ok else np.zeros(1)
    summary = {
        "n_episodes": len(recs),
        "n_valid": len(ok),
        "n_invalid": len(recs) - len(ok),
        "image_decode": a.decode,
        "frames_per_camera": int(sum(r.get("camera_0_frames", 0) for r in ok)),
        "max_joint_step_p50_p99_max": [float(np.median(jumps)), float(np.percentile(jumps, 99)), float(jumps.max())],
        "n_max_joint_step_gt_0.3rad": int((jumps > 0.3).sum()),
        "invalid_examples": [(r["episode"], r["errors"][:3]) for r in recs if r["errors"]][:10],
    }
    print(json.dumps(summary, indent=2))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"summary": summary, "episodes": recs}, f, indent=1)
    sys.exit(0 if len(ok) == len(recs) else 1)


if __name__ == "__main__":
    main()
