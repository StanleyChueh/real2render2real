"""Integrity check + statistics for R2R2R generated rollouts (read-only).

usage: python docs/reproduction/tools/validate_dataset.py OUTPUT_TASK_DIR [--json out.json]
  OUTPUT_TASK_DIR e.g. output_data/yumi_coffee_maker  (expects successes/<episode>/...)
"""
import argparse
import glob
import json
import os

import cv2
import h5py
import numpy as np


def check_episode(ep_dir):
    rec = {"episode": os.path.basename(ep_dir), "errors": []}
    h5 = os.path.join(ep_dir, "robot_data", "robot_data.h5")
    if not os.path.exists(h5):
        rec["errors"].append("missing robot_data.h5")
        return rec
    try:
        with h5py.File(h5, "r") as f:
            for k in ("joint_angles", "ee_poses", "gripper_binary_cmd"):
                if k not in f:
                    rec["errors"].append(f"missing dataset {k}")
                    continue
                a = f[k][:]
                rec[f"{k}_shape"] = list(a.shape)
                if a.dtype != bool and not np.isfinite(a.astype(np.float64)).all():
                    rec["errors"].append(f"non-finite values in {k}")
            ja = f["joint_angles"][:].astype(np.float64)
            g = f["gripper_binary_cmd"][:].astype(np.float64)
            rec["T"] = int(ja.shape[0])
            rec["joint_range"] = [float(ja.min()), float(ja.max())]
            rec["max_joint_step"] = float(np.abs(np.diff(ja, axis=0)).max()) if len(ja) > 1 else 0.0
            rec["gripper_toggles"] = int((np.abs(np.diff(g, axis=0)) > 0).sum())
    except Exception as e:  # corrupt / partially written file
        rec["errors"].append(f"h5 read error: {e}")
        return rec

    rec["cameras"] = {}
    for cam in sorted(glob.glob(os.path.join(ep_dir, "camera_*"))):
        for mod in sorted(os.listdir(cam)):
            files = sorted(glob.glob(os.path.join(cam, mod, "*")))
            key = f"{os.path.basename(cam)}/{mod}"
            info = {"n": len(files)}
            if files:
                bad = 0
                # decode first, middle and last frame; count unreadable ones
                for fp in {files[0], files[len(files) // 2], files[-1]}:
                    im = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
                    if im is None:
                        bad += 1
                    else:
                        info["shape"] = list(im.shape)
                        info["mean_intensity"] = float(im.mean())
                info["unreadable_sampled"] = bad
                if bad:
                    rec["errors"].append(f"unreadable images in {key}")
            if info["n"] != rec.get("T"):
                rec["errors"].append(f"{key}: {info['n']} images vs T={rec.get('T')}")
            rec["cameras"][key] = info
    if not rec["cameras"]:
        rec["errors"].append("no camera images")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir")
    ap.add_argument("--json")
    args = ap.parse_args()
    eps = sorted(glob.glob(os.path.join(args.task_dir, "successes", "*")))
    recs = [check_episode(e) for e in eps]
    ok = [r for r in recs if not r["errors"]]
    Ts = np.array([r["T"] for r in ok]) if ok else np.array([0])
    summary = {
        "n_success_dirs": len(recs),
        "n_valid": len(ok),
        "n_invalid": len(recs) - len(ok),
        "T_mean": float(Ts.mean()), "T_min": int(Ts.min()), "T_max": int(Ts.max()),
        "total_frames_per_camera": int(Ts.sum()),
        "cameras": sorted({k for r in ok for k in r.get("cameras", {})}),
        "image_shape": next((v.get("shape") for r in ok for v in r["cameras"].values()), None),
        "gripper_toggles_hist": {str(k): int(v) for k, v in zip(*np.unique([r["gripper_toggles"] for r in ok], return_counts=True))} if ok else {},
        "max_joint_step_p99": float(np.percentile([r["max_joint_step"] for r in ok], 99)) if ok else None,
        "invalid_examples": [(r["episode"], r["errors"][:3]) for r in recs if r["errors"]][:10],
    }
    print(json.dumps(summary, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"summary": summary, "episodes": recs}, f, indent=1)


if __name__ == "__main__":
    main()
