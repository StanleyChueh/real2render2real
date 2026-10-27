"""Load a v2 checkpoint on CPU from the checkpoint file alone and run it on one real validation window.

    .venv-tinydp/bin/python policy_training/check_checkpoint_inference.py --checkpoint CKPT [--checkpoint EXPORT] \
        --views-root VIEWS --out report.json

For each checkpoint: R2R2RPolicy.load(weights='ema') on CPU, observation = the released loader's pairing for
window start w of the first val episode (images camera_1/camera_0 at rows w, w+2, w+4, w+6; proprio rows
w..w+6 without noise), full DDPM denoising. Reports shapes, finiteness, the right-arm output range and the
episode's own targets (rows w+8..w+38) for unit/frame sanity. This is a contract check, not an evaluation.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import zarr
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2r2r_dp.inference import R2R2RPolicy  # noqa: E402
from r2r2r_dp.normalizer_v2 import absolute_actions  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", required=True)
    ap.add_argument("--views-root", required=True)
    ap.add_argument("--window-start", type=int, default=60)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(8)
    val = os.path.join(a.views_root, "splits", "val")
    ep = os.path.join(val, sorted(os.listdir(val))[0])
    h5 = os.path.join(ep, "robot_data", "robot_data.h5")
    full = absolute_actions(zarr.load(h5.replace(".h5", "_proprio.zarr")), zarr.load(h5.replace(".h5", "_gripper_cmd.zarr")))
    w = a.window_start
    rows = [w, w + 2, w + 4, w + 6]
    target = full[w + 8:w + 39:2, 10:20]
    report = {"episode": json.load(open(os.path.join(ep, "source.json")))["uid"], "window_start": w,
              "observation_rows": rows, "target_rows": list(range(w + 8, w + 39, 2)), "checkpoints": {}}
    outs = {}
    for path in a.checkpoint:
        t0 = time.time()
        pol = R2R2RPolicy.load(path, weights="ema", device="cpu")
        frames = [[Image.open(os.path.join(ep, cam, "rgb", f"{r:04d}.jpg")).convert("RGB") for cam in ("camera_1", "camera_0")]
                  for r in rows]
        obs = pol.preprocess_frames(frames)[None]
        prop = torch.from_numpy(full[rows]).float()[None]
        t1 = time.time()
        out = pol.predict(obs, prop, seed=0)
        y = out["right_action"][0].numpy()
        outs[path] = out["normalized"]
        report["checkpoints"][os.path.basename(path)] = {
            "update": pol.update, "weights": pol.weights, "config_sha256": pol.config_sha256,
            "statistics_sha256": pol.statistics_sha256, "observation_shape": list(obs.shape),
            "proprio_shape": list(prop.shape), "output_shape": list(out["right_action"].shape),
            "finite": bool(np.isfinite(y).all()), "normalized_abs_max": float(out["normalized"].abs().max()),
            "inference_steps": pol.cfg.model_cfg.policy_cfg.num_inference_diffusion_steps,
            "load_s": round(t1 - t0, 1), "predict_s": round(time.time() - t1, 1),
            "pred_first_step": y[0].round(4).tolist(), "target_first_step": target[0].round(4).tolist(),
            "pred_xyz_mean_abs_err_m": float(np.abs(y[:, :3] - target[:, :3]).mean()),
        }
    if len(outs) == 2:
        x, z = list(outs.values())
        report["full_vs_export_identical"] = bool(torch.equal(x, z))
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
