"""Synthetic R2R2R-style episodes for tests (small, deterministic).

Row t of a fixture episode encodes t everywhere so temporal indexing can be read back:
  right EE x = X0 + DX * t, right quaternion = identity (wxyz), left EE constant,
  right gripper closed for rows [close, open), image t is a uniform gray of value gray(t).
"""
import os
from typing import Optional, Tuple

import h5py
import numpy as np
from PIL import Image

from r2r2r_dp.schema import YUMI_JOINT_NAMES, EpisodeSchema

X0, DX = 0.3, 0.01  # DX > 2 * TinyDP proprio_noise (0.003) so rows stay decodable
SMALL = EpisodeSchema(num_steps=60, image_size=(32, 24))


def gray(t: int) -> int:
    return 20 + 3 * t  # distinct, JPEG-stable levels for t < 70


def episode_arrays(T: int, close: int = 20, open_: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=np.float64)
    ee = np.zeros((T, 14), np.float32)
    ee[:, 0:3] = [0.5, 0.37, 0.11]
    ee[:, 3:7] = [0.0945, -0.9658, -0.24, 0.0271] / np.linalg.norm([0.0945, -0.9658, -0.24, 0.0271])
    ee[:, 7] = X0 + DX * t
    ee[:, 8:10] = [-0.2, 0.1]
    ee[:, 10:14] = [1.0, 0.0, 0.0, 0.0]
    ja = (0.01 * t[:, None] + rng.normal(0, 1e-4, (T, 16))).astype(np.float32)
    g = np.zeros((T, 2), bool)
    g[close:open_, 1] = True
    return {"ee_poses": ee, "joint_angles": ja, "gripper_binary_cmd": g}


def write_episode(ep_dir: str, schema: EpisodeSchema = SMALL, arrays: Optional[dict] = None,
                  cameras: Tuple[str, ...] = ("camera_0", "camera_1"), seed: int = 0) -> str:
    T = schema.num_steps
    arrays = arrays if arrays is not None else episode_arrays(T, seed=seed)
    os.makedirs(os.path.join(ep_dir, "robot_data"), exist_ok=True)
    with h5py.File(os.path.join(ep_dir, "robot_data", "robot_data.h5"), "w") as f:
        for k, v in arrays.items():
            f.create_dataset(k, data=v)
    with open(os.path.join(ep_dir, "robot_data", "joint_names.txt"), "w") as f:
        f.write("\n".join(YUMI_JOINT_NAMES))
    w, h = schema.image_size
    for cam in cameras:
        d = os.path.join(ep_dir, cam, "rgb")
        os.makedirs(d, exist_ok=True)
        for t in range(T):
            Image.new("RGB", (w, h), (gray(t),) * 3).save(os.path.join(d, f"{t:04d}.jpg"), quality=95)
    return ep_dir


def make_source_tree(root: str, n: int, schema: EpisodeSchema = SMALL, runs: Tuple[str, ...] = ("run_a", "run_b")):
    """<root>/<run>/output_data/yumi_coffee_maker/successes/env_<i>_<ts> episodes; returns their dirs."""
    eps = []
    for i in range(n):
        run = runs[i % len(runs)]
        ep = os.path.join(root, run, "output_data", "yumi_coffee_maker", "successes", f"env_{i % 3}_2026_10_22_00_00_{i:02d}")
        arrays = episode_arrays(schema.num_steps, seed=i)
        arrays["ee_poses"][:, 8] += 0.01 * i  # distinct contents
        eps.append(write_episode(ep, schema, arrays))
    return eps
