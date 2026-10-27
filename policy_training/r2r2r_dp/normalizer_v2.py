"""Train-only action normalizer v2: min/max over every action target the TinyDP sim loader returns.

Audit M1: TinyDP's calculate_dataset_statistics (and the v1 cross-check) take min/max over
helper_load_action(0, T - s), i.e. rows s, 2s, ... only. Training windows start at every integer row,
so the loader also returns odd-row targets, some of which fall outside those bounds.

Released index arithmetic (tinydp/dataset/image_dataset_sim.py, lines 100 and 135-147, 304):
    length        = T - 1
    window starts = w in range(length - (s * (L + P) - 1))
    item starts   = w, w + s, ..., w + s * (L - 1)
    targets(item) = item + s * m, m = 1..P
Every target the loader returns (rows 2..158 for T=161, s=2, L=4, P=16) is covered here, a superset of
the rows that enter the loss (only each window's last item). v1 statistics files are left untouched.
"""
import hashlib
import json
import os
from typing import Dict, Iterable, List, Sequence

import numpy as np
import zarr

from tinydp.dataset.utils import quat_to_rot_6d

NORMALIZER_VERSION = "r2r2r-dp-norm-v2"
ACTION_LAYOUT = ["left_x", "left_y", "left_z", "left_r00", "left_r01", "left_r02", "left_r10", "left_r11", "left_r12",
                 "left_gripper", "right_x", "right_y", "right_z", "right_r00", "right_r01", "right_r02", "right_r10",
                 "right_r11", "right_r12", "right_gripper"]


def released_window_starts(T: int, s: int, L: int, P: int) -> List[int]:
    return list(range((T - 1) - (s * (L + P) - 1)))


def released_item_starts(T: int, s: int, L: int, P: int) -> List[int]:
    return sorted({w + s * k for w in released_window_starts(T, s, L, P) for k in range(L)})


def released_target_rows(T: int, s: int, L: int, P: int) -> np.ndarray:
    """Rows of every action target the loader returns for an episode of T rows."""
    return np.array(sorted({i + s * m for i in released_item_starts(T, s, L, P) for m in range(1, P + 1)}))


def loss_target_rows(T: int, s: int, L: int, P: int) -> np.ndarray:
    """Rows that enter the loss (the last item of each window)."""
    return np.array(sorted({w + s * (L - 1) + s * m for w in released_window_starts(T, s, L, P) for m in range(1, P + 1)}))


def absolute_actions(ee: np.ndarray, grip: np.ndarray) -> np.ndarray:
    """(T,14) ee_poses + (T,2) gripper -> (T,20) TinyDP absolute action rows (same ops as helper_load_action)."""
    parts = []
    for sl, k in ((slice(0, 7), 0), (slice(7, 14), 1)):
        pose = ee[:, sl]
        parts.append(np.concatenate([pose[:, :3], quat_to_rot_6d(pose[:, 3:]), grip[:, k:k + 1].astype(float)], axis=1))
    return np.concatenate(parts, axis=1)


def view_episode_h5s(view_root: str) -> List[str]:
    return sorted(os.path.join(view_root, d, "robot_data", "robot_data.h5") for d in os.listdir(view_root)
                  if os.path.isfile(os.path.join(view_root, d, "source.json")))


def episode_actions(h5: str) -> np.ndarray:
    ee = zarr.load(h5.replace(".h5", "_proprio.zarr"))
    g = zarr.load(h5.replace(".h5", "_gripper_cmd.zarr"))
    return absolute_actions(ee, g)


def compute_bounds(h5s: Iterable[str], s: int, L: int, P: int):
    mins, maxs, n_rows = [], [], 0
    for h5 in h5s:
        a = episode_actions(h5)
        rows = released_target_rows(len(a), s, L, P)
        mins.append(a[rows].min(0))
        maxs.append(a[rows].max(0))
        n_rows += len(rows)
    if not mins:
        raise ValueError("no episodes")
    return np.min(mins, 0), np.max(maxs, 0), n_rows


def uids_sha256(uids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(uids).encode()).hexdigest()


def file_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_statistics(view_root: str, subset: str, train_uids: Sequence[str], manifest_sha256: str,
                     s: int, L: int, P: int, use_delta_action: bool = False) -> Dict:
    if use_delta_action:
        raise ValueError("normalizer v2 supports absolute actions only")
    h5s = view_episode_h5s(view_root)
    view_uids = [json.load(open(os.path.join(os.path.dirname(os.path.dirname(h)), "source.json")))["uid"] for h in h5s]
    if sorted(view_uids) != sorted(train_uids):
        raise ValueError(f"{view_root} does not contain exactly the {subset} training episodes")
    lo, hi, n_rows = compute_bounds(h5s, s, L, P)
    config = {"action_mode": "absolute", "subsample_steps": s, "seq_length": L, "num_pred_steps": P,
              "target_rows": "all loader-returned targets: item + s*m, m=1..P, item in released item starts",
              "layout": ACTION_LAYOUT}
    return {
        "shape": [20], "min_action": lo.tolist(), "max_action": hi.tolist(),
        "normalizer_version": NORMALIZER_VERSION,
        "provenance": {"subset": subset, "n_episodes": len(h5s), "n_target_rows": int(n_rows),
                       "manifest_sha256": manifest_sha256, "train_uids_sha256": uids_sha256(list(train_uids)),
                       "code_sha256": file_sha256(__file__)},
        "config": config,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
    }


def normalized(a: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """TinyDP scale_action(type='diffusion') followed by nan_to_num (zero-range dims -> 0)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num(2 * (a - lo) / (hi - lo) - 1)


def coverage(h5s: Iterable[str], stats: Dict, s: int, L: int, P: int, dims=slice(10, 20), tol: float = 1e-6) -> Dict:
    """Count loader target rows whose normalized value leaves [-1, 1] (right arm by default)."""
    lo, hi = np.array(stats["min_action"]), np.array(stats["max_action"])
    out, worst, n = 0, 0.0, 0
    for h5 in h5s:
        a = episode_actions(h5)
        z = normalized(a[released_target_rows(len(a), s, L, P)], lo, hi)[:, dims]
        out += int((np.abs(z) > 1 + tol).any(1).sum())
        worst = max(worst, float(np.abs(z).max()))
        n += len(z)
    return {"rows_out_of_bounds": out, "rows_checked": n, "max_abs_normalized": worst}
