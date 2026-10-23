"""Read-only analyses: joint discontinuities, HDF5 row ordering, image/state temporal offset."""
import os
from typing import Dict, List

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

ARM = slice(0, 14)


def joint_jump_profile(ja: np.ndarray, ee: np.ndarray, threshold: float = 0.3) -> Dict:
    """Largest per-step joint change and the simultaneous end-effector motion of each arm.

    TinyDP actions are EE poses, so a joint-space jump only matters to the policy if the EE pose also
    jumps (an IK branch switch at a fixed EE pose is invisible to the action target)."""
    ja = ja.astype(np.float64)
    ee = ee.astype(np.float64)
    dq = np.abs(np.diff(ja[:, ARM], axis=0))
    t, j = np.unravel_index(dq.argmax(), dq.shape)
    out = {"max_joint_step": float(dq[t, j]), "step": int(t), "joint_index": int(j), "flagged": bool(dq[t, j] > threshold)}
    for side, sl in (("left", slice(0, 7)), ("right", slice(7, 14))):
        p, q = ee[:, sl][:, :3], ee[:, sl][:, 3:][:, [1, 2, 3, 0]]  # wxyz -> xyzw for scipy
        dp = np.linalg.norm(np.diff(p, axis=0), axis=1)
        r = Rotation.from_quat(q)
        dr = np.degrees((r[:-1].inv() * r[1:]).magnitude())
        out[f"{side}_ee_step_mm_at_jump"] = float(dp[t] * 1e3)
        out[f"{side}_ee_rot_deg_at_jump"] = float(dr[t])
        out[f"{side}_ee_step_mm_max"] = float(dp.max() * 1e3)
        out[f"{side}_ee_rot_deg_max"] = float(dr.max())
    return out


def row_order_violations(x: np.ndarray, rel: float = 0.3, min_abs: float = 1e-3) -> List[int]:
    """Indices t where swapping rows t and t+1 makes the local path markedly shorter.

    With in-order rows of a smooth trajectory a swap lengthens the path; rows appended out of order by
    concurrent writers produce the opposite pattern. x: (T, D) positions (EE xyz or joint angles)."""
    x = x.astype(np.float64)
    bad = []
    for t in range(1, len(x) - 2):
        seg = x[t - 1:t + 3]
        cur = np.linalg.norm(np.diff(seg, axis=0), axis=1).sum()
        swapped = seg[[0, 2, 1, 3]]
        alt = np.linalg.norm(np.diff(swapped, axis=0), axis=1).sum()
        if cur - alt > max(min_abs, rel * cur):
            bad.append(t)
    return bad


def _gray_small(path: str, size=(160, 90)) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L").resize(size, Image.BILINEAR), dtype=np.float32)


def image_motion_energy(ep_dir: str, camera: str, num_steps: int) -> np.ndarray:
    frames = [_gray_small(os.path.join(ep_dir, camera, "rgb", f"{t:04d}.jpg")) for t in range(num_steps)]
    return np.array([np.abs(frames[t + 1] - frames[t]).mean() for t in range(num_steps - 1)])


def best_lag(state_speed: np.ndarray, image_energy: np.ndarray, max_lag: int = 4) -> Dict:
    """Lag L maximising corr(state_speed[t], image_energy[t + L]); L > 0 means images lag the state."""
    s = (state_speed - state_speed.mean()) / (state_speed.std() + 1e-12)
    v = (image_energy - image_energy.mean()) / (image_energy.std() + 1e-12)
    corr = {}
    n = len(s)
    for lag in range(-max_lag, max_lag + 1):
        a = s[max(0, -lag): n - max(0, lag)]
        b = v[max(0, lag): n - max(0, -lag)]
        corr[lag] = float(np.mean(a * b))
    lag = max(corr, key=corr.get)
    return {"lag": int(lag), "corr": corr}
