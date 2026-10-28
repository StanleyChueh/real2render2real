"""Validated 6-D -> SO(3) conversion for policy outputs (Codex final audit F8).

TinyDP's rot6d is the first two ROWS of the rotation matrix (tinydp.dataset.utils.quat_to_rot_6d), and
its inverse rot_6d_to_rot_mat skips Gram-Schmidt for orthogonal non-unit rows (det != 1) and returns
non-finite or zero matrices for collinear or zero rows. That pinned helper is left unchanged; use this
function wherever a physical rotation is needed:
    r1 = a / |a|,  r2 = (b - (r1.b) r1) / |b - (r1.b) r1|,  r3 = r1 x r2,  R = [r1; r2; r3]
Rows that are non-finite, shorter than `eps`, or (nearly) collinear raise ValueError; every result is
checked for R R^T = I and det R = +1.
"""
import numpy as np


def rot6d_rows_to_matrix(rot6d, eps: float = 1e-6, atol: float = 1e-6) -> np.ndarray:
    v = np.asarray(rot6d, dtype=np.float64)
    if v.shape[-1] != 6:
        raise ValueError(f"rot6d must have a trailing dimension of 6, got shape {v.shape}")
    if not np.isfinite(v).all():
        raise ValueError("rot6d contains non-finite values")
    a, b = v[..., :3], v[..., 3:]
    na = np.linalg.norm(a, axis=-1, keepdims=True)
    if (na < eps).any():
        raise ValueError(f"rot6d first row has norm < {eps}")
    r1 = a / na
    b_perp = b - (r1 * b).sum(-1, keepdims=True) * r1
    nb = np.linalg.norm(b_perp, axis=-1, keepdims=True)
    if (nb < eps * np.maximum(1.0, np.linalg.norm(b, axis=-1, keepdims=True))).any():
        raise ValueError("rot6d rows are zero or (nearly) collinear")
    r2 = b_perp / nb
    R = np.stack([r1, r2, np.cross(r1, r2)], axis=-2)
    eye = np.broadcast_to(np.eye(3), R.shape)
    if not (np.allclose(R @ np.swapaxes(R, -1, -2), eye, atol=atol) and np.allclose(np.linalg.det(R), 1.0, atol=atol)):
        raise ValueError("conversion did not produce a proper rotation")
    return R
