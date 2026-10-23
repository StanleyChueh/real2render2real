"""Discontinuity, row-order and lag detectors on synthetic signals with known answers."""
import numpy as np

from r2r2r_dp.analysis import best_lag, joint_jump_profile, row_order_violations


def _smooth(T=161):
    t = np.linspace(0, 1, T)[:, None]
    return np.hstack([0.3 + 0.2 * t, -0.1 * np.sin(3 * t), 0.1 + 0.05 * t ** 2])


def test_row_order_detects_swapped_rows_only():
    x = _smooth()
    assert row_order_violations(x) == []
    y = x.copy()
    y[[70, 71]] = y[[71, 70]]
    assert 70 in row_order_violations(y)
    z = x.copy()
    z[100:105] = z[100]  # a static segment is not a reordering
    assert row_order_violations(z) == []


def test_joint_jump_profile_flags_jump_and_reports_ee_motion():
    T = 161
    ja = np.cumsum(np.full((T, 16), 0.01), axis=0)
    ja[60:, 9] += 0.8  # 0.8 rad jump between rows 59 and 60 on yumi_joint_4_r
    ee = np.zeros((T, 14))
    ee[:, 3] = ee[:, 10] = 1.0
    ee[:, 7:10] = _smooth(T)
    p = joint_jump_profile(ja, ee)
    assert p["flagged"] and p["step"] == 59 and p["joint_index"] == 9
    assert np.isclose(p["max_joint_step"], 0.81)
    assert p["right_ee_rot_deg_at_jump"] < 1e-6 and p["right_ee_step_mm_at_jump"] < 5
    assert not joint_jump_profile(np.cumsum(np.full((T, 16), 0.01), axis=0), ee)["flagged"]


def test_best_lag_recovers_known_shift():
    rng = np.random.default_rng(0)
    s = np.abs(rng.normal(size=200)).cumsum() % 3
    for lag in (0, 1, 2, 3):
        img = np.r_[np.zeros(lag), s[: len(s) - lag]] + rng.normal(0, 0.05, len(s))
        assert best_lag(s, img)["lag"] == lag
