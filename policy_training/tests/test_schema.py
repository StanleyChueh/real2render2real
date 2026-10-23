"""Strict episode validation (fixes CODEX_AUDIT M1): malformed episodes and missing images must fail."""
import os
import subprocess
import sys

import h5py
import numpy as np
import pytest
from PIL import Image

from fixtures import SMALL, episode_arrays, write_episode
from r2r2r_dp.schema import validate_episode

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLI = os.path.join(REPO, "docs", "reproduction", "tools", "validate_dataset.py")


def _rewrite_h5(ep, **changes):
    p = os.path.join(ep, "robot_data", "robot_data.h5")
    with h5py.File(p, "r") as f:
        arrays = {k: f[k][:] for k in f}
    arrays.update(changes)
    with h5py.File(p, "w") as f:
        for k, v in arrays.items():
            if v is not None:
                f.create_dataset(k, data=v)


@pytest.fixture
def ep(tmp_path):
    return write_episode(str(tmp_path / "ep"))


def test_valid_episode_passes(ep):
    rec = validate_episode(ep, SMALL, decode="all")
    assert rec["errors"] == [] and rec["T"] == SMALL.num_steps and rec["gripper_toggles"] == [0, 2]


def _errors(ep, decode="header"):
    return " | ".join(validate_episode(ep, SMALL, decode=decode)["errors"])


T = SMALL.num_steps
H5_CASES = {
    "short_joint_angles": (dict(joint_angles=np.zeros((T - 1, 16), np.float32)), "joint_angles: shape"),
    "wrong_ee_dim": (dict(ee_poses=np.zeros((T, 13), np.float32)), "ee_poses: shape"),
    "float64_ee": (dict(ee_poses=episode_arrays(T)["ee_poses"].astype(np.float64)), "dtype"),
    "missing_key": (dict(gripper_binary_cmd=None), "keys"),
    "float_gripper": (dict(gripper_binary_cmd=episode_arrays(T)["gripper_binary_cmd"].astype(np.float32)), "dtype"),
    "left_gripper_set": (dict(gripper_binary_cmd=np.ones((T, 2), bool)), "inactive arm"),
    "no_toggle": (dict(gripper_binary_cmd=np.zeros((T, 2), bool)), "toggles"),
}


@pytest.mark.parametrize("case", sorted(H5_CASES))
def test_h5_violations_fail(ep, case):
    change, needle = H5_CASES[case]
    _rewrite_h5(ep, **change)
    assert needle in _errors(ep)


def test_nonfinite_and_quaternion_fail(ep):
    a = episode_arrays(T)
    a["ee_poses"][5, 7] = np.nan
    _rewrite_h5(ep, ee_poses=a["ee_poses"])
    assert "non-finite" in _errors(ep)
    a = episode_arrays(T)
    a["ee_poses"][:, 10:14] *= 1.1
    _rewrite_h5(ep, ee_poses=a["ee_poses"])
    assert "quaternion norm" in _errors(ep)


def test_joint_names_missing_or_reordered(ep):
    p = os.path.join(ep, "robot_data", "joint_names.txt")
    names = open(p).read().split("\n")
    open(p, "w").write("\n".join(names[::-1]))
    assert "joint names" in _errors(ep)
    os.remove(p)
    assert "joint_names.txt missing" in _errors(ep)


def test_missing_camera_and_frames(ep):
    os.remove(os.path.join(ep, "camera_0", "rgb", "0003.jpg"))
    assert "frame ids mismatch" in _errors(ep) and "0003.jpg" in _errors(ep)
    Image.new("RGB", SMALL.image_size).save(os.path.join(ep, "camera_1", "rgb", f"{T:04d}.jpg"))
    assert "extra" in _errors(ep)
    import shutil
    shutil.rmtree(os.path.join(ep, "camera_1"))
    assert "cameras" in _errors(ep)


def test_wrong_resolution_and_format(ep):
    Image.new("RGB", (8, 8)).save(os.path.join(ep, "camera_0", "rgb", "0001.jpg"))
    Image.new("RGB", SMALL.image_size).save(os.path.join(ep, "camera_1", "rgb", "0002.jpg"), format="PNG")
    e = _errors(ep)
    assert "0001.jpg" in e and "0002.jpg" in e


def test_truncated_jpeg_needs_full_decode(ep):
    p = os.path.join(ep, "camera_0", "rgb", "0007.jpg")
    Image.effect_noise(SMALL.image_size, 64).convert("RGB").save(p, quality=95)  # entropy-heavy frame
    data = open(p, "rb").read()
    open(p, "wb").write(data[: len(data) // 2])
    assert "0007.jpg" in _errors(ep, decode="all")


def test_codex_m1_counterexample_fails(tmp_path):
    """The exact malformed episode from CODEX_AUDIT.md M1, which the old validator accepted."""
    ep = str(tmp_path / "bad")
    os.makedirs(os.path.join(ep, "robot_data"))
    with h5py.File(os.path.join(ep, "robot_data", "robot_data.h5"), "w") as f:
        f.create_dataset("joint_angles", data=np.zeros((5, 16), np.float32))
        f.create_dataset("ee_poses", data=np.zeros((1, 3), np.float32))
        f.create_dataset("gripper_binary_cmd", data=np.zeros((2, 1), bool))
    d = os.path.join(ep, "camera_0", "rgb")
    os.makedirs(d)
    for t in range(5):
        Image.new("RGB", (8, 8)).save(os.path.join(d, f"{t:04d}.jpg"))
    open(os.path.join(d, "0001.jpg"), "wb").write(b"\xff\xd8corrupt")
    errors = validate_episode(ep, SMALL, decode="header")["errors"]
    joined = " | ".join(errors)
    for needle in ("joint_angles: shape", "ee_poses: shape", "gripper_binary_cmd: shape", "joint_names.txt missing", "cameras", "0001.jpg"):
        assert needle in joined, needle


def _cli(*args):
    return subprocess.run([sys.executable, CLI, *args], capture_output=True, text=True)


def test_cli_exit_codes(tmp_path):
    assert _cli(str(tmp_path / "nope")).returncode == 2
    task = tmp_path / "task"
    (task / "successes").mkdir(parents=True)
    assert _cli(str(task)).returncode == 2  # empty dataset
    from r2r2r_dp.schema import YUMI_COFFEE_MAKER
    write_episode(str(task / "successes" / "ep_ok"), YUMI_COFFEE_MAKER, episode_arrays(161, close=49, open_=132))
    r = _cli(str(task), "--workers", "1", "--decode", "all")
    assert r.returncode == 0, r.stdout + r.stderr  # full-size valid episode
    os.makedirs(task / "successes" / "ep_bad" / "robot_data")
    assert _cli(str(task), "--workers", "1").returncode == 1  # one invalid episode
