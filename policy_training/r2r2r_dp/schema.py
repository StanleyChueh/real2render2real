"""Expected on-disk schema of one R2R2R rollout and a strict validator for it.

Episode layout written by the unmodified generator (BatchDataLogger):

    <episode>/robot_data/robot_data.h5      ee_poses (T,14) f32, joint_angles (T,16) f32, gripper_binary_cmd (T,2) bool
    <episode>/robot_data/joint_names.txt    16 names, one per line
    <episode>/camera_0/rgb/0000.jpg ... {T-1:04d}.jpg
    <episode>/camera_1/rgb/0000.jpg ...

ee_poses = [left xyz, left quat wxyz, right xyz, right quat wxyz] of {left,right}_dummy_point in the
robot base_link frame (verified against URDF forward kinematics, see 01_dataset_preparation.md).
"""
import dataclasses
import os
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from PIL import Image

YUMI_JOINT_NAMES = (
    "yumi_joint_1_l", "yumi_joint_1_r", "yumi_joint_2_l", "yumi_joint_2_r", "yumi_joint_7_l", "yumi_joint_7_r",
    "yumi_joint_3_l", "yumi_joint_3_r", "yumi_joint_4_l", "yumi_joint_4_r", "yumi_joint_5_l", "yumi_joint_5_r",
    "yumi_joint_6_l", "yumi_joint_6_r", "gripper_l_joint", "gripper_r_joint",
)


@dataclasses.dataclass(frozen=True)
class EpisodeSchema:
    num_steps: int = 161
    cameras: Tuple[str, ...] = ("camera_0", "camera_1")
    image_size: Tuple[int, int] = (1280, 720)  # (width, height) as reported by PIL
    joint_names: Tuple[str, ...] = YUMI_JOINT_NAMES
    ee_dim: int = 14
    joint_dim: int = 16
    gripper_dim: int = 2
    quat_norm_tol: float = 1e-3
    # coffee-maker: only the right arm grasps (left command never set); one close and one open.
    active_gripper: Optional[int] = 1
    expected_gripper_toggles: Optional[int] = 2


YUMI_COFFEE_MAKER = EpisodeSchema()

H5_REL = os.path.join("robot_data", "robot_data.h5")
NAMES_REL = os.path.join("robot_data", "joint_names.txt")


def frame_name(i: int) -> str:
    return f"{i:04d}.jpg"


def _check_h5(path: str, schema: EpisodeSchema, errors: List[str]) -> Dict:
    info: Dict = {}
    try:
        f = h5py.File(path, "r")
    except Exception as e:  # unreadable / truncated file
        errors.append(f"h5 open failed: {e}")
        return info
    with f:
        expected = {
            "ee_poses": (schema.ee_dim, np.float32),
            "joint_angles": (schema.joint_dim, np.float32),
            "gripper_binary_cmd": (schema.gripper_dim, np.bool_),
        }
        if set(f.keys()) != set(expected):
            errors.append(f"h5 keys {sorted(f.keys())} != {sorted(expected)}")
        arrays = {}
        for key, (dim, dtype) in expected.items():
            if key not in f:
                continue
            try:
                a = f[key][:]
            except Exception as e:
                errors.append(f"{key}: read failed: {e}")
                continue
            if a.dtype != dtype:
                errors.append(f"{key}: dtype {a.dtype} != {np.dtype(dtype)}")
            if a.shape != (schema.num_steps, dim):
                errors.append(f"{key}: shape {a.shape} != {(schema.num_steps, dim)}")
            arrays[key] = a
    for key in ("ee_poses", "joint_angles"):
        a = arrays.get(key)
        if a is not None and a.dtype.kind == "f" and not np.isfinite(a).all():
            errors.append(f"{key}: non-finite values")
    ee = arrays.get("ee_poses")
    if ee is not None and ee.ndim == 2 and ee.shape[1] == schema.ee_dim and np.isfinite(ee).all():
        for name, sl in (("left", slice(3, 7)), ("right", slice(10, 14))):
            dev = np.abs(np.linalg.norm(ee[:, sl], axis=1) - 1.0).max()
            if dev > schema.quat_norm_tol:
                errors.append(f"ee_poses {name} quaternion norm deviates by {dev:.2e}")
    g = arrays.get("gripper_binary_cmd")
    if g is not None and g.dtype == np.bool_ and g.ndim == 2 and g.shape[1] == schema.gripper_dim and len(g) > 1:
        toggles = (np.diff(g.astype(np.int8), axis=0) != 0).sum(axis=0)
        info["gripper_toggles"] = toggles.tolist()
        if schema.active_gripper is not None:
            for k in range(schema.gripper_dim):
                if k != schema.active_gripper and g[:, k].any():
                    errors.append(f"gripper[{k}] (inactive arm) is set")
            if schema.expected_gripper_toggles is not None:
                k = schema.active_gripper
                if toggles[k] != schema.expected_gripper_toggles:
                    errors.append(f"gripper[{k}] toggles {toggles[k]} != {schema.expected_gripper_toggles}")
                if g[0, k] or g[-1, k]:
                    errors.append(f"gripper[{k}] must start and end open")
    lengths = {k: len(v) for k, v in arrays.items()}
    if len(set(lengths.values())) > 1:
        errors.append(f"dataset lengths differ: {lengths}")
    info["T"] = int(next(iter(lengths.values()))) if lengths else 0
    return info


def _check_images(ep_dir: str, schema: EpisodeSchema, decode: str, errors: List[str]) -> Dict:
    """decode: 'header' (open + size/mode for every frame), 'all' (full decode of every frame)."""
    info: Dict = {}
    cams = sorted(d for d in os.listdir(ep_dir) if d.startswith("camera_"))
    if tuple(cams) != tuple(sorted(schema.cameras)):
        errors.append(f"cameras {cams} != {list(schema.cameras)}")
    expected = [frame_name(i) for i in range(schema.num_steps)]
    for cam in schema.cameras:
        rgb = os.path.join(ep_dir, cam, "rgb")
        if not os.path.isdir(rgb):
            errors.append(f"{cam}/rgb missing")
            continue
        files = sorted(os.listdir(rgb))
        if files != expected:
            missing = sorted(set(expected) - set(files))
            extra = sorted(set(files) - set(expected))
            errors.append(f"{cam}: frame ids mismatch (missing {missing[:5]}{'...' if len(missing) > 5 else ''}, "
                          f"extra {extra[:5]}{'...' if len(extra) > 5 else ''})")
        bad = 0
        for name in files:
            if name not in expected:
                continue
            fp = os.path.join(rgb, name)
            try:
                with Image.open(fp) as im:
                    if im.format != "JPEG" or im.mode != "RGB" or im.size != schema.image_size:
                        bad += 1
                        if bad <= 3:
                            errors.append(f"{cam}/{name}: {im.format} {im.mode} {im.size} != JPEG RGB {schema.image_size}")
                        continue
                    if decode == "all":
                        im.load()  # full entropy decode; raises on truncated/corrupt data
            except Exception as e:
                bad += 1
                if bad <= 3:
                    errors.append(f"{cam}/{name}: unreadable ({type(e).__name__}: {e})")
        info[f"{cam}_frames"] = len(files)
        if bad > 3:
            errors.append(f"{cam}: {bad} bad frames in total")
    info["image_decode"] = decode
    return info


def validate_episode(ep_dir: str, schema: EpisodeSchema = YUMI_COFFEE_MAKER, decode: str = "header") -> Dict:
    """Return {'episode', 'errors': [...], ...}. An episode is valid iff errors is empty."""
    assert decode in ("header", "all")
    errors: List[str] = []
    rec: Dict = {"episode": ep_dir}
    h5_path = os.path.join(ep_dir, H5_REL)
    if not os.path.isfile(h5_path):
        errors.append("robot_data.h5 missing")
    else:
        rec.update(_check_h5(h5_path, schema, errors))
    names_path = os.path.join(ep_dir, NAMES_REL)
    if not os.path.isfile(names_path):
        errors.append("joint_names.txt missing")
    else:
        names = tuple(open(names_path).read().split("\n"))
        if names != schema.joint_names:
            errors.append(f"joint names {names[:3]}... do not match expected order")
    if os.path.isdir(ep_dir):
        rec.update(_check_images(ep_dir, schema, decode, errors))
    else:
        errors.append("episode directory missing")
    rec["errors"] = errors
    return rec


def load_episode_arrays(ep_dir: str) -> Dict[str, np.ndarray]:
    with h5py.File(os.path.join(ep_dir, H5_REL), "r") as f:
        return {k: f[k][:] for k in ("ee_poses", "joint_angles", "gripper_binary_cmd")}
