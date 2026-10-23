"""Non-destructive TinyDP view of R2R2R episodes.

TinyDP's sim loader (tinydp/dataset/image_dataset_sim.py) globs '**/*.h5' under a dataset root and reads
sidecar arrays next to each file:  <x>.h5 -> <x>_proprio.zarr (ee_poses), <x>_joint.zarr (joint_angles),
<x>_gripper_cmd.zarr (gripper_binary_cmd); images from <episode>/camera_{0,1}/rgb/{t:04d}.jpg where
<episode> = dirname(dirname(<x>.h5)). The authors create the sidecars with scripts/to_zarr_dataset.py,
which writes *into* the episode directories. This module writes the same arrays (same keys, dtypes,
zarr.save_array) into a separate view tree and symlinks the source h5 and camera directories, so the
source episodes are never modified and no image is copied.

View layout:
    <view_root>/<ep_name>/robot_data/robot_data.h5          -> symlink to source h5
    <view_root>/<ep_name>/robot_data/robot_data_{proprio,joint,gripper_cmd}.zarr
    <view_root>/<ep_name>/camera_0 -> source camera_0, camera_1 -> source camera_1
    <view_root>/<ep_name>/source.json                       uid, rel_path, content hash
ep_name = "ep_<split>_<rank:04d>" (contains no 'env', so TinyDP's subsample-by-env code never matches it).
"""
import json
import os
from typing import Dict, Iterable

import h5py
import zarr

from .manifest import content_hash
from .schema import H5_REL

SIDE_CARS = {"ee_poses": "_proprio.zarr", "joint_angles": "_joint.zarr", "gripper_binary_cmd": "_gripper_cmd.zarr"}


def _assert_outside(path: str, source_root: str) -> None:
    p, s = os.path.realpath(path), os.path.realpath(source_root)
    if p == s or p.startswith(s + os.sep) or s.startswith(p + os.sep):
        raise ValueError(f"view root {path} must not overlap the source root {source_root}")


def build_view(view_root: str, episodes: Iterable[Dict], source_root: str) -> int:
    """episodes: manifest entries (uid, rel_path, split, split_rank, content_sha256). Returns count."""
    _assert_outside(view_root, source_root)
    if os.path.exists(view_root):
        raise FileExistsError(f"{view_root} exists; refusing to overwrite")
    os.makedirs(view_root)
    n = 0
    for e in episodes:
        src = os.path.join(source_root, e["rel_path"])
        name = f"ep_{e['split']}_{e['split_rank']:04d}"
        dst = os.path.join(view_root, name)
        os.makedirs(os.path.join(dst, "robot_data"))
        h5_src = os.path.join(src, H5_REL)
        h5_dst = os.path.join(dst, H5_REL)
        os.symlink(h5_src, h5_dst)
        with h5py.File(h5_src, "r") as f:
            arrays = {k: f[k][:] for k in SIDE_CARS}
        if content_hash(arrays) != e["content_sha256"]:
            raise ValueError(f"{e['uid']}: content changed since the manifest was built")
        for key, suffix in SIDE_CARS.items():
            zarr.save_array(h5_dst.replace(".h5", suffix), arrays[key])
        for cam in ("camera_0", "camera_1"):
            os.symlink(os.path.join(src, cam), os.path.join(dst, cam))
        with open(os.path.join(dst, "source.json"), "w") as f:
            json.dump({k: e[k] for k in ("uid", "rel_path", "split", "split_rank", "content_sha256")}, f)
        n += 1
    return n


def build_all_views(manifest: Dict, out_root: str) -> Dict[str, str]:
    """One dataset root per split plus one per nested train subset (train subsets share val/test)."""
    src = manifest["source_root"]
    _assert_outside(out_root, src)
    eps = {e["uid"]: e for e in manifest["episodes"]}
    roots = {}
    for split in ("val", "test"):
        root = os.path.join(out_root, "splits", split)
        build_view(root, [e for e in manifest["episodes"] if e["split"] == split], src)
        roots[split] = root
    for n, uids in manifest["train_subsets"].items():
        root = os.path.join(out_root, "train_subsets", f"n{int(n):04d}")
        build_view(root, [eps[u] for u in uids], src)
        roots[f"train_n{int(n):04d}"] = root
    return roots
