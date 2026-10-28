"""Pre-launch validation of a resolved protocol (audit M4). Runs before any model is allocated.

Verifies, and raises LaunchError listing every problem otherwise:
  * the manifest file hash equals the protocol's pin, and the manifest is semantically valid (version,
    keys, counts, unique contiguous ranks, salt-implied split order, nested subsets; manifest.py);
  * each split root contains exactly the manifest's episodes for that split / train subset, judged by
    the same recursive '**/*.h5' glob TinyDP uses (so stray files cannot enter training);
  * split roots and episode directories are real directories (no symlink aliases) and every episode
    directory has its canonical name ep_<split>_<rank:04d>, so TinyDP's sorted file order -- the sample
    order -- is fixed by the manifest;
  * every view episode's source.json, h5/camera symlinks, sidecar arrays (content hash recomputed) and
    source HDF5 (content hash recomputed) match the manifest; camera frame counts match the episode;
  * every camera's frame bytes match the pinned image-integrity manifest (image_integrity.py), which
    certifies that those bytes decode as RGB JPEGs of the declared resolution;
  * train / eval / test are pairwise disjoint by uid, resolved HDF5 path and content hash;
  * the statistics file is strict JSON, a v2 normalizer bound to this manifest, subset, train
    membership and action convention, its complete config (incl. layout) equals the expected one and
    its config_sha256 is recomputed, and its finite bounds equal a recomputation from the train sidecars;
  * the output directory does not exist yet (or is empty).
"""
import glob
import json
import os
import time
from typing import Dict, List

import h5py
import numpy as np
import zarr

from .adapter import SIDE_CARS
from .image_integrity import ImageIntegrityError, load_image_integrity, verify_episodes
from .manifest import ManifestError, content_hash, load_manifest
from .normalizer_v2 import (NORMALIZER_VERSION, compute_bounds, config_sha256, file_sha256, load_statistics,
                            statistics_config, uids_sha256)
from .schema import H5_REL


class LaunchError(RuntimeError):
    def __init__(self, problems: List[str]):
        self.problems = problems
        super().__init__("launch validation failed:\n  " + "\n  ".join(problems[:50]) +
                         (f"\n  ... {len(problems) - 50} more" if len(problems) > 50 else ""))


def _check_root(root: str, split: str, expected_uids: List[str], by_uid: Dict, source_root: str,
                problems: List[str]) -> Dict[str, Dict]:
    """Return {uid: {'h5_real', 'content'}} for the episodes found under root."""
    found: Dict[str, Dict] = {}
    if not os.path.isdir(root):
        problems.append(f"{root}: missing")
        return found
    if os.path.islink(root):
        problems.append(f"{root}: split root is a symlink alias")
    h5s = sorted(glob.glob("**/*.h5", root_dir=root, recursive=True))  # exactly TinyDP's file list
    for rel in h5s:
        parts = rel.split(os.sep)
        if len(parts) != 3 or parts[1:] != H5_REL.split(os.sep):
            problems.append(f"{root}: unexpected h5 file {rel}")
            continue
        ep = os.path.join(root, parts[0])
        sj = os.path.join(ep, "source.json")
        if not os.path.isfile(sj):
            problems.append(f"{ep}: no source.json")
            continue
        src = json.load(open(sj))
        uid = src.get("uid")
        m = by_uid.get(uid)
        if m is None:
            problems.append(f"{ep}: uid {uid!r} not in manifest")
            continue
        if uid in found:
            problems.append(f"{root}: uid {uid} appears twice")
            continue
        canonical = f"ep_{m['split']}_{m['split_rank']:04d}"
        if parts[0] != canonical:
            problems.append(f"{ep}: directory name must be {canonical} (it fixes TinyDP's sample order)")
        if os.path.islink(ep) or os.path.islink(os.path.join(ep, "robot_data")):
            problems.append(f"{ep}: episode directory is a symlink alias")
        for k in ("rel_path", "split", "split_rank", "content_sha256"):
            if src.get(k) != m[k]:
                problems.append(f"{ep}: source.json {k}={src.get(k)!r} != manifest {m[k]!r}")
        if m["split"] != split:
            problems.append(f"{ep}: manifest split {m['split']} != root split {split}")
        h5 = os.path.join(root, rel)
        src_ep = os.path.join(source_root, m["rel_path"])
        if os.path.realpath(h5) != os.path.realpath(os.path.join(src_ep, H5_REL)):
            problems.append(f"{ep}: h5 does not resolve to the manifest source")
        for cam in ("camera_0", "camera_1"):
            if os.path.realpath(os.path.join(ep, cam)) != os.path.realpath(os.path.join(src_ep, cam)):
                problems.append(f"{ep}: {cam} does not resolve to the manifest source")
        side = {}
        for key, suffix in SIDE_CARS.items():
            p = h5.replace(".h5", suffix)
            if not os.path.isdir(p):
                problems.append(f"{ep}: missing sidecar {os.path.basename(p)}")
                continue
            try:
                side[key] = zarr.load(p)
            except Exception as e:  # corrupted store
                problems.append(f"{ep}: unreadable sidecar {os.path.basename(p)}: {e}")
        if len(side) == len(SIDE_CARS) and content_hash(side) != m["content_sha256"]:
            problems.append(f"{ep}: sidecar contents differ from the manifest (stale or corrupted)")
        try:
            with h5py.File(h5, "r") as f:
                if content_hash({k: f[k][:] for k in SIDE_CARS}) != m["content_sha256"]:
                    problems.append(f"{ep}: source HDF5 contents differ from the manifest")
        except Exception as e:
            problems.append(f"{ep}: unreadable source HDF5: {e}")
        T = m["num_steps"]
        for cam in ("camera_0", "camera_1"):
            d = os.path.join(ep, cam, "rgb")
            names = sorted(os.listdir(d)) if os.path.isdir(d) else []
            if names != [f"{t:04d}.jpg" for t in range(T)]:
                problems.append(f"{ep}: {cam}/rgb does not hold frames 0000..{T - 1:04d}")
        found[uid] = {"h5_real": os.path.realpath(h5), "content": m["content_sha256"], "ep": ep, "T": T}
    got, want = set(found), set(expected_uids)
    if got != want:
        problems.append(f"{root}: membership differs from manifest ({len(got - want)} unexpected, "
                        f"{len(want - got)} missing)")
    return found


STATISTICS_KEYS = {"shape", "min_action", "max_action", "normalizer_version", "provenance", "config", "config_sha256"}
PROVENANCE_KEYS = {"subset", "n_episodes", "n_target_rows", "manifest_sha256", "train_uids_sha256", "code_sha256"}


def check_statistics(stats_path: str, resolved: Dict, train_uids: List[str], manifest_sha: str, train_root: str,
                     problems: List[str], recompute: bool = True) -> Dict:
    if not os.path.isfile(stats_path):
        problems.append(f"{stats_path}: missing statistics file")
        return {}
    try:
        st = load_statistics(stats_path)  # strict JSON, finite ordered 20-D bounds
    except (ValueError, TypeError) as e:
        problems.append(f"{stats_path}: {e}")
        return {}
    if set(st) != STATISTICS_KEYS or set(st.get("provenance", {})) != PROVENANCE_KEYS:
        problems.append(f"{stats_path}: keys must be {sorted(STATISTICS_KEYS)} with provenance {sorted(PROVENANCE_KEYS)}")
        return st
    t = resolved["tinydp"]
    if t["shared_cfg"]["use_delta_action"]:
        problems.append("normalizer v2 supports absolute actions only")
    s, L, P = t["dataset_cfg"]["subsample_steps"], t["shared_cfg"]["seq_length"], t["shared_cfg"]["num_pred_steps"]
    prov = st["provenance"]
    want = {
        "normalizer_version": (st["normalizer_version"], NORMALIZER_VERSION),
        "shape": (st["shape"], [20]),
        "subset": (prov["subset"], f"n{resolved['train_size']:04d}"),
        "n_episodes": (prov["n_episodes"], len(train_uids)),
        "manifest_sha256": (prov["manifest_sha256"], manifest_sha),
        "train_uids_sha256": (prov["train_uids_sha256"], uids_sha256(train_uids)),
        "config": (st["config"], statistics_config(s, L, P)),
        "config_sha256": (st["config_sha256"], config_sha256(st["config"])),
    }
    for k, (got, exp) in want.items():
        if got != exp:
            problems.append(f"{stats_path}: {k}={got!r}, expected {exp!r}")
    if recompute and not problems and os.path.isdir(train_root):
        h5s = sorted(os.path.join(train_root, h) for h in glob.glob("**/*.h5", root_dir=train_root, recursive=True))
        try:
            lo, hi, n_rows = compute_bounds(h5s, s, L, P)
        except ValueError as e:
            problems.append(f"{train_root}: {e}")
            return st
        if not (np.array_equal(lo, st["min_action"]) and np.array_equal(hi, st["max_action"])):
            problems.append(f"{stats_path}: bounds differ from a recomputation over {train_root}")
        if n_rows != prov["n_target_rows"]:
            problems.append(f"{stats_path}: n_target_rows {prov['n_target_rows']} != recomputed {n_rows}")
    return st


def check_launch(resolved: Dict, allow_existing_output: bool = False) -> Dict:
    p, paths = resolved["protocol"], resolved["paths"]
    problems: List[str] = []
    mpath = paths["manifest"]
    if not os.path.isfile(mpath):
        raise LaunchError([f"{mpath}: manifest missing"])
    msha = file_sha256(mpath)
    if msha != p["data"]["manifest_sha256"]:
        raise LaunchError([f"{mpath}: sha256 {msha} != pinned {p['data']['manifest_sha256']}"])
    try:
        m = load_manifest(mpath)
    except ManifestError as e:
        raise LaunchError([str(e)])
    n = str(resolved["train_size"])
    if n not in m["train_subsets"]:
        raise LaunchError([f"train size {n} is not a manifest subset {sorted(m['train_subsets'], key=int)}"])
    roots = {"train": paths["train_root"], "val": paths["eval_root"], "test": paths["test_root"]}
    reals = {k: os.path.realpath(v) for k, v in roots.items()}
    for a in reals:
        for b in reals:
            if a < b and (reals[a] == reals[b] or reals[a].startswith(reals[b] + os.sep)
                          or reals[b].startswith(reals[a] + os.sep)):
                problems.append(f"{a} root {roots[a]} overlaps {b} root {roots[b]}")
    if problems:
        raise LaunchError(problems)
    by_uid = {e["uid"]: e for e in m["episodes"]}
    expected = {"train": m["train_subsets"][n],
                "val": [e["uid"] for e in m["episodes"] if e["split"] == "val"],
                "test": [e["uid"] for e in m["episodes"] if e["split"] == "test"]}
    found = {k: _check_root(roots[k], k, expected[k], by_uid, m["source_root"], problems) for k in roots}
    for key in ("h5_real", "content"):
        seen = {}
        for split, eps in found.items():
            for info in eps.values():
                v = info[key]
                if v in seen and seen[v] != split:
                    problems.append(f"{key} {v} in both {seen[v]} and {split}")
                seen[v] = split
    for a in found:
        for b in found:
            if a < b and set(found[a]) & set(found[b]):
                problems.append(f"uids shared by {a} and {b}: {len(set(found[a]) & set(found[b]))}")
    st = check_statistics(paths["action_statistics"], resolved, expected["train"], msha, roots["train"], problems)
    images = check_images(paths["image_integrity"], p["data"]["image_integrity_sha256"], m, msha, found, problems)
    out = paths["output_dir"]
    if not allow_existing_output and os.path.exists(out) and os.listdir(out):
        problems.append(f"{out}: output directory exists and is not empty (use --resume to continue a run)")
    if problems:
        raise LaunchError(problems)
    return {"manifest_sha256": msha, "counts": {k: len(v) for k, v in found.items()},
            "train_uids_sha256": uids_sha256(expected["train"]),
            "action_statistics_sha256": file_sha256(paths["action_statistics"]),
            "normalizer_version": st["normalizer_version"], "images": images,
            "roots_realpath": reals}


def check_images(path: str, pinned_sha: str, manifest: Dict, manifest_sha: str, found: Dict[str, Dict],
                 problems: List[str], workers: int = min(16, os.cpu_count() or 1)) -> Dict:
    """Recompute every camera digest of every episode found in the split roots (frame bytes only)."""
    if not os.path.isfile(path):
        problems.append(f"{path}: image-integrity manifest missing")
        return {}
    sha = file_sha256(path)
    if sha != pinned_sha:
        problems.append(f"{path}: sha256 {sha} != pinned {pinned_sha}")
        return {}
    try:
        ii = load_image_integrity(path, manifest, manifest_sha)
    except ImageIntegrityError as e:
        problems.append(str(e))
        return {}
    t0 = time.time()
    eps = [(uid, info["ep"], info["T"]) for split in found.values() for uid, info in sorted(split.items())]
    problems.extend(verify_episodes(eps, ii, workers))
    return {"image_integrity_sha256": sha, "episodes_verified": len(eps),
            "frames_verified": 2 * sum(T for _, _, T in eps), "seconds": round(time.time() - t0, 1)}
