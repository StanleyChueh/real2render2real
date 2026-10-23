"""Deterministic, disjoint train/val/test manifest over generated episodes.

Assignment depends only on (salt, uid) where uid = "<run_dir>/<episode_dir>": episodes are ordered by
sha256(salt + ":" + uid); the first n_test go to test, the next n_val to val, the rest to train in
that order. Train subsets for dataset-size experiments are prefixes of the train order (nested), so
val/test are identical for every subset size. Episode content hashes are recorded so duplicated
trajectories can never straddle splits.
"""
import hashlib
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .schema import load_episode_arrays

MANIFEST_VERSION = 1


def episode_uid(ep_dir: str) -> str:
    """'<run>/<episode>' for <root>/<run>/output_data/<task>/successes/<episode>."""
    parts = os.path.normpath(ep_dir).split(os.sep)
    assert parts[-2] == "successes" and parts[-4] == "output_data", ep_dir
    return f"{parts[-5]}/{parts[-1]}"


def order_key(salt: str, uid: str) -> str:
    return hashlib.sha256(f"{salt}:{uid}".encode()).hexdigest()


def content_hash(arrays: Dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(arrays[k])
        h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def build_manifest(
    episode_dirs: Iterable[str],
    source_root: str,
    salt: str = "r2r2r-dp-v1",
    n_val: int = 182,
    n_test: int = 182,
    subset_sizes: Sequence[int] = (50, 100, 150, 1000),
    extra: Optional[Dict] = None,
) -> Dict:
    source_root = os.path.realpath(source_root)
    eps = []
    for d in episode_dirs:
        real = os.path.realpath(d)
        assert real.startswith(source_root + os.sep), f"{d} not under {source_root}"
        arrays = load_episode_arrays(real)
        ja = arrays["joint_angles"].astype(np.float64)
        eps.append({
            "uid": episode_uid(real),
            "rel_path": os.path.relpath(real, source_root),
            "content_sha256": content_hash(arrays),
            "num_steps": int(len(ja)),
            "max_joint_step": float(np.abs(np.diff(ja[:, :14], axis=0)).max()),
        })
    uids = [e["uid"] for e in eps]
    if len(set(uids)) != len(uids):
        raise ValueError("duplicate episode uids")
    hashes = [e["content_sha256"] for e in eps]
    dup = {h for h in hashes if hashes.count(h) > 1}
    if dup:
        raise ValueError(f"{len(dup)} duplicated episode contents; resolve before splitting")
    if n_val + n_test >= len(eps):
        raise ValueError("not enough episodes for the requested val/test sizes")

    eps.sort(key=lambda e: order_key(salt, e["uid"]))
    for i, e in enumerate(eps):
        if i < n_test:
            e["split"], e["split_rank"] = "test", i
        elif i < n_test + n_val:
            e["split"], e["split_rank"] = "val", i - n_test
        else:
            e["split"], e["split_rank"] = "train", i - n_test - n_val
    train = [e["uid"] for e in eps if e["split"] == "train"]
    sizes = sorted(set(int(s) for s in subset_sizes if s <= len(train)) | {len(train)})
    manifest = {
        "version": MANIFEST_VERSION,
        "salt": salt,
        "source_root": source_root,
        "counts": {s: sum(e["split"] == s for e in eps) for s in ("train", "val", "test")},
        "train_subsets": {str(n): train[:n] for n in sizes},
        "episodes": sorted(eps, key=lambda e: (e["split"], e["split_rank"])),
    }
    if extra:
        manifest.update(extra)
    check_manifest(manifest)
    return manifest


def check_manifest(m: Dict) -> None:
    """Raise AssertionError if splits overlap or subsets are inconsistent."""
    by_split: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}
    for e in m["episodes"]:
        by_split[e["split"]].append(e)
    for key in ("uid", "rel_path", "content_sha256"):
        seen = {}
        for s, es in by_split.items():
            for e in es:
                assert e[key] not in seen, f"{key} {e[key]} in both {seen.get(e[key])} and {s}"
                seen[e[key]] = s
    train_order = [e["uid"] for e in sorted(by_split["train"], key=lambda e: e["split_rank"])]
    prev: List[str] = []
    for n in sorted(m["train_subsets"], key=int):
        sub = m["train_subsets"][n]
        assert len(sub) == int(n) and sub == train_order[: int(n)], f"subset {n} is not a train prefix"
        assert sub[: len(prev)] == prev, "subsets are not nested"
        prev = sub
    for s, es in by_split.items():
        assert [e["split_rank"] for e in sorted(es, key=lambda e: e["split_rank"])] == list(range(len(es)))


def save_manifest(m: Dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(m, f, indent=1, sort_keys=True)
        f.write("\n")


def load_manifest(path: str) -> Dict:
    with open(path) as f:
        m = json.load(f)
    check_manifest(m)
    return m
