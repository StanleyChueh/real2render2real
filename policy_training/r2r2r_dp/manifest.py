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


class ManifestError(ValueError):
    pass


MANIFEST_KEYS = {"version", "salt", "source_root", "counts", "train_subsets", "episodes"}
OPTIONAL_KEYS = {"task", "episode_schema"}  # descriptive strings (build_manifest(extra=...))
EPISODE_KEYS = {"uid", "rel_path", "content_sha256", "num_steps", "max_joint_step", "split", "split_rank"}
SPLITS = ("train", "val", "test")


def _is_int(v) -> bool:
    return type(v) is int


def _is_sha256(v) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def check_manifest(m: Dict) -> None:
    """Raise ManifestError unless m is a well-formed version-1 manifest whose splits, ranks, counts and
    subsets are exactly what build_manifest produces for its salt. Explicit checks (effective under -O).

    Split-order policy: episodes sorted by sha256(salt:uid); the first counts.test are test, the next
    counts.val are val, the rest train, ranked in that order; episodes are listed by (split, rank)."""
    def fail(msg):
        raise ManifestError(f"manifest: {msg}")
    if not isinstance(m, dict) or not MANIFEST_KEYS <= set(m) <= MANIFEST_KEYS | OPTIONAL_KEYS:
        fail(f"keys must be {sorted(MANIFEST_KEYS)} plus optional {sorted(OPTIONAL_KEYS)}, "
             f"got {sorted(m) if isinstance(m, dict) else type(m).__name__}")
    if not all(isinstance(m[k], str) for k in OPTIONAL_KEYS & set(m)):
        fail(f"{sorted(OPTIONAL_KEYS)} must be strings")
    if not _is_int(m["version"]) or m["version"] != MANIFEST_VERSION:
        fail(f"unsupported version {m['version']!r} (supported: {MANIFEST_VERSION})")
    if not isinstance(m["salt"], str) or not m["salt"]:
        fail("salt must be a non-empty string")
    if not isinstance(m["source_root"], str) or not os.path.isabs(m["source_root"]):
        fail("source_root must be an absolute path")
    eps = m["episodes"]
    if not isinstance(eps, list) or not eps:
        fail("episodes must be a non-empty list")
    for e in eps:
        if not isinstance(e, dict) or set(e) != EPISODE_KEYS:
            fail(f"episode keys must be {sorted(EPISODE_KEYS)}: {e!r:.200}")
        rp = e["rel_path"]
        if (not isinstance(e["uid"], str) or not isinstance(rp, str) or os.path.isabs(rp)
                or os.path.normpath(rp) != rp or rp.split(os.sep)[0] == ".."):
            fail(f"{e.get('uid')!r}: uid/rel_path must be strings and rel_path a normalized relative path")
        if e["split"] not in SPLITS or not _is_int(e["split_rank"]) or e["split_rank"] < 0:
            fail(f"{e['uid']}: invalid split/split_rank {e['split']!r}/{e['split_rank']!r}")
        if not _is_sha256(e["content_sha256"]) or not _is_int(e["num_steps"]) or e["num_steps"] < 1:
            fail(f"{e['uid']}: invalid content_sha256 or num_steps")
    by_split: Dict[str, List[Dict]] = {s: [e for e in eps if e["split"] == s] for s in SPLITS}
    for key in ("uid", "rel_path", "content_sha256"):
        vals = [e[key] for e in eps]
        if len(set(vals)) != len(vals):
            fail(f"duplicate {key} values")
    for s, es in by_split.items():
        if sorted(e["split_rank"] for e in es) != list(range(len(es))):
            fail(f"{s} ranks are not unique and contiguous from 0")
    if eps != sorted(eps, key=lambda e: (e["split"], e["split_rank"])):
        fail("episodes are not listed in (split, split_rank) order")
    counts = m["counts"]
    if not isinstance(counts, dict) or set(counts) != set(SPLITS) or not all(_is_int(v) for v in counts.values()):
        fail(f"counts must give an integer for each of {list(SPLITS)}")
    actual = {s: len(es) for s, es in by_split.items()}
    if counts != actual:
        fail(f"counts {counts} != episodes per split {actual}")
    if min(actual.values()) < 1:
        fail(f"every split needs at least one episode: {actual}")
    order = sorted(eps, key=lambda e: order_key(m["salt"], e["uid"]))
    n_test, n_val = actual["test"], actual["val"]
    for i, e in enumerate(order):
        want = ("test", i) if i < n_test else ("val", i - n_test) if i < n_test + n_val else ("train", i - n_test - n_val)
        if (e["split"], e["split_rank"]) != want:
            fail(f"{e['uid']}: split/rank {(e['split'], e['split_rank'])} != {want} implied by the salt order")
    subs = m["train_subsets"]
    if not isinstance(subs, dict) or not subs:
        fail("train_subsets must be a non-empty mapping")
    train_order = [e["uid"] for e in by_split["train"]]
    prev: List[str] = []
    for n in sorted(subs, key=lambda k: int(k) if k.isdigit() else -1):
        if not (isinstance(n, str) and n.isdigit() and str(int(n)) == n and int(n) >= 1):
            fail(f"train subset key {n!r} is not a positive integer")
        sub = subs[n]
        if sub != train_order[: int(n)] or len(sub) != int(n):
            fail(f"subset {n} is not the first {n} train episodes in rank order")
        if sub[: len(prev)] != prev:
            fail("subsets are not nested")
        prev = sub
    if str(len(train_order)) not in subs:
        fail("train_subsets must include the full train split")


def save_manifest(m: Dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(m, f, indent=1, sort_keys=True, allow_nan=False)
        f.write("\n")


def _reject_constant(c):
    raise ManifestError(f"manifest: non-standard JSON constant {c}")


def load_manifest(path: str) -> Dict:
    with open(path) as f:
        m = json.load(f, parse_constant=_reject_constant)
    check_manifest(m)
    return m
