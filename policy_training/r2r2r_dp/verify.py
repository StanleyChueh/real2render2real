"""Checks that a TinyDP view is a faithful, non-destructive rendering of the source episodes."""
import hashlib
import json
import os
import shutil
import tempfile
from typing import Dict, Iterable, List

import numpy as np
import zarr

from .adapter import SIDE_CARS
from .schema import H5_REL, load_episode_arrays

OFFICIAL_CONVERTER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "to_zarr_dataset.py")
OFFICIAL_ROOT_LITERAL = 'root_dir = "/home/yujustin/xi/output_data/franka_coffee_maker/successes"'


def tree_fingerprint(dirs: Iterable[str]) -> str:
    """Hash of (relative path, size, mtime_ns) of every file below the given directories, plus h5 bytes."""
    h = hashlib.sha256()
    for d in sorted(dirs):
        for root, subdirs, files in os.walk(d):
            subdirs.sort()
            for name in sorted(files):
                p = os.path.join(root, name)
                st = os.stat(p)
                h.update(f"{os.path.relpath(p, d)}|{st.st_size}|{st.st_mtime_ns}".encode())
                if name.endswith(".h5"):
                    with open(p, "rb") as f:
                        h.update(f.read())
    return h.hexdigest()


def run_official_converter(root_dir: str) -> None:
    """Execute the authors' scripts/to_zarr_dataset.py on root_dir (only its hardcoded path is replaced)."""
    src = open(OFFICIAL_CONVERTER).read()
    assert src.count(OFFICIAL_ROOT_LITERAL) == 1, "official converter changed; refusing to patch"
    src = src.replace(OFFICIAL_ROOT_LITERAL, f"root_dir = {root_dir!r}")
    exec(compile(src, OFFICIAL_CONVERTER, "exec"), {"__name__": "__main__"})


def _zarr_meta(path: str) -> Dict:
    with open(os.path.join(path, ".zarray")) as f:
        return json.load(f)


def verify_view(view_root: str, source_root: str, entries: List[Dict]) -> List[str]:
    """Return a list of problems (empty == verified). entries: manifest entries expected in the view."""
    problems: List[str] = []
    source_root = os.path.realpath(source_root)
    names = sorted(os.listdir(view_root))
    expected = sorted(f"ep_{e['split']}_{e['split_rank']:04d}" for e in entries)
    if names != expected:
        problems.append(f"view entries {len(names)} != expected {len(expected)}")
    by_name = {f"ep_{e['split']}_{e['split_rank']:04d}": e for e in entries}
    tmp = tempfile.mkdtemp(prefix="r2r2r_official_")
    try:
        # copy only the h5 files (KBs) so the official converter can write next to them
        for name in names:
            e = by_name.get(name)
            if e is None:
                continue
            d = os.path.join(tmp, name, "robot_data")
            os.makedirs(d)
            shutil.copy2(os.path.join(source_root, e["rel_path"], H5_REL), d)
        run_official_converter(tmp)
        for name in names:
            e = by_name.get(name)
            if e is None:
                continue
            ep = os.path.join(view_root, name)
            src = os.path.join(source_root, e["rel_path"])
            for link, target in ((os.path.join(ep, H5_REL), os.path.join(src, H5_REL)),
                                 (os.path.join(ep, "camera_0"), os.path.join(src, "camera_0")),
                                 (os.path.join(ep, "camera_1"), os.path.join(src, "camera_1"))):
                if not os.path.islink(link) or os.path.realpath(link) != os.path.realpath(target):
                    problems.append(f"{name}: {os.path.relpath(link, ep)} does not link to the source")
            info = json.load(open(os.path.join(ep, "source.json")))
            if info["uid"] != e["uid"] or info["content_sha256"] != e["content_sha256"]:
                problems.append(f"{name}: source.json mismatch")
            arrays = load_episode_arrays(src)
            for key, suffix in SIDE_CARS.items():
                mine = os.path.join(ep, "robot_data", "robot_data" + suffix)
                theirs = os.path.join(tmp, name, "robot_data", "robot_data" + suffix)
                a, b = zarr.load(mine), zarr.load(theirs)
                if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
                    problems.append(f"{name}: {suffix} differs from the official converter")
                if not np.array_equal(a, arrays[key]) or a.dtype != arrays[key].dtype:
                    problems.append(f"{name}: {suffix} differs from the source h5")
                if _zarr_meta(mine) != _zarr_meta(theirs):
                    problems.append(f"{name}: {suffix} zarr metadata differs from the official converter")
    finally:
        shutil.rmtree(tmp)
    return problems
