"""Deterministic disjoint splits, non-destructive TinyDP views, equivalence with the official converter."""
import glob
import json
import os
import random

import h5py
import numpy as np
import pytest
import zarr

from fixtures import make_source_tree
from r2r2r_dp import adapter, manifest as mf, verify


@pytest.fixture
def source(tmp_path):
    root = str(tmp_path / "src")
    eps = make_source_tree(root, 12)
    return root, eps


def _m(root, eps, **kw):
    kw = {"n_val": 2, "n_test": 2, "subset_sizes": (3, 5), **kw}
    return mf.build_manifest(eps, root, **kw)


def test_manifest_deterministic_and_order_invariant(source):
    root, eps = source
    a = _m(root, eps)
    shuffled = eps[:]
    random.Random(1).shuffle(shuffled)
    b = _m(root, shuffled)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert _m(root, eps, salt="other")["episodes"] != a["episodes"]


def test_manifest_disjoint_fixed_eval_nested_subsets(source):
    root, eps = source
    m = _m(root, eps)
    split = {e["uid"]: e["split"] for e in m["episodes"]}
    assert m["counts"] == {"train": 8, "val": 2, "test": 2}
    assert sorted(m["train_subsets"]) == ["3", "5", "8"]
    for n, uids in m["train_subsets"].items():
        assert all(split[u] == "train" for u in uids)
    assert m["train_subsets"]["5"][:3] == m["train_subsets"]["3"]
    # val/test do not depend on which subset sizes are requested
    other = _m(root, eps, subset_sizes=(1, 2, 7))
    for s in ("val", "test"):
        assert [e["uid"] for e in other["episodes"] if e["split"] == s] == [e["uid"] for e in m["episodes"] if e["split"] == s]


def test_manifest_rejects_duplicate_content_and_tampering(source, tmp_path):
    root, eps = source
    dup = os.path.join(root, "run_c", "output_data", "yumi_coffee_maker", "successes", "env_9_dup")
    import shutil
    shutil.copytree(eps[0], dup)
    with pytest.raises(ValueError, match="duplicated"):
        _m(root, eps + [dup])
    m = _m(root, eps)
    victim = next(e for e in m["episodes"] if e["split"] == "val")
    clone = dict(victim, split="train", split_rank=8)
    m["episodes"].append(clone)
    with pytest.raises(mf.ManifestError):  # explicit exception (was AssertionError, lost under python -O)
        mf.check_manifest(m)


def test_view_matches_official_converter_and_leaves_source_untouched(source, tmp_path):
    root, eps = source
    m = _m(root, eps)
    before = verify.tree_fingerprint(eps)
    roots = adapter.build_all_views(m, str(tmp_path / "views"))
    after = verify.tree_fingerprint(eps)
    assert before == after
    assert not glob.glob(os.path.join(root, "**", "*.zarr"), recursive=True), "sidecars written into the source"
    by_uid = {e["uid"]: e for e in m["episodes"]}
    for key, r in roots.items():
        entries = [e for e in m["episodes"] if e["split"] == key] if key in ("val", "test") else \
            [by_uid[u] for u in m["train_subsets"][str(int(key.split("_n")[1]))]]
        assert verify.verify_view(r, root, entries) == []
    # no source episode appears in two split roots (subsets are all train)
    seen = {}
    for key in ("val", "test", "train_n0008"):
        for s in glob.glob(os.path.join(roots[key], "*", "source.json")):
            uid = json.load(open(s))["uid"]
            assert uid not in seen, f"{uid} in {seen.get(uid)} and {key}"
            seen[uid] = key
    assert len(seen) == 12


def test_view_sidecars_equal_source_arrays(source, tmp_path):
    root, eps = source
    m = _m(root, eps)
    e = m["episodes"][0]
    view = str(tmp_path / "v")
    adapter.build_view(view, [e], root)
    rd = os.path.join(view, f"ep_{e['split']}_{e['split_rank']:04d}", "robot_data")
    with h5py.File(os.path.join(root, e["rel_path"], "robot_data", "robot_data.h5"), "r") as f:
        for key, suffix in adapter.SIDE_CARS.items():
            z = zarr.load(os.path.join(rd, "robot_data" + suffix))
            assert z.dtype == f[key].dtype and np.array_equal(z, f[key][:])


def test_adapter_refuses_overlap_existing_and_changed_source(source, tmp_path):
    root, eps = source
    m = _m(root, eps)
    with pytest.raises(ValueError, match="overlap"):
        adapter.build_view(os.path.join(root, "views"), m["episodes"][:1], root)
    view = str(tmp_path / "v")
    adapter.build_view(view, m["episodes"][:1], root)
    with pytest.raises(FileExistsError):
        adapter.build_view(view, m["episodes"][:1], root)
    e = m["episodes"][1]
    with h5py.File(os.path.join(root, e["rel_path"], "robot_data", "robot_data.h5"), "a") as f:
        f["ee_poses"][0, 0] += 1.0
    with pytest.raises(ValueError, match="content changed"):
        adapter.build_view(str(tmp_path / "v2"), [e], root)
