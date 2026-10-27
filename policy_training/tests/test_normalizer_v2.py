"""Audit M1: the v2 normalizer covers every action target the released sim loader returns."""
import json
import os

import numpy as np
import pytest
import tyro
import zarr

from fixtures import episode_arrays, write_episode
from r2r2r_dp import normalizer_v2 as nv
from r2r2r_dp import tinydp_bridge as tb
from r2r2r_dp.adapter import SIDE_CARS
from tinydp.dataset.utils import default_vision_transform
from tinydp.util.args import ExperimentConfig

REAL_VIEWS = os.path.expanduser("~/Stanley_ws/r2r2r_policy_runs/dataset_v1/views")
ARTIFACTS = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "policy_training", "artifacts")


def _view(tmp, arrays):
    ep = write_episode(os.path.join(tmp, "view", "ep_train_0000"), arrays=arrays)
    h5 = os.path.join(ep, "robot_data", "robot_data.h5")
    for key, suffix in SIDE_CARS.items():
        zarr.save_array(h5.replace(".h5", suffix), arrays[key])
    with open(os.path.join(ep, "source.json"), "w") as f:
        json.dump({"uid": "run/ep"}, f)
    return os.path.join(tmp, "view")


def _dataset(view, stats, out):
    cfg = tyro.cli(ExperimentConfig, args=["--dataset-cfg.is-sim-data"])
    cfg.dataset_cfg.proprio_noise = 0.0
    cfg.logging_cfg.output_dir = out
    os.makedirs(out, exist_ok=True)
    cls = tb.make_manifest_dataset_class(view, view, stats)
    return cls(cfg.dataset_cfg, cfg.shared_cfg, cfg.logging_cfg, default_vision_transform(32), split="train"), cfg


def test_target_row_formula_matches_released_loader(tmp_path):
    view = _view(str(tmp_path), episode_arrays(60))
    st = nv.build_statistics(view, "n0001", ["run/ep"], "m", 2, 4, 16)
    json.dump(st, open(tmp_path / "s.json", "w"))
    ds, cfg = _dataset(view, str(tmp_path / "s.json"), str(tmp_path / "out"))
    s, P = 2, 16
    loader_rows = {r for _, start, end in ds.start_end for r in np.arange(start + s, end + s, s)}
    assert sorted(loader_rows) == nv.released_target_rows(60, 2, 4, 16).tolist()
    assert min(loader_rows) == 2 and max(loader_rows) == 60 - 3 and {r % 2 for r in loader_rows} == {0, 1}
    # windows = consecutive groups of seq_length items; the loss uses each window's last item
    last = {r for k in range(len(ds) // 4) for r in np.arange(ds.start_end[4 * k + 3][1] + s, ds.start_end[4 * k + 3][2] + s, s)}
    assert sorted(last) == nv.loss_target_rows(60, 2, 4, 16).tolist()
    assert sorted({ds.start_end[4 * k][1] for k in range(len(ds) // 4)}) == nv.released_window_starts(60, 2, 4, 16)


def test_odd_row_spike_v1_out_of_bounds_v2_covered(tmp_path):
    """Audit appendix counterexample: ee_poses[21, 7] = 2.0 (odd row, right x)."""
    a = episode_arrays(60)
    a["ee_poses"][21, 7] = 2.0
    view = _view(str(tmp_path), a)
    cfg = tyro.cli(ExperimentConfig, args=["--dataset-cfg.is-sim-data"])
    os.makedirs(tmp_path / "o1")
    tb.compute_action_statistics_tinydp(view, cfg.shared_cfg, cfg.dataset_cfg, str(tmp_path / "v1.json"))
    ds1, _ = _dataset(view, str(tmp_path / "v1.json"), str(tmp_path / "o1"))
    i = next(i for i, (_, start, _) in enumerate(ds1.start_end) if start == 19)
    assert ds1[i]["action"][0, 10].item() == pytest.approx(5.0)  # released normalizer: out of [-1, 1]

    st = nv.build_statistics(view, "n0001", ["run/ep"], "m", 2, 4, 16)
    json.dump(st, open(tmp_path / "v2.json", "w"))
    ds2, _ = _dataset(view, str(tmp_path / "v2.json"), str(tmp_path / "o2"))
    assert ds2[i]["action"][0, 10].item() == pytest.approx(1.0)
    worst = max(float(ds2[j]["action"][:, 10:20].abs().max()) for j in range(len(ds2)))
    assert worst <= 1.0 + 1e-6
    cov = nv.coverage(nv.view_episode_h5s(view), st, 2, 4, 16)
    assert cov["rows_out_of_bounds"] == 0 and cov["max_abs_normalized"] == pytest.approx(1.0)
    cov1 = nv.coverage(nv.view_episode_h5s(view), json.load(open(tmp_path / "v1.json")), 2, 4, 16)
    assert cov1["rows_out_of_bounds"] > 0


def test_v2_statistics_provenance_and_refusals(tmp_path):
    view = _view(str(tmp_path), episode_arrays(60))
    st = nv.build_statistics(view, "n0001", ["run/ep"], "abc", 2, 4, 16)
    assert st["normalizer_version"] == nv.NORMALIZER_VERSION and st["shape"] == [20]
    assert st["provenance"]["manifest_sha256"] == "abc" and st["provenance"]["train_uids_sha256"] == nv.uids_sha256(["run/ep"])
    assert st["config"]["action_mode"] == "absolute" and len(st["config_sha256"]) == 64
    with pytest.raises(ValueError):
        nv.build_statistics(view, "n0001", ["run/other"], "abc", 2, 4, 16)
    with pytest.raises(ValueError):
        nv.build_statistics(view, "n0001", ["run/ep"], "abc", 2, 4, 16, use_delta_action=True)


@pytest.mark.skipif(not os.path.isdir(REAL_VIEWS), reason="prepared real views not available")
def test_real_row15_counterexample(tmp_path):
    """Audit M1: n0150 ep_train_0076, row 15, right r00 (dim 13) normalises to -1.0807 under v1."""
    view = os.path.join(REAL_VIEWS, "train_subsets", "n0150")
    h5 = os.path.join(view, "ep_train_0076", "robot_data", "robot_data.h5")
    a = nv.episode_actions(h5)
    v1 = json.load(open(os.path.join(ARTIFACTS, "action_statistics_train_n0150.json")))
    v2 = json.load(open(os.path.join(ARTIFACTS, "normalizer_v2", "action_statistics_v2_train_n0150.json")))
    z1 = nv.normalized(a[15], np.array(v1["min_action"]), np.array(v1["max_action"]))[13]
    z2 = nv.normalized(a[15], np.array(v2["min_action"]), np.array(v2["max_action"]))[13]
    assert z1 == pytest.approx(-1.080700, abs=1e-5)
    assert -1.0 - 1e-9 <= z2 <= 1.0 + 1e-9
    assert 15 in nv.released_target_rows(161, 2, 4, 16)
