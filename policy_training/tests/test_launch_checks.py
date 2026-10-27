"""Audit M4: the v2 launch path rejects every unsafe data / configuration combination before model allocation."""
import copy
import json
import os
import shutil

import numpy as np
import pytest
import zarr

import train_r2r2r_dp
from r2r2r_dp.launch_checks import LaunchError, check_launch
from r2r2r_dp.protocol import ProtocolError, config_from_dict, config_to_dict, resolve, validate_protocol
from run_fixtures import make_run_fixture


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    return make_run_fixture(tmp_path_factory.mktemp("launch"))


@pytest.fixture
def fresh(tmp_path):
    """A private copy for tests that mutate views/statistics."""
    return make_run_fixture(tmp_path)


def _run(fx, out, n=3, protocol=None):
    return resolve(protocol or fx["protocol"], n, out)


def _problems(fn):
    with pytest.raises(LaunchError) as e:
        fn()
    return "\n".join(e.value.problems)


def test_valid_launch_passes(fx, tmp_path):
    rep = check_launch(_run(fx, str(tmp_path / "out")))
    assert rep["counts"] == {"train": 3, "val": 2, "test": 2}
    assert rep["normalizer_version"] == "r2r2r-dp-norm-v2"


def test_train_eval_overlap_rejected(fx, tmp_path):
    run = _run(fx, str(tmp_path / "out"))
    run["paths"]["eval_root"] = run["paths"]["train_root"]
    assert "overlaps" in _problems(lambda: check_launch(run))


def test_eval_on_test_split_rejected(fx, tmp_path):
    run = _run(fx, str(tmp_path / "out"))
    run["paths"]["eval_root"] = run["paths"]["test_root"]
    assert "overlaps" in _problems(lambda: check_launch(run))
    run["paths"]["test_root"] = os.path.join(fx["views"], "splits", "val")
    msg = _problems(lambda: check_launch(run))
    assert "manifest split test != root split val" in msg and "membership differs" in msg


def test_larger_subset_as_train_root_rejected(fx, tmp_path):
    run = _run(fx, str(tmp_path / "out"))
    run["paths"]["train_root"] = os.path.join(fx["views"], "train_subsets", "n0006")
    assert "membership differs" in _problems(lambda: check_launch(run))


def test_wrong_subset_statistics_rejected(fx, tmp_path):
    run = _run(fx, str(tmp_path / "out"))
    run["paths"]["action_statistics"] = os.path.join(fx["stats_dir"], "action_statistics_v2_train_n0006.json")
    msg = _problems(lambda: check_launch(run))
    assert "subset='n0006'" in msg and "train_uids_sha256" in msg


def test_v1_statistics_rejected(fx, tmp_path):
    v1 = tmp_path / "v1.json"
    st = json.load(open(os.path.join(fx["stats_dir"], "action_statistics_v2_train_n0003.json")))
    json.dump({k: st[k] for k in ("shape", "min_action", "max_action")}, open(v1, "w"))  # released file format
    run = _run(fx, str(tmp_path / "out"))
    run["paths"]["action_statistics"] = str(v1)
    assert "normalizer_version=None" in _problems(lambda: check_launch(run))


def test_tampered_statistics_rejected(fresh, tmp_path):
    p = os.path.join(fresh["stats_dir"], "action_statistics_v2_train_n0003.json")
    st = json.load(open(p))
    st["max_action"][10] += 0.5  # provenance intact, numbers wrong
    json.dump(st, open(p, "w"))
    assert "differ from a recomputation" in _problems(lambda: check_launch(_run(fresh, str(tmp_path / "out"))))


def test_absolute_statistics_with_delta_actions_rejected(fx, tmp_path):
    p = copy.deepcopy(fx["protocol"])
    p["tinydp"]["shared_cfg.use_delta_action"] = True
    with pytest.raises(ProtocolError, match="use_delta_action"):
        resolve(p, 3, str(tmp_path / "out"))
    # a resolved run altered after protocol validation is still caught by the statistics binding
    run = _run(fx, str(tmp_path / "out"))
    run["tinydp"]["shared_cfg"]["use_delta_action"] = True
    assert "action_mode='absolute', expected 'delta'" in _problems(lambda: check_launch(run))


@pytest.mark.parametrize("mutate,match", [
    (lambda p: p.update(load_config="x.yaml"), "unsupported protocol keys"),
    (lambda p: p["tinydp"].update({"dataset_cfg.data_subsample_num_traj": 50}), "unsupported TinyDP overrides"),
    (lambda p: p["tinydp"].update({"logging_cfg.log_name": "wandb-run"}), "unsupported TinyDP overrides"),
    (lambda p: p["tinydp"].update({"dataset_cfg.dataset_root": "/x"}), "unsupported TinyDP overrides"),
    (lambda p: p["tinydp"].update({"model_cfg.policy_cfg.pred_right_only": False}), "pred_right_only"),
    (lambda p: p["tinydp"].update({"dataset_cfg.is_sim_data": False}), "is_sim_data"),
    (lambda p: p["data"].update(eval_split="test"), "eval_split"),
    (lambda p: p["data"].update(normalizer="v1"), "normalizer"),
    (lambda p: p["data"].update(timing_variant="aligned"), "timing_variant"),
    (lambda p: p["schedule"].update(extra=1), "unsupported"),
    (lambda p: p["schedule"].pop("total_updates"), "missing"),
    (lambda p: p["schedule"].update(warmup_updates=10 ** 9), "warmup_updates"),
    (lambda p: p["tinydp"].update({"shared_cfg.batch_size": "16"}), "expected number"),
])
def test_protocol_rejections(fx, tmp_path, mutate, match):
    p = copy.deepcopy(fx["protocol"])
    mutate(p)
    with pytest.raises(ProtocolError, match=match):
        resolve(p, 3, str(tmp_path / "out"))  # validates the protocol, then type-checks every override


def test_manifest_pin_rejected(fx, tmp_path):
    p = copy.deepcopy(fx["protocol"])
    p["data"]["manifest_sha256"] = "0" * 64
    with pytest.raises(LaunchError, match="sha256"):
        check_launch(resolve(p, 3, str(tmp_path / "out")))


def test_unknown_train_size_rejected(fx, tmp_path):
    with pytest.raises(LaunchError, match="not a manifest subset"):
        check_launch(_run(fx, str(tmp_path / "out"), n=4))


def _ep(fx, root="train_subsets/n0003", i=0):
    d = os.path.join(fx["views"], root)
    return os.path.join(d, sorted(os.listdir(d))[i])


def test_mutated_sidecar_rejected(fresh, tmp_path):
    z = os.path.join(_ep(fresh), "robot_data", "robot_data_proprio.zarr")
    a = zarr.load(z)
    a[5, 7] += 1e-3
    zarr.save_array(z, a)
    assert "sidecar contents differ" in _problems(lambda: check_launch(_run(fresh, str(tmp_path / "out"))))


def test_stale_sidecar_from_other_episode_rejected(fresh, tmp_path):
    a, b = _ep(fresh, i=0), _ep(fresh, i=1)
    name = os.path.join("robot_data", "robot_data_joint.zarr")
    shutil.rmtree(os.path.join(a, name))
    shutil.copytree(os.path.join(b, name), os.path.join(a, name))
    assert "sidecar contents differ" in _problems(lambda: check_launch(_run(fresh, str(tmp_path / "out"))))


def test_missing_and_corrupted_sidecars_rejected(fresh, tmp_path):
    shutil.rmtree(os.path.join(_ep(fresh, i=0), "robot_data", "robot_data_gripper_cmd.zarr"))
    with open(os.path.join(_ep(fresh, i=1), "robot_data", "robot_data_proprio.zarr", ".zarray"), "w") as f:
        f.write("{not json")
    msg = _problems(lambda: check_launch(_run(fresh, str(tmp_path / "out"))))
    assert "missing sidecar robot_data_gripper_cmd.zarr" in msg and "unreadable sidecar" in msg


def test_stray_h5_and_missing_episode_rejected(fresh, tmp_path):
    root = os.path.join(fresh["views"], "train_subsets", "n0003")
    os.makedirs(os.path.join(root, "junk", "x"))
    shutil.copy(os.path.join(_ep(fresh), "robot_data", "robot_data.h5"), os.path.join(root, "junk", "x", "extra.h5"))
    shutil.rmtree(_ep(fresh, i=2))
    msg = _problems(lambda: check_launch(_run(fresh, str(tmp_path / "out"))))
    assert "unexpected h5 file" in msg and "1 missing" in msg


def test_source_data_change_rejected(fresh, tmp_path):
    import h5py
    h5 = os.path.realpath(os.path.join(_ep(fresh), "robot_data", "robot_data.h5"))
    with h5py.File(h5, "r+") as f:
        f["ee_poses"][3, 0] += 1.0
    assert "source HDF5 contents differ" in _problems(lambda: check_launch(_run(fresh, str(tmp_path / "out"))))


def test_missing_frame_rejected(fresh, tmp_path):
    os.remove(os.path.join(os.path.realpath(os.path.join(_ep(fresh), "camera_1")), "rgb", "0007.jpg"))
    assert "camera_1/rgb does not hold frames" in _problems(lambda: check_launch(_run(fresh, str(tmp_path / "out"))))


def test_nonempty_output_dir_rejected(fx, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "old.txt").write_text("x")
    assert "not empty" in _problems(lambda: check_launch(_run(fx, str(out))))


def test_cli_rejects_before_model_allocation(fx, tmp_path, monkeypatch):
    import r2r2r_dp.driver as driver
    monkeypatch.setattr(driver, "Trainer", lambda *a, **k: pytest.fail("model must not be built"))
    out = tmp_path / "out"
    out.mkdir()
    (out / "x").write_text("x")
    with pytest.raises(LaunchError):
        train_r2r2r_dp.main(["--protocol", fx["protocol_path"], "--train-size", "3", "--output-dir", str(out), "--device", "cpu"])


def test_config_roundtrip_and_hash(fx, tmp_path):
    a = _run(fx, str(tmp_path / "a"))
    b = resolve(fx["protocol"], 3, str(tmp_path / "b"), views_root=fx["views"])
    assert a["config_sha256"] == b["config_sha256"]  # output dir / views location are not part of the identity
    assert config_to_dict(config_from_dict(a["tinydp"])) == a["tinydp"]
    cfg = config_from_dict(a["tinydp"])
    assert cfg.model_cfg.policy_cfg.down_dims == (8, 16) and cfg.trainer_cfg.accum_iter == 4
    assert cfg.logging_cfg.log_name is None and cfg.dataset_cfg.action_statistics is None
    assert resolve(fx["protocol"], 6, str(tmp_path / "a"))["config_sha256"] != a["config_sha256"]
    p = copy.deepcopy(fx["protocol"])
    p["tinydp"]["optimizer_cfg.lr"] = 1e-4
    assert resolve(p, 3, str(tmp_path / "a"))["config_sha256"] != a["config_sha256"]


def test_frozen_repository_protocols_are_valid():
    here = os.path.join(os.path.dirname(__file__), "..", "configs")
    for name in sorted(os.listdir(here)):
        p = json.load(open(os.path.join(here, name)))
        validate_protocol(p)
        assert p["tinydp"]["shared_cfg.batch_size"] == 16 and p["schedule"]["accum_micro_batches"] == 4
        assert p["tinydp"]["optimizer_cfg.lr"] == 2e-4 and p["tinydp"]["optimizer_cfg.weight_decay"] == 0.09
        assert p["tinydp"]["shared_cfg.image_size"] == 448 and p["tinydp"]["model_cfg.policy_cfg.pred_right_only"]
    base = json.load(open(os.path.join(here, "baseline_v2.json")))
    for name in ("lifecycle_u20_v2.json", "lifecycle_u100_v2.json"):
        p = json.load(open(os.path.join(here, name)))
        assert p["data"] == base["data"] and p["tinydp"] == base["tinydp"]  # only the schedule differs
