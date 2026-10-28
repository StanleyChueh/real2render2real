"""Codex final audit F2, F3, F4, F5, F8: invalid inputs are rejected before they can be used.

Each case was accepted on 43ab363 (see docs/policy_training/CODEX_FINDINGS_REPAIR.md). Fixtures are the
synthetic CPU fixtures of run_fixtures.py; nothing here touches production data.
"""
import copy
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
import zarr
from PIL import Image

from r2r2r_dp import image_integrity as ii
from r2r2r_dp import manifest as mf
from r2r2r_dp import normalizer_v2 as nv
from r2r2r_dp.inference import R2R2RPolicy
from r2r2r_dp.launch_checks import LaunchError, check_launch
from r2r2r_dp.protocol import ProtocolError, load_protocol, resolve
from r2r2r_dp.rotation import rot6d_rows_to_matrix
from run_fixtures import make_run_fixture

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    return make_run_fixture(tmp_path_factory.mktemp("cfi"))


@pytest.fixture
def fresh(tmp_path):
    return make_run_fixture(tmp_path / "fixture")


def _problems(fx, out, protocol=None):
    with pytest.raises(LaunchError) as e:
        check_launch(resolve(protocol or fx["protocol"], 3, str(out)))
    return "\n".join(e.value.problems)


def _first_train_episode(fx):
    root = os.path.join(fx["views"], "train_subsets", "n0003")
    return os.path.join(root, sorted(os.listdir(root))[0])


# ---------- F2: non-finite values in normalization ----------

@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_normalized_rejects_nonfinite_actions_and_bounds(bad):
    with pytest.raises(ValueError, match="non-finite"):
        nv.normalized(np.array([[bad]]), np.array([0.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="non-finite"):
        nv.normalized(np.array([[0.5]]), np.array([bad]), np.array([1.0]))
    with pytest.raises(ValueError, match="min <= max"):
        nv.normalized(np.array([[0.5]]), np.array([1.0]), np.array([0.0]))


def test_constant_dimension_mapping_kept_and_counted():
    z = nv.normalized(np.array([[2.0, 0.5], [3.0, 0.5]]), np.array([2.0, 0.0]), np.array([2.0, 1.0]))
    assert z[0, 0] == 0.0 and z[1, 0] > 1e300  # TinyDP nan_to_num mapping for a finite zero-range dim
    assert z[0, 1] == z[1, 1] == 0.0


def test_nan_sidecar_cannot_be_certified(fresh):
    ep = _first_train_episode(fresh)
    z = os.path.join(ep, "robot_data", "robot_data_proprio.zarr")
    a = zarr.load(z)
    a[21, 7] = np.nan
    zarr.save_array(z, a)
    root = os.path.dirname(ep)
    with pytest.raises(ValueError, match="non-finite"):
        nv.build_statistics(root, "n0003", fresh["manifest"]["train_subsets"]["3"], "0" * 64, 2, 4, 16)
    st = json.load(open(os.path.join(fresh["stats_dir"], "action_statistics_v2_train_n0003.json")))
    with pytest.raises(ValueError, match="non-finite"):
        nv.coverage(nv.view_episode_h5s(root), st, 2, 4, 16)


def test_statistics_json_is_strict(fresh, tmp_path):
    p = os.path.join(fresh["stats_dir"], "action_statistics_v2_train_n0003.json")
    st = json.load(open(p))
    st["min_action"][0] = float("nan")
    open(p, "w").write(json.dumps(st))  # writes a NaN literal
    with pytest.raises(ValueError):
        nv.load_statistics(p)
    assert "non-standard JSON constant NaN" in _problems(fresh, tmp_path / "o")
    with pytest.raises(ValueError):
        nv.dump_statistics(st, str(tmp_path / "x.json"))


# ---------- F3: manifest semantics, sample order, aliases, statistics config ----------

@pytest.mark.parametrize("mutate,match", [
    (lambda m: m.update(version=2), "unsupported version"),
    (lambda m: m.update(version=True), "unsupported version"),
    (lambda m: m["counts"].update(train=999), "counts"),
    (lambda m: m.update(salt="other-salt"), "salt order"),
    (lambda m: m.update(unexpected=1), "keys must be"),
    (lambda m: m["episodes"].reverse(), "order"),
    (lambda m: m["episodes"][0].update(split_rank=1), "ranks"),
    (lambda m: m["episodes"][0].update(rel_path="../escape"), "rel_path"),
    (lambda m: m["train_subsets"].update({"0": []}), "positive integer"),
    (lambda m: m["train_subsets"].pop("6"), "full train split"),
])
def test_manifest_semantics(fx, mutate, match):
    m = copy.deepcopy(fx["manifest"])
    mf.check_manifest(m)
    mutate(m)
    with pytest.raises(mf.ManifestError, match=match):
        mf.check_manifest(m)


def test_repinned_bad_manifest_rejected_at_launch(fresh, tmp_path):
    m = json.load(open(fresh["manifest_path"]))
    m["counts"]["val"] = 999
    open(fresh["manifest_path"], "w").write(json.dumps(m))
    p = copy.deepcopy(fresh["protocol"])
    p["data"]["manifest_sha256"] = nv.file_sha256(fresh["manifest_path"])
    assert "counts" in _problems(fresh, tmp_path / "o", p)


def test_renamed_view_rejected(fresh, tmp_path):
    ep = _first_train_episode(fresh)
    os.rename(ep, os.path.join(os.path.dirname(ep), "zz_renamed"))
    assert "directory name must be ep_train_0000" in _problems(fresh, tmp_path / "o")


def test_symlinked_episode_alias_rejected(fresh, tmp_path):
    ep = _first_train_episode(fresh)
    real = os.path.join(str(tmp_path), "moved_episode")
    os.rename(ep, real)
    os.symlink(real, ep)  # same name, same content, but an alias outside the view tree
    assert "symlink alias" in _problems(fresh, tmp_path / "o")


def test_symlinked_split_root_rejected(fresh, tmp_path):
    root = os.path.join(fresh["views"], "splits", "val")
    os.rename(root, root + "_real")
    os.symlink(root + "_real", root)
    assert "split root is a symlink alias" in _problems(fresh, tmp_path / "o")


@pytest.mark.parametrize("mutate,match", [
    (lambda st: st["config"].update(layout=list(reversed(st["config"]["layout"]))), "config="),
    (lambda st: st["config"].update(target_rows="every second row"), "config="),
    (lambda st: (st["config"].update(layout=list(reversed(st["config"]["layout"]))),
                 st.update(config_sha256=nv.config_sha256(st["config"]))), "config="),  # consistent digest, wrong layout
    (lambda st: st.update(config_sha256="0" * 64), "config_sha256"),
    (lambda st: st["provenance"].update(n_episodes=99), "n_episodes"),
    (lambda st: st["provenance"].update(n_target_rows=1), "n_target_rows"),
    (lambda st: st.update(extra=1), "keys must be"),
])
def test_statistics_declared_config_enforced(fresh, tmp_path, mutate, match):
    p = os.path.join(fresh["stats_dir"], "action_statistics_v2_train_n0003.json")
    st = json.load(open(p))
    mutate(st)
    json.dump(st, open(p, "w"))
    assert match in _problems(fresh, tmp_path / "o")


def test_valid_launch_reports_images_and_roots(fx, tmp_path):
    rep = check_launch(resolve(fx["protocol"], 3, str(tmp_path / "o")))
    assert rep["images"]["episodes_verified"] == 7 and rep["images"]["frames_verified"] == 7 * 2 * 60
    assert set(rep["roots_realpath"]) == {"train", "val", "test"}


# ---------- F4: image content ----------

def _frame(fx, t=7):
    return os.path.join(_first_train_episode(fx), "camera_0", "rgb", f"{t:04d}.jpg")


def test_corrupt_wrong_size_and_missing_frames_rejected(fresh, tmp_path):
    f = _frame(fresh)
    good = open(f, "rb").read()
    open(f, "wb").write(b"not a jpeg")
    assert "differ from the image-integrity manifest" in _problems(fresh, tmp_path / "a")
    Image.new("RGB", (64, 48), (100, 100, 100)).save(f, quality=95)  # valid JPEG, wrong resolution
    assert "differ from the image-integrity manifest" in _problems(fresh, tmp_path / "b")
    open(f, "wb").write(good)
    check_launch(resolve(fresh["protocol"], 3, str(tmp_path / "c")))  # restored bytes pass again
    os.remove(f)
    assert "does not hold" in _problems(fresh, tmp_path / "d")


def test_builder_decodes_and_checks_resolution(fresh, tmp_path):
    f = _frame(fresh)
    rgb = os.path.dirname(f)
    Image.new("RGB", (64, 48)).save(f, quality=95)
    with pytest.raises(ii.ImageIntegrityError, match=r"\(64, 48\)"):
        ii.certify_camera(rgb, 60, (32, 24))
    data = open(_frame(fresh, 8), "rb").read()
    open(f, "wb").write(data[: len(data) // 2])  # truncated JPEG
    with pytest.raises(ii.ImageIntegrityError, match="not decodable"):
        ii.certify_camera(rgb, 60, (32, 24))
    Image.new("L", (32, 24)).save(f, quality=95)  # grayscale
    with pytest.raises(ii.ImageIntegrityError, match="expected JPEG RGB"):
        ii.certify_camera(rgb, 60, (32, 24))


def test_integrity_file_pinned_and_bound_to_manifest(fresh, tmp_path):
    p = copy.deepcopy(fresh["protocol"])
    p["data"]["image_integrity_sha256"] = "0" * 64
    assert "pinned" in _problems(fresh, tmp_path / "a", p)
    rec = json.load(open(fresh["image_integrity"]))
    rec["manifest_sha256"] = "1" * 64
    other = str(tmp_path / "ii.json")
    json.dump(rec, open(other, "w"))
    p = copy.deepcopy(fresh["protocol"])
    p["data"].update(image_integrity=other, image_integrity_sha256=nv.file_sha256(other))
    assert "certifies manifest" in _problems(fresh, tmp_path / "b", p)


def test_launch_failure_exits_nonzero(fresh, tmp_path):
    open(_frame(fresh), "wb").write(b"not a jpeg")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, os.path.join(HERE, "..", "train_r2r2r_dp.py"), "--protocol",
                        fresh["protocol_path"], "--train-size", "3", "--output-dir", str(tmp_path / "o"), "--device", "cpu"],
                       capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode != 0 and "LaunchError" in r.stderr and "image-integrity" in r.stderr


# ---------- F5: protocol numeric domains ----------

@pytest.mark.parametrize("section,key,value", [
    ("schedule", "grad_clip", -1.0), ("schedule", "grad_clip", 0.0), ("schedule", "grad_clip", float("inf")),
    ("schedule", "total_updates", True), ("schedule", "accum_micro_batches", 2.0), ("schedule", "warmup_updates", False),
    ("schedule", "ema_decay_per_update", float("nan")), ("schedule", "val_seed", -1), ("schedule", "sampler_seed", 2 ** 32),
    ("tinydp", "shared_cfg.batch_size", 2.5), ("tinydp", "shared_cfg.batch_size", 0), ("tinydp", "shared_cfg.batch_size", True),
    ("tinydp", "optimizer_cfg.lr", float("nan")), ("tinydp", "optimizer_cfg.lr", 0.0), ("tinydp", "optimizer_cfg.lr", -1e-4),
    ("tinydp", "optimizer_cfg.weight_decay", -0.1), ("tinydp", "dataset_cfg.proprio_noise", float("inf")),
    ("tinydp", "dataset_cfg.subsample_steps", 0), ("tinydp", "shared_cfg.num_cameras", 3),
    ("tinydp", "shared_cfg.image_size", 8), ("tinydp", "shared_cfg.seed", 1.5), ("tinydp", "trainer_cfg.num_workers", -1),
    ("tinydp", "model_cfg.policy_cfg.down_dims", [8, 0]), ("tinydp", "model_cfg.policy_cfg.down_dims", 8),
    ("tinydp", "model_cfg.policy_cfg.num_inference_diffusion_steps", 1000),
    ("tinydp", "model_cfg.policy_cfg.gripper_loss_w", -5.0),
    ("tinydp", "shared_cfg.num_pred_steps", 6),  # with down_dims [8, 16, 32] (set below): 6 % 2**2 != 0
    ("data", "manifest_sha256", "ABC"),
])
def test_invalid_protocol_values_rejected(fx, tmp_path, section, key, value):
    p = copy.deepcopy(fx["protocol"])
    if key == "shared_cfg.num_pred_steps":
        p["tinydp"]["model_cfg.policy_cfg.down_dims"] = [8, 16, 32]  # two halvings need num_pred_steps % 4 == 0
    p[section][key] = value
    with pytest.raises(ProtocolError):
        resolve(p, 3, str(tmp_path / "o"))


def test_supported_zero_values_accepted(fx, tmp_path):
    p = copy.deepcopy(fx["protocol"])
    p["schedule"]["warmup_updates"] = 0
    p["tinydp"]["dataset_cfg.proprio_noise"] = 0
    p["tinydp"]["trainer_cfg.num_workers"] = 0
    p["tinydp"]["optimizer_cfg.weight_decay"] = 0.0
    resolve(p, 3, str(tmp_path / "o"))
    with pytest.raises(ProtocolError, match="train_size"):
        resolve(fx["protocol"], True, str(tmp_path / "o"))


def test_protocol_file_is_strict_json(fx, tmp_path):
    text = open(fx["protocol_path"]).read().replace('"grad_clip": 1.0', '"grad_clip": NaN')
    assert "NaN" in text
    (tmp_path / "p.json").write_text(text)
    with pytest.raises(ProtocolError, match="NaN"):
        load_protocol(str(tmp_path / "p.json"))


def test_frozen_configs_pin_the_committed_integrity_file():
    repo = os.path.dirname(os.path.dirname(HERE))
    for name in ("baseline_v3.json", "lifecycle_u20_v3.json", "lifecycle_u100_v3.json"):
        p = load_protocol(os.path.join(HERE, "..", "configs", name))
        assert nv.file_sha256(os.path.join(repo, p["data"]["image_integrity"])) == p["data"]["image_integrity_sha256"]


# ---------- F8: inference inputs/outputs and physical rotations ----------

@pytest.fixture(scope="module")
def policy(tmp_path_factory):
    import train_r2r2r_dp
    tmp = tmp_path_factory.mktemp("cfi_pol")
    f = make_run_fixture(tmp, total_updates=2, warmup_updates=0, full_ckpt_every_updates=2,
                         inference_export_every_updates=2, val_every_updates=100, full_val_every_updates=100)
    out = os.path.join(str(tmp), "run")
    train_r2r2r_dp.main(["--protocol", f["protocol_path"], "--train-size", "3", "--output-dir", out, "--device", "cpu"])
    return os.path.join(out, "policy_ema_u0000002.pt")


def _inputs():
    return torch.zeros(1, 4, 2, 3, 32, 32), torch.zeros(1, 4, 20)


@pytest.mark.parametrize("case", ["nan_proprio", "inf_obs", "int_obs", "image_size", "batch0", "steps0", "steps_big",
                                  "steps_bool", "seed_neg", "numpy"])
def test_inference_rejects_invalid_inputs(policy, case):
    pol = R2R2RPolicy.load(policy)
    obs, prop = _inputs()
    kw = {"num_inference_steps": 2}
    if case == "nan_proprio":
        prop[0, 0, 0] = float("nan")
    elif case == "inf_obs":
        obs[0, 0, 0, 0, 0, 0] = float("inf")
    elif case == "int_obs":
        obs = obs.long()
    elif case == "image_size":
        obs = torch.zeros(1, 4, 2, 3, 16, 16)
    elif case == "batch0":
        obs, prop = obs[:0], prop[:0]
    elif case == "steps0":
        kw["num_inference_steps"] = 0
    elif case == "steps_big":
        kw["num_inference_steps"] = 101
    elif case == "steps_bool":
        kw["num_inference_steps"] = True
    elif case == "seed_neg":
        kw["seed"] = -1
    elif case == "numpy":
        prop = prop.numpy()
    with pytest.raises((ValueError, TypeError)):
        pol.predict(obs, prop, **kw)


def test_nonfinite_output_and_weights_rejected(policy, tmp_path, monkeypatch):
    pol = R2R2RPolicy.load(policy)
    obs, prop = _inputs()
    out = pol.predict(obs, prop, num_inference_steps=2, seed=0)
    assert torch.isfinite(out["right_action"]).all()
    monkeypatch.setattr(pol.model, "forward_inference", lambda b: torch.full((1, 16, 10), float("nan")))
    with pytest.raises(FloatingPointError):
        pol.predict(obs, prop, num_inference_steps=2, seed=0)
    ck = torch.load(policy, weights_only=False)
    k = next(n for n, v in ck["model"].items() if v.is_floating_point())
    ck["model"][k] = ck["model"][k].clone().fill_(float("nan"))
    with pytest.raises(ValueError, match="not finite"):
        R2R2RPolicy(ck)
    ck = torch.load(policy, weights_only=False)
    ck["action_statistics"]["min_action"][12], ck["action_statistics"]["max_action"][12] = 1.0, 0.0
    with pytest.raises(ValueError, match="min_action > max_action"):
        R2R2RPolicy(ck)


def test_right_poses_are_proper_rotations(policy):
    pol = R2R2RPolicy.load(policy)
    out = pol.predict(*_inputs(), num_inference_steps=2, seed=0)
    poses = pol.right_poses(out["right_action"])
    R = poses["rotation"]
    assert R.shape == (1, 16, 3, 3) and np.allclose(R @ np.swapaxes(R, -1, -2), np.eye(3), atol=1e-6)
    assert np.allclose(np.linalg.det(R), 1.0)


def test_rotation_converter_cases():
    from scipy.spatial.transform import Rotation
    from tinydp.dataset.utils import quat_to_rot_6d
    assert np.allclose(rot6d_rows_to_matrix([2.0, 0, 0, 0, 3.0, 0]), np.eye(3))  # Codex 'nonunit' case
    for v in ([1.0, 0, 0, 1.0, 0, 0], [0.0] * 6, [1.0, 0, 0, 2.0, 1e-9, 0], [np.nan, 0, 0, 0, 1, 0], [1e-9, 0, 0, 0, 1, 0]):
        with pytest.raises(ValueError):
            rot6d_rows_to_matrix(v)
    q = Rotation.random(64, random_state=0)
    wxyz = q.as_quat()[:, [3, 0, 1, 2]]
    assert np.allclose(rot6d_rows_to_matrix(quat_to_rot_6d(wxyz)), q.as_matrix(), atol=1e-12)  # TinyDP row convention


def test_repair_layer_never_uses_inherited_rotation_helper():
    src = os.path.join(HERE, "..", "r2r2r_dp")
    for name in os.listdir(src):
        if name.endswith(".py"):
            text = open(os.path.join(src, name)).read()
            assert "rot_6d_to_rot_mat" not in text.replace("its inverse rot_6d_to_rot_mat", ""), name


def test_critical_checks_survive_python_optimize(fresh, tmp_path):
    """python -O strips assert statements; manifest, launch and protocol checks must still reject."""
    ep = _first_train_episode(fresh)
    os.rename(ep, os.path.join(os.path.dirname(ep), "zz_renamed"))
    code = f"""
import copy, json, sys
sys.path[:0] = [{os.path.join(HERE, '..')!r}]
from r2r2r_dp import manifest as mf
from r2r2r_dp.launch_checks import LaunchError, check_launch
from r2r2r_dp.protocol import ProtocolError, resolve
assert False, "asserts must be stripped in this interpreter"
p = json.load(open({fresh['protocol_path']!r}))
m = json.load(open({fresh['manifest_path']!r})); m["counts"]["train"] = 999
out = []
for fn in (lambda: mf.check_manifest(m), lambda: check_launch(resolve(p, 3, {str(tmp_path / 'o')!r})),
           lambda: resolve(dict(p, schedule=dict(p["schedule"], grad_clip=-1.0)), 3, "x")):
    try:
        fn(); out.append("accepted")
    except (mf.ManifestError, LaunchError, ProtocolError) as e:
        out.append(type(e).__name__)
print(json.dumps(out))
"""
    r = subprocess.run([sys.executable, "-O", "-c", code], capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1]) == ["ManifestError", "LaunchError", "ProtocolError"]
