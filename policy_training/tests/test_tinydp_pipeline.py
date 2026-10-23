"""TinyDP's unmodified sim loader on manifest views: temporal indexing, action/observation pairing,
orientation conversion and train-only normalisation statistics."""
import json
import os

import numpy as np
import pytest
import torch
import tyro
from scipy.spatial.transform import Rotation
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from fixtures import DX, X0, SMALL, gray, make_source_tree
from r2r2r_dp import adapter, manifest as mf, tinydp_bridge as tb
from tinydp.dataset.image_dataset import CollateFunction, VideoSampler
from tinydp.dataset.utils import default_vision_transform, quat_to_rot_6d, unscale_action
from tinydp.util.args import ExperimentConfig

T = SMALL.num_steps


@pytest.fixture
def views(tmp_path):
    root = str(tmp_path / "src")
    eps = make_source_tree(root, 6)
    m = mf.build_manifest(eps, root, n_val=2, n_test=1, subset_sizes=(2,))
    roots = adapter.build_all_views(m, str(tmp_path / "views"))
    return m, roots, tmp_path


def _args(tmp_path, *extra):
    a = tyro.cli(ExperimentConfig, args=["--dataset-cfg.is-sim-data", *extra])
    a.logging_cfg.output_dir = str(tmp_path / "out")
    os.makedirs(a.logging_cfg.output_dir, exist_ok=True)
    return a


def _dataset(views, split, *extra, scale=False):
    m, roots, tmp_path = views
    a = _args(tmp_path, *extra)
    a.dataset_cfg.scale_action = scale
    stats = str(tmp_path / "stats.json")
    if not os.path.exists(stats):
        tb.compute_action_statistics_tinydp(roots["train_n0003"], a.shared_cfg, a.dataset_cfg, stats)
    tb.install_ordered_glob()
    cls = tb.make_manifest_dataset_class(roots["train_n0003"], roots["val"], stats)
    return cls(a.dataset_cfg, a.shared_cfg, a.logging_cfg, default_vision_transform(64), split=split), a


def _frame_index(obs: torch.Tensor) -> np.ndarray:
    """Recover t from the uniform gray level of each (normalised) camera image."""
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD).view(3, 1, 1)
    level = ((obs * std + mean) * 255).mean(dim=(-3, -2, -1)).numpy()
    return np.rint((level - 20) / 3).astype(int)


def test_observation_proprio_action_alignment(views):
    ds, a = _dataset(views, "train")
    s, npred = a.dataset_cfg.subsample_steps, a.shared_cfg.num_pred_steps
    assert (s, npred, a.shared_cfg.seq_length) == (2, 16, 4)  # TinyDP defaults
    checked = 0
    for idx in range(0, len(ds), 7):
        h5, start, end = ds.start_end[idx]
        item = ds[idx]
        j = int(round((item["proprio"][10].item() - X0) / DX))   # proprio = right EE x of row j
        assert j == start
        assert (_frame_index(item["observation"]) == start).all()  # both cameras show frame j
        act_rows = np.rint((item["action"][:, 10].numpy() - X0) / DX).astype(int)
        assert act_rows.tolist() == list(range(start + s, start + s * (npred + 1), s))  # rows j+2 ... j+32
        closed = (np.arange(T) >= 20) & (np.arange(T) < 40)
        assert item["action"][:, 19].numpy().tolist() == closed[act_rows].astype(float).tolist()
        assert (item["action"][:, 9] == 0).all()  # left gripper never closes
        checked += 1
    assert checked > 10


def test_windows_cover_valid_range_and_sequences_are_consecutive(views):
    ds, a = _dataset(views, "train")
    s, L = 2, a.shared_cfg.seq_length
    n_windows = (T - 1) - (s * (L + 16) - 1)
    assert len(ds) == 3 * n_windows * L
    for k in range(0, len(ds), L):  # dataset order groups each observation sequence contiguously
        starts = [ds.start_end[k + i][1] for i in range(L)]
        assert starts == list(range(starts[0], starts[0] + s * L, s))
        assert len({ds.start_end[k + i][0] for i in range(L)}) == 1
    assert max(e for _, _, e in ds.start_end) + s <= T  # last action row index stays inside the episode


def test_sampler_and_collate_shapes(views):
    ds, a = _dataset(views, "train")
    B, L = 2, a.shared_cfg.seq_length
    sampler = VideoSampler(ds, batch_size=B, sequence_length=L)
    batch_idx = next(iter(sampler))
    batch = CollateFunction(L, pred_right_only=True)([ds[i] for i in batch_idx])
    assert list(batch["observation"].shape) == [B, L, 2, 3, 64, 64]
    assert list(batch["proprio"].shape) == [B, L, 20]
    assert list(batch["action"].shape) == [B, L, 16, 20]
    for b in range(B):  # each sample is one episode's consecutive observation steps
        fr = [_frame_index(batch["observation"][b, t])[0] for t in range(L)]
        assert fr == list(range(fr[0], fr[0] + 2 * L, 2))


def test_quaternion_convention_and_sign_invariance():
    rng = np.random.default_rng(0)
    q_xyzw = Rotation.random(50, random_state=1).as_quat()
    q_wxyz = q_xyzw[:, [3, 0, 1, 2]]
    r6 = quat_to_rot_6d(q_wxyz)  # TinyDP default format = wxyz, matching ee_poses
    expected = Rotation.from_quat(q_xyzw).as_matrix()[:, :2, :].reshape(-1, 6)
    assert np.allclose(r6, expected, atol=1e-7)
    assert np.allclose(quat_to_rot_6d(-q_wxyz), r6, atol=1e-7)  # q and -q are the same rotation
    assert np.allclose(quat_to_rot_6d(np.array([[1.0, 0, 0, 0]])), [[1, 0, 0, 0, 1, 0]])
    assert not np.allclose(quat_to_rot_6d(q_xyzw), r6, atol=1e-3)  # misreading the order is detectable
    del rng


def test_statistics_use_train_only(views):
    m, roots, tmp_path = views
    # make validation episodes extreme (view-side sidecars only): if they leaked into statistics, max x would move
    import zarr
    for ep in os.listdir(roots["val"]):
        p = os.path.join(roots["val"], ep, "robot_data", "robot_data_proprio.zarr")
        z = zarr.load(p)
        z[:, 7] += 100.0
        zarr.save_array(p, z)
    train_ds, a = _dataset(views, "train", scale=True)
    val_ds, _ = _dataset(views, "val", scale=True)
    stats = json.load(open(tmp_path / "stats.json"))
    ref = tb.compute_action_statistics_numpy(roots["train_n0003"], 16, 2)
    assert np.allclose(stats["min_action"], ref["min_action"]) and np.allclose(stats["max_action"], ref["max_action"])
    assert stats["max_action"][10] < 1.0
    for ds in (train_ds, val_ds):
        assert torch.allclose(ds.stats["max"], torch.tensor(stats["max_action"], dtype=ds.stats["max"].dtype))
    # scaled train actions lie in [-1, 1]; unscaling recovers the raw rows
    item = train_ds[0]
    assert item["action"][:, 10:19].abs().max() <= 1.0 + 1e-6
    raw = unscale_action(item["action"].clone().double(), {k: v.clone() for k, v in train_ds.stats.items()}, type="diffusion")
    rows = np.rint((raw[:, 10].numpy() - X0) / DX)
    assert np.allclose(rows, np.arange(2, 34, 2) + train_ds.start_end[0][1])


def test_wrapper_refuses_missing_or_conflicting_statistics(views, tmp_path):
    m, roots, _ = views
    with pytest.raises(AssertionError, match="train-only statistics"):
        tb.make_manifest_dataset_class(roots["train_n0003"], roots["val"], str(tmp_path / "missing.json"))
    ds, a = _dataset(views, "train")
    stale = os.path.join(a.logging_cfg.output_dir, "action_statistics.json")
    json.dump({"shape": [20], "min_action": [0.0] * 20, "max_action": [1.0] * 20}, open(stale, "w"))
    with pytest.raises(ValueError, match="differs"):
        _dataset(views, "train")
    a.dataset_cfg.data_subsample_num_traj = 2
    cls = tb.make_manifest_dataset_class(roots["train_n0003"], roots["val"], str(tmp_path / "stats.json"))
    os.remove(stale)
    with pytest.raises(ValueError, match="manifest train subsets"):
        cls(a.dataset_cfg, a.shared_cfg, a.logging_cfg, default_vision_transform(64), split="train")


def test_dataset_is_picklable_for_spawn_workers(views):
    """train.py uses the 'spawn' start method: the dataset object must pickle with its configuration."""
    import pickle
    ds, _ = _dataset(views, "train")
    clone = pickle.loads(pickle.dumps(ds))
    assert clone.start_end == ds.start_end and clone.split_root == ds.split_root
    assert torch.equal(clone[3]["action"], ds[3]["action"])
