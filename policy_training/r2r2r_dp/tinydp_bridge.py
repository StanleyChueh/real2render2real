"""Use TinyDP's unmodified sim dataset with manifest-defined split roots and train-only statistics.

TinyDP builds train and val datasets from ONE dataset_root, splits by a seeded shuffle of glob order
(train_split=0.95), and, when no statistics file is given, computes min/max over ALL files before the
split. ManifestSimDataset keeps the released class and changes only which root each split reads:
split='train' -> train root with train_split=1.0 (all files), any other split -> eval root with
train_split=0.0 (all files). Statistics must be supplied (computed from the train root only).

TinyDP's file list comes from glob(), whose order follows the file system. ordered_glob() sorts it so
the per-epoch window order is reproducible; membership is unaffected.
"""
import copy
import glob as _glob
import json
import os
from typing import Dict, Optional

import numpy as np

import tinydp.dataset.image_dataset_sim as _sim
from tinydp.dataset.utils import convert_multi_step_np, quat_to_rot_6d

_OriginalSimDataset = _sim.SequenceDataset


def ordered_glob(*args, **kwargs):
    return sorted(_glob.glob(*args, **kwargs))


def install_ordered_glob() -> None:
    _sim.glob = ordered_glob


class ManifestSimDataset(_OriginalSimDataset):
    """Module-level (picklable for spawn-started DataLoader workers); configure via make_manifest_dataset_class."""
    _config: Optional[Dict[str, str]] = None

    def __init__(self, dataset_config, shared_config, logging_config, vision_transform, split="train", debug=False):
        c = type(self)._config
        assert c is not None, "call make_manifest_dataset_class() first"
        cfg = copy.deepcopy(dataset_config)
        cfg.dataset_root = c["train_root"] if split == "train" else c["eval_root"]
        cfg.train_split = 1.0 if split == "train" else 0.0
        if cfg.data_subsample_ratio > 0 or cfg.data_subsample_num_traj > 0 or cfg.balance_data:
            raise ValueError("use manifest train subsets instead of TinyDP subsampling/balancing")
        stale = os.path.join(logging_config.output_dir, "action_statistics.json")
        if os.path.exists(stale) and _read_json(stale) != _read_json(c["action_statistics"]):
            raise ValueError(f"{stale} differs from {c['action_statistics']}; TinyDP would silently prefer it")
        cfg.action_statistics = c["action_statistics"]
        super().__init__(cfg, shared_config, logging_config, vision_transform, split=split, debug=debug)
        self.split_root = cfg.dataset_root


def make_manifest_dataset_class(train_root: str, eval_root: str, action_statistics: str):
    """Configure and return the drop-in replacement for tinydp.dataset.image_dataset_sim.SequenceDataset."""
    assert os.path.isfile(action_statistics), f"train-only statistics required: {action_statistics}"
    ManifestSimDataset._config = {"train_root": train_root, "eval_root": eval_root, "action_statistics": action_statistics}
    return ManifestSimDataset


def _read_json(p: str) -> Dict:
    with open(p) as f:
        return json.load(f)


def compute_action_statistics_tinydp(train_root: str, shared_cfg, dataset_cfg, out_path: str) -> Dict:
    """Run TinyDP's own calculate_dataset_statistics over the train root only."""
    install_ordered_glob()
    ds = _OriginalSimDataset.__new__(_OriginalSimDataset)
    ds.seed = shared_cfg.seed
    ds.num_pred_steps = shared_cfg.num_pred_steps
    ds.subsample_steps = dataset_cfg.subsample_steps
    ds.use_delta_action = shared_cfg.use_delta_action
    ds.proprio_noise = 0.0  # proprio only enters the delta-action path; keep stats deterministic
    ds.h5_files = ordered_glob(os.path.join(train_root, "**", "*.h5"), recursive=True)
    if not ds.h5_files:
        raise ValueError(f"no h5 files under {train_root}")
    n = len(ds.h5_files)
    ds.calculate_dataset_statistics(out_path)
    assert len(ds.h5_files) == n, "TinyDP dropped files while computing statistics"
    return _read_json(out_path)


def compute_action_statistics_numpy(train_root: str, num_pred_steps: int, subsample_steps: int) -> Dict:
    """Independent re-implementation (absolute actions) used to cross-check TinyDP's statistics."""
    import zarr
    mins, maxs = [], []
    for h5 in ordered_glob(os.path.join(train_root, "**", "*.h5"), recursive=True):
        ee = zarr.load(h5.replace(".h5", "_proprio.zarr"))
        g = zarr.load(h5.replace(".h5", "_gripper_cmd.zarr")).astype(float)
        idx = np.arange(subsample_steps, len(ee), subsample_steps)  # == helper_load_action(0, T - s)
        parts = []
        for sl, k in ((slice(0, 7), 0), (slice(7, 14), 1)):
            pose = ee[idx, sl]
            parts.append(np.concatenate([pose[:, :3], quat_to_rot_6d(pose[:, 3:]), g[idx, k:k + 1]], axis=1))
        a = convert_multi_step_np(np.concatenate(parts, axis=1), num_pred_steps)
        mins.append(a.min((0, 1)))
        maxs.append(a.max((0, 1)))
    return {"min_action": np.min(mins, 0).tolist(), "max_action": np.max(maxs, 0).tolist()}
