"""Synthetic manifest + views + v2 statistics + protocol for CPU tests of the v2 training path.

Source: 10 episodes of T=60 rows (fixtures.make_source_tree): 2 test, 2 val, 6 train (subsets 3 and 6).
Model: the released DiffusionPolicy with small UNet widths and 32 px images so a few optimizer
updates run in seconds on CPU.
"""
import copy
import json
import os

from fixtures import make_source_tree
from r2r2r_dp.adapter import build_all_views
from r2r2r_dp.manifest import build_manifest, save_manifest
from r2r2r_dp.normalizer_v2 import build_statistics, file_sha256
from r2r2r_dp.protocol import PROTOCOL_VERSION

TRAIN_SIZES = (3, 6)


def tiny_protocol(manifest, manifest_sha, views, stats_dir, **schedule):
    sched = {"accum_micro_batches": 4, "total_updates": 6, "warmup_updates": 2, "lr_schedule": "cosine",
             "ema_decay_per_update": 0.9, "grad_clip": 1.0, "amp_dtype": "bfloat16", "sampler_seed": 42,
             "log_every_updates": 1, "val_every_updates": 2, "val_subset_window_stride": 10, "val_seed": 7,
             "full_val_every_updates": 6, "full_ckpt_every_updates": 3, "keep_full_checkpoints": 5,
             "inference_export_every_updates": 3}
    sched.update(schedule)
    return {
        "protocol_version": PROTOCOL_VERSION, "name": "tiny_test", "purpose": "unit tests",
        "data": {"manifest": manifest, "manifest_sha256": manifest_sha, "views_root": views, "eval_split": "val",
                 "normalizer": "v2", "statistics_dir": stats_dir, "timing_variant": "released_pairing"},
        "tinydp": {"dataset_cfg.is_sim_data": True, "dataset_cfg.scale_action": True, "dataset_cfg.proprio_noise": 0.003,
                   "dataset_cfg.subsample_steps": 2, "model_cfg.policy_cfg.pred_right_only": True,
                   "model_cfg.policy_cfg.down_dims": [8, 16], "model_cfg.policy_cfg.num_inference_diffusion_steps": 4,
                   "shared_cfg.batch_size": 2, "shared_cfg.seq_length": 4, "shared_cfg.num_pred_steps": 16,
                   "shared_cfg.num_cameras": 2, "shared_cfg.image_size": 32, "shared_cfg.use_delta_action": False,
                   "shared_cfg.seed": 0, "optimizer_cfg.lr": 2e-4, "optimizer_cfg.weight_decay": 0.09,
                   "trainer_cfg.num_workers": 0, "trainer_cfg.pin_memory": False},
        "schedule": sched,
    }


def make_run_fixture(tmp, n_episodes=10, **schedule):
    """Returns dict(source, manifest, views, stats_dir, protocol, protocol_path)."""
    tmp = str(tmp)
    src = os.path.join(tmp, "src")
    eps = make_source_tree(src, n_episodes)
    m = build_manifest(eps, src, n_val=2, n_test=2, subset_sizes=TRAIN_SIZES)
    mpath = os.path.join(tmp, "manifest.json")
    save_manifest(m, mpath)
    views = os.path.join(tmp, "views")
    build_all_views(m, views)
    stats_dir = os.path.join(tmp, "stats")
    os.makedirs(stats_dir)
    msha = file_sha256(mpath)
    for n, uids in m["train_subsets"].items():
        st = build_statistics(os.path.join(views, "train_subsets", f"n{int(n):04d}"), f"n{int(n):04d}", uids, msha, 2, 4, 16)
        with open(os.path.join(stats_dir, f"action_statistics_v2_train_n{int(n):04d}.json"), "w") as f:
            json.dump(st, f)
    proto = tiny_protocol(mpath, msha, views, stats_dir, **schedule)
    ppath = os.path.join(tmp, "protocol.json")
    with open(ppath, "w") as f:
        json.dump(proto, f)
    return {"source": src, "manifest": m, "manifest_path": mpath, "views": views, "stats_dir": stats_dir,
            "protocol": copy.deepcopy(proto), "protocol_path": ppath}
