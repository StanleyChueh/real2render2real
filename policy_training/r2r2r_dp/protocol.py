"""Frozen, versioned training protocol -> fully resolved TinyDP ExperimentConfig.

The TinyDP configuration is built from the released dataclass defaults plus an explicit allowlist of
overrides taken from one protocol JSON file. There is no CLI pass-through and no --load-config path,
so the resolved configuration is exactly what the protocol file and this module say (audit M4).
"""
import copy
import dataclasses
import hashlib
import json
import os
from typing import Any, Dict

from tinydp.util.args import (DatasetConfig, ExperimentConfig, LoggingConfig, ModelConfig, OptimizerConfig,
                              PolicyConfig, SharedConfig, TrainerConfig, VisionEncoderConfig)

PROTOCOL_VERSION = "r2r2r-dp-protocol-v2"
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# TinyDP fields a protocol may set. Everything else keeps the released default.
TINYDP_ALLOWED = {
    "dataset_cfg.is_sim_data", "dataset_cfg.proprio_noise", "dataset_cfg.subsample_steps", "dataset_cfg.scale_action",
    "model_cfg.policy_cfg.pred_right_only", "model_cfg.policy_cfg.down_dims",
    "model_cfg.policy_cfg.num_train_diffusion_steps", "model_cfg.policy_cfg.num_inference_diffusion_steps",
    "model_cfg.policy_cfg.gripper_loss_w",
    "shared_cfg.batch_size", "shared_cfg.seq_length", "shared_cfg.num_pred_steps", "shared_cfg.num_cameras",
    "shared_cfg.image_size", "shared_cfg.use_delta_action", "shared_cfg.seed",
    "optimizer_cfg.lr", "optimizer_cfg.weight_decay",
    "trainer_cfg.num_workers", "trainer_cfg.pin_memory",
}
# Values the protocol must pin (the sim schema and the supported action convention).
TINYDP_REQUIRED = {"dataset_cfg.is_sim_data": True, "dataset_cfg.scale_action": True,
                   "shared_cfg.use_delta_action": False, "model_cfg.policy_cfg.pred_right_only": True}
DATA_KEYS = {"manifest", "manifest_sha256", "views_root", "eval_split", "normalizer", "statistics_dir", "timing_variant"}
SCHEDULE_KEYS = {"accum_micro_batches", "total_updates", "warmup_updates", "lr_schedule", "ema_decay_per_update",
                 "grad_clip", "amp_dtype", "sampler_seed", "log_every_updates", "val_every_updates",
                 "val_subset_window_stride", "val_seed", "full_val_every_updates", "full_ckpt_every_updates",
                 "keep_full_checkpoints", "inference_export_every_updates"}
TOP_KEYS = {"protocol_version", "name", "purpose", "data", "tinydp", "schedule"}
NORMALIZERS = {"v2": "action_statistics_v2_train_n{n:04d}.json"}
TIMING_VARIANTS = {"released_pairing"}


class ProtocolError(ValueError):
    pass


def default_experiment_config() -> ExperimentConfig:
    return ExperimentConfig(dataset_cfg=DatasetConfig(), model_cfg=ModelConfig(PolicyConfig(), VisionEncoderConfig()),
                            optimizer_cfg=OptimizerConfig(), shared_cfg=SharedConfig(), logging_cfg=LoggingConfig(),
                            trainer_cfg=TrainerConfig())


def _set(cfg, dotted: str, value):
    *parents, leaf = dotted.split(".")
    obj = cfg
    for p in parents:
        obj = getattr(obj, p)
    default = getattr(obj, leaf)
    if isinstance(default, tuple):
        value = tuple(value)
    elif isinstance(default, bool) or isinstance(value, bool):
        if not (isinstance(default, bool) and isinstance(value, bool)):
            raise ProtocolError(f"{dotted}: expected bool, got {value!r}")
    elif isinstance(default, (int, float)) and not isinstance(value, (int, float)):
        raise ProtocolError(f"{dotted}: expected number, got {value!r}")
    setattr(obj, leaf, value)


def config_to_dict(cfg) -> Dict[str, Any]:
    return json.loads(json.dumps(dataclasses.asdict(cfg)))


def config_from_dict(d: Dict[str, Any]) -> ExperimentConfig:
    """Inverse of config_to_dict (tuples restored from the released defaults' field types)."""
    cfg = default_experiment_config()

    def fill(obj, values):
        for k, v in values.items():
            cur = getattr(obj, k)
            if dataclasses.is_dataclass(cur):
                fill(cur, v)
            elif isinstance(cur, tuple) and v is not None:
                setattr(obj, k, tuple(v))
            elif k == "task_names" and v is not None:
                setattr(obj, k, tuple(v))
            else:
                setattr(obj, k, v)
    fill(cfg, d)
    return cfg


def load_protocol(path: str) -> Dict:
    with open(path) as f:
        p = json.load(f)
    validate_protocol(p)
    return p


def validate_protocol(p: Dict) -> None:
    unknown = set(p) - TOP_KEYS
    if unknown:
        raise ProtocolError(f"unsupported protocol keys: {sorted(unknown)}")
    if p.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError(f"protocol_version must be {PROTOCOL_VERSION}")
    for section, allowed in (("data", DATA_KEYS), ("schedule", SCHEDULE_KEYS)):
        got = set(p.get(section, {}))
        if got != allowed:
            raise ProtocolError(f"{section}: missing {sorted(allowed - got)}, unsupported {sorted(got - allowed)}")
    t = p.get("tinydp", {})
    bad = set(t) - TINYDP_ALLOWED
    if bad:
        raise ProtocolError(f"unsupported TinyDP overrides: {sorted(bad)}")
    for k, v in TINYDP_REQUIRED.items():
        if t.get(k) != v:
            raise ProtocolError(f"{k} must be {v} in this protocol (got {t.get(k)!r})")
    d, s = p["data"], p["schedule"]
    if d["normalizer"] not in NORMALIZERS:
        raise ProtocolError(f"normalizer must be one of {sorted(NORMALIZERS)}")
    if d["timing_variant"] not in TIMING_VARIANTS:
        raise ProtocolError(f"timing_variant must be one of {sorted(TIMING_VARIANTS)}")
    if d["eval_split"] != "val":
        raise ProtocolError("eval_split must be 'val' (the test split is reserved for final evaluation)")
    if s["lr_schedule"] != "cosine" or s["amp_dtype"] not in ("bfloat16", "none"):
        raise ProtocolError("unsupported lr_schedule / amp_dtype")
    for k in ("accum_micro_batches", "total_updates", "log_every_updates", "val_every_updates",
              "val_subset_window_stride", "full_val_every_updates", "full_ckpt_every_updates",
              "keep_full_checkpoints", "inference_export_every_updates"):
        if not (isinstance(s[k], int) and s[k] >= 1):
            raise ProtocolError(f"schedule.{k} must be a positive integer")
    if not (isinstance(s["warmup_updates"], int) and 0 <= s["warmup_updates"] < s["total_updates"]):
        raise ProtocolError("schedule.warmup_updates must be in [0, total_updates)")
    if not 0.0 < s["ema_decay_per_update"] < 1.0:
        raise ProtocolError("schedule.ema_decay_per_update must be in (0, 1)")


def resolve(p: Dict, train_size: int, output_dir: str, views_root: str = None) -> Dict:
    """Return {'protocol', 'train_size', 'paths', 'tinydp' (resolved dict), 'config_sha256'}."""
    validate_protocol(p)
    cfg = default_experiment_config()
    for k, v in sorted(p["tinydp"].items()):
        _set(cfg, k, v)
    cfg.dataset_cfg.action_statistics = None  # supplied per split by the bridge, never from output_dir
    cfg.logging_cfg.output_dir = os.path.abspath(output_dir)
    cfg.logging_cfg.log_dir = None
    cfg.logging_cfg.log_name = None
    cfg.trainer_cfg.accum_iter = p["schedule"]["accum_micro_batches"]
    views = os.path.abspath(os.path.expanduser(views_root or p["data"]["views_root"]))
    stats_dir = p["data"]["statistics_dir"]
    stats_dir = stats_dir if os.path.isabs(stats_dir) else os.path.join(REPO, stats_dir)
    manifest = p["data"]["manifest"]
    manifest = manifest if os.path.isabs(manifest) else os.path.join(REPO, manifest)
    paths = {
        "manifest": manifest,
        "views_root": views,
        "train_root": os.path.join(views, "train_subsets", f"n{train_size:04d}"),
        "eval_root": os.path.join(views, "splits", p["data"]["eval_split"]),
        "test_root": os.path.join(views, "splits", "test"),
        "action_statistics": os.path.join(stats_dir, NORMALIZERS[p["data"]["normalizer"]].format(n=train_size)),
        "output_dir": os.path.abspath(output_dir),
    }
    tinydp = config_to_dict(cfg)
    hashed = copy.deepcopy(tinydp)
    hashed["logging_cfg"] = {}
    key = {"protocol": {k: p[k] for k in ("protocol_version", "data", "tinydp", "schedule")}, "train_size": train_size,
           "tinydp": hashed}
    key["protocol"]["data"] = {k: v for k, v in key["protocol"]["data"].items() if k != "views_root"}
    return {"protocol": p, "train_size": train_size, "paths": paths, "tinydp": tinydp,
            "config_sha256": hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()}
