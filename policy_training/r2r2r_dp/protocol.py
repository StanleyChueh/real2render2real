"""Frozen, versioned training protocol -> fully resolved TinyDP ExperimentConfig.

The TinyDP configuration is built from the released dataclass defaults plus an explicit allowlist of
overrides taken from one protocol JSON file. There is no CLI pass-through and no --load-config path,
so the resolved configuration is exactly what the protocol file and this module say (audit M4).

Every value is checked against a field-specific domain before any dataset or model is built (Codex
final audit F5): integers exclude booleans, floats must be finite, counts/dimensions are positive and
cross-field constraints (UNet depth vs. horizon, inference vs. training diffusion steps) hold. Protocol
files are parsed as strict JSON (NaN/Infinity literals are rejected).
"""
import copy
import dataclasses
import hashlib
import json
import math
import os
from typing import Any, Callable, Dict, Tuple

from tinydp.util.args import (DatasetConfig, ExperimentConfig, LoggingConfig, ModelConfig, OptimizerConfig,
                              PolicyConfig, SharedConfig, TrainerConfig, VisionEncoderConfig)

PROTOCOL_VERSION = "r2r2r-dp-protocol-v3"
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
DATA_KEYS = {"manifest", "manifest_sha256", "views_root", "eval_split", "normalizer", "statistics_dir", "timing_variant",
             "image_integrity", "image_integrity_sha256"}
SCHEDULE_KEYS = {"accum_micro_batches", "total_updates", "warmup_updates", "lr_schedule", "ema_decay_per_update",
                 "grad_clip", "amp_dtype", "sampler_seed", "log_every_updates", "val_every_updates",
                 "val_subset_window_stride", "val_seed", "full_val_every_updates", "full_ckpt_every_updates",
                 "keep_full_checkpoints", "inference_export_every_updates"}
TOP_KEYS = {"protocol_version", "name", "purpose", "data", "tinydp", "schedule"}
NORMALIZERS = {"v2": "action_statistics_v2_train_n{n:04d}.json"}
TIMING_VARIANTS = {"released_pairing"}


class ProtocolError(ValueError):
    pass


SEED_MAX = 2 ** 32 - 1  # numpy.random.seed range


def _is_int(v, lo=None, hi=None) -> bool:
    return type(v) is int and (lo is None or v >= lo) and (hi is None or v <= hi)


def _is_real(v, lo=None, hi=None, lo_open=False, hi_open=False) -> bool:
    if type(v) not in (int, float) or not math.isfinite(v):
        return False
    if lo is not None and (v <= lo if lo_open else v < lo):
        return False
    return hi is None or (v < hi if hi_open else v <= hi)


def _is_bool(v) -> bool:
    return type(v) is bool


def _is_sha256(v) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


# Domain of every TinyDP field a protocol may set, checked on the resolved value (protocol or default).
TINYDP_DOMAINS: Dict[str, Tuple[str, Callable[[Any], bool]]] = {
    "dataset_cfg.is_sim_data": ("bool", _is_bool),
    "dataset_cfg.proprio_noise": ("finite real >= 0", lambda v: _is_real(v, 0)),
    "dataset_cfg.subsample_steps": ("integer >= 1", lambda v: _is_int(v, 1)),
    "dataset_cfg.scale_action": ("bool", _is_bool),
    "model_cfg.policy_cfg.pred_right_only": ("bool", _is_bool),
    "model_cfg.policy_cfg.down_dims": ("non-empty sequence of integers >= 1",
                                       lambda v: isinstance(v, (list, tuple)) and len(v) >= 1 and all(_is_int(x, 1) for x in v)),
    "model_cfg.policy_cfg.num_train_diffusion_steps": ("integer >= 1", lambda v: _is_int(v, 1)),
    "model_cfg.policy_cfg.num_inference_diffusion_steps": ("integer >= 1", lambda v: _is_int(v, 1)),
    "model_cfg.policy_cfg.gripper_loss_w": ("finite real >= 0", lambda v: _is_real(v, 0)),
    "shared_cfg.batch_size": ("integer >= 1", lambda v: _is_int(v, 1)),
    "shared_cfg.seq_length": ("integer >= 1", lambda v: _is_int(v, 1)),
    "shared_cfg.num_pred_steps": ("integer >= 1", lambda v: _is_int(v, 1)),
    "shared_cfg.num_cameras": ("2 (camera_1, camera_0)", lambda v: _is_int(v, 2, 2)),
    "shared_cfg.image_size": ("integer >= 32", lambda v: _is_int(v, 32)),
    "shared_cfg.use_delta_action": ("bool", _is_bool),
    "shared_cfg.seed": (f"integer in [0, {SEED_MAX}]", lambda v: _is_int(v, 0, SEED_MAX)),
    "optimizer_cfg.lr": ("finite real > 0", lambda v: _is_real(v, 0, lo_open=True)),
    "optimizer_cfg.weight_decay": ("finite real >= 0", lambda v: _is_real(v, 0)),
    "trainer_cfg.num_workers": ("integer >= 0", lambda v: _is_int(v, 0)),
    "trainer_cfg.pin_memory": ("bool", _is_bool),
}
_POSITIVE_SCHEDULE_INTS = ("accum_micro_batches", "total_updates", "log_every_updates", "val_every_updates",
                           "val_subset_window_stride", "full_val_every_updates", "full_ckpt_every_updates",
                           "keep_full_checkpoints", "inference_export_every_updates")


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
        if not isinstance(value, (list, tuple)):
            raise ProtocolError(f"{dotted}: expected a list, got {value!r}")
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


def _reject_constant(c):
    raise ProtocolError(f"non-standard JSON constant {c} (protocols are strict JSON)")


def _get(cfg, dotted: str):
    obj = cfg
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def validate_resolved_tinydp(cfg) -> None:
    """Field domains and cross-field constraints of the resolved TinyDP configuration."""
    bad = [f"{k}={_get(cfg, k)!r} (expected {d})" for k, (d, ok) in sorted(TINYDP_DOMAINS.items()) if not ok(_get(cfg, k))]
    if bad:
        raise ProtocolError("invalid TinyDP values: " + "; ".join(bad))
    pc, sc = cfg.model_cfg.policy_cfg, cfg.shared_cfg
    if pc.num_inference_diffusion_steps > pc.num_train_diffusion_steps:
        raise ProtocolError("num_inference_diffusion_steps must not exceed num_train_diffusion_steps")
    levels = len(pc.down_dims) - 1  # ConditionalUnet1D halves the horizon at every level but the last
    if sc.num_pred_steps % (2 ** levels):
        raise ProtocolError(f"num_pred_steps {sc.num_pred_steps} must be divisible by 2**{levels} for down_dims {pc.down_dims}")


def load_protocol(path: str) -> Dict:
    with open(path) as f:
        p = json.load(f, parse_constant=_reject_constant)
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
    for k in ("name", "purpose"):
        if not isinstance(p.get(k), str):
            raise ProtocolError(f"{k} must be a string")
    for k in ("manifest", "views_root", "statistics_dir", "image_integrity"):
        if not isinstance(d[k], str) or not d[k]:
            raise ProtocolError(f"data.{k} must be a non-empty path string")
    for k in ("manifest_sha256", "image_integrity_sha256"):
        if not _is_sha256(d[k]):
            raise ProtocolError(f"data.{k} must be a lowercase hex sha256")
    if d["normalizer"] not in NORMALIZERS:
        raise ProtocolError(f"normalizer must be one of {sorted(NORMALIZERS)}")
    if d["timing_variant"] not in TIMING_VARIANTS:
        raise ProtocolError(f"timing_variant must be one of {sorted(TIMING_VARIANTS)}")
    if d["eval_split"] != "val":
        raise ProtocolError("eval_split must be 'val' (the test split is reserved for final evaluation)")
    if s["lr_schedule"] != "cosine" or s["amp_dtype"] not in ("bfloat16", "none"):
        raise ProtocolError("unsupported lr_schedule / amp_dtype")
    for k in _POSITIVE_SCHEDULE_INTS:
        if not _is_int(s[k], 1):
            raise ProtocolError(f"schedule.{k} must be a positive integer (not a bool), got {s[k]!r}")
    if not (_is_int(s["warmup_updates"], 0) and s["warmup_updates"] < s["total_updates"]):
        raise ProtocolError("schedule.warmup_updates must be an integer in [0, total_updates)")
    if not _is_real(s["ema_decay_per_update"], 0, 1, lo_open=True, hi_open=True):
        raise ProtocolError("schedule.ema_decay_per_update must be a finite real in (0, 1)")
    if not _is_real(s["grad_clip"], 0, lo_open=True):
        raise ProtocolError(f"schedule.grad_clip must be a finite positive gradient-norm bound, got {s['grad_clip']!r}")
    for k in ("sampler_seed", "val_seed"):
        if not _is_int(s[k], 0, SEED_MAX):
            raise ProtocolError(f"schedule.{k} must be an integer in [0, {SEED_MAX}]")


def _repo_path(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def resolve(p: Dict, train_size: int, output_dir: str, views_root: str = None) -> Dict:
    """Return {'protocol', 'train_size', 'paths', 'tinydp' (resolved dict), 'config_sha256'}."""
    validate_protocol(p)
    if not _is_int(train_size, 1):
        raise ProtocolError(f"train_size must be a positive integer, got {train_size!r}")
    cfg = default_experiment_config()
    for k, v in sorted(p["tinydp"].items()):
        _set(cfg, k, v)
    validate_resolved_tinydp(cfg)
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
        "image_integrity": _repo_path(p["data"]["image_integrity"]),
        "output_dir": os.path.abspath(output_dir),
    }
    tinydp = config_to_dict(cfg)
    hashed = copy.deepcopy(tinydp)
    hashed["logging_cfg"] = {}
    key = {"protocol": {k: p[k] for k in ("protocol_version", "data", "tinydp", "schedule")}, "train_size": train_size,
           "tinydp": hashed}
    key["protocol"]["data"] = {k: v for k, v in key["protocol"]["data"].items() if k != "views_root"}
    return {"protocol": p, "train_size": train_size, "paths": paths, "tinydp": tinydp,
            "config_sha256": hashlib.sha256(json.dumps(key, sort_keys=True, allow_nan=False).encode()).hexdigest()}
