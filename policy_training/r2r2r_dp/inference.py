"""Self-contained inference adapter for r2r2r-dp-ckpt-v2 checkpoints (audit M5).

TinyDP's DiffusionWrapper cannot be imported (it imports the non-existent tinydp.dataset.dataset), reads
the normalizer from <folder>/action_statistics.json and loads the raw 'model' weights although
validation uses EMA weights. This adapter needs only the checkpoint file: the resolved TinyDP
configuration and the exact action statistics are stored inside it.

Weights: 'ema' (default; the weights train-time validation evaluates) or 'raw' (full checkpoints only).
Inputs (TinyDP sim contract, see VALIDATED_TRAINING_PROTOCOL.md):
  observation  (B, obs_horizon=4, 2, 3, H, W), camera order [camera_1, camera_0], each frame passed through
               tinydp.dataset.utils.default_vision_transform(image_size) (use preprocess_frames)
  proprio      (B, 4, 20) = [left xyz, left rot6d, left grip, right xyz, right rot6d, right grip]
               (absolute, base_link, metres; rot6d = first two ROWS of R; grip 1 = close command)
Output: right-arm actions (B, 16, 10) = [xyz, rot6d, gripper] in the same frame and units, i.e. absolute
targets for state rows t+2, t+4, ..., t+32 relative to the newest observation row t (subsample 2).
right_poses() converts them to validated rotation matrices (rotation.py), never via TinyDP's helper.

Validation (Codex final audit F8): weights and bounds must be finite with min <= max (finite zero-range
dims keep TinyDP's constant denormalisation); inputs must be finite tensors of the exact shapes above
(H = W = image_size); num_inference_steps must be an integer in [1, num_train_diffusion_steps]; a
non-finite policy output raises instead of being returned.
"""
import hashlib
import json
from typing import Dict, List, Optional

import numpy as np
import torch

from .driver import INFERENCE_FORMATS
from .normalizer_v2 import absolute_actions, require_bounds
from .rotation import rot6d_rows_to_matrix


class R2R2RPolicy:
    def __init__(self, ckpt: Dict, weights: str = "ema", device: str = "cpu"):
        from tinydp.dataset.utils import default_vision_transform
        from tinydp.policy.model import DiffusionPolicy
        from .protocol import config_from_dict

        if ckpt.get("format") not in INFERENCE_FORMATS or ckpt.get("kind") not in ("full", "inference"):
            raise ValueError(f"not a {'/'.join(INFERENCE_FORMATS)} full or inference checkpoint")
        if weights not in ("ema", "raw"):
            raise ValueError("weights must be 'ema' or 'raw'")
        if ckpt["kind"] == "inference" and weights != ckpt["weights"]:
            raise ValueError(f"inference export holds {ckpt['weights']} weights only")
        self.cfg = cfg = config_from_dict(ckpt["run"]["tinydp"])
        if not cfg.model_cfg.policy_cfg.pred_right_only or cfg.shared_cfg.use_delta_action:
            raise ValueError("adapter supports the right-only absolute-action protocol only")
        self.device = torch.device(device)
        self.model = DiffusionPolicy(shared_cfg=cfg.shared_cfg, model_cfg=cfg.model_cfg)
        self.model.load_state_dict(ckpt["model"])
        if weights == "ema" and ckpt["kind"] == "full":
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    p.copy_(ckpt["ema"][name])
        bad = [k for k, v in self.model.state_dict().items() if v.is_floating_point() and not torch.isfinite(v).all()]
        if bad:
            raise ValueError(f"checkpoint weights are not finite: {bad[:5]}")
        self.model.to(self.device).eval()
        st = ckpt["action_statistics"]
        if st.get("shape") != [20]:
            raise ValueError(f"action statistics shape {st.get('shape')!r}, expected [20]")
        lo, hi = require_bounds(st["min_action"], st["max_action"])
        self.right_min, self.right_max = lo[10:20], hi[10:20]
        # zero-range dims: training targets were nan_to_num'd to 0 and denormalise to the constant (as TinyDP)
        self.degenerate_dims = [int(i) for i in np.flatnonzero(self.right_max - self.right_min <= 0)]
        self.statistics_sha256 = hashlib.sha256(json.dumps(st, sort_keys=True).encode()).hexdigest()
        self.weights, self.update = weights, ckpt["update"]
        self.config_sha256 = ckpt["run"]["config_sha256"]
        self.vision_transform = default_vision_transform(resolution=cfg.shared_cfg.image_size)
        self.obs_horizon, self.action_horizon = cfg.shared_cfg.seq_length, cfg.shared_cfg.num_pred_steps

    @classmethod
    def load(cls, path: str, weights: str = "ema", device: str = "cpu") -> "R2R2RPolicy":
        return cls(torch.load(path, map_location="cpu", weights_only=False), weights=weights, device=device)

    def preprocess_frames(self, frames: List[List]) -> torch.Tensor:
        """frames[t][c]: PIL images for obs step t and camera c in [camera_1, camera_0] order -> (4, 2, 3, H, W)."""
        return torch.stack([torch.stack([self.vision_transform(im) for im in cams]) for cams in frames])

    @staticmethod
    def proprio_from_state(ee_poses: np.ndarray, gripper_cmd: np.ndarray) -> np.ndarray:
        """(T,14) ee_poses (xyz + wxyz per arm) and (T,2) gripper commands -> (T,20) proprio (no noise)."""
        return absolute_actions(np.asarray(ee_poses, np.float32), np.asarray(gripper_cmd))

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        lo = torch.as_tensor(self.right_min, dtype=z.dtype, device=z.device)
        hi = torch.as_tensor(self.right_max, dtype=z.dtype, device=z.device)
        return (z + 1) / 2 * (hi - lo) + lo

    def right_poses(self, right_action) -> Dict[str, np.ndarray]:
        """(..., 10) right-arm actions -> xyz (..., 3), validated rotation matrices (..., 3, 3), gripper (...)."""
        a = np.asarray(right_action.detach().cpu() if torch.is_tensor(right_action) else right_action, dtype=np.float64)
        if a.shape[-1] != 10 or not np.isfinite(a).all():
            raise ValueError("right_action must be finite with a trailing dimension of 10")
        return {"xyz": a[..., :3], "rotation": rot6d_rows_to_matrix(a[..., 3:9]), "gripper": a[..., 9]}

    def _check_inputs(self, observation, proprio, num_inference_steps, seed) -> int:
        if not (torch.is_tensor(observation) and torch.is_tensor(proprio)):
            raise TypeError("observation and proprio must be torch tensors")
        S, L = self.cfg.shared_cfg.image_size, self.obs_horizon
        B = proprio.shape[0] if proprio.ndim == 3 else -1
        if B < 1 or tuple(proprio.shape) != (B, L, 20) or tuple(observation.shape) != (B, L, 2, 3, S, S):
            raise ValueError(f"expected observation (B,{L},2,3,{S},{S}) and proprio (B,{L},20) with B >= 1, "
                             f"got {tuple(observation.shape)} and {tuple(proprio.shape)}")
        for name, x in (("observation", observation), ("proprio", proprio)):
            if not x.is_floating_point() or not torch.isfinite(x).all():
                raise ValueError(f"{name} must be a finite floating-point tensor")
        n_train = self.cfg.model_cfg.policy_cfg.num_train_diffusion_steps
        if num_inference_steps is not None and not (type(num_inference_steps) is int and 1 <= num_inference_steps <= n_train):
            raise ValueError(f"num_inference_steps must be an integer in [1, {n_train}], got {num_inference_steps!r}")
        if seed is not None and not (type(seed) is int and seed >= 0):
            raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
        return B

    @torch.no_grad()
    def predict(self, observation: torch.Tensor, proprio: torch.Tensor, num_inference_steps: Optional[int] = None,
                seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
        B = self._check_inputs(observation, proprio, num_inference_steps, seed)
        steps = num_inference_steps or self.cfg.model_cfg.policy_cfg.num_inference_diffusion_steps
        self.model.noise_scheduler.set_timesteps(steps)
        batch = {"observation": observation.to(self.device).float(), "proprio": proprio.to(self.device).float()}
        devs = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devs, enabled=seed is not None):
            if seed is not None:
                torch.manual_seed(seed)
            z = self.model.forward_inference(batch)
        if z.shape != (B, self.action_horizon, 10):
            raise RuntimeError(f"unexpected policy output shape {tuple(z.shape)}")
        action = self.denormalize(z.double()).float()
        if not (torch.isfinite(z).all() and torch.isfinite(action).all()):
            raise FloatingPointError("policy output is not finite")
        return {"normalized": z, "right_action": action}
