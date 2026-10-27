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
"""
import hashlib
import json
from typing import Dict, List, Optional

import numpy as np
import torch

from .driver import CKPT_FORMAT
from .normalizer_v2 import absolute_actions


class R2R2RPolicy:
    def __init__(self, ckpt: Dict, weights: str = "ema", device: str = "cpu"):
        from tinydp.dataset.utils import default_vision_transform
        from tinydp.policy.model import DiffusionPolicy
        from .protocol import config_from_dict

        if ckpt.get("format") != CKPT_FORMAT:
            raise ValueError(f"not a {CKPT_FORMAT} checkpoint")
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
        self.model.to(self.device).eval()
        st = ckpt["action_statistics"]
        lo = np.array(st["min_action"], dtype=np.float64).reshape(st["shape"])
        hi = np.array(st["max_action"], dtype=np.float64).reshape(st["shape"])
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

    @torch.no_grad()
    def predict(self, observation: torch.Tensor, proprio: torch.Tensor, num_inference_steps: Optional[int] = None,
                seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
        B = proprio.shape[0]
        if observation.shape[:3] != (B, self.obs_horizon, 2) or proprio.shape[1:] != (self.obs_horizon, 20):
            raise ValueError(f"expected observation (B,{self.obs_horizon},2,3,H,W) and proprio (B,{self.obs_horizon},20)")
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
        return {"normalized": z, "right_action": self.denormalize(z.double()).float()}
