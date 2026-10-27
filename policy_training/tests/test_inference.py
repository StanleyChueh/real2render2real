"""Audit M5: checkpoint -> self-contained reload -> denoise -> denormalise, on CPU."""
import json
import os
import shutil

import numpy as np
import pytest
import torch

import train_r2r2r_dp
from r2r2r_dp.inference import R2R2RPolicy
from run_fixtures import make_run_fixture


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("inf")
    fx = make_run_fixture(tmp, total_updates=3, full_ckpt_every_updates=3, inference_export_every_updates=3,
                          ema_decay_per_update=0.5)
    out = os.path.join(str(tmp), "run")
    train_r2r2r_dp.main(["--protocol", fx["protocol_path"], "--train-size", "3", "--output-dir", out, "--device", "cpu"])
    # move the two checkpoint files somewhere else and delete the run directory: nothing else may be needed
    keep = os.path.join(str(tmp), "isolated")
    os.makedirs(keep)
    for f in ("ckpt_full_u0000003.pt", "policy_ema_u0000003.pt"):
        shutil.copy(os.path.join(out, f), keep)
    stats = json.load(open(os.path.join(out, "action_statistics.json")))
    shutil.rmtree(out)
    return {"full": os.path.join(keep, "ckpt_full_u0000003.pt"), "export": os.path.join(keep, "policy_ema_u0000003.pt"),
            "stats": stats, "fx": fx}


def _batch(B=2, res=32):
    g = torch.Generator().manual_seed(0)
    return torch.randn(B, 4, 2, 3, res, res, generator=g), torch.randn(B, 4, 20, generator=g)


def test_reload_denoise_denormalise_matches_training_model(trained):
    from tinydp.dataset.utils import unscale_action
    from tinydp.policy.model import DiffusionPolicy
    from r2r2r_dp.protocol import config_from_dict
    ck = torch.load(trained["full"], weights_only=False)
    pol = R2R2RPolicy.load(trained["full"], weights="ema")
    obs, prop = _batch()
    out = pol.predict(obs, prop, num_inference_steps=4, seed=11)
    assert out["right_action"].shape == (2, 16, 10) and out["normalized"].shape == (2, 16, 10)
    assert torch.isfinite(out["right_action"]).all() and out["normalized"].abs().max() <= 1.0 + 1e-6  # clip_sample
    # reference: the training model with the EMA weights applied (what validation evaluates)
    cfg = config_from_dict(ck["run"]["tinydp"])
    ref = DiffusionPolicy(shared_cfg=cfg.shared_cfg, model_cfg=cfg.model_cfg)
    ref.load_state_dict(ck["model"])
    with torch.no_grad():
        for n, p in ref.named_parameters():
            p.copy_(ck["ema"][n])
    ref.eval()
    ref.noise_scheduler.set_timesteps(4)
    torch.manual_seed(11)
    with torch.no_grad():
        z = ref.forward_inference({"observation": obs, "proprio": prop})
    assert torch.equal(z, out["normalized"])
    # denormalisation equals TinyDP's own unscale_action on the right half of the 20-D statistics
    st = {"min": torch.tensor(trained["stats"]["min_action"]), "max": torch.tensor(trained["stats"]["max_action"])}
    full = unscale_action(torch.cat([z.double(), z.double()], -1), stat=st, type="diffusion")[..., 10:20]
    assert torch.allclose(full.float(), out["right_action"], atol=1e-6)


def test_export_and_full_checkpoint_give_identical_ema_policy(trained):
    obs, prop = _batch()
    a = R2R2RPolicy.load(trained["full"], weights="ema").predict(obs, prop, num_inference_steps=4, seed=3)
    b = R2R2RPolicy.load(trained["export"], weights="ema").predict(obs, prop, num_inference_steps=4, seed=3)
    assert torch.equal(a["normalized"], b["normalized"])
    raw = R2R2RPolicy.load(trained["full"], weights="raw").predict(obs, prop, num_inference_steps=4, seed=3)
    assert not torch.equal(a["normalized"], raw["normalized"])  # EMA and raw weights are distinct, chosen explicitly
    with pytest.raises(ValueError, match="ema weights only"):
        R2R2RPolicy.load(trained["export"], weights="raw")


def test_input_contract_enforced(trained):
    pol = R2R2RPolicy.load(trained["full"])
    obs, prop = _batch()
    with pytest.raises(ValueError):
        pol.predict(obs, prop[..., :10])  # proprio must stay 20-D
    with pytest.raises(ValueError):
        pol.predict(obs[:, :3], prop[:, :3])  # four observation steps
    assert pol.weights == "ema" and pol.update == 3 and len(pol.config_sha256) == 64
    assert pol.degenerate_dims == [2, 3, 4, 5, 6, 7, 8]  # fixture: right z and rotation (identity) constant
    ee = np.zeros((5, 14), np.float32)
    ee[:, 3] = ee[:, 10] = 1.0  # identity wxyz quaternions
    p = R2R2RPolicy.proprio_from_state(ee, np.zeros((5, 2), bool))
    assert p.shape == (5, 20) and np.allclose(p[0, 3:9], [1, 0, 0, 0, 1, 0]) and np.allclose(p[0, 13:19], [1, 0, 0, 0, 1, 0])


def test_released_wrapper_still_unimportable():
    """Inherited TinyDP defect (unchanged submodule); the adapter above replaces it."""
    with pytest.raises(ModuleNotFoundError, match="tinydp.dataset.dataset"):
        __import__("tinydp.policy.diffusion_wrapper")
