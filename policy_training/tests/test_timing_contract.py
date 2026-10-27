"""Audit M6: the frozen 'released_pairing' timing contract, on a fixture whose images lag the state by 2 rows.

Filename arithmetic (image file j paired with proprio row j and targets j+2..j+32) is what the released
loader does and what the baseline keeps. These tests show that this arithmetic is preserved exactly AND
that it does not imply sensing alignment: with a 2-row render delay, image file j depicts state row j-2.
"""
import os

import numpy as np
import pytest
import torch
import zarr
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from fixtures import DX, X0, episode_arrays, write_episode
from r2r2r_dp import analysis
from r2r2r_dp import tinydp_bridge as tb
from r2r2r_dp.adapter import SIDE_CARS
from r2r2r_dp.normalizer_v2 import build_statistics

LAG, T = 2, 60


def level(t):
    return 20 + 3 * max(t - LAG, 0)  # frame t shows state row t - LAG (clamped at the start)


@pytest.fixture
def lagged_view(tmp_path):
    a = episode_arrays(T)
    ep = write_episode(str(tmp_path / "view" / "ep_train_0000"), arrays=a)
    for cam in ("camera_0", "camera_1"):
        for t in range(T):
            Image.new("RGB", (32, 24), (level(t),) * 3).save(os.path.join(ep, cam, "rgb", f"{t:04d}.jpg"), quality=95)
    h5 = os.path.join(ep, "robot_data", "robot_data.h5")
    for key, suffix in SIDE_CARS.items():
        zarr.save_array(h5.replace(".h5", suffix), a[key])
    open(os.path.join(ep, "source.json"), "w").write('{"uid": "run/ep"}')
    return str(tmp_path / "view"), a


def _decode_state_row(img: torch.Tensor) -> int:
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD).view(3, 1, 1)
    g = float(((img * std + mean) * 255).mean())
    return int(round((g - 20) / 3))


def test_released_pairing_is_filename_arithmetic_not_sensing_alignment(lagged_view, tmp_path):
    import json
    import tyro
    from tinydp.dataset.utils import default_vision_transform
    from tinydp.util.args import ExperimentConfig
    view, a = lagged_view
    json.dump(build_statistics(view, "n0001", ["run/ep"], "m", 2, 4, 16), open(tmp_path / "s.json", "w"))
    cfg = tyro.cli(ExperimentConfig, args=["--dataset-cfg.is-sim-data", "--dataset-cfg.no-scale-action"])
    cfg.dataset_cfg.proprio_noise = 0.0
    cfg.logging_cfg.output_dir = str(tmp_path / "out")
    os.makedirs(cfg.logging_cfg.output_dir)
    ds = tb.make_manifest_dataset_class(view, view, str(tmp_path / "s.json"))(
        cfg.dataset_cfg, cfg.shared_cfg, cfg.logging_cfg, default_vision_transform(32), split="train")
    checked = 0
    for i, (_, j, _) in enumerate(ds.start_end):
        if j < LAG:
            continue
        item = ds[i]
        assert int(round((item["proprio"][10].item() - X0) / DX)) == j  # proprio row j
        rows = np.round((item["action"][:, 10].numpy() - X0) / DX).astype(int)
        assert rows.tolist() == list(range(j + 2, j + 33, 2))  # absolute targets j+2 .. j+32
        for cam in range(2):  # both cameras: file j, which depicts state row j - 2
            assert _decode_state_row(item["observation"][cam]) == j - LAG
        checked += 1
    assert checked > 50


def test_lag_estimator_recovers_render_delay(tmp_path):
    """The motion-energy estimator used on the real data recovers a known 2-row delay (and 0 without delay)."""
    for lag in (0, 2):
        ep = str(tmp_path / f"lag{lag}")
        a = episode_arrays(T)
        x = np.where(np.arange(T) < 30, 0.3, 0.4)  # one sharp state event between rows 29 and 30
        a["ee_poses"][:, 7] = x
        write_episode(ep, arrays=a)
        for cam in ("camera_0", "camera_1"):
            for t in range(T):
                s = max(t - lag, 0)
                Image.new("RGB", (32, 24), ((60 if s < 30 else 160),) * 3).save(
                    os.path.join(ep, cam, "rgb", f"{t:04d}.jpg"), quality=95)
        speed = np.abs(np.diff(x))
        for cam in ("camera_0", "camera_1"):
            assert analysis.best_lag(speed, analysis.image_motion_energy(ep, cam, T))["lag"] == lag
