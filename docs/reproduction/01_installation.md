# 01 — Installation (Phase 1): data-generation environment only

Date: 2026-10-22 · Branch `repro/baseline-datagen` · Host: see `00_environment_audit.md`

Only the **data-generation** environment was installed. The real-to-sim (`r2r2r_rsrd`) and 3DGS-to-mesh (`r2r2r_sugar`) environments were **not** installed, because data generation consumes the authors' released processed assets.

## 1. Environment layout

| Item | Location |
|---|---|
| venv | `real2render2real/.venv` (uv, CPython 3.10.15). Isolated; no conda env touched. |
| Assets | `real2render2real/data/` (data.zip, full) and `real2render2real/outputs/coffee_maker/` (subset of outputs.zip) |
| Asset staging | `~/Stanley_ws/r2r2r_assets_staging/` (data.zip kept for re-extraction, sha256 below) |
| Run logs / telemetry | `~/Stanley_ws/r2r2r_repro_runs/<run>/` (`run.log`, `gpu.csv`, `gpu_procs.log`, `metrics.json`, `dataset_validation.json`, `output_data/`) |
| Reproduction tooling (committed) | `docs/reproduction/tools/` — `launch_run.sh`, `bench_scaling.sh`, `run_with_num_envs.py`, `viser_null_client.py`, `tsline.py`, `analyze_run.py`, `validate_dataset.py`, `metrics_table.py`, `make_figures.py`, `remotezip.py` |
| Generated data | `real2render2real/output_data/yumi_coffee_maker/` (gitignored) |

`.gitignore` additions (repro branch): `/data/`, `/outputs/`, `/.venv/`.

## 2. Commands executed (in order)

All commands run from the repo root unless noted. **conda `base` was active in the shell → `CONDA_PREFIX` must be unset**, otherwise `isaaclab.sh` installs into the miniforge base env instead of the venv (see `03_known_issues.md`, KI-1).

```bash
git checkout -b repro/baseline-datagen          # from clean dev @ 0ac79eb
uv venv --python 3.10.15 .venv
source .venv/bin/activate
unset CONDA_PREFIX
export UV_HTTP_TIMEOUT=600                         # first torch attempt timed out at uv's default

uv pip install --upgrade pip                       # pip 26.2.2 (isaaclab.sh calls `python -m pip`)
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121          # ~5 min
uv pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com   # ~7.5 min

cd dependencies/IsaacLab && ./isaaclab.sh --install && cd ../..  # ~6 min, egl_probe built OK with apt cmake

cd dependencies/rsrd/dependencies && uv pip install -e jaxmp -e jaxls && cd ../..
uv pip install -e trajgen
uv pip install viser
uv pip install "viser[examples]"
uv pip install -U "jax[cuda12]"==0.5.3
cd .. && uv pip install -e .
uv pip install mujoco==3.3.1
```

Not executed from `env_data_gen.sh`: `pip install uv` (uv already present), `git submodule init/update` (already checked out at pinned commits), `uv sync` (lockfile conflicts, see KI-2).

### Post-install fix (documented by the authors in `asyncio_error.md`)

```bash
P=.venv/lib/python3.10/site-packages/isaacsim/extscache/omni.kit.pip_archive-0.0.0+d02c707b.lx64.cp310/pip_prebundle
mv $P/websockets $P/websockets-12.0.dist-info ~/Stanley_ws/r2r2r_repro_runs/backup_isaacsim_prebundle/
```
(moved, not deleted → reversible.) Without it, `import viser` after AppLauncher fails with `ModuleNotFoundError: No module named 'websockets.asyncio'` (observed in run01).

### Assets

| Asset | Method | Size on disk | Integrity |
|---|---|---|---|
| data.zip | full download (curl, Drive `confirm=t`) | 1 646 963 428 B zip → 2.9 GB | `zipfile.testzip()` OK; sha256 `fc22f70087f83f805a9a994cee7d06a9e43661a4b051b1bdd9917d0e97989704` |
| outputs.zip (coffee_maker subset) | HTTP-Range remote extraction of 289 members (0.12 GB compressed) | 279 MB | CRC-checked by `zipfile` on extraction |

Extracted outputs members: `outputs/coffee_maker/dig/2025-03-18_154136/{config.yml,dataparser_transforms.json}`, `outputs/coffee_maker/track/`, `outputs/coffee_maker/state_rigid_*` (both parts + `.pt`). **Not** extracted (not read by data gen): DiG checkpoint (888 MB), GARField, older DiG run, `sam_data.hdf5`, `dino.npy`, `track2/`, `track_trimmed/`, other tasks. Helper: `docs/reproduction/tools/remotezip.py` (`list URL` / `extract URL OUT_DIR PREFIX…`), run with `python3 -I`.

## 3. Resulting key versions (`uv pip freeze`)

| Package | Version | Pinned by | Note |
|---|---|---|---|
| python | 3.10.15 | env_data_gen.sh | ✅ |
| torch / torchvision / torchaudio | 2.5.1+cu121 / 0.20.1+cu121 / 2.5.1+cu121 | env_data_gen.sh, IsaacLab setup.py | ✅ unchanged through all later steps |
| nvidia-cudnn-cu12 / cublas | 9.1.0.70 / 12.1.3.1 | torch | ✅ not upgraded by `jax[cuda12]` |
| isaacsim (+core, rl, …) | 4.5.0.0 | env_data_gen.sh | ✅ |
| IsaacLab (`isaaclab`) | 0.36.6 (repo VERSION 2.0.2, submodule 7c23afbd) | submodule | ✅ |
| isaaclab_assets / rl / tasks / mimic | 0.2.1 / 0.1.4 / 0.10.27 / 1.0.3 | submodule | |
| numpy | 1.26.4 | pyproject | isaacsim install downgraded 2.2.6 → 1.26.4 |
| jax / jaxlib / jax-cuda12-plugin | 0.5.3 | env_data_gen.sh | ✅ |
| jaxlie / jaxtyping / jax-dataclasses | 1.5.0 / 0.3.2 / 1.6.3 | jaxmp, pyproject | |
| viser | 1.1.1 | unpinned | ⚠️ newer than lockfile's 1.0.6 (unpinned in script) |
| websockets | 16.1.1 | via viser | Isaac's prebundled 12.0 moved aside |
| mujoco / mujoco-mjx | 3.3.1 / 3.2.3 | env_data_gen.sh / viser[examples] | version skew; mjx not used by data gen |
| warp-lang | 1.18.0 | IsaacLab (unpinned) | ⚠️ pip warp is newer than Isaac Sim 4.5's bundled warp; not exercised by this task (no error observed) |
| trimesh | 5.1.1 | IsaacLab | |
| transformers | 4.41.2 | robomimic (isaaclab_mimic) | |
| protobuf | 3.20.3 | wandb (rl-games) | |
| nvidia-cuda-nvcc-cu12 | 12.9.86 | jax[cuda12] | added |
| egl-probe | 1.0.2 | robomimic | built from source OK |

Full freeze: kept in the session log (`freeze_final.txt`); regenerate with `uv pip freeze`.

## 4. Validation (each step tested before the next)

| Check | Command / evidence | Result |
|---|---|---|
| Torch GPU | `torch.cuda.is_available()`, device_count=2, matmul on cuda:0, cuDNN 90100 | ✅ |
| Isaac Sim startup (headless) | `SimulationApp({"headless": True})` smoke test | ✅ app ready 6.4 s; first launch "Startup Complete" at 155 s (shader cache build). Driver 580.178.04 accepted (Vulkan). |
| Isaac Sim GPU enumeration | Kit GPU table | Both 4090s enumerated, Intel iGPU skipped; with the default *non-rendering* kit both were "Active" (multi-GPU) |
| IsaacLab install | `./isaaclab.sh --install` exit 0; all 5 extensions editable | ✅ |
| JAX GPU | `jax.devices()` → `[CudaDevice(0), CudaDevice(1)]`, matmul OK alongside torch in one process | ✅ |
| R2R2R imports | `import jaxmp, jaxls, jaxlie, trajgen, viser, mujoco, real2render2real; from trajgen.traj_interp import traj_interp_batch` | ✅ |
| IsaacLab + R2R2R under AppLauncher | `scripts/run.py` | ❌ run01 (websockets) → ✅ run02 after fix |
| Rendering | run02 produces 1280×720 RGB from 2 calibrated cameras | ✅ (see 02) |
| GPU actually used | `nvidia-smi pmon` / Kit GPU table | ⚠️ physics/torch on GPU 0 but **RTX renderer on GPU 1** by default — see 02 and KI-6 |

## 5. Dependency adjustments vs. the authors' script (complete list)

1. `UV_HTTP_TIMEOUT=600` (network only).
2. `unset CONDA_PREFIX` before `isaaclab.sh --install` (installer targeting).
3. Isaac Sim prebundled `websockets 12.0` moved out of `pip_prebundle` (authors' documented fix).
4. No version of any package was changed from what the script resolves; no source file of R2R2R, IsaacLab or submodules was modified.
