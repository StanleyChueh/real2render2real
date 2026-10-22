# 00 — Environment & Repository Audit (Phase 0)

Date: 2026-10-22
Branch: `repro/baseline-datagen` (created from `dev` @ `0ac79eb`, worktree verified clean)
Fork: https://github.com/StanleyChueh/real2render2real · Upstream: https://github.com/uynitsuj/real2render2real

## 1. System

| Item | Value |
|---|---|
| OS | Ubuntu 22.04.5 LTS (jammy) |
| Kernel | 6.8.0-138-generic |
| glibc | 2.35 |
| CPU | Intel i7-13700, 24 threads |
| RAM | 62 GiB (≈34 GiB available at audit time), 2 GiB swap |
| Disk | `/` 3.7 TB, 3.2 TB free (single NVMe; `~` and `/tmp` on same FS) |
| GPU 0 | RTX 4090 24 GB, PCI 01:00.0, idle (Xorg 4 MiB only) |
| GPU 1 | RTX 4090 24 GB, PCI 05:00.0, **drives the desktop** (Xorg, gnome-shell, Firefox ≈ 0.6 GB) |
| NVIDIA driver | 580.178.04 (reports CUDA 13.0 runtime capability) |
| System CUDA toolkit | none (`nvcc` not on PATH, nothing in `/usr/local/cuda*`) |
| gcc / cmake | 11.4.0 / 3.22.1 (`/usr/bin/cmake`, apt-provided — matters for the README's `egl_probe` note) |
| `DISPLAY` | `:0` (a display exists, but run.py forces headless) |
| inotify `max_user_watches` | 65536 (see Known Issues: carb `errno=28`) |
| `CUDA_VISIBLE_DEVICES` | unset |

## 2. Existing Python environments (left untouched)

| Env | Python | Notes |
|---|---|---|
| conda `base` (miniforge3) | 3.14.6 | conda 26.5.3, mamba available |
| conda `env_isaaclab` | 3.11.16 | **Isaac Sim 5.1.0 + IsaacLab 0.54.4** (from `~/Stanley_ws/IsaacLab`, v2.3.2), torch 2.7.0+cu128. **Incompatible** with R2R2R (needs Isaac Sim 4.5 / Py3.10 / IsaacLab 2.0.2). Not used, not modified. |
| conda `lerobot`, `lerobot_smolvla` | — | unrelated |
| `uv` | 0.12.18 at `~/.local/bin/uv` | used for the data-gen venv |
| `~/.local/share/ov/data/exts` | — | Omniverse ext cache shared with the existing Isaac Sim 5.1 install. Isaac Sim 4.5 pip uses versioned kit dirs; watched as a possible conflict source. |

No `.venv` existed in the repo — the R2R2R data-gen env will be created at `real2render2real/.venv` (repo-local, isolated).

## 3. Repository

- `dev` up to date with `origin/dev`, clean. Latest: `0ac79eb Fix directory name in env_3dgs_to_mesh.sh`.
- Remote `origin` = fork (StanleyChueh). No `upstream` remote configured.
- Repo size 1.4 GB (mostly submodules).

### Submodules (recursive, all checked out, none dirty)

| Path | Commit | Notes |
|---|---|---|
| dependencies/IsaacLab | `7c23afbd` (VERSION **2.0.2**) | uynitsuj fork; "franka articulation cfg changes". IsaacLab 2.0.x ↔ Isaac Sim 4.5 ✔ |
| dependencies/trajgen | `6b19f56` "CoRL submit code" | needed for data gen |
| dependencies/rsrd | `cca2256` | only `jaxmp`, `jaxls` needed for data gen |
| rsrd/dependencies/jaxmp | `d1ab9c7` | IK solver (data gen) |
| rsrd/dependencies/jaxls | `2d1035c` (py310 branch) | data gen |
| rsrd/dependencies/{dig,garfield,hamer,hamer_helper} | — | real-to-sim only |
| dependencies/SuGaR | `bc2c49f` | 3DGS→mesh only |
| dependencies/openpi (+aloha, libero), tinydp | — | policy training only |

### Entry points & inputs for the target task (YuMi coffee maker)

- `scripts/run.py` → `YumiCoffeeMakerCfg(num_envs=2, env_spacing=1.5)` + `CoffeeMaker` simulator, `save_data=True`, output `output_data/yumi_coffee_maker/`. Forces `headless=True`, `enable_cameras=True`. Loop stops at 1000 successful envs.
- Reads from `data/`: `assets/object_scans/{coffee_maker,mug}/*.usd`, `assets/table2/table2_instanceable.usd`, `assets/skyboxes/*`, `camera_poses/camera_poses_calib_031425.h5`, `yumi_description/urdf/yumi.urdf`, YuMi USD.
- Reads from `outputs/coffee_maker/`: `dig/2025-03-18_154136/{config.yml,dataparser_transforms.json}`, `track/keyframes.txt`, `state_rigid_*/grasps.txt` (directory enumeration order matters: sorted by the time stamp in the dir name).

### Released assets (Google Drive folder `14I_Y66uaiAbUDb5nO-uf_mLw4_OBgvdQ`, public, no auth)

| File | Drive ID | Compressed | Notes |
|---|---|---|---|
| data.zip | `1JA_8RcskQcFQKfjou4Sg9YzN8kDgsoeC` | 1.5 GB | 13 362 entries; all data-gen assets. Download fully. |
| outputs.zip | `1r8nYx3WHf_6axkZAGn1RJJS6JmAKe6Jr` | 13.1 GB (14 046 662 575 B) | ≈54 GB unzipped; `coffee_maker/` = 7.1 GB, of which 3.3 GB `sam_data.hdf5`, 1.1 GB GARField, 1.2 GB an older DiG run are **not** read by data gen. |

The Drive endpoint supports HTTP `Range`, so only the required coffee-maker members are extracted remotely (central-directory read + per-member range fetches; ≈1.3 GB instead of 13 GB). Tool: `remotezip.py` (kept outside the repo; documented in 01_installation.md).

### Dependency pin sources

| Source | Pins |
|---|---|
| `env_data_gen.sh` | Py 3.10.15, torch 2.5.1/tv 0.20.1/ta 2.5.1 (cu121), `isaacsim[all,extscache]==4.5.0`, IsaacLab `./isaaclab.sh --install`, `jaxmp`, `jaxls`, `trajgen` editable, viser, `jax[cuda12]==0.5.3`, `mujoco==3.3.1` |
| `pyproject.toml` | numpy==1.26.4, jax[cuda12]==0.5.3, jaxtyping==0.3.2, viser, tyro; requires-python `>=3.10,<3.12` |
| `uv.lock` | Resolves only the project's own deps; contains **torch 2.8.0** (and 2.0.1) and no isaacsim → **conflicts** with the shell-script pins. `uv sync` must NOT be used; follow `env_data_gen.sh` with `uv pip`. |

## 4. Compatibility table

| Component | Required | Installed | Status |
|---|---|---|---|
| OS | Ubuntu 22.04 (Isaac Sim 4.5 supports 20.04/22.04) | 22.04.5 | ✅ |
| Kernel | — | 6.8.0-138 | ✅ (HWE kernel; no known Isaac Sim 4.5 issue) |
| glibc | ≥ 2.34 (isaacsim pip wheels) | 2.35 | ✅ |
| NVIDIA driver | Isaac Sim 4.5 recommends 535.129.03 (min) | 580.178.04 | ⚠️ newer than validated; expected to work, **must be verified by a startup/render test**. No driver change will be made. |
| CUDA (torch) | 12.1 runtime via wheels | driver supports ≤13.0; no system toolkit | ✅ cu121 wheels bundle their runtime |
| CUDA (jax) | `jax[cuda12]==0.5.3` (pip CUDA libs) | — | ✅ pip-bundled |
| GPU | RTX-class, ≥ 8 GB (README: 50 envs fit on a 4090) | 2× RTX 4090 24 GB | ✅ use GPU 0 (headless, idle) |
| RAM | Isaac Sim recommends 32 GB+ | 62 GB | ✅ (≈34 GB free: other workloads on box) |
| Disk | ≈ 50 GB isaacsim+venv, 1.5 GB data, ≈1.3 GB outputs subset, output data | 3.2 TB free | ✅ |
| Python | 3.10.15 | uv-managed 3.10.15 (to be installed into `.venv`) | ✅ planned |
| Isaac Sim | 4.5.0 (pip) | 5.1.0 only in another conda env | ➖ install 4.5.0 into isolated `.venv` |
| IsaacLab | pinned submodule 2.0.2 @ 7c23afbd | present | ✅ |
| PyTorch | 2.5.1+cu121 | — | ➖ install |
| cmake | compiled (not pip) for egl_probe | apt 3.22.1 | ✅ |
| gcc | for egl_probe / native ext | 11.4 | ✅ |
| Network | pypi.nvidia.com, download.pytorch.org, Drive | all reachable (HTTP 200) | ✅ |

## 5. Known / anticipated installation problems

1. `uv.lock` resolves torch 2.8.0 → do not `uv sync`; use `env_data_gen.sh` steps.
2. README refers to `./isaaclab_alt.sh` as an `egl_probe` fallback, but **that file does not exist** in the pinned IsaacLab submodule.
3. `websockets` shadowing: Isaac Sim 4.5's `omni.kit.pip_archive` prebundle ships websockets 12.0; viser needs ≥13.1 (`asyncio_error.md`). Expected to hit at `import viser` after AppLauncher.
4. carb `Failed to create change watch … errno=28` with inotify limit 65536 (`asyncio_error.md`). A fix would need `sysctl fs.inotify.max_user_watches` (system change, requires sudo) — only if it actually blocks.
5. `jax[cuda12]` and torch-cu121 both ship CUDA libs into the same venv (nvidia-* wheels); possible cuDNN/cublas version conflicts — check JAX GPU init.
6. Isaac Sim 4.5 first launch compiles shaders (several minutes) and requires accepting the Omniverse EULA (`OMNI_KIT_ACCEPT_EULA=YES`).
7. GPU selection: Isaac Sim picks its rendering GPU via `--/renderer/activeGpu` / `multi_gpu` settings and AppLauncher `--device`; `CUDA_VISIBLE_DEVICES` alone is not guaranteed to restrict the RTX renderer. Will be checked with `nvidia-smi`.
8. Table randomization needs NVIDIA vMaterials (sudo `.run` installer) — **not needed** for the original coffee-maker run (randomize_table is off by default); skipped.

## 6. Verdict

Environment is compatible. **No major system changes required** (no driver/CUDA/kernel changes, no sudo). Proceeding to Phase 1 with an isolated repo-local uv venv, data-gen environment only. The real-to-sim (`r2r2r_rsrd`) and SuGaR environments are **not** installed — data generation uses the authors' released processed assets.
