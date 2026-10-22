# 02 — Original Data Generation: YuMi Coffee Maker (Phases 2–3)

Date: 2026-10-22 · Branch `repro/baseline-datagen` · Env: `01_installation.md`

## 1. What was run

- Entry point: **unmodified** `scripts/run.py` → `YumiCoffeeMakerCfg` + `CoffeeMaker` simulator, `save_data=True`, headless, cameras on. Original algorithm, original task configuration, authors' released assets (`data.zip`, coffee-maker subset of `outputs.zip`).
- For num_envs ≠ 2, `docs/reproduction/tools/run_with_num_envs.py` executes run.py's source with only the literal `YumiCoffeeMakerCfg(num_envs=2, …)` replaced (asserts a single match; run.py on disk untouched). The authors' own comment in run.py instructs editing this value per GPU.
- Launcher `docs/reproduction/tools/launch_run.sh` (runs used its earlier versions v1–v4, same measurement logic): timestamps every log line, samples `nvidia-smi` every 1 s (per GPU) and per-process every 10 s, attaches a viser client (required, KI-7), stops via `timeout --signal=INT`, then kills orphaned logger workers (KI-8).
- Runtime settings that are **not** defaults (all outside the algorithm):
  - `CUDA_VISIBLE_DEVICES=0` — pins CUDA, JAX, PhysX **and** (verified) the RTX renderer to GPU 0 (KI-6).
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false` — JAX allocates on demand instead of grabbing 25 % of VRAM (KI-16).
  - viser client: headless Chrome (`--disable-gpu`) or a minimal Python websocket client (`tools/viser_null_client.py`).

### Commands

```bash
# baseline, as shipped (2 envs; GPU placement left to Isaac Sim)
L=docs/reproduction/tools/launch_run.sh
CLIENT=chrome $L run02_env2 2400
# single-GPU (GPU 0)
CLIENT=chrome CUDA_VISIBLE_DEVICES=0 $L run04_env2_gpu0 1200
# scaling
CLIENT=null CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false R2R2R_NUM_ENVS=N $L run07_envN_… 480   # = tools/bench_scaling.sh
# metrics / integrity / figures
python docs/reproduction/tools/analyze_run.py RUN_DIR --json RUN_DIR/metrics.json
python docs/reproduction/tools/validate_dataset.py RUN_DIR/output_data/yumi_coffee_maker --json …
python docs/reproduction/tools/metrics_table.py RUN_DIR …
python docs/reproduction/tools/make_figures.py --out FIG_DIR --dataset … --timeline-runs … --scaling "label=run,run,…"
```
(run02 used the first launcher version; identical apart from timestamps and client handling.)

## 2. Validation checklist (Phase 2)

| # | Check | Evidence | Result |
|---|---|---|---|
| 1 | Environment initialisation | `viser (listening *:8080)` 15.6–21 s after launch (warm shader cache; first-ever launch 155 s); "Found 2 saved camera poses"; jaxmp controller built | ✅ |
| 2 | Robot & object rendering | `samples_rgb.png`: YuMi, coffee maker, mug, table, randomized skybox/background and lights visible from both calibrated cameras | ✅ |
| 3 | Trajectory generation | `trajectories.png`, `joint_angles.png`: right arm reaches, grasps (gripper closes at step ≈48), carries the mug to the coffee maker, releases (≈131), retracts; per-episode variation in start pose/trajectory | ✅ |
| 4 | RGB image output | `camera_0/rgb`, `camera_1/rgb`, 1280×720×3 JPEG, 161 frames per camera per episode | ✅ |
| 5 | Robot states & actions | `robot_data.h5`: `joint_angles` (161×16), `ee_poses` (161×14 = left xyz+wxyz, right xyz+wxyz, base frame), `gripper_binary_cmd` (161×2 bool, [left, right]); `joint_names.txt` | ✅ |
| 6 | Successful rollout directories | `output_data/yumi_coffee_maker/successes/env_<i>_<timestamp>/`; failed envs deleted automatically | ✅ |
| 7 | Data integrity | `validate_dataset.py` over **every** saved rollout of every run: h5 readable, finite, image count = T for both cameras, sampled images decodable, exactly 2 gripper toggles | ✅ 0 invalid (see table) |

Notes on what is saved: actions are not stored separately — downstream converters (`scripts/yumi_convert_dpgs_data_fast.py`, openpi `convert_dpgs_data.py`) derive action chunks from the joint/EE trajectories. Depth is off by default.

## 3. Results

All runs on RTX 4090 GPU 0 unless noted; one run per configuration (8–40 min each), so differences of a few % in success rate are within run-to-run noise. "Success" = passes the simulator's kinematic feasibility filter (see 03, O-4). Steady-state throughput excludes initialisation and the first batch. Full machine-readable table: `docs/reproduction/metrics_table.md`; per-run `metrics.json` / `dataset_validation.json` in `~/Stanley_ws/r2r2r_repro_runs/<run>/`.

| run | envs | GPU placement | JAX alloc | viser client | batches | succ / attempted | success % | init → viser [s] | 1st batch done [s] | median batch [s] | s / attempted traj | **succ traj / h** | GPU0 peak [GiB] | GPU0 util % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| run02 | 2 | default (render+JAX on GPU1) | 25 % | chrome | 129 | 222/258 | 86.0 | – | – | 13.0 | 6.50 | 456 | 9.7 (+6.1 on GPU1) | 17 |
| run04 | 2 | GPU 0 | 25 % | chrome | 92 | 160/184 | 87.0 | 15.6 | 43.4 | 12.4 | 6.22 | 495 | 14.4 | 19 |
| run05 | 5 | GPU 0 | 25 % | chrome | 34 | 155/170 | 91.2 | 20.7 | 60.7 | 19.1 | 3.82 | 839 | 21.0 | 54 |
| run05 | 10 | GPU 0 | 25 % | chrome | 0 | **OOM** | – | – | – | – | – | – | 23.5 (full) | – |
| run06 | 2 | GPU 0 | lazy | chrome | 35 | 67/70 | 95.7 | 20.6 | 49.7 | 12.5 | 6.25 | 551 | 8.6 | 41 |
| run06 | 5 | GPU 0 | lazy | chrome | 22 | 97/110 | 88.2 | 16.9 | 51.0 | 19.1 | 3.83 | 797 | 14.0 | 54 |
| run06 | 8 | GPU 0 | lazy | chrome | 24 | 166/192 | 86.5 | 16.9 | 65.7 | 23.5 | 2.94 | 1078 | 16.7 | 70 |
| run06 | 10 | GPU 0 | lazy | chrome | 18 | 164/180 | 91.1 | 17.5 | 65.7 | 30.2 | 3.02 | 1065 | 19.3 | 63 |
| run06 | 12 | GPU 0 | lazy | chrome | 15 | 153/180 | 85.0 | 17.7 | 74.0 | 37.4 | 3.12 | 991 | 22.9 | 66 |
| run07 | 2 | GPU 0 | lazy | null | 45 | 80/90 | 88.9 | 18.7 | 40.0 | 9.8 | 4.88 | 641 | 8.5 | 53 |
| run07 | 5 | GPU 0 | lazy | null | 27 | 122/135 | 90.4 | 15.9 | 44.9 | 16.2 | 3.24 | 969 | 13.9 | 63 |
| run07 | 8 | GPU 0 | lazy | null | 19 | 140/152 | 92.1 | 19.1 | 57.7 | 22.0 | 2.75 | 1174 | 16.9 | 70 |
| run07 | 10 | GPU 0 | lazy | null | 17 | 151/170 | 88.8 | 16.5 | 53.9 | 26.5 | 2.65 | 1204 | 19.5 | 72 |
| run07 | 12 | GPU 0 | lazy | null | 14 | 148/168 | 88.1 | 16.8 | 62.4 | 30.4 | 2.53 | **1228** | 22.8 | 75 |

### Pooled dataset statistics (all runs)

| Quantity | Value |
|---|---|
| Attempted trajectories | 2 059 |
| Successful (saved) trajectories | 1 825 (**88.6 %**) |
| Saved rollouts passing integrity check | 1 825 / 1 825 (0 invalid) |
| Steps per rollout | 161 (all rollouts) |
| RGB frames | 293 825 per camera, 587 650 total (1280×720 JPEG) |
| Disk | ≈ 156 GB (≈ 86 MB per rollout) under `~/Stanley_ws/r2r2r_repro_runs/*/output_data` |
| Arm used | right arm in every inspected rollout (left arm static) |
| Max single-step joint change per rollout | median 0.13 rad, p90 0.18, p99 0.45, max 1.05; 2.2 % of rollouts > 0.3 rad |

### Findings

1. **The original pipeline reproduces end-to-end** with the released assets and unmodified code. ≈ 86–96 % of attempted trajectories pass the feasibility filter, consistently across env counts (the filter rate does not depend on parallelism).
2. **GPU placement (KI-6).** As shipped on this dual-GPU desktop, Isaac Sim renders on the display GPU (GPU 1) and JAX pre-allocates on it, while physics/torch run on GPU 0 — i.e. both GPUs are used without being asked. `CUDA_VISIBLE_DEVICES=0` pins everything to GPU 0 (verified: Kit active device PCI 01:00, `nvidia-smi pmon` shows only a 6 MiB graphics context on GPU 1). Throughput was not hurt by single-GPU placement (495 vs 456 succ/h at 2 envs).
3. **Memory.** Linear fit over the null-client runs: ≈ 1.37 GB per env (two 1280×720 tiled cameras) + ≈ 6.2 GB fixed; JAX's 25 % pre-allocation adds ≈ 6 GB on top. With pre-allocation, **10 envs OOM**; without it, 12 envs fit (22.8 GB). **50 envs (README) is far beyond a single 24 GB card in this configuration.**
4. **Throughput saturates at ≈ 1.2 k successful trajectories/h (≈ 20/min) from 8 envs on**, with GPU 0 at only 70–75 % utilisation. The main Python process runs at 125–200 % CPU while the 8 JPEG writer processes sit at 15–60 % → the bound is the single-threaded per-step loop (render read-back, deepcopy of camera buffers, viser image push, IK dispatch), not GPU compute or JPEG encoding (hypothesis from `top` sampling; not profiled).
5. **The viser client matters.** A GPU-less headless Chrome client consumes ≈ 10 CPU cores (software WebGL) and costs 8–19 % throughput (e.g. 641 vs 551 succ/h at 2 envs, 1228 vs 991 at 12). A browser on a different machine (as the authors likely used) or the null client avoids this.
6. **Initialisation** takes 16–21 s to the viser server (warm caches; 155 s on the very first Isaac Sim launch) and the first batch completes 40–75 s after launch.
7. **Measurement caveats:** shared workstation — an unrelated `rerun` viewer (≈1.3 cores), a `gjs` desktop process (≈1 core, 20 GB RSS) and kswapd activity were present during all runs; one run per configuration.

### Figures (generated, not committed; `~/Stanley_ws/r2r2r_repro_runs/figures/`)

| File | Content |
|---|---|
| `samples_rgb.png` | 3 episodes × 2 cameras × 4 time steps (run04) |
| `trajectories.png` | EE x–y paths and EE height for 60 episodes; gripper command profile |
| `joint_angles.png` | per-joint median and 5–95 % band over 60 episodes |
| `gpu_memory_timeline.png` | per-GPU memory over time: run02 (default placement), run04 (GPU 0), run05_env10 (OOM), run07_env12 |
| `scaling_runtime_gpu.png` | succ traj/h, batch time, GPU 0 peak memory, success rate vs num_envs for 3 configurations |

Regenerate: `python docs/reproduction/tools/make_figures.py --out … --dataset … --timeline-runs … --scaling "label=run,…" …` (exact invocation in the session log; see §1).

## 4. Comparison with paper / README

References: arXiv 2505.09601 (HTML version, Sec. 4.2/5 and appendix), repository README, `outputs/coffee_maker/track/` (tracked human demonstration).

| Claim / artefact | Source | Our observation | Status |
|---|---|---|---|
| Data generation from a single human video + object scans, without dynamics simulation | paper | Pipeline runs from released 4D-DPM track (`keyframes.txt`) + grasps + SuGaR meshes; objects are kinematic (`kinematic_enabled=True`, collisions disabled) | **A — reproduced** |
| Coffee maker task "put the mug on the coffee maker" with YuMi | paper | Robot grasps the mug and places it at the coffee maker; matches the tracked human demo's object motion (qualitative, `samples_rgb.png` vs `track/camopt_render_final_opt.png`) | **A — reproduced** (qualitative) |
| Two third-person cameras ("Views From Both Cameras") | paper, Fig. 24 | 2 calibrated cameras from `camera_poses_calib_031425.h5`, 1280×720 RGB | **A — reproduced** |
| Domain randomisation: lighting (sphere lights, colour/intensity), background skybox, object initial pose, robot initial joint perturbation | paper Sec./appendix | All visible in renders (`samples_rgb.png`): different backgrounds/lighting per episode, varied mug start pose, jittered left-arm home pose | **A — reproduced** (qualitative; distributions not measured) |
| Camera pose randomisation (≤ 2 cm, ≤ 5°) | paper | Not verified (camera extrinsics are not logged per episode) | **C — not checked** |
| Table texture randomisation | README (needs vMaterials) | Off by default; not installed | **C — not reproduced (optional ablation)** |
| Output layout `output_data/<task>/successes/` | README | Exactly as described | **A — reproduced** |
| "num_envs=50 should work on a 4090" | run.py comment | Single 4090: OOM at 10 envs with defaults, max 12 envs with lazy JAX allocation | **C — not reproduced** (KI-16) |
| 51 demonstrations / min on one RTX 4090 (27× teleop at 1.7/min) | paper Sec. 4.2 | Best single-GPU: 20.5 successful / 23.0 attempted per min (12 envs, null client) → **40–45 % of the paper's rate**, still ≈ 12× the paper's teleop baseline. Paper does not state num_envs, resolution or whether 51/min counts attempted or filtered demos. | **B — partially reproduced** |
| Throughput scales linearly with rendering GPUs | paper | Not tested (single-GPU only, as instructed) | **C — not tested** |
| Policy results (e.g. coffee maker: DP 53.3 %, π0-FAST 80.0 % with 1 000 R2R2R demos; 15 trials) | paper Sec. 5 | No policy trained, no real-robot evaluation | **C — not reproduced; no claim made** |
| Released converted datasets (HF) | README TODO | Not released at time of writing | **C — unavailable** |

### Summary

- **A. Successfully reproduced:** installation of the data-gen env with the pinned stack (Isaac Sim 4.5.0, IsaacLab 2.0.2 fork, torch 2.5.1+cu121, Py 3.10.15); end-to-end YuMi coffee-maker data generation with the original algorithm and assets; output format; rendering from both calibrated cameras; domain randomisation (qualitative); 1 825 valid rollouts.
- **B. Partially reproduced:** generation throughput (≈ 40–45 % of the paper's 51 demos/min on a single 4090; configuration of the paper's measurement unknown).
- **C. Not reproduced / not tested:** 50 parallel envs on one 4090; multi-GPU scaling; table-texture randomisation; camera-pose jitter verification; any policy training or real-robot result.
