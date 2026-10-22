# Independent reproduction audit

Audit date: 2026-10-22 (Asia/Taipei). Reviewer: Codex.

**Verdict: the available evidence supports a limited YuMi coffee-maker data-generation reproduction using released processed assets. It does not establish full-paper reproduction, complete dataset validity, or a controlled reproduction of the paper's runtime results.** No CRITICAL defect was established in the inspected saved outputs. The MAJOR findings below affect validation and comparability; they should be resolved before strengthening the claims.

Severity: **CRITICAL** invalidates reproduction; **MAJOR** affects reliability or comparability; **MINOR** concerns documentation or implementation details. An inherited baseline issue is identified explicitly and is not attributed to the reproduction branch.

## Scope and source comparison

Compared local `dev` / `origin/dev` at `0ac79eb` with `repro/baseline-datagen` at `ccaa68e`. The reproduction branch contains one additional commit. `git diff --stat dev...HEAD` reports 16 changed files, 1,119 insertions and one deletion: reproduction documentation, ten helper scripts, and `.gitignore`. There are no committed changes to `scripts/`, `real2render2real/`, `env_data_gen.sh`, `pyproject.toml`, `uv.lock`, `.gitmodules`, or submodule pointers. A separate `git diff --exit-code dev HEAD -- ...` over those implementation/configuration paths exited 0.

Recursive submodule status matches the recorded pins, including IsaacLab `7c23afbd`, trajgen `6b19f56`, rsrd `cca2256`, jaxmp `d1ab9c7`, and jaxls `2d1035c`. The pre-existing dirty trajgen worktree contains only `trajgen.egg-info/PKG-INFO` and `SOURCES.txt`: metadata format 2.2 → 2.4 and addition of the already existing `traj_resampling.py` to the packaging inventory. No algorithm source edit was found there. This dirt was left untouched.

The runtime wrapper changes `num_envs` by replacing one exact source literal before `exec`; it does change the executed configuration even though `scripts/run.py` remains byte-identical. Its assertion limits substitution to one match. `CUDA_VISIBLE_DEVICES=0`, disabled JAX preallocation, and the null Viser client are material execution differences, documented in the reproduction. Static inspection supports the client's camera-independence **when the two calibrated frustums load**: the calibrated-camera branch uses the frustums, not `client.camera`. A matched-seed browser/null output-equivalence test has not been run.

The implemented path retains trajectory interpolation/resampling, sampled grasps, JAX IK, kinematic object motion, collision-disabled objects, and tiled rendering. These are consistent at a high level with [paper §§4.1–4.2](https://arxiv.org/html/2505.09601v1#S4). Source identity against `dev` is verified; numerical equivalence to the authors' experiments is unverified. In particular, the paper describes PyRoki IK, while this unchanged baseline imports `jaxmp`; no equivalence evidence bridges that naming/implementation difference.

## Executed checks and artifact evidence

No dependencies were installed, no simulation or training was launched, and no assets were downloaded. CPU-only checks used the existing `.venv/bin/python -B`. Only this audit document was added to the repository; temporary audit scripts/fixtures and results were written under `/tmp`.

Existing evidence root: `/home/stanley/Stanley_ws/r2r2r_repro_runs/`. Each listed successful run has `run.log`, `gpu.csv`, `gpu_procs.log`, `metrics.json`, `dataset_validation.json`, and saved outputs. The logs were treated as evidence of prior execution, not tests executed by this reviewer.

| Executed by reviewer | Observed result and limit |
|---|---|
| Branch diff, recursive submodule status, dirty submodule diff | No substantive baseline source changes found; details above. |
| Installed package metadata inspection | Python 3.10.15; torch/vision/audio 2.5.1+cu121 / 0.20.1+cu121 / 2.5.1+cu121; NumPy 1.26.4; JAX/jaxlib/CUDA plugin 0.5.3; Isaac Sim 4.5.0.0; isaaclab package 0.36.6. |
| `.venv/bin/python -B -m pip check` | Exit 0, `No broken requirements found.` This checks declared package requirements, not runtime or ABI compatibility. |
| Independent parsing of all 14 tabulated run logs, including multiline tensors; recomputation with `analyze_run.analyze` | 491 completed batches, 2,059 completed attempts, 1,825 accepted, 234 rejected. Recomputed metrics equal all 14 stored `metrics.json` records; equality does not validate the formulas. |
| Logged move/delete events versus actual episode directories | All 1,825 accepted episodes match directory names and move events; 234 deletion messages match rejected completed attempts. No queue-full/save/move/delete error strings found in those 14 logs. |
| Independent scan of every saved HDF5 file and camera filename inventory | All 1,825 have shapes `(161,16)`, `(161,14)`, `(161,2)`, finite arrays, Boolean gripper data with two transitions, left command always false, 16 joint names, and EE quaternion norms within `1e-4` of one. Every episode has exactly `camera_0` and `camera_1`, each with names `0000.jpg`–`0160.jpg`. |
| Arm joint arrays versus named joint limits in the released YuMi URDF | No arm joint limit violations above `1e-5` found across all saved timesteps. Gripper limits, velocity limits, FK agreement, contact, and image/state alignment were not validated by this check. |
| Deterministic JPEG sampling: first/middle/last episode of each successful run, frames 0/80/160 in both cameras | 234 images decoded, all `(720,1280,3)`. The other 587,416 JPEG files were inventoried but **not decoded by this reviewer**. |
| Inspection of existing `figures/samples_rgb.png` | Visible robot, mug, coffee maker, two views, differing backgrounds and mug positions. The collage supports a qualitative rendering claim, not geometric accuracy or physical success. |
| Temporary negative fixtures for `validate_dataset.py` | Malformed episode returned `errors=[]`; nonexistent dataset CLI returned exit 0. See M1. |
| CPU fixtures executing the unchanged AST-extracted `_target_state_error` and `_joint_pos_limit_check` methods | NaN target position and NaN joint position each left `success_envs=[True]`. See M5. These were isolated method tests, not simulator tests. |
| Untimestamped three-batch analyzer fixture, last batch all false | Attempted throughput reported 1,440/h for a constructed 20-second steady interval containing four attempts, whose correct rate is 720/h. See M2. |
| `bash -n` on both reproduction shell tools; AST parsing of all eight reproduction Python tools; `git diff --check` | All exited successfully / parsed without errors. These are syntax/whitespace checks only. |

The independent scan was executed as `.venv/bin/python -B /tmp/codex_reproduction_audit.py`; its detailed local results are `/tmp/codex_reproduction_evidence.json`. Temporary artifacts are not a portable evidence bundle. The checks above state their actual coverage so they can be repeated without treating the existing validator as sufficient.

### Counts reconciled against actual files

`runNN/envN` below abbreviates the corresponding run directory under the evidence root. Full names appear in [metrics_table.md](metrics_table.md). The OOM run has no saved episodes or completed batches.

| Run | Completed accepted/attempted | Actual saved episodes |
|---|---:|---:|
| run02/env2 | 222/258 | 222 |
| run04/env2 | 160/184 | 160 |
| run05/env5 | 155/170 | 155 |
| run05/env10, OOM | 0/0 | 0 |
| run06/env2, lazy | 67/70 | 67 |
| run06/env5, lazy | 97/110 | 97 |
| run06/env8, lazy | 166/192 | 166 |
| run06/env10, lazy | 164/180 | 164 |
| run06/env12, lazy | 153/180 | 153 |
| run07/env2, lazy/null | 80/90 | 80 |
| run07/env5, lazy/null | 122/135 | 122 |
| run07/env8, lazy/null | 140/152 | 140 |
| run07/env10, lazy/null | 151/170 | 151 |
| run07/env12, lazy/null | 148/168 | 148 |
| **Total** | **1,825/2,059 = 88.64%** | **1,825** |

This accounts for 293,825 frames per camera, 587,650 total. An additional 60 `env_*` directories exist outside `successes/` across the runs; these are unfinalized output directories, not accepted episodes or evidence of additional successes. The denominator is **completed and classified attempts**, excluding interrupted batches and startup failures. The pooled acceptance rate mixes configurations and is not a matched experimental comparison.

## Findings

### M1 — MAJOR: the integrity validator can certify malformed data

Evidence: [validate_dataset.py](tools/validate_dataset.py), especially `check_episode` and `main`; [02_generation_results.md](02_generation_results.md), validation checklist item 7.

The tool records shapes and transition counts but does not enforce expected dimensions, common time lengths, Boolean gripper dtype, two toggles, joint-name presence/order, two required cameras, expected resolution, consecutive frame IDs, or quaternion normalization. It samples only first/middle/last images; its displayed image shape is not a dataset-wide resolution check. Zero episodes and invalid episodes do not cause a nonzero CLI exit.

Executed counterexample: an episode with `joint_angles=(5,16)`, `ee_poses=(1,3)`, `gripper_binary_cmd=(2,1)` with zero toggles, no joint names, only one camera, 8×8 black JPEGs, and a corrupt unsampled `0001.jpg` returned `errors=[]`. A nonexistent task directory returned exit 0 and `n_valid=0, n_invalid=0`.

**Impact:** “1,825 valid rollouts” is stronger than this validator establishes. The independent scan above closes several structural gaps for the current artifacts, but neither scan proves all-frame decodability, temporal alignment, correct action semantics, or physical feasibility.

Minimal remedy: enforce the task schema and nonempty dataset, make validation failures observable to automation, and describe sampled decoding accurately. Add negative fixtures before using the validator as an acceptance gate; do not alter the baseline generation algorithm to address an audit-tool defect.

### M2 — MAJOR: GPU utilization and legacy timing are not consistently bounded

Evidence: [analyze_run.py, `gpu_stats` and `analyze`](tools/analyze_run.py); raw `run04_env2_gpu0/{gpu.csv,run.log,t_start,t_end}`; `03_known_issues.md` KI-13.

`gpu_stats` accepts `t0` but does not use it and has no end bound. Its steady utilization/power include every row after the first batch, including shutdown, orphan workers, and unrelated later activity; peak memory uses the entire CSV. In `run04`, the GPU CSV extends **1,453.1 seconds after the final completed batch**. GPU0 utilization recomputed only between first and last batch completions is **42.04%**, versus the report's **18.69%**. Consequently, the 19% value in the tables is not an active-generation average. The same formula contaminates power measurements and can contaminate peaks; not every reported peak was shown to be wrong.

For untimestamped `run02`, the fallback deduplicates timestamps from success-directory names. It sees only **127 unique successful batch timestamps for 129 classified batches**: two batches were all false. This loses batch timing events and uses second-resolution local wall time. First and last batches do contain successes, so the observed 456/h interval-rate calculation retains matching endpoints, but the batch median/percentiles are approximations. The synthetic all-failed-final-batch fixture demonstrates that the general fallback can overstate throughput by 2×. `run02`'s 2,400-second wall/exit record was reconstructed after the launcher failed; it is not an independent captured completion time.

Minimal remedy: restrict GPU samples to explicit phase boundaries, report device-wide versus process memory separately, use monotonic event timestamps for every batch including all-false batches, and label reconstructed timing. Recompute tables and figures from the bounded data.

### M3 — MAJOR: the paper comparison uses the wrong task baseline

Evidence: [02_generation_results.md, comparison table](02_generation_results.md); [paper §4.2 and Appendix §9.6, Table 7](https://arxiv.org/html/2505.09601v1#A9.SS6).

The paper's 51 demos/minute is an average; Table 7 gives **38.22 minutes for 1,000 coffee-maker demonstrations**, or **26.16/minute**, excluding upfront preprocessing. The reproduction's best steady rate is **1,228.39/h = 20.47/minute**, approximately **78.25% of the task-specific rate**. The reported 40–45% comparison to 51/minute is arithmetically understandable but does not characterize coffee-maker reproduction performance fairly.

Using saved episodes divided by recorded process wall time, run07/env12 yields **1,107.69/h = 18.46/minute**, and run07/env10 yields **1,130.15/h = 18.84/minute**. Neither run actually generated 1,000 demonstrations in one uninterrupted configuration. Different startup, persistence, resolution, environment count, cache, and client conditions still prevent a matched benchmark conclusion. A local ratio to the paper's teleoperation rate is an extrapolation, not a reproduced teleoperation experiment.

Minimal remedy: use the coffee-maker number as the primary comparator; publish steady and process-wall throughput separately and keep the 51/minute average as context. Do not label either result matched-paper throughput until configurations and timing boundaries are reconciled.

### M4 — MAJOR: launcher failures and run isolation are not enforced

Evidence: [launch_run.sh](tools/launch_run.sh), lines 7–16 and 31–40; [bench_scaling.sh](tools/bench_scaling.sh), lines 5–14. This finding is from static inspection; the launcher was not executed during the audit.

The launcher writes the child status to a file but always exits 0. The bench script suppresses analyzer/validator errors and does not check their status. Runs reuse one fixed repository `output_data` path, accept an existing run directory, and move whatever output tree exists; there is no enforced empty-start check or run identifier in the data. Cleanup searches all matching Python command lines with `pgrep` and sends SIGKILL, rather than limiting cleanup to this run's process tree. Concurrent runs or interrupted/reused directories can therefore kill another experiment, mix datasets, overwrite evidence, or continue after validation failure. These failure modes were **not observed** in the reconciled saved runs.

All 13 productive tabulated runs were stopped by timeout rather than normal completion: twelve have recorded exit 124; `run02` has a reconstructed exit record. The OOM run records exit 137, and its log contains `torch.OutOfMemoryError` at line 204. These are bounded sample runs, not proof of clean shutdown or the normal 1,000-success stopping path.

Minimal remedy: make timeout/expected bounded-stop distinct from crash/OOM, fail on analysis/validation errors, reject existing run/output directories, record a run manifest, and clean up only owned processes. Test orchestration with stub commands in a temporary directory before another expensive run.

### M5 — MAJOR: success filtering and asynchronous logging need stronger validation

Evidence: unchanged [yumi_coffee_maker.py](../../real2render2real/isaaclab_viser/yumi_simulators/yumi_coffee_maker.py), lines 184–207, 432–437, 586–738 and 956–1001; unchanged [batch_data_logger.py](../../real2render2real/isaaclab_viser/data/batch_data_logger.py), lines 94–133, 143–188 and 193–218.

The documented filter thresholds match the source: minimum EE height 0.025 m; selected-arm target error limits 0.02 m and 0.45 rad; joint position checks during the grasp phase. Tracking checks run immediately before/during grasp, not over the entire rollout. There is no test of mug placement/contact or collision-free motion, by design. The measured 88.64% is a **filter acceptance rate**, not a real-robot policy/task success rate. The report largely acknowledges this distinction and must retain it.

Two inherited risks remain:

- Threshold comparisons fail open for NaNs. The executed isolated-method fixtures left success true for a NaN target position and a NaN joint position. Saved arrays are finite in all inspected episodes, so no observed episode is invalidated by this counterexample; unlogged target errors could not be checked.
- Eight workers consume a shared queue and append HDF5 rows without per-episode ownership, timestep IDs, or an ordering barrier. Reset prints success and redirects/deletes directories without waiting for persistence acknowledgements. Multiple workers may reach the same episode file out of order or while it is being renamed. Equal file counts and finite values cannot prove temporal order. Rendering/buffer capture also occurs before the state update and logging in the main loop; the RGB/state/action offset needs an explicit replay/timestep test. These are source-supported risks, **not a claim that corruption or misalignment was observed**.

Observed maximum per-step joint changes are median **0.132 rad**, p99 **0.451 rad**, maximum **1.047 rad**; **41/1,825 (2.25%)** exceed 0.3 rad. Position-limit compliance does not establish velocity/smoothness compliance. Neither the timestep-specific velocity envelope nor downstream conversion/replay was tested.

Minimal remedy: add CPU filter boundary/nonfinite tests, an asynchronous writer ordering/drain test with sequence-tagged tasks, and one short image/state/FK/action alignment replay. Preserve the baseline during review; any future filter/logger fix must be identified as a deviation and compared separately.

### M6 — MAJOR: environment and artifact provenance cannot be replayed exactly

Evidence: [01_installation.md](01_installation.md), [03_known_issues.md](03_known_issues.md), local package metadata and backup directory, ignored `data/`, `outputs/`, `.venv/`, and external run artifacts.

The installed key versions match the report. Editable trajgen, jaxmp, jaxls and IsaacLab metadata point to this checkout. The Isaac Sim prebundled websockets directories are absent, and both backup directories are present under the evidence root, consistent with the documented author workaround in `asyncio_error.md`. That is an actual installed-environment modification, despite unchanged repository dependency files. The original failure is supported by `run01` log line 74; successful dataset generation after the workaround is supported by later outputs. No fresh AppLauncher/import/render check was executed here.

Unpinned dependencies resolved to viser 1.1.1, websockets 16.1.1, warp-lang 1.18.0, and trimesh 5.1.1; mujoco/mujoco-mjx are 3.3.1/3.2.3. `pip check` succeeds, but does not validate native/Isaac extension behavior. The branch correctly documents avoiding its unchanged `uv.lock`; therefore that lockfile is not an exact record of this run environment. The stated reason needs a correction: on this Linux x86_64/Python 3.10 target, the lock's markers select **torch 2.0.1**, not 2.8.0; either differs from the installed/script-pinned 2.5.1. The referenced `freeze_final.txt` was not found under the supplied run root. Current metadata does not prove the environment was unchanged between all historical runs.

The local `data.zip` is 1,646,963,428 bytes and SHA-256 matches the recorded value:

```text
fc22f70087f83f805a9a994cee7d06a9e43661a4b051b1bdd9917d0e97989704
```

Current calibration HDF5 and YuMi URDF are byte-identical to their local archive members. This supports consistency with the retained download, not independent authentication against an author-published hash. The coffee-maker output subset contains 280 files totaling 291,248,387 bytes; no retained full `outputs.zip` was used for an independent CRC/provenance check. No remote extraction was repeated. Current key input hashes are recorded below to make future drift detectable:

| Input relative to `outputs/coffee_maker/` | SHA-256 |
|---|---|
| `track/keyframes.txt` | `f3c4335abc27ee9f20c7c420f50c1a9431c263d8826e8172efba15d3a087d019` |
| `dig/2025-03-18_154136/dataparser_transforms.json` | `1cc66231bbdb94e6c7b2b4789148a716b33a9a0df501917e0b9fa6e8f4f6226e` |
| `state_rigid_20250217_151412/grasps.txt` | `8f320bd6a4a275842354cb36c935a07f00174666adab3e500430b125fa8706d2` |

Raw logs, datasets, figures, full package state, and historical launcher versions are outside the commit. Logs do not capture RNG seeds, all rendering settings, hardware/driver manifests, or checksums of the effective executed sources and assets per run. The present source diff cannot prove historical runtime source identity. Minimal remedy: retain a small immutable manifest and raw logs/summary JSON with checksums and artifact locations; capture the resolved environment plus the websockets workaround. No reinstall is necessary to record current state.

### M7 — MAJOR: runtime scaling and causal explanations exceed experimental support

Evidence: [02_generation_results.md](02_generation_results.md), run environment/client files, per-process telemetry, and the timing analysis above.

The late single-GPU runs have corroborating Kit GPU tables and process samples: the main process uses compute/render memory on GPU0, with only a 6 MiB graphics context on GPU1. Default `run02` renders on GPU1. Thus those placement observations are credible for this host; `CUDA_VISIBLE_DEVICES` alone should not be treated as portable proof of RTX device selection elsewhere.

Only one run per configuration was made, with 14–45 completed batches in each null-client run, no recorded seed, different clients/allocators across series, warm caches, and a shared workstation. “Differences of a few percent are noise,” “filter rate does not depend on parallelism,” the 8–19% causal browser penalty, the exact saturation point, and the CPU-bottleneck explanation lack repeated controlled trials/profiling. The report labels the CPU explanation a hypothesis; it should remain one. Peak-memory fits are observations over 2–12 environments, not measurements at 50 environments. The default-allocation OOM at 10 is real; 12 is the largest successful lazy-allocation configuration **tested**, not a proven maximum.

Minimal remedy: repeat only the key comparison on an otherwise idle host with recorded seeds/configurations and bounded telemetry. Include matched client/allocator runs and uncertainty across independent runs. A profiler is needed before assigning the bottleneck; neither GPU utilization alone nor a linear memory extrapolation establishes it.

### N1 — MINOR: scope and documentation need narrower wording

`02_generation_results.md` says the original pipeline reproduces “end-to-end” and marks generation from a human video and scans as reproduced. The actual starting point is released meshes, tracks, grasps, and calibration; the reproduction explicitly skipped reconstruction/tracking and mesh-conversion environments. Use “released-assets-to-rendered-dataset” consistently. This is a qualification of a documented partial reproduction, not evidence that the reported data were fabricated.

`01_installation.md` says no submodule source file was modified; KI-18 later explains generated tracked metadata. Prefer “no algorithm source modified” and retain the dirty-metadata disclosure. `03_known_issues.md` calls p99 joint jumps approximately 0.67 rad, whereas the pooled data give 0.451 rad; identify the sample/run used for the former. The figures tool docstring advertises a nonexistent `--runs` argument; the actual CLI uses `--timeline-runs` and `--scaling`.

## Claims that remain unverified

| Claim or validation target | Audit assessment |
|---|---|
| Full scan/video → reconstruction → segmentation → tracking → meshes → generation | Upstream processing was not rerun; released intermediates were consumed. Full original-pipeline reproduction remains unverified. |
| Exact algorithm equivalence to the paper | Branch parity with `dev` is established; no author-reference trajectory/IK outputs or matched configuration establish paper-level numerical equivalence. |
| All 1,825 episodes are usable, correctly aligned training demonstrations | Structural consistency substantially corroborated; all-frame decoding, sequence correctness, FK, image/state timing, action conversion, velocity limits and training ingestion remain untested. |
| Camera randomization matches the paper | [Paper §4.2](https://arxiv.org/html/2505.09601v1#S4.SS2) specifies up to 2 cm/5° extrinsic jitter. Baseline `base.py:283–297` uses per-axis ±0.015 m translation and additive ±0.008 quaternion-component noise; calibrated frustum poses are shared across the batch. No recorded per-episode extrinsics or distribution test establish agreement with the paper's parameterization. |
| Exact asset/track/grasp accuracy and generated endpoint accuracy | Current input existence/hashes are verified; no independent trajectory/registration/mesh accuracy or final placement tolerance was measured. |
| Browser/null client preserves generated data | Supported by calibrated-camera control flow, not by a matched-seed numerical/render comparison. |
| 50 parallel environments on one 4090 | Unverified; OOM at 10 with default allocation and successful lazy-allocation runs through 12 are the available evidence. The 25-environment run was aborted, not a successful capacity test. |
| Clean 1,000-success run, shutdown, restart or recovery | Not tested; productive runs were time limited. Pooling 1,825 episodes across configurations does not test this lifecycle. |
| Other tasks, bimanual manipulation, Franka, or multi-GPU scaling | Not exercised by these saved coffee-maker runs; the right arm is hardcoded for this task. |
| Table texture randomization | Disabled in baseline reset path; acknowledged as not reproduced. |
| Policy training, policy success rates and real-robot transfer | Explicitly not performed; no downstream policy result is established. |
| Installation GPU matmul smoke checks, first-launch 155-second timing, old freeze/session claims | Reported historical checks; not independently rerun here, and the cited complete session evidence is not in this commit. |

## Minimal regression tests to add

These are recommendations, **not a claim that the proposed suite exists or passed**. Keep them separate from baseline implementation edits.

| Priority | Minimal test | Required assertion |
|---|---|---|
| P0 | Tiny valid dataset plus malformed variants | Enforce both cameras, synchronized shapes and frame IDs, expected dimensions, Boolean grippers/toggles, finite values, joint names and quaternion norms; missing/empty/invalid datasets fail the CLI. Include the exact M1 fixture. State explicitly whether decoding is exhaustive or sampled. |
| P0 | Synthetic metric logs/telemetry | Cover wrapped tensors, all-false first/middle/last batches, no batches/OOM, truncated lines, and telemetry outside phase boundaries. Check counts and rates against hand-calculated values; appending post-run GPU rows must not change active metrics. |
| P0 | CPU success-filter boundary fixtures | Exercise 0.025 m, 0.02 m, 0.45 rad, joint bounds, selected/inactive arms, persistence of earlier failure, and NaN/Inf inputs. Snapshot baseline behavior separately from any proposed correction. |
| P0 | Small asynchronous logging sequence test | With tagged frame/state IDs and deliberately delayed writes, require ordered, unique rows, exact frame-to-state pairing, no save errors, and completed writes before redirect; terminate mid-batch and verify that partial data cannot enter accepted outputs. This tests the actual multiworker logger rather than a serial mock. |
| P1 | Source/configuration parity guard | Compare algorithm paths and recursive submodule pins to `dev`; inspect effective wrapper source to confirm only the requested positive `num_envs` literal changes. Record launcher/tool/source/input hashes and the exact installed versions in a per-run manifest. |
| P1 | Small trajectory interpolation/resampling fixture | Check identity transforms, requested start/end poses, quaternion convention/sign equivalence and normalization, finite outputs, deterministic seeded sampling, and expected resampled length against the unchanged baseline. A separate author-reference fixture is needed to establish paper equivalence. |
| P1 | One short fixed-seed GPU render/replay test when authorized to run generation | Validate paired camera/state/action timestep semantics, joint-name mapping, FK/EE agreement, finite target errors, and correct downstream converter conventions. Check that changing the Viser client camera does not move the two saved calibrated views. No GPU test was executed in this audit. |
| P1 | Stubbed launcher lifecycle test | Child crash/OOM, absent client, failed validator, reused output directory, interrupted logger, and a separate unrelated process. Require accurate status, no mixed output, and cleanup restricted to owned children. |
| P1 | Focused repeated benchmark | At a fixed configuration, report both complete classified/saved episodes per process wall time and steady throughput; include first-batch costs, bounded GPU samples, seeds, device PCI IDs, and independent run variation. Exercise the normal 1,000-success exit once before claiming it. |

The regression priorities address the demonstrated validator/timing failures first, then the untested logging, lifecycle and scientific-comparison boundaries. No baseline fixes, dependency reinstalls, or new generation runs were performed as part of this review.
