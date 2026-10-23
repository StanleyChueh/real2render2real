# 01 — Dataset preparation for TinyDP (YuMi coffee maker)

Date: 2026-10-23 · Branch `repro/dp-training` (from `repro/baseline-datagen` @ `334e83d`) · Source data: the 1,825 accepted episodes under `~/Stanley_ws/r2r2r_repro_runs/run0*/output_data/yumi_coffee_maker/successes/` (never modified; see §6).

Scope: make the released-asset R2R2R rollouts consumable by the authors' TinyDP Diffusion Policy (`dependencies/tinydp` @ `76d8f9c`) with deterministic splits and train-only normalisation. No generator code, generated episode, or TinyDP source file was changed.

## 1. Environment

A separate venv, `real2render2real/.venv-tinydp` (uv, CPython 3.10.15, gitignored), is used because TinyDP's pins conflict with the data-generation venv (`transformers 4.48.2`, `tyro 0.8.14`, `zarr 2.12.0`). The generation venv `.venv` and conda envs were not touched. The torch wheel came from the local uv cache, and nothing else large was downloaded. ResNet-18 is randomly initialised in TinyDP (`weights=None`), so no weights were fetched.

```bash
uv venv --python 3.10.15 .venv-tinydp
uv pip install --python .venv-tinydp/bin/python torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv-tinydp/bin/python -e dependencies/tinydp
uv pip install --python .venv-tinydp/bin/python line_profiler pytest
uv pip install --python .venv-tinydp/bin/python huggingface_hub==0.25.2 numpy==1.26.4 h5py==3.16.0   # adjustments, below
```

| Adjustment vs `tinydp/pyproject.toml` | Reason (observed failure) |
|---|---|
| `torch 2.5.1+cu121`, `torchvision 0.20.1+cu121` | Same version as the pin. The cu121 build matches the generation stack and was already cached. |
| `huggingface_hub==0.25.2` (unpinned upstream; resolved to 0.36.2) | `diffusers 0.18.2` → `ImportError: cannot import name 'cached_download'` (removed in hub ≥ 0.26). 0.25.2 satisfies `transformers 4.48.2` (≥ 0.24). |
| `numpy==1.26.4` (unpinned upstream; resolved to 2.2.6) | `zarr 2.12.0` → `AttributeError: module 'numpy' has no attribute 'product'`; `scikit-image 0.19.3` ABI error. Same numpy as the generation venv. |
| `h5py==3.16.0` (pinned 3.12.1) | The generator wrote with h5py 3.16 / **HDF5 2.0.0** (`libver='latest'`). HDF5 1.14.4 cannot open the Boolean `gripper_binary_cmd` (`bad version number for datatype message`). TinyDP's sim loader never opens `.h5`; only the adapter needs this. |
| `line_profiler` (missing upstream) | Imported unconditionally by `image_dataset_sim.py`. |
| `pytest` | Tests only. |

Full resolved environment: `artifacts/freeze_tinydp_env.requirements`.

## 2. Actual episode schema (verified on all 1,825 episodes)

| Item | Value |
|---|---|
| `robot_data/robot_data.h5` keys | exactly `ee_poses` (161,14) float32, `joint_angles` (161,16) float32, `gripper_binary_cmd` (161,2) bool. No attributes. Chunked per row (1,D). |
| `ee_poses` row layout | `[left x y z, left qw qx qy qz, right x y z, right qw qx qy qz]` of `{left,right}_dummy_point` in the robot **`base_link`** frame |
| quaternion convention | **wxyz** (scalar first). Verified by URDF forward kinematics with `yourdfpy` on 5 episodes × 17 steps × 2 arms: position error < 1e-5 m, wxyz rotation error median 4e-5° (max 8e-5°) vs 165° if read as xyzw. The right-arm quaternion takes both signs (q and −q) within and across episodes; this is harmless because TinyDP converts to a rotation matrix (6-D rows), which is sign-invariant (tested). |
| `joint_angles` columns | `joint_names.txt` (identical in all episodes): `yumi_joint_{1,2,7,3,4,5,6}_{l,r}` interleaved L/R in that order (14 arm joints), then `gripper_l_joint`, `gripper_r_joint` (finger positions, m) |
| `gripper_binary_cmd` | `[left, right]`, True = **close command** (state machine `gripper_closed`). Left is never set. Right toggles at diff indices 48 and 131 in every episode, i.e. closed for rows 49–131. This is the commanded value; the finger joint (col 15) follows it physically (0.024 → ≥0.005 m). |
| images | `camera_0/rgb`, `camera_1/rgb`, `0000.jpg`…`0160.jpg`, JPEG RGB 1280×720. **All 587,650 files fully decoded** with PIL (the library TinyDP uses), with no failures. |
| row ↔ frame index | Frame file `k` and HDF5 row `k` are both written for logger step `k` (count − setup_phase − 1). Rows are appended by 8 asynchronous workers; see §5.2 for the ordering check. |
| step period | One control step = one `sim.step` (physics dt). The setup phase (35 steps) is not logged. |

Strict validation (`docs/reproduction/tools/validate_dataset.py`, rewritten on this branch to fix CODEX_AUDIT M1) enforces all of the above per episode and exits 1 on any invalid episode or 2 on an empty or missing dataset:

```bash
.venv-tinydp/bin/python docs/reproduction/tools/validate_dataset.py ~/Stanley_ws/r2r2r_repro_runs/run0*/output_data/yumi_coffee_maker --decode all --workers 16
# → n_episodes 1825, n_valid 1825, n_invalid 0, frames_per_camera 293825, exit 0
```

## 3. TinyDP compatibility

| TinyDP component | Expects | Our data | Resolution |
|---|---|---|---|
| README converter `script/convert_dataset_to_mp4.py` | real-robot HDF5 keys `action/cartesian_pose`, `state/cartesian/cartesian_pose`, `state/joint/joint_angle_rad`, `observation/<cam>/image/camera_rgb` | none of these keys | **Incompatible; not used.** It targets TinyDP's real-data loader. |
| Sim loader `tinydp/dataset/image_dataset_sim.py` (`--dataset-cfg.is-sim-data`) | per `<x>.h5`: `<x>_proprio.zarr` (=`ee_poses`), `<x>_gripper_cmd.zarr`, `<x>_joint.zarr`; images at `dirname(dirname(<x>.h5))/camera_{1,0}/rgb/{t:04d}.jpg` | h5 only | The authors' converter for sim data is `scripts/to_zarr_dataset.py` (R2R2R repo). It writes these sidecars **into** the episode directories with a hardcoded path. We reproduce its output out of place (§4). |
| Quaternion handling | `quat_to_rot_6d(q, format="wxyz")`, xyz at `[:3]`, quaternion at `[3:7]` per arm | xyz then wxyz | Compatible. (The `# w, x, y, z, -- x, y, z` comment in the loader is stale; the code uses xyz first.) |
| Gripper | `_gripper_cmd.zarr` bool → float, used as an absolute target (1 = close; the "Binary invert" comment is not implemented) | bool close command | Compatible |
| Split | one `dataset_root`, `train_split=0.95` on a seeded shuffle of **glob order** | — | Not deterministic across file systems, and the split changes with subset size. Replaced by manifest roots (§4, 02). |
| Statistics | computed in `__init__` over **all** files before the split, unless a JSON is given (or `output_dir/action_statistics.json` exists, which silently wins) | — | Leakage risk. Precomputed from train views only; the wrapper refuses a conflicting `output_dir` file (§4.3). |
| Subsampling | `data_subsample_num_traj` parses `int(name.split("_")[1])` of `env_*` folders | episode names collide across runs | Not used. Nested manifest subsets instead; view folders are named `ep_<split>_<rank>` (no `env`), and the wrapper rejects the flags. |

## 4. Non-destructive adapter

Code: `policy_training/r2r2r_dp/{schema,manifest,adapter,verify,tinydp_bridge,analysis}.py`, CLI `policy_training/prepare_dataset.py`. Outputs: `~/Stanley_ws/r2r2r_policy_runs/dataset_v1/` (outside the repo; 208 MB of zarr sidecars and symlinks, **0 JPEGs copied**).

### 4.1 Manifest (`artifacts/manifest_v1.json`, sha256 `2c58b6f9…99b1031`)

- uid = `<run_dir>/<episode_dir>`; content hash = sha256 over the three arrays (names, dtypes, shapes, bytes). There are no duplicate uids or contents.
- Order = sha256(`"r2r2r-dp-v1:" + uid`). The first **182 → test**, the next **182 → val**, the remaining **1,461 → train** in that order.
- Train subsets are prefixes of the train order (**nested**): **50, 100, 150, 1,000, 1,461**, matching the paper's 50/100/150/1,000 plus all train. **Val and test are identical for every subset size.**
- Rebuilding is byte-identical. The input order does not matter (tested).
- Not stratified by generation run. Per-run val+test share is 15–25%. All runs share the same generator, task and algorithm; they differ only in env count, client and allocator.

### 4.2 Views

`prepare_dataset.py views` builds one TinyDP dataset root per split and per train subset:

```
dataset_v1/views/splits/{val,test}/ep_<split>_<rank>/
dataset_v1/views/train_subsets/n{0050,0100,0150,1000,1461}/ep_train_<rank>/
    robot_data/robot_data.h5 -> <source>/robot_data/robot_data.h5   (symlink)
    robot_data/robot_data_{proprio,joint,gripper_cmd}.zarr           (zarr.save_array of ee_poses / joint_angles / gripper_binary_cmd)
    camera_0 -> <source>/camera_0, camera_1 -> <source>/camera_1    (symlinks)
    source.json                                                       (uid, rel_path, split, rank, content hash)
```

Verification (`r2r2r_dp/verify.py`, run for every view): symlink targets equal the source; the content hash still matches the manifest; each sidecar equals the source array (dtype, shape, values). The authors' `scripts/to_zarr_dataset.py` was **executed** on copies of the same HDF5 files (only its hardcoded `root_dir` literal replaced). Our sidecars are value-, dtype- and `.zarray`-metadata-identical to its output. Result: `artifacts/views_verification_v1.json`, with 0 problems in all 7 roots.

Staged rollout, as required:

| Stage | Episodes | Result |
|---|---|---|
| trial 1 | 2 | 0 problems; TinyDP built 968 items (= 2 × 121 windows × 4 obs steps); shapes action (16,20), proprio (20,), observation (2,3,448,448); finite; source fingerprint unchanged |
| trial 2 | 10 | 0 problems; 4,840 items; source fingerprint unchanged |
| full | 1,825 (7 roots) | 0 problems; source fingerprint `dd23bd44…2f2bcd` unchanged before/after |

Source fingerprint = sha256 over (relative path, size, mtime_ns) of every source file, plus the bytes of every HDF5 file.

### 4.3 Train-only normalisation statistics (`artifacts/action_statistics_train_n*.json`)

Computed per train subset by **TinyDP's own `calculate_dataset_statistics`** run on the train view only (`proprio_noise=0` inside the computation; absolute actions do not depend on proprio). Each is cross-checked against an independent numpy re-implementation; all 5 match within 1e-6. Shape (20,): `[left xyz, left rot6d, left grip, right xyz, right rot6d, right grip]`, min/max over all 16-step chunks. Val/test episodes never enter these files (tested with a poisoned val set).

Degenerate dimensions: the left gripper has min = max = 0. TinyDP's `(a−min)/(max−min)` gives NaN, which its `nan_to_num` sets to 0. Left-arm EE dimensions span only the home-pose jitter (per-episode left EE range median 1.8 mm, max 6.0 mm), which min–max scaling would stretch to [−1, 1]. Right-arm example (n1461): x ∈ [0.140, 0.632], y ∈ [−0.464, 0.126], z ∈ [0.025, 0.187] m, gripper ∈ {0, 1}.

## 5. Temporal semantics and alignment

### 5.1 What TinyDP pairs (verified with frame-encoded fixtures, `tests/test_tinydp_pipeline.py`)

With TinyDP defaults (`subsample_steps=2`, `seq_length=4`, `num_pred_steps=16`, `use_delta_action=False`):

- One item for row `j` gives image frame `j` from both cameras (`camera_1`, `camera_0` order), proprio = EE pose of row `j` (+ uniform noise), and action = absolute EE poses + gripper at rows `j+2, j+4, …, j+32`.
- A training sample is 4 consecutive items `j, j+2, j+4, j+6` of one episode. The loss uses only the action chunk of the **last** observation step (rows `j+8 … j+38`).
- Valid `j`: 0…126 per 161-step episode (121 windows × 4 = 484 items per episode). The last action row used is 158; rows 159–160 are never targets.
- Proprio noise (`proprio_noise=0.003`: ±3 mm on xyz and ±0.003 rad on Euler angles) is applied in **both** train and val datasets, because TinyDP does not condition it on the split.

### 5.2 HDF5 row order (CODEX_AUDIT M5)

An adjacent-swap detector flags any row pair whose swap would shorten the local EE path by > 30% and > 1 mm. It is tested on synthetic swaps. On all 1,825 episodes: **0 violations on the right-EE trajectory**, which moves 4–40 mm per step. EE and joints share one write task per row, so a reordering would show in both.

The same detector on joint angles fires in 8 episodes at isolated steps. There the joint zig-zags (0.04–0.4 rad) while the EE advances uniformly (4–6 mm per step): IK null-space chatter, not reordering (`analysis_summary_v1.json: joint_zigzag_episodes`). The count of 161 rows and the gripper toggles at exactly rows 48/131 in every episode are consistent with in-order writes. This is evidence, not proof: the logger has no sequence ids.

### 5.3 Image vs state timing (inherited; not changed)

The main loop reads the camera output rendered in the previous iteration, then steps physics, then logs the robot state. Measured on the data:

- **Sharp events:** for the 41 joint-jump episodes, a state discontinuity between rows `t` and `t+1` appears as the image change between frames `t+2` and `t+3` in 34/41 (camera_0) and 36/41 (camera_1) cases.
- **Cross-correlation** of right-EE speed with image motion energy over 61 evenly spaced episodes: lag 1–2 frames (camera_0: 2 in 35 cases, 1 in 20; camera_1: 1 in 46, 2 in 10). This peak is weak (median margin 0.018).

**Conclusion: frame `k` shows the robot approximately at state row `k−2`.** Under TinyDP's pairing, image `j` shows ≈ state `j−2`, proprio is row `j`, and the first action target is row `j+2`. Training reproduces the authors' pairing as-is. Whether real-robot camera latency matches this offset is an open question for later work.

## 6. Joint discontinuities (> 0.3 rad per step): reported, not excluded

`prepare_dataset.py analyze` → `artifacts/analysis_summary_v1.json`.

| Quantity | Value |
|---|---|
| Flagged episodes | **41 / 1,825** (2.25%), the same 41 as CODEX_AUDIT M5 |
| By split | train 33, val 4, test 4. Per train subset: n50 → 2, n100 → 4, n150 → 5, n1000 → 21, n1461 → 33 |
| Joint | `yumi_joint_4_r` 35, `yumi_joint_6_r` 6 |
| When | mid-trajectory, steps 55–120 (mostly 57–64 and 99–106), not at the start |
| Jump size | flagged median 0.41, p99 0.98, max 1.05 rad (unflagged: median 0.13, max 0.30) |
| Right-EE motion at the jump step | median **5.2 mm, 0.43°** (max 6.7 mm, 1.4°), i.e. smaller than ordinary per-step motion (unflagged episodes at their largest joint step: median 16 mm, 7.7°) |
| Effect on EE-space actions/proprio | Essentially none locally: the jumps are IK reconfigurations at an almost unchanged EE pose. Rendered images do show the arm changing configuration. |
| Effect on normalisation (n1461 train) | **Large for rotation.** These episodes are the only ones with extreme right-wrist orientations. With vs without the 33 flagged train episodes: r00 ∈ [−0.091, 1.000] vs [0.630, 1.000]; r01 max 0.479 vs 0.351; r02 ∈ [−0.849, 0.992] vs [−0.708, 0.667]; r12 min −0.745 vs −0.394; z min 0.025 vs 0.028; max abs bound change 0.72. Range ratio for the right rot6d dims: 2.95, 1.13, 1.34, 1.00, 1.00, 1.26. |

They are **kept** in all splits. Their effect is that ≈2% of episodes define the min–max range of 4 of 6 right-rotation dimensions, compressing the remaining data into a smaller part of [−1, 1]. A "flagged-excluded" or robust-statistics run is a candidate ablation and must be labelled as a deviation (02, §6).

## 7. Reproduce

```bash
cd ~/Stanley_ws/real2render2real
P=.venv-tinydp/bin/python; O=~/Stanley_ws/r2r2r_policy_runs/dataset_v1
$P docs/reproduction/tools/validate_dataset.py ~/Stanley_ws/r2r2r_repro_runs/run0*/output_data/yumi_coffee_maker --decode all --workers 16
$P policy_training/prepare_dataset.py manifest --source-root ~/Stanley_ws/r2r2r_repro_runs --out $O/manifest.json   # == artifacts/manifest_v1.json
$P policy_training/prepare_dataset.py trial --manifest $O/manifest.json --n 2  --out ~/Stanley_ws/r2r2r_policy_runs/trials/trial_n2
$P policy_training/prepare_dataset.py trial --manifest $O/manifest.json --n 10 --out ~/Stanley_ws/r2r2r_policy_runs/trials/trial_n10
$P policy_training/prepare_dataset.py views --manifest $O/manifest.json --out-root $O/views
$P policy_training/prepare_dataset.py stats --manifest $O/manifest.json --views-root $O/views --out-dir $O/stats
$P policy_training/prepare_dataset.py analyze --manifest $O/manifest.json --views-root $O/views --out $O/analysis_report.json
$P -m pytest policy_training/tests -q
```
