# Independent TinyDP pre-training audit

Date: 2026-10-23, Asia/Taipei. Reviewer: Codex.

**Verdict: NO-GO for full training on the current preparation/protocol.** The current dataset views, trajectory splits, coordinate conventions and right-arm selection are substantially corroborated. However, the action bounds miss real training targets; accumulated training changes scheduler/EMA behavior; resume is incorrect; and the checkpoint-to-inference contract is broken. The launch wrapper also does not enforce the split/statistics guarantees established during preparation. These are actionable blockers before committing expensive compute.

**CRITICAL:** no existing split contamination, baseline algorithm modification, or invalid source dataset was established. **MAJOR:** seven findings below affect reliability or comparability. **MINOR:** a window-index documentation ambiguity is identified separately. An executable counterexample is distinguished from evidence of corruption in the current artifacts.

No generator, TinyDP, adapter, environment, or existing dataset was modified during this audit. No dependencies were installed, no assets downloaded, and no GPU training was launched. Fresh execution consisted of CPU tests, read-only artifact inspection, a reduced-model diagnostic, and instrumented scalar-model lifecycle tests. Temporary fixtures/results are under `/tmp`; only this report was added to the repository.

## 1. Provenance and baseline preservation

The active branch is `repro/dp-training` at **`f2ce4551c0dd3ca3c19efb15080a686822e5a565`**, with parent/baseline `repro/baseline-datagen` at **`334e83d4d1be7a4a5e78768cd5344051d4a892fa`**. The latter contains the [prior audit](../reproduction/CODEX_AUDIT.md). The preparation commit identifies Claude Opus 5.5 as co-author; this is commit metadata, not independent proof of how every historical command ran.

Executed `git diff --exit-code repro/baseline-datagen HEAD -- scripts real2render2real dependencies pyproject.toml uv.lock env_data_gen.sh`: exit 0. The preparation commit adds `policy_training/`, documentation and artifacts, adds the isolated venv ignore rule, and replaces the reproduction **validation helper** at `docs/reproduction/tools/validate_dataset.py`. That helper replacement is a real baseline-branch difference, but does not modify the original R2R2R algorithm. TinyDP remains at **`76d8f9cd14affe0066df3dbb38759b086613f617`**. Recursive submodule pins match the earlier audit.

The pre-existing dirty trajgen files remain only generated `trajgen.egg-info/{PKG-INFO,SOURCES.txt}` metadata. No substantive source modification was found. The source-data fingerprint was recomputed and matches the preparation artifact:

```text
dd23bd44ce626485a9f3ec8f78ce3c75cfeadb90785efdcf456c8f997c2f2bcd
```

This fingerprint covers paths, sizes, mtimes and HDF5 bytes, **not every JPEG's bytes**. It is good consistency evidence, not a cryptographically immutable image dataset. The committed manifest and external `dataset_v1/manifest.json` are byte-identical; SHA-256:

```text
2c58b6f975103c085ed09ec852fa9bcbbc3d22f30d3d77386dcc4678899b1031
```

The existing `.venv-tinydp` metadata confirms Python's prepared package stack: torch 2.5.1+cu121, torchvision 0.20.1+cu121, NumPy 1.26.4, zarr 2.12.0, h5py 3.16.0, diffusers 0.18.2, transformers 4.48.2 and huggingface_hub 0.25.2. An executed direct-requirements check finds the documented **h5py 3.16.0 versus TinyDP's `==3.12.1`** deviation. Source HDF5 and sidecar reads succeeded in the installed environment. No dependency reinstall is needed for the findings here. `python -m pip check` could not run because this venv has no pip; `uv pip check` could not acquire a lock in its read-only default cache. Neither is reported as passed; the metadata check is not a complete transitive/ABI check.

## 2. Checks actually executed

All fresh Python execution used the existing `.venv-tinydp/bin/python -B`, with `PYTHONDONTWRITEBYTECODE=1`; CPU model tests used two OpenMP/MKL threads. Commands below reference local temporary evidence, which is not committed as a portable test suite.

| Fresh check | Result and coverage |
|---|---|
| `python -B -m pytest policy_training/tests -q -p no:cacheprovider --basetemp=/tmp/codex_pretrain_pytest` | **37 passed in 4.38 s**, 544 zarr/NumPy deprecation warnings. Includes malformed HDF5/images, prior-audit M1 fixture, CLI exit codes, converter equivalence on copies, split isolation, poisoned validation data, indexing, quaternion conversion and wrapper guards. |
| `/tmp/codex_pretrain_inventory.py` | Independently checked all **1,825** HDF5 arrays, manifest hashes, names, camera filenames, split assignment and **all 3,125 view entries / 9,375 sidecars** against source values/dtypes and symlink destinations. No inventory/identity errors. Results: `/tmp/codex_pretrain_inventory.json`. |
| Independent train-only normalization calculation using source arrays and SciPy rotations | All five JSONs match the training subset's **even-row** extrema. Actual target support also contains odd rows; observed violations are quantified in M1. No validation/test source arrays were used in this calculation. |
| Fresh JPEG sampling | **130 full PIL decodes**: 13 deterministically spaced source episodes × 2 cameras × frames 0/49/80/131/160, all JPEG/RGB/1280×720. All 587,650 filenames were inventoried. The reviewer did **not** repeat the complete 587,650-frame decode. |
| Independent XML-URDF FK | **238 pose comparisons**, seven episodes × 17 rows × two arms. Maximum position error **6.40e-7 m**, maximum wxyz rotation error **8.20e-5°**. Treating the same quaternions as xyzw gives median **163.18°** error. |
| Actual TinyDP train/val datasets at N=50/150/1000 | Train items **24,200 / 72,600 / 484,000**; each val dataset has **182 trajectories / 88,088 items**, with identical resolved membership and no train overlap. Manifest/view inspection separately verified the fixed 182-trajectory test split. |
| Actual real-data batch at 448 px | B=2: observations `(2,4,2,3,448,448)`, proprio `(2,4,20)`, actions `(2,4,16,20)`, all finite. Verified camera order against transformed source JPEGs and recovered raw right-arm targets against their source rows. |
| Reduced CPU DiffusionPolicy diagnostic | Same released model/forward code, but **64 px, B=1 and UNet widths `(16,32,64)`**, not the proposed full model. Loss **1.780018**, 208 gradient tensors all finite; output action dimension 10. Changing only left targets by +999 with fixed noise leaves loss exactly unchanged (**1.561083** both times). Two-step denoising returns finite `(1,16,10)`. No expensive full-model/GPU performance conclusion follows. |
| Negative normalization fixtures and actual loader counterexample | An odd-row spike normalizes to **5.0**; an actual n150 loader target is **−1.080700**. See M1. |
| Negative wrapper tests | Identical train/eval roots accepted and produced **50 overlapping files**; absolute statistics accepted with the delta-action flag. Output statistics file not created. See M4/M5. |
| `/tmp/codex_pretrain_engine.py` | Executed the released epoch loop on CPU scalar models with CUDA autocast/synchronization replaced by no-ops. Six micro-batches / accumulation 4 produce **1 optimizer step, 6 scheduler steps, 6 EMA updates**. Instrumented released `train.main` with toy data/model and stubbed epoch work reproduces incorrect resume. Log: `/tmp/codex_pretrain_engine.log`. This was not full training. |
| Inference module import | Executed import fails with `ModuleNotFoundError: No module named 'tinydp.dataset.dataset'`. |
| Image/state sharp-event check | Recomputed full-frame motion-energy series for three flagged episodes, both cameras: all six image-event peaks lag their joint event by **2 rows**. See M6 for the limits of this estimator. |

Historical evidence inspected separately: `/home/stanley/Stanley_ws/r2r2r_policy_runs/validation_full_decode.json` and `logs/02_validation_full_decode.log` record 1,825 valid episodes and full decoding of all 587,650 JPEGs, exit 0. Their episode set matches the current manifest. This is **retained execution evidence**, not a fresh full-decode pass by this reviewer.

Likewise, the committed [smoke_test_b16_acc4.json](artifacts/smoke_test_b16_acc4.json) is byte-identical to the external report, and its stdout contains matching values: four finite micro-losses averaging **1.662107**, gradient norm **15.49136**, **254/254** parameter tensors changed, finite val loss **1.687158**, inference `(2,16,10)`, and PyTorch peak **14.29 GiB**. The stdout ends with a worker-abort traceback during teardown, as disclosed. A separately retained exit-status file was not found in the smoke directories. I did not rerun this full-size GPU test or certify its teardown. It also used the default optimizer values, not the proposed lr/weight decay, and bypassed the real EMA/epoch/checkpoint path.

## 3. Verified data and policy contract

| Component | Actual contract / assessment |
|---|---|
| HDF5 | Exactly `ee_poses: (161,14) float32`, `joint_angles: (161,16) float32`, `gripper_binary_cmd: (161,2) bool`. All source arrays finite; hashes match the manifest. The sim loader reads zarr sidecars rather than these HDF5 datasets directly. |
| Sidecars | `robot_data_{proprio,joint,gripper_cmd}.zarr` match the official converter's three source keys. The new adapter preserves source data by writing out of place and symlinking HDF5/cameras. The real-robot MP4 converter is correctly excluded. |
| Cartesian frame | `[left xyz, left wxyz, right xyz, right wxyz]`, meters, poses of the dummy-point frames relative to `base_link`. Independent FK confirms this sampled contract. These are neither camera-frame coordinates nor joint-space action targets. |
| Joint order | `[1_l,1_r,2_l,2_r,7_l,7_r,3_l,3_r,4_l,4_r,5_l,5_r,6_l,6_r,gripper_l_joint,gripper_r_joint]`. Arm angles are radians; final finger coordinates are meters. The sim loader uses binary-command sidecars when present, not the finger coordinates. |
| Rotation | TinyDP converts scalar-first quaternions into the first **two rows** of a rotation matrix, flattened to six values. q/−q invariance is verified by the executed suite. Do not substitute a column-based 6D convention in later deployment. |
| Gripper | Boolean command **1=close, 0=open**; no inversion actually occurs despite the stale loader comment. Left is false throughout. All 1,825 right commands change after rows **48 and 131**, so rows **49–131** are closed. This is command state, not a measured aperture. |
| Camera order | TinyDP stacks **camera_1, then camera_0**, converts JPEGs through its torchvision/timm transform, resizes to a square and normalizes. The resize stretches aspect ratio; it does not add the padding described in an upstream comment. |
| Dataset item | For item start `j`: image file `j`, proprio row `j`, absolute targets at `j+2,j+4,…,j+32`. Proprio noise is enabled by default, including validation; it was disabled only in independent exact-value probes. |
| Sequence / loss target | Window start `i` supplies items `i,i+2,i+4,i+6`. The model trains on **the last item's** target chunk: `i+8,…,i+38`. Window starts are **0–120**; item starts range **0–126**; final possible target is row **158**. There are **121 windows × 4 items = 484 items/episode**. |
| Nominal timing | Unchanged IsaacLab `SimulationCfg` defaults to `dt=1/60 s`; one logged manipulation row follows one physics step. Subsampling 2 corresponds nominally to 30 Hz targets. Saved data lack acquisition timestamps, so this is code-derived simulation cadence, not verified real-camera/deployment latency. |
| Right-only configuration | Dataset/collate retain **20D proprio and 20D action arrays**. `DiffusionPolicy.forward_loss` selects the last timestep's **right slice `[10:20]`**; the network predicts **10D**. The CPU target-invariance test confirms left targets do not contribute. The constant left-gripper normalization NaN is masked upstream and is not in this right-arm loss. |

The right-only/absolute-action protocol is a **released-code-compatible single-arm variant**. The [paper's Table 8](https://arxiv.org/html/2505.09601v1#A9.SS7) specifies 20D YuMi actions, batch 64, 500 warm-up steps and 100,000 total steps; its action-format wording differs from §4.3's absolute-EE description. The preparation discloses these ambiguities. It must not be called an exact Table-8 reproduction without choosing and freezing the interpretation.

### Split and normalization membership

Independently reproduced the SHA-256 UID ordering: first **182 test**, next **182 val**, remaining **1,461 train**. UIDs, resolved paths and recorded numerical-content hashes are unique across splits. Every current sidecar maps to the expected source. Subsets **50/100/150/1000/1461** are nested prefixes of the train order. Val/test trajectories are unchanged across those sizes. The wrapper's current correctly supplied roots produce the intended memberships; M4 concerns missing enforcement at launch, not demonstrated contamination in these prepared roots.

All five normalizers contain only their own training subset's data. The poisoned-validation test passed in the fresh suite. However, **train-only membership and complete training-target coverage are separate requirements**; the latter fails in M1. Content hashes include the three numerical arrays, not JPEG bytes. Trajectory-level separation also does not establish independence of shared generation-batch lighting, grasps, or the common original human demonstration; offline results concern held-out synthetic trajectories.

## 4. MAJOR findings and minimal fixes

### M1 — Normalization misses the odd-row targets actually used in training

**Inherited TinyDP defect, repeated by the new cross-check.** References: [sim loader](../../dependencies/tinydp/tinydp/dataset/image_dataset_sim.py), lines 135–146, 186–217 and 301–310; [tinydp_bridge.py](../../policy_training/r2r2r_dp/tinydp_bridge.py), lines 84–98.

`calculate_dataset_statistics` calls `helper_load_action(start=0, end=T−2)`, yielding rows **2,4,…,160**. Training windows start at every integer row, so their targets include **both parities**, rows 2–158. The allegedly independent preparation check explicitly copies the same even-row index sequence. Agreement between those calculations therefore does not establish coverage of actual training targets.

Fresh real-data results below count unique `(episode,row)` right-arm target rows outside the JSON bounds with normalized tolerance `1e-6`, before window multiplicity. They are not counts of optimizer samples or independent trials.

| Train N | Out-of-bound target rows / candidate rows | Affected episodes | Largest normalized absolute value |
|---:|---:|---:|---:|
| 50 | 8 / 7,850 | 6 | 1.028846 |
| 100 | 8 / 15,700 | 6 | 1.080700 |
| 150 | 10 / 23,550 | 8 | 1.080700 |
| 1,000 | 11 / 157,000 | 9 | 1.061949 |
| 1,461 | 12 / 229,377 | 10 | 1.061949 |

**Small real counterexample:** n150 `ep_train_0076`, source `run06_env2_gpu0_noprealloc/env_1_2026_10_22_20_06_53`, row **15**, right rotation component `r00` (global dimension 13): raw **0.67616565**, stored bounds **[0.68872507,0.99998644]**, normalized **−1.080700**. The actual loader returns this target from the last observation at row 13 of window start 7; this is not merely an unused source row. DDPM has `clip_sample=True`, so its final normalized output cannot exactly reproduce that component.

**Small synthetic counterexample:** with the existing 60-row fixture, change only `ee_poses[21,7]=2.0`. Statistics retain maximum right x **0.88** from even rows, but the item starting at 19 returns the row-21 target normalized to **5.0**. The complete reproduction is in the appendix.

**Fix:** calculate bounds outside the baseline implementation over the union of actual training action indices, including both parities; assert coverage of all trained right-arm target components. Regenerate every subset's statistics and record their membership/configuration hashes. If preserving the released normalizer is scientifically required, retain it as an explicitly labeled compatibility run and document this defect; do not claim complete `[-1,1]` target coverage. Changing bounds is a documented training-preparation deviation, not a silent TinyDP edit.

### M2 — Accumulation 4 is not equivalent to the intended optimizer-step schedule

**Inherited behavior amplified by the proposed memory workaround.** Reference: [engine.py](../../dependencies/tinydp/tinydp/util/engine.py), lines 53–65 and 96–99; [train.py](../../dependencies/tinydp/script/train.py), lines 168–176. The preparation correctly discloses the scheduler and EMA issue, but leaves it unresolved.

Executed scalar-model counterexample using the released epoch loop: six micro-batches with accumulation 4 cause **one optimizer update, six scheduler updates, six EMA updates**, and leave an unapplied gradient `−0.900000` at epoch end. The next epoch's initial `zero_grad` drops it. Thus the proposed setting gives a 500-**micro-batch** warm-up, approximately **125 optimizer updates**, not 500. EMA also advances between optimizer updates. GroupNorm's lack of batch statistics does not make these optimizer trajectories equivalent to batch 64 / accumulation 1.

At N=50, 378 micro-batches produce 94 optimizer updates and discard the final two micro-batches' gradients each epoch. Sampler order repeats because `set_epoch` is never called; consequently the same tail is discarded. The published epoch-count arithmetic uses floor division consistently, but it is not an exact global 100,000-update stopping rule.

The loop additionally divides metric values by accumulation even in validation. The executed toy model's raw val loss **0.810000** is returned as **0.202500** at accumulation 4. `log.txt` and selected TensorBoard loss tags therefore have different units from the smoke report's raw loss; some TensorBoard tags retain the unscaled value.

**Fix:** in a separately identified training driver, advance scheduler and EMA once per actual optimizer update, specify the partial-accumulation policy, stop/count in optimizer updates, and log raw losses separately from backprop-scaled losses. Add the six-micro-batch counterexample as a regression. Alternatively freeze the micro-batch behavior as an explicit variant and adjust all schedule/EMA/reporting claims accordingly. Do not silently modify the authors' submodule.

### M3 — Resume repeats epochs and loses the EMA/schedule state

**Inherited and executed with a CPU-instrumented `train.main`.** References: [train.py](../../dependencies/tinydp/script/train.py), lines 46, 165–178 and 211–219; [misc.py](../../dependencies/tinydp/tinydp/util/misc.py), lines 370–389.

The local `start_epoch` is copied before `resume_from_ckpt` changes `args.shared_cfg.start_epoch`. EMA is created after resume and not passed to the helper. The scheduler is recreated from step zero; checkpoints save neither scheduler nor scaler/RNG/sampler state.

**Counterexample:** a one-parameter checkpoint at epoch **4**, model value **3**, EMA value **9**, with requested final epoch count 6. Instrumenting the real orchestration with toy datasets and stubbed epoch work produces training calls **[0,1,2,3,4,5]**, although the loaded args correctly say start epoch 5. Expected calls are **[5]**. The saved EMA is **3**, not the restored **9**; checkpoint keys are only `ema,epoch,model,optimizer`. No expensive model training was involved in this probe.

**Fix:** capture the resumed start epoch after loading; restore existing EMA and scheduler/global update count, scaler and RNG/sampler state; validate uninterrupted versus interrupted CPU runs. Until then reject `--shared-cfg.resume` for scientific continuation, rather than representing it as a faithful resume. A long run that cannot be resumed should have that limitation explicit before scheduling it.

### M4 — Preparation guarantees are not enforced by the actual launch wrapper

**New preparation-layer gap.** References: [train_tinydp.py](../../policy_training/train_tinydp.py), lines 46–85; [ManifestSimDataset](../../policy_training/r2r2r_dp/tinydp_bridge.py), lines 38–57.

The wrapper checks directory/file existence and some flags, but does not load the manifest, verify exact root membership or source/sidecar hashes, reject train/eval overlap, bind a statistics file to a specific subset/action convention, or validate the statistics' numerical schema. The default command's roots are currently correct; these omissions make the accepted interface unsafe for the claimed leakage/provenance contract.

**Executed counterexamples:** passing n0050 as **both** roots is accepted; instantiating both datasets gives **50 overlapping trajectories**. Adding `--shared-cfg.use-delta-action` with the existing absolute-action JSON is also accepted. The normalizer JSON has no action-mode/subsampling/manifest binding. `--load-config` is not guarded, and the released entry point loads it after CLI parsing, so inspecting only the raw CLI cannot certify the final configuration.

**Fix:** before model allocation, load one frozen manifest/configuration and verify exact expected UIDs, resolved disjointness, nonempty roots, matching sidecar/source fingerprints, normalizer provenance and permitted action/sampling settings. Reject unsupported config-file overrides or validate the fully resolved configuration. Refuse ambiguous reuse of output directories/provenance. Add overlap, wrong-subset statistics, absolute/delta mismatch and mutated-sidecar negative tests. Preserve the current genuinely disjoint views.

### M5 — The trained artifact is not ready for the documented inference path

**Inherited import/loading defects plus a new artifact omission.** References: [diffusion_wrapper.py](../../dependencies/tinydp/tinydp/policy/diffusion_wrapper.py), lines 5 and 54–78; [train_tinydp.py](../../policy_training/train_tinydp.py), lines 75–84; sim loader lines 102–114.

1. Executed import `from tinydp.policy.diffusion_wrapper import DiffusionWrapper` fails because `tinydp.dataset.dataset` does not exist.
2. With supplied statistics, the sim dataset reads the external JSON and never writes `output_dir/action_statistics.json`. The wrapper also only records its path/hash. Fresh dataset/wrapper probes create no output statistics file; the historical smoke output directory is empty. `DiffusionWrapper` requires that exact file in the checkpoint folder.
3. Validation uses EMA weights, but the inference loader selects checkpoint `model` and ignores `ema`. Therefore the evaluated/saved-model choice is not the same contract. The proposed short smoke calls `model.forward_inference` directly, bypassing all three issues.

**Fix:** provide a versioned external inference adapter with the correct imports, copy the exact normalizer/config/manifest hashes into a self-contained run artifact, and explicitly choose EMA or raw weights consistently for validation/checkpoint selection/inference. Require a CPU save/load/denormalize round-trip before full training. A complete robot harness is not necessary for this check. Do not interpret the existing finite 10D denoising output as evidence that checkpoint loading/deployment works.

### M6 — Image/state/action timing remains intentionally misaligned and insufficiently validated

**Inherited data-generation behavior, not introduced by the view adapter.** References: [generator](../../real2render2real/isaaclab_viser/yumi_simulators/yumi_coffee_maker.py), lines 184–206, 350–380 and 956–1001; [preparation §5](01_dataset_preparation.md).

The code reads a previous render/buffer before the physics/state update and then labels it with the current logger index. Preserving filenames does not synchronize sensing. Fresh event checks on `run02` episodes ending **17_43_10**, **17_43_51**, **17_55_33** find the row-58 joint jump's image-motion event two rows later in both cameras. This corroborates the historical lag analysis, but motion energy is a proxy: it does not establish an exact per-frame registration for every episode or real-camera latency.

**Counterexample pairing:** item `j=60` presents image 60 (approximately state 58), proprio 60 and first target 62. For sequence start 54, those are the final observation/first target actually used by the loss. This differs from synchronized image/proprio at a common physical instant. The current adapter faithfully reproduces the authors' indexing; faithfulness to a released implementation does not establish correct deployment timing.

The adjacent-swap detector is also a heuristic. Zero detections is not proof of ordered writes; coordinated block swaps or exchanges in static segments can evade it. The original logger still has eight workers and no sequence IDs. The 41 joint-jump episodes are retained and influence some rotation bounds; this is documented, not silently filtered.

**Fix:** freeze an explicit timing contract before training: either a labeled released-pairing reproduction with this limitation, or a separately versioned aligned dataset view with justified frame/state offsets and boundary handling. Validate representative projected FK/visual events at grasp/release and record the expected deployment cadence/latency. Do not shift images silently or regenerate the baseline during this audit. Add synthetic image-delay tests that distinguish correct filename arithmetic from actual sensing alignment.

### M7 — The scientific training/evaluation protocol is not yet frozen or exercised end to end

References: [02_training_protocol.md](02_training_protocol.md), decisions D1–D5 and blockers B1–B3; [03_pretraining_checks.md](03_pretraining_checks.md), §4.

Right-only 10D actions, absolute versus delta semantics, altered accumulation and optimizer constants need one explicit versioned baseline choice. The existing GPU smoke is useful execution evidence for the full model's data/gradient path, but it does not exercise the wrapper → resolved-config → actual epoch loop → EMA validation → checkpoint → inference chain. My CPU instrumentation finds M2/M3 precisely in those bypassed paths. The reduced CPU model cannot establish full-size resource stability or the proposed optimizer schedule.

Every-epoch full validation is especially disproportionate across sizes: the proposed N=50 budget entails **1,464,064 validation micro-batches** versus **402,192 training micro-batches**. Checkpoint cadence also remains epoch-based. Choose fixed optimizer-update-based validation/checkpoint intervals or a declared fixed evaluation sample. Validation proprio noise and diffusion noise should be seeded/repeated for comparable reporting; identical trajectory membership alone does not make individual denoising losses deterministic. The sampler's repeated order/fixed dropped tail should be disclosed or corrected in the external driver.

The one-step **1.28 s** historical timing is not sustained throughput: model/loader setup, epoch validation, checkpoints and teardown are excluded or incompletely represented. It does not justify a reliable full-run duration. No policy success, real-world transfer or convergence is established by finite losses.

**Fix:** freeze a resolved run configuration for each N with shared held-out membership and optimizer-update budget; specify action/gripper/timing semantics and checkpoint selection. Then execute a bounded lifecycle smoke including EMA validation and a CPU-compatible checkpoint round-trip before scaling. A fresh full 100k-step run is neither needed nor authorized to resolve these issues.

## 5. MINOR documentation finding

**N1 — distinguish window start from flattened item start.** Preparation §5.1 calls `j=0…126` valid while also describing four-step samples. The code uses **window starts 0…120**, with flattened item starts through 126. Its count of 121 windows/episode is correct. Use separate symbols consistently and show the final-row calculation `120+6+32=158`; otherwise readers can accidentally construct an out-of-range window starting at 126.

The disclosed h5py override, isolated venv, right-only output shape and retained flagged episodes are not undisclosed baseline edits. The new strict validator resolves the prior audit's demonstrated malformed-episode acceptance cases; it does not resolve temporal ordering, action-bound coverage or checkpoint lifecycle correctness.

## 6. Release gates for a GO verdict

These are proposed checks/fixes, **not tests claimed to have passed**:

1. Cover both-parity training targets in every normalizer, or explicitly approve/freeze the defective released-normalizer behavior as a compatibility experiment. Add the odd-row spike and actual row-15 regression.
2. Enforce manifest membership, resolved train/eval disjointness and statistics/action-mode provenance at launch. Reject all M4 negative cases before allocating the full model.
3. Fix or explicitly constrain optimizer/scheduler/EMA/update-count semantics in an external versioned driver; regression-test incomplete accumulation and raw loss logging.
4. Demonstrate faithful resume and a self-contained checkpoint/normalizer/config load, with consistent EMA selection. At minimum, explicitly disable unsupported resume until corrected.
5. Freeze the right-only versus 20D, absolute versus delta, timing, validation cadence and checkpoint-selection decisions. Preserve the exact 50/150/1000 held-out trajectory membership already verified.
6. Run a bounded actual entry-point lifecycle test, including validation and checkpoint loading, after the above changes. Report full-size GPU checks only if actually executed and capture process exit/worker cleanup. No lengthy training is required for this gate.

The preparation has correctly solved several important problems: source preservation, sim-schema conversion, fixed trajectory splits, and exclusion of held-out data from the normalizer's source membership. Those successes do not remove the current normalization and training-lifecycle blockers. **Full training remains NO-GO.**

## Appendix: compact normalization counterexample

Run from the repository root with the existing TinyDP venv. This creates only temporary files and loads one item; it starts no training:

```python
import os, sys, tempfile, zarr, tyro
sys.path[:0] = ["policy_training", "policy_training/tests"]
from fixtures import episode_arrays, write_episode
from r2r2r_dp.adapter import SIDE_CARS
from r2r2r_dp import tinydp_bridge as tb
from tinydp.util.args import ExperimentConfig
from tinydp.dataset.utils import default_vision_transform

with tempfile.TemporaryDirectory(dir="/tmp") as d:
    a = episode_arrays(60)
    a["ee_poses"][21, 7] = 2.0  # odd row, right x
    ep = write_episode(d + "/view/ep", arrays=a)
    h5 = ep + "/robot_data/robot_data.h5"
    for key, suffix in SIDE_CARS.items():
        zarr.save_array(h5.replace(".h5", suffix), a[key])
    cfg = tyro.cli(ExperimentConfig, args=["--dataset-cfg.is-sim-data"])
    cfg.dataset_cfg.proprio_noise = 0.0
    cfg.logging_cfg.output_dir = d + "/out"
    os.mkdir(cfg.logging_cfg.output_dir)
    tb.compute_action_statistics_tinydp(
        d + "/view", cfg.shared_cfg, cfg.dataset_cfg, d + "/stats.json")
    cls = tb.make_manifest_dataset_class(d + "/view", d + "/view", d + "/stats.json")
    ds = cls(cfg.dataset_cfg, cfg.shared_cfg, cfg.logging_cfg,
             default_vision_transform(32), split="train")
    i = next(i for i, (_, start, _) in enumerate(ds.start_end) if start == 19)
    print(ds[i]["action"][0, 10].item())  # observed: 5.0
```
