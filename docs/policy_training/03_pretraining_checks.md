# 03 — Pre-training checks

> **Update (2026-10-27):** the readiness verdict in §5 was superseded by the audit ([CODEX_PRETRAIN_AUDIT.md](CODEX_PRETRAIN_AUDIT.md), NO-GO) and its repair ([AUDIT_REPAIR_REPORT.md](AUDIT_REPAIR_REPORT.md), [LIFECYCLE_TEST_RESULTS.md](LIFECYCLE_TEST_RESULTS.md)). The smoke test below exercised the legacy released loop and is historical evidence only.

Date: 2026-10-23 · Env `.venv-tinydp` (01 §1) · GPU 0 only (`CUDA_VISIBLE_DEVICES=0`).

## 1. Deterministic tests (`policy_training/tests`, 37 tests, all passing)

```bash
.venv-tinydp/bin/python -m pytest policy_training/tests -q      # 37 passed in ~5 s (CPU only, synthetic fixtures)
```

| Requirement | Tests |
|---|---|
| Malformed episodes | `test_schema.py`: wrong shapes/dims/dtypes, missing key, non-finite values, non-unit quaternions, Boolean gripper violations (dtype, inactive arm set, toggle count), missing or reordered joint names; **the exact CODEX_AUDIT M1 counterexample** now fails on 6 separate checks |
| Missing / bad images | missing camera, missing frame, extra frame, wrong resolution, wrong format, truncated JPEG (caught only with full decode) |
| CLI failure visibility | `validate_dataset.py` exit 2 for a missing or empty dataset, 1 for any invalid episode, 0 for a valid full-size (161 × 1280×720) episode |
| Temporal indexing | `test_tinydp_pipeline.py::test_windows_cover_valid_range_and_sequences_are_consecutive`: window count, observation steps `j, j+2, j+4, j+6` from one episode, last action row inside the episode |
| Action/observation pairing | `::test_observation_proprio_action_alignment`: with row-encoded fixtures (image gray level, EE x and gripper encode the row), TinyDP returns image frame `j` (both cameras), proprio row `j`, actions rows `j+2…j+32`, gripper targets from the same rows, left gripper 0. `::test_sampler_and_collate_shapes`: batch shapes and per-sample frame sequences after `VideoSampler` + `CollateFunction` |
| Orientation conversion | `::test_quaternion_convention_and_sign_invariance`: TinyDP `quat_to_rot_6d(wxyz)` equals the first two rows of scipy's rotation matrix for 50 random rotations; q and −q give identical rot6d; identity → `[1,0,0,0,1,0]`; misreading xyzw is detectable. Real-data convention verified separately by URDF FK (01 §2). |
| Data leakage | `test_manifest_adapter.py`: deterministic and order-invariant manifest; disjoint uids, paths and content hashes across splits; val/test independent of subset sizes; nested subsets; duplicate content rejected; tampered manifest rejected; no episode in two split roots. `test_tinydp_pipeline.py::test_statistics_use_train_only`: poisoned val sidecars (+100 m) do not move the statistics, and both datasets use the given train file. `::test_wrapper_refuses_missing_or_conflicting_statistics` |
| Adapter fidelity / non-destructiveness | sidecars identical to the **official** `scripts/to_zarr_dataset.py` output and to the source arrays; source fingerprint unchanged; no `.zarr` written into the source; refuses overlapping or existing output; detects source content changes |
| Analysis detectors | `test_analysis.py`: adjacent row swap detected (static segments are not), joint-jump profile, lag recovery 0–3 |
| Spawn-safety | `::test_dataset_is_picklable_for_spawn_workers` (train.py uses `spawn` workers) |
| Training wrapper | `test_train_wrapper.py`: rejects missing `is-sim-data`, `log-name` (WandB), dataset-root, subsampling, statistics overrides; installs `ManifestSimDataset` into TinyDP; disables WandB; writes provenance |

## 2. Checks on the real data

| Check | Result | Evidence |
|---|---|---|
| Strict validation, full JPEG decode of all 587,650 frames | 1,825 / 1,825 valid, exit 0 | `~/Stanley_ws/r2r2r_policy_runs/validation_full_decode.json` |
| Quaternion convention | wxyz; FK rotation error ≤ 8e-5° | 01 §2 |
| HDF5 row order | 0 EE violations; 8 joint-chatter episodes explained | 01 §5.2 |
| Image/state lag | images ≈ 2 steps behind state | 01 §5.3 |
| Joint discontinuities | 41 flagged; kept; normalisation effect quantified | 01 §6, `artifacts/analysis_summary_v1.json` |
| Adapter staged 2 → 10 → all | 0 problems; source unchanged | 01 §4.2, `artifacts/views_verification_v1.json` |
| Train-only statistics | TinyDP == independent numpy, 5/5 subsets | 01 §4.3 |

## 3. Smoke test: one dataloader batch + forward/backward on GPU 0

`policy_training/smoke_test.py` builds exactly what `train.py` builds: both datasets, `VideoSampler`, `CollateFunction(pred_right_only)`, `MultiEpochsDataLoader` (spawn workers, prefetch 2, pinned memory), `DiffusionPolicy`, AdamW with TinyDP's weight-decay groups, `NativeScalerWithGradNormCount`, and the cosine scheduler. It then runs one full optimiser step (accum micro-batches, bf16 autocast, clip 1.0), one no-grad val loss, and a 10-step denoising call. No checkpoint is written.

Attempts (train = `train_subsets/n1461`, eval = `splits/val`, stats = `action_statistics_train_n1461.json`):

| # | batch × accum | workers | Outcome |
|---|---|---|---|
| 1 | 64 × 1 (released default) | 20 (default) | **Host OOM:** the kernel OOM-killed one DataLoader worker (16.9 GB shared memory). 20 workers × 2 prefetched batches × 1.23 GB (64×4×2×3×448×448 fp32) exceeds 62 GB RAM. No other process was killed (kernel log). |
| 2 | 64 × 1 | 4 | Loader OK (host RAM peak 16.2 GB). **CUDA OOM** in the ResNet-18 forward (512 images at 448², GroupNorm in fp32 under autocast) |
| 3 | 32 × 2 | 4 | **CUDA OOM** in the forward pass (256 images) |
| 4 | **16 × 4** | 4 | **All 15 checks pass** (`artifacts/smoke_test_b16_acc4.json`) |

Run 4 results:

| Check | Value |
|---|---|
| Device | RTX 4090, 1 visible device; GPU 1 stayed at its desktop baseline (143 MiB peak) |
| Train / val files | 1,461 / 182, disjoint (realpath) |
| Items / micro-batches per epoch | train 707,124 / 11,048; val 88,088 / 1,376 |
| Statistics in use | equal to the given train-only file (train and val datasets) |
| Batch shapes | observation (16, 4, 2, 3, 448, 448); proprio (16, 4, 20); action (16, 4, 16, 20) |
| Batch finite / scaled right-arm actions in [−1, 1] | yes / max abs 1.0 |
| Model | DiffusionPolicy(action_dim=10, obs_horizon=4, action_horizon=16, unet), 148,892,490 parameters |
| Train micro-losses (4) | 1.604, 1.645, 1.714, 1.685 (mean 1.662), all finite |
| Gradient norm before clipping | 15.49 (finite; clipped to 1.0) |
| Parameters updated / finite | 254 / 254 tensors changed; all finite |
| Val loss (no grad) | 1.687, finite |
| Inference (10 denoising steps, 2 samples) | shape (2, 16, 10), finite |
| Peak GPU memory | 14.29 GiB allocated by PyTorch; 20.7 GB device total (nvidia-smi) |
| Peak host RAM | 10.3 GB |
| Time for one optimiser step (4 micro-batches incl. 3 loads) | 1.28 s (single measurement) |

Known noise: at interpreter exit, TinyDP's persistent `MultiEpochsDataLoader` workers report `DataLoader worker ... killed by signal: Aborted` while being torn down after the report was written. The process exit status was 0. During training the loaders live for the whole run.

Command:

```bash
O=~/Stanley_ws/r2r2r_policy_runs; V=$O/dataset_v1/views
CUDA_VISIBLE_DEVICES=0 .venv-tinydp/bin/python policy_training/smoke_test.py \
  --train-root $V/train_subsets/n1461 --eval-root $V/splits/val \
  --action-statistics docs/policy_training/artifacts/action_statistics_train_n1461.json \
  --out $O/smoke/smoke_b16_acc4_w4/report.json -- \
  --dataset-cfg.is-sim-data --model-cfg.policy-cfg.pred-right-only \
  --trainer-cfg.num-workers 4 --shared-cfg.batch-size 16 --trainer-cfg.accum-iter 4 \
  --logging-cfg.output-dir $O/smoke/smoke_b16_acc4_w4/tinydp_out
```

(The smoke test used TinyDP's default optimiser lr 1e-4 / wd 0.01; the protocol's lr 2e-4 / wd 0.09 only changes optimiser constants.)

## 4. Not checked yet

- Sustained training throughput and long-run stability (no full or bounded training run was launched).
- Full `train_tinydp.py` → `train.py` execution path. The wrapper's guards and dataset substitution are unit-tested and the smoke test builds the same objects, but `train.py`'s epoch loop has not been run end to end.
- EMA validation path, checkpoint save/resume, `DiffusionWrapper` inference loading.
- Closed-loop policy evaluation (no simulator rollout harness or robot).

## 5. Readiness

Ready for **independent audit** of the data path and protocol. **Not ready for full training** until B1 (validation cost) and B2 (checkpoint cadence) are decided and decisions D1–D3 are confirmed (02 §5).
