# Validated TinyDP training protocol — `baseline_v2`

Status: **frozen and validated for launch; no full training run has been started.** This supersedes the proposal in [02_training_protocol.md](02_training_protocol.md). How each audit finding was resolved is in [AUDIT_REPAIR_REPORT.md](AUDIT_REPAIR_REPORT.md); the bounded GPU run is in [LIFECYCLE_TEST_RESULTS.md](LIFECYCLE_TEST_RESULTS.md).

## 1. Identity

| Item | Value |
|---|---|
| Protocol file | `policy_training/configs/baseline_v2.json` (`protocol_version` `r2r2r-dp-protocol-v2`) |
| Entry point | `policy_training/train_r2r2r_dp.py` → `r2r2r_dp.driver.Trainer` |
| TinyDP | submodule `dependencies/tinydp` @ `76d8f9cd14affe0066df3dbb38759b086613f617`, **unmodified** |
| Environment | `.venv-tinydp` (torch 2.5.1+cu121, numpy 1.26.4, zarr 2.12.0, h5py 3.16.0, diffusers 0.18.2, huggingface_hub 0.25.2), unchanged in this task |
| Manifest | `docs/policy_training/artifacts/manifest_v1.json`, sha256 `2c58b6f975103c085ed09ec852fa9bcbbc3d22f30d3d77386dcc4678899b1031` (pinned in the protocol) |
| Views | `~/Stanley_ws/r2r2r_policy_runs/dataset_v1/views` (symlinks + zarr sidecars; the 1,825 source episodes are never written) |
| Normalizer | v2, `docs/policy_training/artifacts/normalizer_v2/action_statistics_v2_train_nXXXX.json` |
| Resolved configuration hash per N | `docs/policy_training/artifacts/launch_checks_baseline_v2.json` (`config_sha256`). Identical protocol and N ⇒ identical hash; output and view paths are excluded. |

The resolved TinyDP `ExperimentConfig` is built from the released dataclass defaults plus the protocol's `tinydp` section. Only an allowlist of fields may be overridden (`protocol.TINYDP_ALLOWED`). There is no CLI pass-through and no `--load-config`, and `logging_cfg.log_name` (WandB) cannot be set. No WandB, TensorBoard upload or other remote logging exists in the v2 path.

## 2. Data and splits

| Split | Episodes | Use |
|---|---|---|
| train | nested prefixes N = 50, 100, 150, 1000, 1461 (`views/train_subsets/nXXXX`) | optimisation and normalisation (train subset only) |
| val | 182 (`views/splits/val`), identical for every N | frequent subset validation and milestone full validation |
| test | 182 (`views/splits/test`) | **reserved** for final evaluation; never loaded by the driver |

Before any dataset or model object is built, `launch_checks.check_launch` verifies:
- the manifest hash;
- exact membership of the train, val and test roots, using TinyDP's own `**/*.h5` glob;
- source.json and symlinks for every episode;
- sidecar and source-HDF5 content hashes;
- camera frame counts;
- disjointness of the splits by UID, resolved path and content hash;
- statistics provenance, plus a numerical recomputation of the bounds;
- that the output directory is new.

On the real data this takes 0.9–4.3 s per N (`artifacts/launch_checks_baseline_v2.json`).

## 3. Observation, action and timing contract (`timing_variant: released_pairing`)

For an episode of T = 161 rows, subsample s = 2, observation horizon L = 4 and prediction horizon P = 16:

| Symbol | Range | Meaning |
|---|---|---|
| window start `w` | 0 … 120 (121 windows) | one training sample |
| item start `i` | `w, w+2, w+4, w+6` (0 … 126) | one observation step |
| observation of item `i` | `camera_1/rgb/{i:04d}.jpg`, `camera_0/rgb/{i:04d}.jpg` → 448×448 (aspect stretched), ImageNet-normalised; colour jitter on train only | |
| proprio of item `i` | row `i`: `[left xyz, left rot6d, left grip, right xyz, right rot6d, right grip]` (20-D); uniform noise ±0.003 on xyz and Euler angles (train and val, as released) | |
| targets of item `i` | absolute rows `i+2, i+4, …, i+32` (20-D each) | all returned targets: rows 2 … 158 |
| loss | last item only (`i = w+6`), right half `[10:20]` of its 16 targets: rows `w+8 … w+38` | 10-D ε-prediction loss, gripper weight 5 |

Units and frames: metres; rotations are the first two **rows** of the rotation matrix of the scalar-first (wxyz) quaternion; the robot `base_link` frame; gripper 1 = close command.

**Known timing property (kept, not corrected):** the generator logs the image rendered before the state update, so image file `i` depicts approximately state row `i − 2`. This was measured by motion energy at joint events (34/41 and 36/41 events; three episodes re-verified by the audit). The baseline keeps the authors' filename pairing; `test_timing_contract.py` pins both that pairing and the fact that it differs from sensing alignment. An aligned dataset variant would need its own `timing_variant`, its own normalizer and an independent comparison; it is not implemented.

## 4. Normalizer v2

For each train subset: per-dimension min/max over **every target row the loader returns** (rows 2–158, both parities) of every training episode. That is a superset of the loss rows. Diffusion scaling to [−1, 1]; zero-range dimensions (the left gripper) are NaN → 0 as in TinyDP. On the real data, 0 right-arm training target rows fall outside [−1, 1] for every N. The released even-row v1 files are kept unchanged as a labelled compatibility baseline (`artifacts/action_statistics_train_nXXXX.json`); the v2 path rejects them.

## 5. Model and optimisation

| Setting | Value | Source |
|---|---|---|
| Policy | TinyDP `DiffusionPolicy`: ResNet-18 + GroupNorm (random init) ×2 cameras, ConditionalUnet1D (256, 256, 512, 1024), DDPM 100 steps, squaredcos_cap_v2, ε-prediction, `clip_sample=True`; 148,892,490 parameters | released, unchanged |
| Action output | right arm only, 10-D `[xyz, rot6d, grip]`, absolute; proprio input stays 20-D | `pred_right_only` (deviation D1) |
| Micro-batch × accumulation | 16 × 4 = 64 samples per optimizer update | deviation (24 GB GPU) |
| Optimizer | AdamW (0.9, 0.999), lr 2e-4, weight decay 0.09 on ≥2-D weights (TinyDP `add_weight_decay`) | paper Table 8 |
| LR schedule | diffusers cosine, 500 warm-up **optimizer updates**, 100,000 total updates; stepped once per update. As released, update 1 runs at lr 0. | paper; semantics fixed (M2) |
| EMA | decay 0.9999 **per optimizer update** (TinyDP `ModelEMA`, parameters only) | M2 |
| Precision / clipping | bf16 autocast + GradScaler, gradient norm clipped to 1.0; GradScaler-skipped steps are counted and logged | released |
| Accumulation policy | each update = 4 consecutive micro-batches of one continuous stream; groups may span an epoch boundary; nothing is dropped | M2 |
| Sampling | `VideoSampler(seed=42)`, `set_epoch(e)` every epoch. Batch composition is fixed by TinyDP's seeded window shuffle; windows that do not fill a final micro-batch are never sampled (N=50: 2 of 6,050; N=1461: 13 of 176,781) | released (inherited) |
| Item randomness | proprio noise and colour jitter seeded from (seed, stream, epoch, index), independent of worker count | M3 |
| Seeds | model/torch 0, sampler 42, validation 1234 | |
| Workers | 4 per loader, spawn | host RAM |

## 6. Validation, checkpoints and selection

| Event | Cadence (optimizer updates) | Content |
|---|---|---|
| Train log | every 10 | raw micro-batch losses, lr used, gradient norm, epoch position, timing |
| Subset validation | every 1,000 and at the end | EMA weights, fixed windows (start % 20 = 0 → 7 per val episode, 1,274 windows, 80 batches), fixed RNG seed; raw mean loss per window |
| Full validation | every 25,000 and at the end | EMA weights, all 22,022 val windows (1,377 batches) |
| Full checkpoint | every 5,000 and at the end, keep the last 2 | model, EMA, optimizer, scheduler, scaler, RNG, update count, data position, run record, action statistics |
| Inference export | every 10,000 and at the end | EMA-applied weights + run record + action statistics |

Checkpoint selection: the **final** EMA export is the reported policy. The validation loss is a denoising loss; it is logged for monitoring but not used to pick checkpoints, so no selection on held-out data takes place. Test-split evaluation happens once, after training, on that final checkpoint.

Inference: `r2r2r_dp.inference.R2R2RPolicy.load(checkpoint, weights="ema")` needs only the checkpoint file. It returns denormalised 10-D right-arm absolute targets for rows t+2 … t+32 relative to the newest observation row t.

## 7. Deviations from the paper (R2R2R §4.3, Table 8)

| # | Paper | This protocol | Reason / status |
|---|---|---|---|
| D1 | 20-D YuMi action space (Table 8) | right-arm 10-D output (`pred_right_only`), 20-D proprio | Left arm never grips and moves ≤ 6 mm per episode; the left gripper range is degenerate. Released-code-compatible single-arm variant. **Not an exact Table-8 reproduction.** |
| D2 | §4.3 text: absolute EE poses; Table 8: "delta EEF xyz" | absolute | TinyDP default and the authors' own command line (`--shared-cfg.no-use-delta-action`). Delta is rejected by the v2 path; a delta variant needs its own normalizer and protocol. |
| D3 | batch 64 | 16 × 4 accumulation | Measured CUDA OOM at 64 and 32 on a 24 GB 4090. GroupNorm, so no batch-statistic difference. Diffusion noise draws differ from a true 64 batch. |
| D4 | 1× GH200, ~3 h | 1× RTX 4090 | Throughput is in LIFECYCLE_TEST_RESULTS.md. |
| D5 | not reported | normalizer v2 (all targets), update-based schedule/EMA, fixed validation subset, seeded item randomness | Corrections of released-code behaviour (audit M1–M3). The released behaviour is kept reproducible via the v1 statistics and the legacy launcher, both labelled. |
| D6 | not reported | image ≈ 2 rows behind state (released pairing kept) | generator property, documented (M6) |

Retained without change: the 41 episodes with > 0.3 rad joint discontinuities (they set part of the right-rotation range; see 01 §6).

## 8. Launch commands (not executed beyond the bounded lifecycle test)

```bash
cd ~/Stanley_ws/real2render2real
N=50   # one of 50 100 150 1000 1461
OUT=~/Stanley_ws/r2r2r_policy_runs/train/baseline_v2_n$(printf %04d $N)_seed0
CUDA_VISIBLE_DEVICES=0 .venv-tinydp/bin/python policy_training/train_r2r2r_dp.py \
  --protocol policy_training/configs/baseline_v2.json --train-size $N --output-dir $OUT
# after an interruption (same command plus --resume):
CUDA_VISIBLE_DEVICES=0 .venv-tinydp/bin/python policy_training/train_r2r2r_dp.py \
  --protocol policy_training/configs/baseline_v2.json --train-size $N --output-dir $OUT --resume
```

Before launching, check that the chosen GPU is free (`nvidia-smi`) and that no concurrent experiment is assigned to it. The driver refuses more than one visible CUDA device.

**Workers:** `--num-workers 8` is a validated operational override. The data stream is invariant to worker count (bitwise on CPU), and the override is recorded in `run_config.json` → `runtime`. It raises throughput from 1,857 to 3,255 updates/h, with 21 GB host RAM at peak (LIFECYCLE_TEST_RESULTS.md §4). The configuration hash is unaffected.

## 9. Cost per run (projection from the 100-update lifecycle runs)

| | 4 workers | 8 workers |
|---|---|---|
| 100,000 updates incl. validation and checkpoints | ≈ 55 h | ≈ 32 h |
| GPU 0 | 15.4 GiB allocated, up to 23.5 GB on the device | 21.7 GB |
| Disk | ≈ 10.8 GB per run | same |

One run at a time on GPU 0. Nothing else should run on that GPU.

## 10. Recommended next experiment

1. **Prerequisite (process):** an independent review of branch `fix/tinydp-audit-gates`, consistent with this project's practice of auditing before expensive compute.
2. **Run 1 — `baseline_v2`, N = 1000, seed 0, 100k updates, `--num-workers 8`, GPU 0 (≈ 32 h).** N = 1000 is the paper's reported coffee-maker data point. Check `metrics.jsonl` after ≈ 2k updates (loss trend, `optimizer_step_skipped`, memory) and at each 25k full validation. Use `--resume` after any interruption. Only the final EMA export is reported.
3. **Afterwards:** test-split denoising loss / action error, computed once for that final export.
4. **Then:** N = 50, 100 and 150 with the identical protocol; seeds 1–2 if variance matters.
5. **In parallel, for any success-rate claim:** build a closed-loop evaluation harness (IsaacLab rollout of the YuMi coffee-maker task with the same 2-camera, 4-step and 2-row-lag observation contract). Without it, results are offline losses only.
