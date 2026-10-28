# TinyDP pre-training audit: repair report (M1–M7)

> **Superseded status (2026-10-28).** The independent Codex final audit of this report's commit `43ab363` ([CODEX_FINAL_AUDIT.md](CODEX_FINAL_AUDIT.md)) graded M2 and M4 **FAIL** and M1, M3, M5, M6, M7 **PARTIAL**, with eight MAJOR findings (F1–F8), and returned **NO-GO**. The PASS grades and the GO decision below were therefore premature; they are kept unchanged as the historical record of commit `43ab363`. The repairs, re-grading and current decision are in [CODEX_FINDINGS_REPAIR.md](CODEX_FINDINGS_REPAIR.md). In particular, the invariant "`micro_batches` = 4 × `update`" in §6 holds only when no optimizer step was skipped; the repaired invariant is `micro_batches` = 4 × `attempted_updates`, `attempted_updates` = `update` + `skipped_updates`.

Branch `fix/tinydp-audit-gates`, based on `repro/dp-training` @ `2a053668` (local HEAD = GitHub HEAD, verified 2026-10-27). The audit is in [CODEX_PRETRAIN_AUDIT.md](CODEX_PRETRAIN_AUDIT.md). The frozen result is [VALIDATED_TRAINING_PROTOCOL.md](VALIDATED_TRAINING_PROTOCOL.md), and the bounded GPU run is in [LIFECYCLE_TEST_RESULTS.md](LIFECYCLE_TEST_RESULTS.md).

**Not modified:** the R2R2R generator, `scripts/`, `real2render2real/`, every submodule (TinyDP stays clean at `76d8f9c`), the environments, the 1,825 source episodes, the dataset views, and the v1 statistics files. Phase 4 below verifies this. No dependency was installed or upgraded.

## 1. Corrections to earlier documents found in Phase 0

| Claim | Where | Finding |
|---|---|---|
| "`set_epoch` never called; batch order repeats every epoch" | 02_training_protocol.md §3; audit M2/M7 | **Incorrect.** TinyDP's `MultiEpochsDataLoader` wraps the batch sampler in `_RepeatSampler`, which calls `set_epoch(epoch)` before each pass. Batch order changes per epoch and equals `VideoSampler.set_epoch(e)` (`test_driver.py::test_released_multiepoch_loader_reshuffles_each_epoch`). What *is* fixed is batch **composition**: TinyDP's seeded window shuffle groups windows into the same micro-batches every epoch, and the windows that do not fill a final micro-batch are never sampled. The v2 driver keeps this inherited property and documents it. |
| "window starts j = 0…126" | 01 §5.1 (audit N1) | Window starts are 0…120; item starts are 0…126; the final target row is 120 + 6 + 32 = 158. Stated with separate symbols in VALIDATED_TRAINING_PROTOCOL.md §3 and pinned by `test_normalizer_v2.py::test_target_row_formula_matches_released_loader`. |

## 2. Findings, fixes and evidence

**Status key:** PASS means fixed and verified by execution. PARTIAL means fixed with a documented, remaining limitation. "Inherited" means the defect is in released TinyDP or the generator; "new" means it was introduced by the preparation layer.

### M1 — action normalisation misses odd-row targets (inherited, repeated by the v1 cross-check) — **PASS**

- **Fix:** `r2r2r_dp/normalizer_v2.py` derives the target-row set from the released index arithmetic. That covers every target the loader returns (rows 2–158, both parities), a superset of the loss rows. A test checks the formula against an instantiated TinyDP dataset.
- **Output:** `docs/policy_training/artifacts/normalizer_v2/action_statistics_v2_train_nXXXX.json` (written by `prepare_dataset.py stats-v2`). Each file carries `normalizer_version`, a provenance block (manifest sha256, subset, sha256 of the ordered train UIDs, code sha256) and an action-convention block with its own `config_sha256`.
- **v1 untouched:** the v1 files are unchanged and labelled as the released-normalizer compatibility baseline.

**Evidence on the real data** (`normalizer_v2_summary.json`): right-arm training target rows outside [−1, 1] (tolerance 1e-6).

| N | v1 (released, even rows) | v2 | rows checked | val rows outside v2 train bounds (information only) |
|---|---|---|---|---|
| 50 | 8, max \|z\| 1.028846 | **0**, max 1.0 | 7,850 | 83 / 28,574 |
| 100 | 8, max 1.080700 | **0** | 15,700 | 65 |
| 150 | 10, max 1.080700 | **0** | 23,550 | 56 |
| 1000 | 11, max 1.061949 | **0** | 157,000 | 6 |
| 1461 | 12, max 1.061949 | **0** | 229,377 | 3 |

The v1 column reproduces the audit's table exactly.

**Regressions** (`test_normalizer_v2.py`):
- **Audit odd-row spike:** v1 gives **5.0**; v2 gives **1.0**, and every item of the loader is within [−1, 1].
- **Real n0150 `ep_train_0076` row 15 r00:** v1 gives **−1.080700**; v2 is within [−1, 1].
- **Provenance and refusals:** membership mismatch and delta actions are rejected.

**Note:** validation trajectories may legitimately exceed the train bounds. They are not clipped. The DDPM `clip_sample` limits what the model can output.

### M2 — accumulation, scheduler, EMA and loss reporting (inherited, amplified by the 16×4 workaround) — **PASS**

**Fix:** a separate driver, `r2r2r_dp/driver.py`. TinyDP's `engine.train_one_epoch` and `script/train.py` are not used; the released dataset, sampler, collate, model, `add_weight_decay`, `NativeScalerWithGradNormCount`, `ModelEMA` and diffusers scheduler are reused unchanged.
- **One update** = exactly `accum` consecutive micro-batches of a continuous stream across epochs, so nothing is dropped.
- **Scheduler and EMA** each advance once per update; warm-up and the total are counted in updates.
- **Losses:** raw losses are logged (each micro-batch plus the mean), and validation reports the raw per-window mean.
- **Skipped steps:** GradScaler-skipped steps are detected and counted.

**Evidence:**
- `test_released_engine_counterexample` reproduces the audit's six-micro-batch case on the released loop:
  - 1 optimizer step, 6 scheduler steps, 6 EMA updates;
  - a tail gradient that is dropped by the next epoch's `zero_grad`;
  - validation loss reported as raw/4.
- `test_update_semantics` runs the v2 driver for 9 updates over 30 micro-batches per epoch (not a multiple of 4) and checks:
  - Adam step = scheduler step = EMA updates = 9; 36 micro-batches;
  - the stream is exactly epoch-0 batches 0–29, then epoch-1 batches 0–5, with `set_epoch` applied, and update 8 spans the boundary;
  - the logged lr equals the diffusers cosine with warm-up 3, per update;
  - the logged losses are exactly the model outputs.

**Note:** as in the released code, the diffusers warm-up starts at lr 0, so update 1 does not move the weights (documented, unchanged).

### M3 — checkpoint and resume (inherited) — **PASS on CPU; GPU continuation measured (see LIFECYCLE_TEST_RESULTS.md)**

**Fix:** full checkpoints contain model, EMA, optimizer, scheduler, GradScaler, Python/NumPy/torch (CPU+CUDA) RNG, update and micro-batch counters, the data position (epoch, batch) and the resolved run record. `--resume`:
- requires the same resolved configuration hash and identical action statistics;
- truncates the metrics log to the checkpoint (keeping a backup);
- continues the data stream at the exact micro-batch.

Writes are atomic (temporary file + rename).

**Evidence:** `test_interrupted_resume_equals_uninterrupted` compares two runs.
- **Run B:** stops after update 5, its u5 checkpoint is deleted (a crash after the u3 checkpoint), and it resumes from u3 **with 2 spawn workers instead of 0**.
- **Result:** run B equals the uninterrupted 6-update run A **bitwise** in model, EMA and optimizer tensors, and in scheduler state, data position and every train/validation log record.
- **Refusals:** a changed configuration, changed statistics or a missing checkpoint are rejected (`test_resume_rejects_changed_configuration`).

**Defect found and fixed during this repair (new, never committed):** the first version of per-item seeding reseeded the *training* process's global RNG when `num_workers=0`. Diffusion noise then depended on where items were produced; runs with 0 and 2 workers differed from update 1. Item seeding now restores the caller's RNG and seeds only the CPU generator inside a fork. Regression: `test_item_seeding_is_keyed_and_leaves_caller_rng_untouched`. The bitwise resume test above uses different worker counts, so it would catch a recurrence.

**Legacy path:** `train_tinydp.py` is retained as the v1 launcher of the released loop. It now refuses to run without `--legacy-v1-released-loop`. Its resume defect is unchanged (released code) and documented.

### M4 — launch-time enforcement (new, preparation layer) — **PASS**

**Fix:** `r2r2r_dp/protocol.py` resolves one protocol JSON into a TinyDP `ExperimentConfig` from defaults plus an allowlist, and pins the sim schema, right-only and absolute actions. `r2r2r_dp/launch_checks.py` runs before any dataset or model is built (see VALIDATED_TRAINING_PROTOCOL.md §2 for the checks).

**Evidence:** `test_launch_checks.py`, 33 tests, rejects:
- train=eval roots and eval=test roots;
- a swapped test/val root, a larger subset under the train-root name, and an unknown train size;
- statistics for another subset, v1-format statistics, numerically tampered v2 statistics, and absolute statistics with delta actions (both at the protocol level and after resolution);
- `load_config`, subsampling, WandB `log_name`, `dataset_root`, `pred_right_only=false`, `is_sim_data=false`, `eval_split=test`, unknown normalizer or timing variants, extra or missing schedule keys, and wrong value types;
- a manifest hash mismatch;
- mutated, stale (copied from another episode), missing and corrupted sidecars;
- a stray `.h5`, a missing episode, a changed source HDF5 and a missing JPEG frame;
- a non-empty output directory.

The CLI raises before `Trainer` (model allocation) is reached. All five real train sizes pass in 0.9–4.3 s (`artifacts/launch_checks_baseline_v2.json`).

### M5 — inference checkpoint contract (inherited import/loader defects + new artifact omission) — **PASS (offline contract only)**

**Fix:** `r2r2r_dp/inference.py` (`R2R2RPolicy`) loads one checkpoint file, either full or an EMA export, with correct imports.
- **Weights:** `ema` (the default, equal to what validation evaluates) or `raw` (full checkpoints only). An EMA export refuses `raw`.
- **Contract:** 20-D proprio and 4×2 camera input, 10-D right-arm absolute output. Denormalisation equals TinyDP's `unscale_action` on dims 10–19 of the embedded statistics.
- **Run directory:** it also holds `action_statistics.json` and `run_config.json`, but inference does not need them.

**Evidence:** `test_inference.py`. After 3 updates the checkpoint files are copied out and **the run directory is deleted**.
- The reloaded policy's normalised output equals the training model with EMA applied, bitwise, under the same seed.
- Denormalisation matches TinyDP within 1e-6.
- The EMA export and the full checkpoint give identical outputs; raw weights differ.
- Wrong proprio width and a wrong horizon are rejected.

The released `DiffusionWrapper` still fails to import (`ModuleNotFoundError: tinydp.dataset.dataset`; inherited, the submodule is unchanged). **Limitation:** this establishes the offline load/denoise/denormalise contract only. It is not deployment readiness: there is no robot, no rollout harness, and no latency or frame-timing validation (see M6).

### M6 — image/state timing (inherited from the generator) — **PARTIAL by design**

- **Decision:** the released pairing is kept for the baseline (`timing_variant: released_pairing`, the only accepted value). The exact frame, proprio, target and loss indexing is in VALIDATED_TRAINING_PROTOCOL.md §3.
- **Evidence:** `test_timing_contract.py` uses a fixture whose images lag the state by 2 rows.
  - The loader pairs image file j (which depicts state j−2) with proprio row j and targets j+2…j+32, for both cameras.
  - The motion-energy estimator used on the real data recovers lag 2, and lag 0 on an unlagged control.
- **Effect on the contract:** the policy is trained to map images ≈2 rows (≈33 ms at the nominal 60 Hz) older than its proprio to actions. At deployment, images must be supplied with the same delay relative to proprio, or the mismatch must be studied.
- **Why PARTIAL:** no aligned variant was built, because it requires an independent comparison and is outside a faithful baseline.

### M7 — frozen protocol, update-based cadence — **PASS**

- **Protocol:** `policy_training/configs/baseline_v2.json` is the single versioned protocol (deviations from the paper in VALIDATED_TRAINING_PROTOCOL.md §7).
- **Cadence:** validation and checkpoint intervals are in optimizer updates. Frequent validation uses a deterministic fixed subset of 1,274 windows. Full validation (22,022 windows) runs at milestones. The test split is never loaded.
- **Bounded variants:** the two lifecycle protocols differ from the baseline only in their `schedule`, which a test enforces. The bounded GPU lifecycle run is reported in LIFECYCLE_TEST_RESULTS.md.

## 3. Regression tests (executed)

| Command | Result |
|---|---|
| `.venv-tinydp/bin/python -m pytest policy_training/tests -q -p no:cacheprovider` | **89 passed** in 40 s (CPU only; CUDA hidden by `conftest.py`) |
| of which the original 37 (`test_schema` 15, `test_manifest_adapter` 6, `test_tinydp_pipeline` 7, `test_analysis` 3, `test_train_wrapper` 6) | all pass. Only change: `conftest.py` hides CUDA. |
| new: `test_normalizer_v2` 4, `test_launch_checks` 33, `test_driver` 8, `test_inference` 4, `test_timing_contract` 2, legacy-gate test in `test_train_wrapper` 1 | 52 pass |
| `prepare_dataset.py stats-v2 --manifest docs/policy_training/artifacts/manifest_v1.json --views-root VIEWS --v1-dir docs/policy_training/artifacts --out-dir docs/policy_training/artifacts/normalizer_v2` | 0 right-arm training targets out of bounds for all five N (§2 M1) |
| launch checks of `baseline_v2.json` for N = 50, 100, 150, 1000, 1461 on the real views | all pass, 0.9–4.3 s each (`artifacts/launch_checks_baseline_v2.json`) |

**Requirement mapping:**

| Phase-2 requirement | Test(s) | Result |
|---|---|---|
| 1. existing 37 tests | original files | PASS |
| 2. odd/even target normalisation | `test_normalizer_v2.py` (4, incl. the real row-15 case) | PASS |
| 3. train/val overlap rejected | `test_train_eval_overlap_rejected`, `test_eval_on_test_split_rejected`, `test_larger_subset_as_train_root_rejected` | PASS |
| 4. incorrect action conventions rejected | `test_absolute_statistics_with_delta_actions_rejected`, `test_protocol_rejections[pred_right_only / is_sim_data]` | PASS |
| 5. scheduler/EMA update counts | `test_update_semantics`, `test_released_engine_counterexample` | PASS |
| 6. interrupted/resumed training | `test_interrupted_resume_equals_uninterrupted` (CPU bitwise); GPU A/B run in LIFECYCLE_TEST_RESULTS §2 | PASS |
| 7. self-contained checkpoint inference | `test_inference.py`; real GPU checkpoint loaded on CPU with the run directory removed | PASS |
| 8. dataset source files unchanged | `verify_repair_invariants.py`: source fingerprint `dd23bd44…2f2bcd` unchanged, 0 view files modified | PASS |

## 4. Independent verification (Phase 4, adversarial)

| Check | Command / evidence | Result |
|---|---|---|
| Original counterexamples reproduced on the released code | M1 appendix → **5.0**; real n0150 row 15 → **−1.080700**; v1 out-of-bound counts equal the audit table exactly; M2 six micro-batches → 1/6/6 + dropped tail + val/4; M3 released `train.main` (`policy_training/repro_released_resume.py`) → epochs **[0,1,2,3,4,5]** instead of [5], saved EMA **3.0** instead of 9.0, keys `ema, epoch, model, optimizer`; M4 the legacy launcher still accepts train = eval roots (unchanged legacy `prepare`, now gated); M5 `DiffusionWrapper` import → `ModuleNotFoundError` | all reproduced; all absent from the v2 path |
| Every fix is load-bearing (mutation testing) | 21 single-line mutations re-introducing the defects into a scratch copy (control run on the unmutated copy: 27 targeted tests pass). Tool: `policy_training/tools/mutation_check.py`; results in `artifacts/mutation_check_v2.json`. | **21 / 21 detected** |
| R2R2R algorithm and baseline untouched | `git diff --ignore-submodules=dirty repro/baseline-datagen -- scripts real2render2real dependencies pyproject.toml uv.lock env_data_gen.sh` | empty |
| No submodule change | `git submodule status --recursive`; TinyDP worktree clean at `76d8f9c` | at pins; only the pre-existing `trajgen.egg-info` drift (documented since the data-generation audit) |
| Original generated data and statistics preserved | source fingerprint; `find views -newermt 2026-10-24` → 0 files; `git diff 2a05366 -- manifest_v1.json action_statistics_train_n*.json` → empty | preserved |
| Data leakage impossible under the validated launch configuration | launch checks for all five N (membership by TinyDP's own glob, disjointness by UID, path and content hash, statistics bound to the train-UID hash) and 33 negative tests; the driver never constructs a dataset on the test root | holds |
| Checkpoints independent of transient directories | `check_checkpoint_inference.py` on the real u20 checkpoint and export, with the run directory renamed away | loads and predicts; full = export bitwise |
| Reported optimizer steps = actual updates | `verify_repair_invariants.py`: Adam `step` = scheduler `last_epoch` = `update` and `micro_batches` = 4 × `update` for u20_B (20), u100_w4 (100) and u100_w8 (100); GradScaler-skipped steps counted (0) | holds |

Invariant report: `artifacts/repair_invariants_v2.json` (`all_ok: true`).

## 5. Inherited versus newly introduced defects

| Defect | Origin | Status in the v2 path |
|---|---|---|
| Even-row-only statistics (M1) | TinyDP `calculate_dataset_statistics`, copied by the v1 numpy cross-check (preparation) | replaced by normalizer v2 |
| Scheduler/EMA per micro-batch, dropped accumulation tail, val loss / accum (M2) | TinyDP `engine.py` | separate driver |
| Resume repeats epochs, loses EMA, no scheduler/scaler/RNG/data state (M3) | TinyDP `train.py` / `misc.py` | full-state checkpoints, bitwise-tested resume |
| Wrapper accepts overlapping roots and unbound statistics (M4) | preparation layer (`train_tinydp.py`) | launch checks; legacy launcher gated |
| `DiffusionWrapper` import, statistics file missing from the run dir, raw vs EMA mismatch (M5) | TinyDP (import, raw weights) + preparation (statistics not copied) | `R2R2RPolicy`, statistics embedded |
| Image ≈ 2 rows behind state (M6) | R2R2R generator | documented contract, kept |
| Per-item seeding reseeded the training RNG with `num_workers=0` | **this repair**, found by its own resume test before commit | fixed, regression-tested |
| `set_epoch` claim, window-start notation | preparation documents (and the audit) | corrected (§1) |
| Batch composition fixed across epochs; the windows that do not fill the last micro-batch are never sampled (N=50: 2, N=1461: 13 windows) | TinyDP sampler | kept, documented |
| Update 1 runs at lr 0 (diffusers warm-up starts at 0) | TinyDP/diffusers | kept, documented |

## 6. Status summary

| Finding | Status |
|---|---|
| M1 normalisation | **PASS** |
| M2 accumulation / scheduler / EMA / loss reporting | **PASS** |
| M3 checkpoint and resume | **PASS** (bitwise on CPU; on GPU every stateful quantity is exact and numerical differences stay at the run-to-run nondeterminism level) |
| M4 launch validation | **PASS** |
| M5 inference contract | **PASS** (offline load/denoise/denormalise; not deployment) |
| M6 timing | **PARTIAL** (released pairing frozen and tested; lag not corrected; no aligned variant) |
| M7 frozen protocol | **PASS** |

## 7. Remaining limitations

1. **No closed-loop evaluation exists.** There is neither a simulator rollout harness nor a robot, so no success rate, and no comparison with the paper's 53.3 % (coffee maker, DP, 1000 demos), can be produced. Validation and test losses are denoising losses.
2. **Not an exact Table-8 reproduction:** right-arm-only output, the absolute-vs-delta ambiguity, and batch 16 × 4 instead of 64.
3. **Timing:** images lag the state by ≈ 2 rows (≈ 33 ms nominal) in all training data. Real-camera latency is unknown.
4. **Reproducibility:** GPU runs are not bitwise reproducible (`cudnn.benchmark`, as released). Seeds fix the data stream, initialisation and noise draws, but kernel nondeterminism accumulates; one run per N gives no variance estimate.
5. **Cost:** throughput is measured over 100 updates only. A 100k-update run is projected at ≈ 32 h (8 workers) to ≈ 55 h (4 workers) per N, and long-run stability (memory, data loader, numerical) is not yet observed.
6. **Validation-split variety:** episodes are separated by trajectory, but all come from one human demonstration and shared generation batches, so held-out loss measures in-distribution generalisation only.
7. **Legacy path:** `train_tinydp.py` (released loop) still has the inherited M2/M3/M4 behaviour. It is gated behind `--legacy-v1-released-loop` and must not be used for reported results.

## 8. Decision

**GO for full TinyDP training under `baseline_v2`** (single GPU, one run at a time, via `train_r2r2r_dp.py`), conditional on the independent review this project requires before expensive compute.

- **Resolved:** every engineering blocker the audit raised (M1–M5, M7).
- **Frozen, not corrected:** M6, the timing contract (the released pairing, documented and tested).
- **Not covered by this GO:** any claim of policy success, deployment readiness or an exact Table-8 reproduction (§7).
- **Recommended first run:** VALIDATED_TRAINING_PROTOCOL.md §10.
