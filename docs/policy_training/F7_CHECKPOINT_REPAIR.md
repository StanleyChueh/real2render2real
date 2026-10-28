# Targeted F7 repair: strict optimizer-state validation on resume (Codex re-audit N1)

| | |
|---|---|
| Audited commit | `986a3fab059ff982d6fbc23b59875e1bae7857cf` (branch `fix/codex-final-findings`, unchanged) |
| Re-audit | [CODEX_REAUDIT_986a3fa.md](CODEX_REAUDIT_986a3fa.md): verbatim copy of `/tmp/r2r2r-codex-reaudit-986a3fa/CODEX_REAUDIT.md` (sha256 `35bfe0a1…`). Its small evidence files (logs, JSON, probe scripts; not the multi-GB fixtures or `.pt` files) are copied unchanged to `artifacts/codex_reaudit_986a3fa/original_evidence/`, with checksums in `artifacts/codex_reaudit_986a3fa/SHA256SUMS.original_evidence`. The `/tmp` originals were only read. |
| Repair branch | `fix/f7-optimizer-state-validation`, created from `986a3fa` |
| Implementation commit | `f4715cefbf91f5213a18177ed94ceba6d76ded8f` |
| Scope | One file of training code (`policy_training/r2r2r_dp/driver.py`), plus new tests, tools and evidence. The generator, `scripts/`, `real2render2real/`, all submodules (TinyDP clean at `76d8f9c`), `repro/baseline-datagen` (`334e83d`), the 1,825 source episodes, views, splits, manifest, v1/v2 normalizers, image-integrity certificate, protocol configs and the existing checkpoints are unchanged (§7). No dependency installed, no full training, no data generation. |

Codex graded F7 PARTIAL and the training path NO-GO because of **N1**: a checkpoint missing one parameter's whole Adam state was accepted, resumed, reported `finished=true` and produced a checkpoint that its own validator rejected. N2 (the mutation harness counts any non-zero exit as a detection) is MINOR and not the subject of this repair; see §8.

## 1. Reproduction on 986a3fa (before any edit)

All runs used the unmodified audited code (`artifacts/f7_repair/reproduction_on_986a3fa/code_under_test.json`).

**Codex's probes, vendored verbatim** into `policy_training/audit/codex_reaudit_986a3fa/`, with sha256 equal to the `/tmp` originals (`SHA256SUMS.original`). Only the output path of `checkpoint_effect.py` was redirected, in a scratch copy.

| Case (`checkpoint_effect.py`, fixture trained to u3, resume to u4) | Preflight problems | Outcome on 986a3fa |
|---|---|---|
| valid | `[]` | finishes u4, final checkpoint valid |
| missing_adam_state (one parameter's state deleted) | `[]` | **finishes u4, `finished=true`**; final checkpoint rejected by its own check: `optimizer step counts [1, 4] != update 4` |
| bad_moment_shape (`exp_avg` (2,3)) | `[]` | accepted; `RuntimeError` tensor size mismatch inside AdamW |
| invalid_rng (torch RNG length 1) | `[]` | accepted; `RuntimeError` RNG size 5048/5056 vs 1 at train start |
| nan_lr (group lr NaN) | `[]` | accepted; one optimizer step with lr NaN runs, then `ValueError` in strict-JSON logging |

`test_extra.py` gave **0 passed / 5 failed**: `optimizer_nan_lr`, `optimizer_missing_moment`, `optimizer_bad_moment_shape`, `scheduler_nan_lr` and `invalid_rng_payload` were all accepted by `load_full`. This is identical to Codex's `extra.log`. Logs: `reproduction_on_986a3fa/{checkpoint_effect.log, checkpoint_effect_results.json, test_extra.log}`.

**Entrypoint probe** (`tools/f7_resume_probe.py`, new). It uses the real `train_r2r2r_dp.main`, with checkpoints after every update and `keep_full_checkpoints=2`. It trains to u3 (files u2 and u3), corrupts u3 in place, then runs `--resume`. Results on 986a3fa (`reproduction_on_986a3fa/f7_resume_probe.json`):

| Case | Resume | Reported | Checkpoints | Valid u2 | Notes |
|---|---|---|---|---|---|
| missing_adam_state | **completed** | `final_update 4, finished true` | +u4, **−u2** | **deleted by rotation** | u4 self-invalid: `optimizer step counts [1, 4] != update 4`; model differs from intact continuation by 4.84e-05 (Codex: 4.839897155761719e-05) |
| scheduler_nan_lr | **completed** | `finished true` | +u4, −u2 | deleted | `_last_lr` NaN accepted silently |
| missing_exp_avg / _exp_avg_sq / _step | failed late | — | unchanged | kept | raw `KeyError` from AdamW after the checkpoint had been applied |
| bad_moment_shape, invalid_rng, optimizer_nan_lr | failed late | — | unchanged | kept | `RuntimeError` / `ValueError` after loading |
| missing_all_adam_state, nan_moment | rejected | — | unchanged | kept | the only two cases caught by the old checks |

**Validator census** (`audit/f7_repair/validator_cases.py`, 45 corruption cases of the new regression test applied to a u3 checkpoint written by the code under test). On 986a3fa, of the 40 cases applicable to its format, `checkpoint_problems` **accepted 29**, crashed on 3 (`KeyError: 'step'`, a non-scalar step, a NaN step), and rejected 8. The 5 history cases do not apply to that format (`reproduction_on_986a3fa/validator_cases_on_986a3fa.json`).

The new regression file run against 986a3fa: 58 of 58 tests fail (`reproduction_on_986a3fa/new_f7_tests_on_986a3fa.log`). Some fail only because the old Trainer has no optimizer history; the census above is the precise measure of what the old validator accepted.

## 2. Root cause

`Trainer.checkpoint_problems` at 986a3fa (`driver.py` lines 333–349) checked the optimizer state by:

1. comparing only the **number** of parameters per group;
2. comparing the **set** of `int(step)` values over the state entries **present** with `{update}`;
3. checking finiteness of floating tensors present.

A deleted entry leaves the set at `{u}`, so it passed. Missing keys, wrong moment shapes, dtypes and hyperparameters, `lr`, the scheduler's `_last_lr`/`_step_count`/`base_lrs`, grad-scaler settings and RNG payloads were never examined. AdamW lazily re-initialises a parameter with no state (`step 0`, zero moments), so the resume ran with a silently reset optimizer. Three more gaps let the damage escape the process:

- `load_full` applied components one after another, so a failure part-way left a partially loaded Trainer.
- `save_full` wrote whatever state it held, then rotated older checkpoints out.
- Malformed entries could make the validator itself raise instead of reporting a problem.

The validator could not do better than a uniform-step test, because the trainer kept no independent record of which parameters had been stepped.

## 3. Implementation (`policy_training/r2r2r_dp/driver.py`)

**Optimizer history.** `Trainer.param_steps[g][j]` holds the number of executed optimizer steps in which optimizer parameter *j* of group *g* had a gradient. That is exactly the Adam `step` AdamW gives it, since AdamW skips parameters whose `grad is None`. The count is recorded inside the existing `optimizer.step` wrapper, which only runs when GradScaler really calls the optimizer, so skipped steps are excluded as before. Full checkpoints gain `optimizer_history = {"param_names": [[names per group]], "steps": [[counts per group]]}`. The format string stays `r2r2r-dp-ckpt-v3` (additive field, see §6). Parameters need not share one history: a parameter without a gradient in some update legitimately carries fewer steps, or no state at all.

**Resume validation** (`checkpoint_problems`). It never raises: any unexpected structure is reported as `malformed checkpoint (…)`, and the function never touches runtime state.

| Component | Required |
|---|---|
| optimizer structure | exactly `{state, param_groups}`; per group the serialized parameter ids equal this optimizer's, in order |
| parameter mapping | `optimizer_history.param_names` equal this model's parameter names per group |
| group hyperparameters | same key set; `betas`, `eps`, `weight_decay`, `amsgrad`, `maximize`, `foreach`, `capturable`, `differentiable`, `fused`, `initial_lr` equal to this optimizer's; `lr` finite and equal to the schedule at the checkpoint's update (`base_lr × λ(u)`, rel. tol 1e-12) |
| history | one integer (not bool) in [0, u] per parameter; the most-updated parameter has exactly u steps (u = 0: all zero) |
| state coverage | an entry for every parameter with history > 0 and none for history 0; no unknown ids |
| state entry | keys exactly `{step, exp_avg, exp_avg_sq}` (+`max_exp_avg_sq` with amsgrad); `step` a finite scalar equal to that parameter's history; moments with the parameter's shape and dtype, finite, second moment ≥ 0 |
| scheduler | same key set; `last_epoch` = u and `_step_count` = u + 1 (ints); `base_lrs`, `lr_lambdas`, `verbose`, `_get_lr_called_within_step` as this scheduler's; `_last_lr` finite and equal to the schedule at u |
| grad scaler | `{}` only if this run's scaler is disabled; else exactly the 5 GradScaler keys; scale finite > 0, growth factor > 1, 0 < backoff < 1, interval int ≥ 1, 0 ≤ tracker < interval; settings equal to this run's when enabled |
| RNG | `python`, `numpy`, `torch` (+`cuda` on CUDA) present, nothing unknown; each payload accepted by a scratch generator (`random.Random`, `np.random.RandomState`, `torch.Generator`, CUDA `torch.Generator`), with exact uint8 length for torch/CUDA; the global RNG is untouched |

The existing checks (format, configuration hash, statistics, sample order, model/EMA tensors, counters, data position) are unchanged.

**Fresh versus corrupted.** A u = 0 checkpoint must have empty optimizer state and an all-zero history; a fresh `save_full()` at u = 0 is valid and resumes. A checkpoint claiming u > 0 needs a parameter stepped in every update and Adam state matching every recorded history. An empty state claiming three updates is rejected, and so is zeroing both state and history.

**Transactional loading** (`load_full`):

- An unreadable file or a non-dict payload raises `CheckpointError`.
- After validation, the components are applied by `_apply_checkpoint`, whose statements are unchanged from 986a3fa. If one raises, the previous model, EMA, optimizer, scheduler, scaler, counters, data position and history are restored, and `CheckpointError(… previous state restored)` is raised.
- If the rollback itself fails, the Trainer is marked `unusable` and `train()` refuses to start.

**No inconsistent checkpoint is published.** `save_full` runs the same validation on the state it is about to write. If anything fails, it raises `CheckpointError("refusing to write …")` before the file is created and before older checkpoints are rotated out; the session then ends with an exception, never with a completion summary.

**Entrypoint.** `train_r2r2r_dp.py --resume` already calls `load_full` before `truncate_metrics`, `train()` and the session summary, so a rejected resume now ends with `CheckpointError` (non-zero exit) before any update, without touching the metrics, the summaries or any checkpoint. The one file it does append to beforehand is `resume_history.jsonl`, which records the attempt (unchanged behaviour). No change to the entrypoint was needed.

**Compatibility of diagnostic drivers.** `param_steps` has a class-level default of `None` and is initialised when the step counter is installed, like the other counters. Codex's original suite builds Trainers with `Trainer.__new__` and needs this; the first draft broke 2 of its tests and was corrected before the commit (§5).

## 4. Regression tests (`policy_training/tests/test_f7_checkpoint_state.py`, 58 tests)

Real Trainer on the synthetic CPU fixture, with a real `torch.amp.GradScaler('cpu')`; no optimizer mocking.

| Requirement | Test(s) |
|---|---|
| valid resume, bitwise equal to uninterrupted (model, EMA, every Adam tensor, param groups, scheduler, scaler, data position, history, counters, RNG) | `test_valid_checkpoint_has_full_history_and_resumes_bitwise` |
| missing complete Adam state (one parameter / all parameters) | `…rejected_before_resume[missing_one_adam_state]`, `[missing_all_adam_state]` |
| missing `exp_avg` / `exp_avg_sq` / `step`; extra key | `[missing_exp_avg]`, `[missing_exp_avg_sq]`, `[missing_step]`, `[extra_state_key]` |
| wrong tensor shapes / dtype / step form | `[exp_avg_shape]`, `[exp_avg_sq_transposed]`, `[exp_avg_dtype]`, `[step_not_scalar]`, `[step_fractional]`, `[step_behind]` |
| incompatible parameter groups / metadata | `[group_moved_param]`, `[group_count]`, `[group_weight_decay]`, `[group_betas]`, `[group_amsgrad]`, `[group_missing_key]`, `[group_initial_lr]`, `[group_lr_off_schedule]`, `[unknown_param_id]`, `[not_an_optimizer]`, `[history_names_permuted]` |
| non-finite optimizer values | `[step_nan]`, `[exp_avg_nan]`, `[exp_avg_sq_inf]`, `[exp_avg_sq_negative]`, `[group_lr_nan]` |
| scheduler / scaler / RNG (Codex's extra cases and more) | `[scheduler_last_lr_nan]`, `[scheduler_step_count]`, `[scheduler_base_lrs]`, `[scheduler_missing_key]`, `[scaler_missing_key]`, `[scaler_tracker_range]`, `[scaler_empty]`, `[scaler_backoff]`, `[rng_torch_length]`, `[rng_torch_dtype]`, `[rng_python]`, `[rng_numpy]`, `[rng_unexpected_key]` |
| fresh checkpoint with zero updates is legitimate; same file claiming history is rejected | `test_fresh_zero_update_checkpoint_is_legitimate` |
| claims completed updates while lacking history | `[missing_all_adam_state]`, `[history_all_zero]`, `[history_step_too_large]`, `[history_bool]`, `[history_missing_without_uniform_state]` |
| per-parameter histories, not one shared step count; interrupted vs uninterrupted with non-uniform history | `test_nonuniform_parameter_histories_are_validated_per_parameter`: one parameter has no gradient in attempts 1 and 3, so history 2 of 4, no state at u1; resume from u2 equals uninterrupted bitwise; forcing step 4, dropping the history or its entry is rejected |
| pre-repair checkpoints (no history) keep loading | `test_checkpoint_without_history_written_before_the_repair_still_loads` |
| rejected before resumed optimization begins | every corruption case asserts model, EMA, optimizer (empty), scheduler, scaler, counters, `optimizer_calls == 0`, history and pending RNG unchanged |
| rejected resume cannot report completion or overwrite valid checkpoints | `test_entrypoint_resume_rejects_and_preserves_valid_checkpoints[5 cases]` (real `--resume`: `CheckpointError`; every file except `resume_history.jsonl` byte-identical); `test_inconsistent_live_state_is_never_published` (in-process loss of a parameter's state: `refusing to write`, no new or `.tmp` file, older checkpoints unchanged, no checkpoint record after u2) |
| transactional apply | `test_failure_while_applying_restores_previous_state`, `test_failed_rollback_makes_the_trainer_refuse_to_train`, `test_unreadable_or_non_dict_checkpoint_is_a_checkpoint_error` |

Every corruption case also asserts that the reported problem names the defect where that is checkable (`EXPECT` table), not merely that something was rejected. No existing test was modified.

## 5. Results at f4715ce

Environment for all CPU commands: `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=$PWD/dependencies/tinydp:$PWD/policy_training:$PWD/policy_training/tests`. Logs in `artifacts/f7_repair/verification/`.

| Check | Command | Result |
|---|---|---|
| focused F7 tests | `.venv-tinydp/bin/python -m pytest -q -p no:cacheprovider policy_training/tests/test_f7_checkpoint_state.py` | **58 passed**, exit 0 |
| full repository suite | `… -m pytest -q -p no:cacheprovider policy_training/tests` | **252 passed** (194 previous + 58 new), exit 0 |
| Codex re-audit extra suite (verbatim) | `… -m pytest -q policy_training/audit/codex_reaudit_986a3fa/test_extra.py` | **5 passed** (986a3fa: 0/5) |
| Codex re-audit continuation probe (verbatim, output path redirected) | `checkpoint_effect.py` | valid finishes u4 with a valid checkpoint; missing_adam_state, bad_moment_shape, invalid_rng and nan_lr: preflight reports the defect, `load_full` raises `CheckpointError` |
| Codex original adversarial suite (43ab363, verbatim) | `bash policy_training/audit/run_codex_final_suite.sh OUT` | **25 passed / 3 failed**, the same three inherited `rot_6d_to_rot_mat` cases as at 986a3fa (upstream helper, intentionally unmodified; §8) |
| validator census | `… policy_training/audit/f7_repair/validator_cases.py` | **45 / 45 rejected** (986a3fa: 29 accepted, 3 crashes, 8 rejected of 40) |
| entrypoint probe | `… policy_training/tools/f7_resume_probe.py OUT.json` | `all_ok`: valid resume completes, model difference to intact 0.0; all 10 corruptions `CheckpointError` before any update, no checkpoint added or removed, u2 and `metrics.jsonl` byte-identical, no session summary |
| mutation check (48 previous + 11 F7) | `… policy_training/tools/mutation_check.py OUT.json` | control **128 passed**; **59 / 59 detected**, 0 timeouts, 0 inapplicable; every F7 mutant fails at the intended assertion (`first_error` field, e.g. `missing_one_adam_state accepted`) |
| legacy checkpoints | `… policy_training/tools/legacy_ckpt_compat.py OUT.json lifecycle_v3/u20_A/run_config.json <5 ckpts>` | 5/5 lifecycle_v3 checkpoints (u10/u20 of A and B, relocated copy): no problems, applied, file hashes unchanged |
| CUDA skip probe | `CUDA_VISIBLE_DEVICES=0 … policy_training/tools/cuda_skip_probe.py OUT.json` | all 5 checks pass (5 attempts / 4 updates / 1 skip, exact resume, model diff 0.0), peak 1.16 GB |
| CUDA rejection probe | `CUDA_VISIBLE_DEVICES=0 … policy_training/audit/f7_repair/cuda_rejection_probe.py OUT.json` | valid applied; CUDA RNG missing/wrong length, missing Adam entry, empty scaler, NaN scale all rejected with state unchanged |
| repository invariants | `… policy_training/verify_repair_invariants.py --base 2a05366 --checkpoint lifecycle_f7/u20_B/ckpt_full_u0000020.pt --out OUT.json` | `all_ok`: protected paths equal `repro/baseline-datagen`, submodules at pins, TinyDP clean, v1 statistics/manifest and source fingerprint unchanged, views unmodified, launch checks pass for all five N, resumed checkpoint counters consistent |
| full-width GPU lifecycle (bounded, 20 updates) | `bash policy_training/audit/f7_repair/run_lifecycle_f7.sh ROOT` | A, B1, B2, C1 exit 0; corrupted resume C2 exit 1 before any update, files unchanged; A ≈ B at GPU-nondeterminism level (§6.2) |

## 6. Checkpoint / resume consistency

### 6.1 CPU (bitwise)

- **Uniform history.** Resume from u3 to u4 equals the uninterrupted run bitwise: model, EMA, all 160 Adam entries, param groups, scheduler, scaler, data position, counters, history and RNG.
- **Non-uniform history.** Resume from u2 equals uninterrupted bitwise, including the parameter that carries 2 of 4 steps and has no state at u1.
- **Existing tests.** The earlier resume tests (`test_interrupted_resume_equals_uninterrupted` with 0 → 2 workers, `test_resume_after_skip_equals_uninterrupted`) still pass, now under the stricter validation and the write-time self-check.

### 6.2 Full-width GPU lifecycle (real data, N = 50)

Bounded: 20 updates, GPU 0 only after confirming it was idle (18 MiB used, no compute process, no COAST job), real entrypoint, `configs/lifecycle_u20_v3.json`, N = 50, full model, bf16, real `GradScaler('cuda')`. Script: `policy_training/audit/f7_repair/run_lifecycle_f7.sh ROOT [STAGES]`, which wraps `tools/run_lifecycle.sh`. Run outputs are in `~/Stanley_ws/r2r2r_policy_runs/lifecycle_f7/` (not committed); logs, metrics and comparisons are in `artifacts/f7_repair/verification/lifecycle_f7/`.

| Session | What | Exit | Updates | GPU 0 peak |
|---|---|---|---|---|
| A | 20 uninterrupted updates | 0 | 0 → 20, `finished` | 21,703 MiB |
| B1 / B2 | stop after 10 / `--resume` to 20 | 0 / 0 | 0 → 10 / 10 → 20, `finished` | 21,555 / 21,767 MiB |
| C1 / C2 | separate stop after 10 / one Adam entry deleted from its u10 checkpoint, then `--resume` | 0 / **1** | 0 → 10 / **none** | 21,505 / 1,565 MiB |

- **C2 (negative case).** It ended after 8.3 s, before any update, with `CheckpointError: …/ckpt_full_u0000010.pt: optimizer: no Adam state for 1 parameters with recorded updates, e.g. ['vision_encoder.bn1.weight (10 recorded updates)']`. Every file in the run directory except `resume_history.jsonl` is byte-identical before and after (`logs/C.sha256.{before,after}`). There is no new checkpoint, no truncated metrics and no session summary.
- **A vs B** (`compare_A_B.json`):
  - Identical: the learning rate at all 20 updates, the scheduler state, data position, update/micro-batch counters (u10: 10/40, u20: 20/80), validation window counts and the configuration hash.
  - Train loss differs by at most 2.5e-4. Full validation at u20: A 1.723201, B 1.723193.
  - Model weights differ by at most 4.2e-4 at u10 (*before any resume*) and 4.4e-4 at u20. EMA differs by at most 8.3e-7. Adam steps are equal; the moments differ by at most 1.2e-4.
  - These are the same GPU kernel-nondeterminism levels as the v3 lifecycle (2.9e-4 / 3.4e-4); bitwise GPU equality is not claimed.
- **Checkpoint validation** (`checkpoint_validation.json`). All 5 new full-width checkpoints (A and B at u10/u20, and X) carry `optimizer_history`. In each, 254 of 254 parameters have state with step = update and the recorded history equals that step. Each validates with no problems and applies on a fresh Trainer. The save-time self-check passed at every save.
- **Cost of the self-check.** Checkpoint saves took 1.24–1.43 s, against 1.26–1.51 s for the same saves in lifecycle_v3, so the cost is not measurable at this scale.
- **Script fault, disclosed.** The first version of the script had a defect in stage C. Its corruption step called `torch.load` without `map_location` under `CUDA_VISIBLE_DEVICES=''` and failed, and the script continued without `set -e`. That session therefore resumed an *uncorrupted* copy of B's u10 checkpoint and completed normally. It is kept, renamed `u20_X_unintended_second_resume_of_B1_copy` (logs `X_unintended_second_resume.*`), and is not counted as the negative test. As a by-product, two resumes from the same u10 file (B vs X, `compare_B_X.json`) are bitwise equal at u10 and differ by 2.1e-5 at u20. The script was fixed (CPU `map_location`, abort on failure, C as its own session) and only stage C was rerun. A and B used the same commands in both versions.
- **Not re-run at full width.** Inference export and reload, and AMP overflow. Inference code is unchanged by this repair. Overflow and skip accounting on CUDA is covered by the small-model `cuda_skip_probe` (§5); at full width it was shown at 966d459 (CODEX_FINDINGS_REPAIR.md §4).

## 7. Preservation

- `git diff 986a3fa f4715ce` touches only `policy_training/` (driver, tests, tools, vendored audit probes) and `docs/policy_training/artifacts/f7_repair/`.
- The protected-path diff (`scripts real2render2real dependencies pyproject.toml uv.lock env_data_gen.sh .gitmodules`) is empty.
- `policy_training/configs`, `artifacts/normalizer_v2`, `manifest_v1.json`, `image_integrity_v1` and `codex_final_43ab363` evidence are unchanged.
- Submodules are at their pins; TinyDP is `76d8f9c` and clean. `dependencies/trajgen` shows the same pre-existing egg-info drift as before.
- `repro/baseline-datagen` is still `334e83d4d1be7a4a5e78768cd5344051d4a892fa`.
- The existing `lifecycle_v2`/`lifecycle_v3` checkpoints were only read; their hashes are recorded in `legacy_v3_checkpoint_compat.json`.

## 8. Remaining limitations

1. **Coordinated forgery is not detectable.** Consider a checkpoint where one parameter's Adam state is deleted *and* its history entry is set to 0, while another parameter still carries u steps. It is indistinguishable from a parameter that never received a gradient, and it is accepted. Single-record corruption (Codex's case) is detected. For this policy every one of the 254 optimizer parameters is stepped in every update, which the existing real checkpoints and the fixture show. A stricter policy-specific rule ("all parameters must have history u") would close this, but the task asked not to assume identical histories.
2. **Values are checked for form, not authenticity.** The validator checks structure, shapes, dtypes, finiteness, domains and the agreement of independent records (history, Adam steps, scheduler, counters, data position). It cannot tell a finite, in-range moment tensor that was altered from the one training produced. A content digest or signature would be a separate feature.
3. **Legacy checkpoints get the uniform rule.** Checkpoints without `optimizer_history` (986a3fa and earlier) are validated with every parameter at step u. All 5 existing lifecycle_v3 checkpoints satisfy this. A legacy checkpoint of a model with gradient-free parameters would be rejected rather than guessed at (fail-safe).
4. **A rejected resume still leaves one log entry.** `resume_history.jsonl` records the attempt even when the resume is rejected. It is a log of attempts, not of completions.
5. **N2 (MINOR) is not resolved.** The harness still counts any non-zero pytest exit, and a timeout, as a detection. It now records each mutant's first failure line, so the reason for every detection can be audited. For the 11 new F7 mutants, those lines show the intended assertions.
6. **Inherited rotation helper.** The 3 failures in Codex's original suite exercise TinyDP's own `rot_6d_to_rot_mat` in the unmodified submodule. The repair layer's `rotation.rot6d_rows_to_matrix` is used for physical poses. This limitation is unchanged and documented in CODEX_FINDINGS_REPAIR.md §5.
7. **Bitwise equality is CPU-only.** GPU resume equality is at kernel-nondeterminism level (§6.2), not bitwise. No convergence, robot-success or paper-reproduction claim follows from these lifecycle tests.

## 9. Checkpoint compatibility implications

| Checkpoint | Continuation (`load_full`) | Inference |
|---|---|---|
| v3 written by f4715ce (with `optimizer_history`) | yes, strict per-parameter rules | yes |
| v3 written by 966d459 / 986a3fa (no history) | yes, if every parameter has full uniform Adam state (all existing lifecycle_v3 checkpoints do) | yes |
| v3 with corrupted optimizer, scheduler, scaler or RNG state | rejected with `CheckpointError` before any state changes | yes (inference reads only weights and statistics) |
| v2 (`r2r2r-dp-ckpt-v2`) | rejected (unchanged) | yes (unchanged) |

Older code (986a3fa) ignores the new `optimizer_history` key, so new checkpoints still load there under its weaker checks. The configuration hashes, protocol v3 and the data are unchanged, so no launch artifact needs regenerating.

## 10. Decision

- **F7 / N1:** repaired, but this is a self-assessment, not an independent grade.
  - Every Codex counterexample is now rejected before any state changes: the entire-Adam-state deletion, the missing moment, the wrong moment shape, NaN `lr`, NaN `_last_lr` and the invalid RNG payload.
  - A rejected resume exits non-zero and leaves every checkpoint, the metrics log and the session summaries untouched.
  - No inconsistent checkpoint can be written.
  - Valid resumes, including non-uniform per-parameter histories, remain bitwise exact on CPU and at nondeterminism level on GPU.
  - Existing v3 checkpoints still load.
- **F1–F6, F8:** unchanged and still passing (252-test suite; Codex's original suite 25/3 with only the inherited rotation-helper failures; 59/59 mutations).
- **Remaining blockers:**
  1. Independent Codex re-audit of this branch.
  2. The open MINOR N2 (§8.5).
  3. Optionally, the stricter all-parameters history rule (§8.1).
- **Recommendation:**
  - **GO** for independent Codex review of `fix/f7-optimizer-state-validation`.
  - **NO-GO** for full TinyDP training until that re-audit confirms F7; after a passing re-audit, a **CONDITIONAL GO** for the baseline run under the documented scientific limits.
  - No training was started.
