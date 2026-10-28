# Repair of the Codex final-audit findings F1–F8 (protocol v3)

> **Update (F7):** Codex's re-audit of `986a3fa` ([CODEX_REAUDIT_986a3fa.md](CODEX_REAUDIT_986a3fa.md)) graded F7 PARTIAL (N1: missing Adam state accepted on resume). The targeted repair is documented in [F7_CHECKPOINT_REPAIR.md](F7_CHECKPOINT_REPAIR.md) (branch `fix/f7-optimizer-state-validation`). The F7 rows below describe the state at 966d459/986a3fa.

| | |
|---|---|
| Audited commit | `43ab363058c4c9a5a45207b1414315320ef8438d` (branch `fix/tinydp-audit-gates`, kept unchanged as the reference) |
| Pre-repair reference | `2a0536689e81b1e01876c8cdf0550fa9fae20227` |
| Audit | [CODEX_FINAL_AUDIT.md](CODEX_FINAL_AUDIT.md): verbatim copy of `/tmp/r2r2r-codex-final-evidence/CODEX_FINAL_AUDIT.md`. All evidence files are copied unchanged to `artifacts/codex_final_43ab363/original_evidence/`, with their sha256 in `artifacts/codex_final_43ab363/SHA256SUMS.original_evidence`. The `/tmp` originals were only read. |
| Repair branch | `fix/codex-final-findings`, created from `43ab363` |
| Scope | Only the repair layer `policy_training/` and `docs/policy_training/` changed. No changes to the generator, `scripts/`, `real2render2real/`, any submodule (TinyDP stays at `76d8f9c`, clean), the environments, the 1,825 source episodes, the dataset views, the v1/v2 statistics files or the manifest. No dependency was installed. No full training was run. |

## 1. Reproduction before any change

Codex's independent suite is vendored verbatim as `policy_training/audit/codex_final_43ab363/{test_independent_final.py,conftest.py}`; the sha256 values recorded in `SHA256SUMS.original` are checked on every run. `policy_training/audit/run_codex_final_suite.sh OUT` runs it against the current checkout. Its only transformation is to replace the hard-coded evidence directory (`/tmp/r2r2r-codex-final-evidence`, which the suite's `observe()` and `conftest.py` write to) with `OUT`, in a temporary copy, so that Codex's evidence is never overwritten. The script prints the diff to prove nothing else changes.

Run on the audited commit `43ab363` before any edit (the vendored copy under `policy_training/audit/` was the only untracked addition, hence `"dirty": true` in `suite_exit.json`):

| | Codex (host run) | Reproduction |
|---|---|---|
| Result | 6 passed, 22 failed, exit 1 | **6 passed, 22 failed, exit 1** (same 22 test IDs) |
| Observations | 28 records (`observations_host.json`) | 28 records, **all 28 identical** to Codex's (timings excluded) |

Evidence: `artifacts/codex_final_repair/reproduction_on_43ab363/`. Examples of the reproduced values: the AMP skip run had 0 optimizer steps, 1 scheduler step, 1 EMA update, 1 reported update and scale 65536 → 32768; `normalized(NaN)` gave 0; a NaN sidecar gave "0 of 168 rows out of bounds"; the missing EMA key `vision_encoder.conv1.weight` was accepted.

## 2. Findings matrix and resolution

| # | Codex tests (failing on `43ab363`) | Responsible source | Root cause | Correction | Regression tests (new) | Status |
|---|---|---|---|---|---|---|
| **F1** AMP skip consumes the update budget | `test_amp_skipped_step_does_not_advance_successful_update` | `driver.py` `Trainer.train` (`43ab363` lines 284–289) | Success was inferred from a decrease of the grad scale; scheduler, EMA and update counter advanced unconditionally after every accumulation group | Calls that actually reach `optimizer.step()` are counted by an instance-level wrapper. This works for subclasses and does not rely on GradScaler's return value, which is `None` for AdamW either way. A skipped group consumes its micro-batches, logs `train_skipped`, and does **not** advance the scheduler, EMA, update counter or validation/checkpoint cadence. Bounded retry: abort (`SkippedStepError`) after 20 consecutive or 1,000 total skips. New counters `attempted_updates` and `skipped_updates` are stored in the checkpoint, and `finished` means successful updates = total. Zero-lr steps count, because they execute. Invariants: `attempted = updates + skipped` and `micro_batches = accum × attempted`. | `test_codex_final_training.py`: `test_skipped_step_advances_nothing_but_data`, `test_zero_learning_rate_step_is_a_successful_update`, `test_persistent_overflow_aborts_without_counting`, `test_resume_after_skip_equals_uninterrupted`; CUDA probe §5 | **Resolved** |
| **F2** non-finite normalization hidden | `test_normalization_rejects_nonfinite[nan/inf/neg_inf]`, `test_bound_builder_cannot_certify_nan_data` | `normalizer_v2.py` `normalized`, `compute_bounds`, `coverage` | A blanket `nan_to_num` mapped NaN→0 and ±Inf→±max. NaN bounds make every comparison false, so coverage reported 0 rows out of bounds. | `require_finite` / `require_bounds` on actions, bounds and inputs (finite, shape (20,), min ≤ max) before any reduction or scaling. TinyDP's mapping for finite zero-range dimensions is kept, and coverage still counts a value ≠ the constant as out of bounds. Strict JSON `load_statistics` / `dump_statistics`. | `test_normalized_rejects_nonfinite_actions_and_bounds[×3]`, `test_constant_dimension_mapping_kept_and_counted`, `test_nan_sidecar_cannot_be_certified`, `test_statistics_json_is_strict` | **Resolved** |
| **F3** metadata, order and layout not enforced | `test_manifest_metadata_rejected_even_when_re_pinned[version/counts/salt]`, `test_changed_view_order_rejected`, `test_wrong_action_layout_rejected` | `manifest.py` `check_manifest`; `launch_checks.py` `_check_root`, `check_statistics`; `driver.py` `load_full` | The manifest was checked by `assert` for overlap and prefixes only. Episode directory names, which fix TinyDP's sorted sample order, were not checked. Statistics were compared on selected fields only, and resume compared only `batches_per_epoch`. | `ManifestError` (explicit exceptions, effective under `-O`) for: required/optional keys; version 1 (no bool); counts = episodes per split; unique contiguous ranks; split/rank equal to the assignment implied by the salt; listing order; normalized relative `rel_path` (no `..`); positive nested subsets including the full split. Strict JSON. Launch: canonical names `ep_<split>_<rank:04d>`; split roots and episode directories must not be symlink aliases; realpaths are reported. Statistics: exact key set, full `config` equality (incl. 20-D layout), recomputed `config_sha256`, `n_episodes` and `n_target_rows`. The trainer stores `train_order_sha256` (digest of TinyDP's actual `(uid, start, end)` window order) in checkpoints, and resume requires equality. | `test_manifest_semantics[×10]`, `test_repinned_bad_manifest_rejected_at_launch`, `test_renamed_view_rejected`, `test_symlinked_episode_alias_rejected`, `test_symlinked_split_root_rejected`, `test_statistics_declared_config_enforced[×7]` (incl. wrong layout with a *consistent* digest), `test_corrupt_checkpoint_rejected_before_any_state_changes[order]`, `test_critical_checks_survive_python_optimize` | **Resolved** |
| **F4** JPEG content not enforced | `test_corrupt_jpeg_rejected_before_model` | `launch_checks.py` (frame names only) | The image identity was not bound to anything | New `image_integrity.py`. A versioned integrity manifest (`r2r2r-dp-images-v1`) is bound to the episode manifest by sha256 and pinned by sha256 in the protocol. It is built once, read-only: every frame is fully decoded by PIL from the same bytes that are hashed, and must be an RGB JPEG of 1280×720 with exactly frames `0000…T−1`; per camera it stores sha256 over (name, sha256(bytes)). Every launch recomputes all camera digests of all three split roots from the current bytes. Byte equality transfers the decode certificate, so frames are not re-decoded. Real data: 1,825 episodes, 587,650 frames, all valid (build 3 min 14 s); file `artifacts/image_integrity_v1/image_integrity_v1.json`, sha256 `63eecde5…814e`. | `test_corrupt_wrong_size_and_missing_frames_rejected` (corrupt bytes, valid JPEG at a wrong resolution, restore, missing frame), `test_builder_decodes_and_checks_resolution` (wrong size, truncated, grayscale), `test_integrity_file_pinned_and_bound_to_manifest`, `test_launch_failure_exits_nonzero` | **Resolved** |
| **F5** invalid numeric protocol values | `test_protocol_invalid_numerics_rejected[negative_clip/fractional_batch/nan_lr/bool_updates]` | `protocol.py` `_set`, `validate_protocol` | `isinstance(True, int)`; no domains for TinyDP fields; `grad_clip` unchecked | Field-specific domain table for every allowed TinyDP field, checked on the *resolved* value (protocol or released default) before any allocation. Integers exclude booleans; reals must be finite; lr > 0; weight decay, noise and gripper weight ≥ 0; positive counts and dimensions; workers ≥ 0; seeds in [0, 2³²−1]; exactly 2 cameras; image size ≥ 32. Cross-field: inference steps ≤ training steps, and `num_pred_steps` divisible by 2^(levels−1) of `down_dims`. Schedule: positive integer counters, warm-up in [0, total), EMA in (0, 1), `grad_clip` finite > 0, seeds in range. Data: path strings and lowercase-hex sha256 pins. `train_size` must be a positive int. Strict JSON. Bumped `protocol_version` to `r2r2r-dp-protocol-v3` (the data section gained `image_integrity` and `image_integrity_sha256`). The CLI validates `--num-workers` ≥ 0 and `--stop-after-updates` ≥ 1. | `test_invalid_protocol_values_rejected[×28]`, `test_supported_zero_values_accepted`, `test_protocol_file_is_strict_json`, `test_frozen_configs_pin_the_committed_integrity_file` | **Resolved** |
| **F6** validation not transactional | `test_validation_exception_restores_weights_and_mode`, `test_validation_nonfinite_loss_rejected` | `driver.py` `evaluate`, `save_inference` | Restoration was not in `finally`; mode was forced to train; losses were not checked | `evaluate`: raw weights and the *previous* train/eval mode are restored in `finally`. Loader workers are shut down in `finally`. A non-finite batch loss or mean raises `FloatingPointError`; empty batches or windows raise `ValueError`. Failures propagate, so nothing is logged as a result and training stops with a non-zero exit. `save_inference`: same `finally` restore; a non-finite EMA export raises. | `test_validation_failure_restores_weights_and_mode[train/eval]`, `test_nonfinite_and_empty_validation_raise`, `test_nonfinite_validation_aborts_training`, `test_export_failure_restores_raw_weights` | **Resolved** |
| **F7** incomplete EMA checkpoint loads | `test_checkpoint_missing_ema_parameter_rejected` | `driver.py` `load_full`; inherited `ModelEMA.load_state_dict` ignores absent keys | No validation before mutation | `checkpoint_problems()` checks everything before any state is changed: format `r2r2r-dp-ckpt-v3` (v2 files load for inference only); required keys; config hash; statistics; sample order. Model and EMA: exact key sets, shapes, dtypes and finite values. Counters: non-negative ints (no bool), `attempted = updates + skipped`, `micro = accum × attempted = epoch × batches_per_epoch + batch`. Scheduler `last_epoch` = update. Optimizer: group sizes, Adam step counts = {update}, finite state. Grad scale positive and finite. Complete RNG state. Raises `CheckpointError`. | `test_corrupt_checkpoint_rejected_before_any_state_changes[×17]` (also asserts nothing was mutated), `test_valid_checkpoint_loads` | **Resolved** |
| **F8** non-finite inference; unsafe inherited rotation | `test_inference_nonfinite_input_rejected`, `test_inherited_physical_rotation_conversion_is_valid_or_rejects[nonunit/collinear/zero]` | `inference.py` `predict`; pinned `tinydp.dataset.utils.rot_6d_to_rot_mat` | No input, weight, bound or output checks. The inherited helper skips Gram–Schmidt for orthogonal non-unit rows and returns NaN or zero matrices for degenerate rows. | Adapter: rejects non-finite weights; non-finite, reversed or non-20-D bounds (finite constant dimensions keep TinyDP's denormalisation); non-tensor, non-float or non-finite inputs; shapes other than (B ≥ 1, 4, 2, 3, S, S) with S = `image_size` and (B, 4, 20); inference steps outside [1, `num_train_diffusion_steps`] or bool; negative seeds. A non-finite output raises. New `rotation.rot6d_rows_to_matrix`, outside the submodule: TinyDP row convention; normalise row 1, Gram–Schmidt row 2, rejection of non-finite, short or (nearly) collinear rows, then checks R Rᵀ = I and det R = +1. `R2R2RPolicy.right_poses` uses it. **The pinned TinyDP helper is unchanged** (the submodule may not be modified), so Codex's three tests of that helper still fail, as they must. | `test_inference_rejects_invalid_inputs[×10]`, `test_nonfinite_output_and_weights_rejected`, `test_right_poses_are_proper_rotations`, `test_rotation_converter_cases` (Codex's three inputs, plus a round trip against SciPy on 64 random rotations via TinyDP's own `quat_to_rot_6d`), `test_repair_layer_never_uses_inherited_rotation_helper` | **Resolved in the repair layer.** Inherited helper: documented limitation (§6) |

The MINOR documentation finding is also resolved. Superseded-status banners were added to [AUDIT_REPAIR_REPORT.md](AUDIT_REPAIR_REPORT.md) and [LIFECYCLE_TEST_RESULTS.md](LIFECYCLE_TEST_RESULTS.md); their original text is kept as the record of `43ab363`. The invariant is now stated as `micro_batches = accum × attempted_updates`. [VALIDATED_TRAINING_PROTOCOL.md](VALIDATED_TRAINING_PROTOCOL.md) now describes `baseline_v3`.

M6 is untouched by design. The released image/state pairing (`timing_variant: released_pairing`) is not transformed, and `test_timing_contract.py` still pins it.

### Changes to existing tests (none weakened)

| Test | Change | Why |
|---|---|---|
| `test_manifest_adapter.py::test_manifest_rejects_duplicate_content_and_tampering` | `pytest.raises(AssertionError)` → `pytest.raises(mf.ManifestError)` | The check is now an explicit exception, so it is not stripped by `python -O` |
| `test_launch_checks.py::test_v1_statistics_rejected` | Expected message "normalizer_version=None" → "keys must be" | A v1 file is now rejected one step earlier (exact key set); still rejected |
| `test_launch_checks.py::test_absolute_statistics_with_delta_actions_rejected` | Expected message → "normalizer v2 supports absolute actions only" | A dedicated check now precedes the config comparison; still rejected |
| `test_launch_checks.py::test_frozen_repository_protocols_are_valid` | `*_v2.json` → `*_v3.json` | Configs renamed; same assertions |
| `test_driver.py::test_update_semantics` | `optimizer_steps_skipped` → `skipped_updates_total`, plus `attempted == optimizer_calls == 9` | Summary field renamed; one assertion added |
| `run_fixtures.py` | Builds the fixture's image-integrity file and adds the two new protocol keys; statistics written with `dump_statistics` | Protocol v3 |

### Frozen configs

`configs/{baseline,lifecycle_u20,lifecycle_u100}_v2.json` were renamed (`git mv`) to `*_v3.json`. In each file the only differences are `protocol_version`, `name` and the two `image_integrity*` data keys. Data, TinyDP and schedule values are identical, so new configuration hashes follow (`artifacts/launch_checks_baseline_v3.json`).

## 3. Mutation check (each repair undone in a scratch copy)

`policy_training/tools/mutation_check.py` copies `policy_training/`, applies one exact string replacement and runs the targeted test. It covers the 21 original M1–M5 mutations (two strings updated to the edited source) and 27 new F1–F8 mutations, each undoing one repair: for example "skipped GradScaler step counted as an update", "no bound on consecutive skips", "statistics config not compared", "frame bytes not verified", "booleans accepted as integers", "validation not transactional", "EMA key set not checked" and "collinear rot6d accepted". A timeout counts as detected; one is reported separately if it occurs.

Result at `966d459`: control run 117 passed (exit 0); **48 of 48 mutations detected** (21 M1–M5 + 27 F1–F8), 0 timeouts (`artifacts/codex_final_repair/verification_966d459/mutation_check_v3.json`). An earlier full run on the uncommitted tree detected 46 of 48. Two F-mutations removed only one of two overlapping checks, so the test still passed: the F2 sidecar check, which the action-level finite check backs up, and the F7 `micro == accum × attempted` check, which the data-position check backs up. The F2 mutation now disables the shared `require_finite` check. A checkpoint case that only the micro-batch identity can catch (`micro_with_matching_position`) was added. Both were then detected.

## 4. Verification at the repair commit

All runs below are at implementation commit **`966d45980bd7fd2654fd9f79c62c0dd844a181d5`** (clean tree; the commit that follows adds only this table and the evidence files). Evidence: `artifacts/codex_final_repair/verification_966d459/`.

| Check | Command | Result |
|---|---|---|
| Repository tests (89 original + 105 new) | `CUDA_VISIBLE_DEVICES="" .venv-tinydp/bin/python -m pytest -q -p no:cacheprovider policy_training/tests` | **194 passed**, exit 0 (71.9 s) |
| Codex independent suite, unchanged (28 cases) | `policy_training/audit/run_codex_final_suite.sh OUT` | **25 passed, 3 failed**, exit 1 (`suite_exit.json`: commit `966d459`, dirty false). The 3 failures are `test_inherited_physical_rotation_conversion_is_valid_or_rejects[nonunit/collinear/zero]`. They test the pinned, unmodifiable TinyDP helper, and their observations equal Codex's (det 36, non-finite, det 0); see §6.1. All 22 other previously failing cases pass. The 6 previously passing cases still pass, including bitwise cross-epoch CPU resume and relocated inference. |
| Mutation check (48) | `.venv-tinydp/bin/python policy_training/tools/mutation_check.py OUT.json` | 48 / 48 detected (§3) |
| Preservation and invariants | `.venv-tinydp/bin/python policy_training/verify_repair_invariants.py --base 43ab363… --checkpoint <v3 u20_A> --checkpoint <v3 u20_B> --checkpoint <v2 u100_w4> --out …` | `all_ok: true`, exit 0. Protected paths (`scripts`, `real2render2real`, `dependencies`, `pyproject.toml`, `uv.lock`, `env_data_gen.sh`) are equal to `repro/baseline-datagen` (`334e83d`, unchanged). Submodules are at their pins and TinyDP is clean; the only drift is the pre-existing `trajgen.egg-info`. The v1 statistics and manifest are unchanged vs `43ab363`. Source fingerprint `dd23bd44…2f2bcd`; no view file modified. Launch checks pass for N = 50/100/150/1000/1461. Checkpoint counters: Adam steps = scheduler = update, attempted = update + skipped, micro = 4 × attempted = epoch × bpe + batch (v3 u20 ×2 and legacy v2 u100). |
| Real-data launch checks, protocol v3 | `.venv-tinydp/bin/python policy_training/tools/launch_check_all.py policy_training/configs/baseline_v3.json …` | All 5 N pass, 6.3–38.4 s; N = 1461 verifies 1,825 episodes / 587,650 frames (`artifacts/launch_checks_baseline_v3.json`) |
| Released-code M3 counterexample | `.venv-tinydp/bin/python -I policy_training/repro_released_resume.py` | Still reproduces on unchanged TinyDP: epochs `[0..5]` retrained (expected `[5]`), EMA 9 → 3 |
| Codex evidence preserved | `sha256sum -c` against `artifacts/codex_final_43ab363/SHA256SUMS.original_evidence` | `/tmp/r2r2r-codex-final-evidence` unchanged; repo copies identical; `CODEX_FINAL_AUDIT.md` byte-identical |
| v2 statistics, manifest and earlier audit | `git diff 43ab363 HEAD -- docs/policy_training/artifacts/{normalizer_v2,manifest_v1.json,action_statistics_train_n*.json} docs/policy_training/CODEX_PRETRAIN_AUDIT.md` | empty |

The repository's `pytest` reports deprecation/user warnings from zarr, NumPy and torch's scheduler-order check; their number grew only because there are more tests. None is new in kind, and none is suppressed. Non-zero exit is ensured for: failed launch checks (`test_launch_failure_exits_nonzero`, subprocess); non-finite loss or validation (exception); skipped-step abort (`SkippedStepError`); and a session ending before its target update (`SystemExit` in `train_r2r2r_dp.py`).

## 5. Bounded GPU checks (GPU 0 only)

Before the runs, GPU 0 held only the NX desktop process (423 MiB, 0 %). There was no COAST or other experiment process; GPU 1 was not used and stayed at its desktop baseline (≤ 367 MiB).

**CUDA skip probe** (`tools/cuda_skip_probe.py`, synthetic fixture, small model, bf16 autocast, real `GradScaler('cuda')`; `artifacts/codex_final_repair/cuda_skip_probe.json`):
- An Inf gradient in attempt 2 gives 5 attempts, 4 updates, 1 skip, 4 optimizer calls, scheduler 4, Adam steps {4}, 20 micro-batches and scale 65536 → 32768, plus one `train_skipped` record for update 2.
- Resuming from the u2 checkpoint written after the skip (attempts 3, skipped 1, micro 12) gives the same final counters, scheduler, scaler and data position as the uninterrupted run, with a max model difference of 0.0.

**Real-data lifecycle, protocol v3** (`configs/lifecycle_u20_v3.json`, N = 50, full model, real entry point; run A = 20 uninterrupted updates, run B = stop after 10, then `--resume` to 20; `artifacts/codex_final_repair/lifecycle_v3/`):

| | A | B1 | B2 (resume) |
|---|---|---|---|
| exit / tracebacks | 0 / 0 | 0 / 0 | 0 / 0 |
| launch image check | 414 episodes, 133,308 frames | same | same |
| config / sample-order digest | `4e800408…` / `40d6e761…` | same | same |
| start update / attempt | 0 / 0 | 0 / 0 | 10 / 10 |
| skipped updates | 0 | 0 | 0 |
| PyTorch peak allocated | 15.43 GiB | 15.43 GiB | 15.43 GiB |
| GPU 0 peak (nvidia-smi) | 22,335 MiB | 22,293 MiB | 22,287 MiB |

A vs B: the learning rate at every update, the scheduler, the data position, the update and micro-batch counters (u10: 10/40, u20: 20/80), the validation windows and the config hash are **identical**. The train loss differs by at most 2.6e-4, model weights by at most 2.9e-4 (u10, before any resume) and 3.4e-4 (u20), EMA by at most 9.5e-7. Full validation at u20: A 1.723198, B 1.723190. This matches the GPU kernel-nondeterminism level measured for v2; bitwise GPU equality is not claimed. These runs overlapped with CPU test jobs (load average up to 37 on 24 cores), so their wall times are **not** throughput measurements; the v2 figures in LIFECYCLE_TEST_RESULTS.md §4 remain the throughput reference.

The u20 checkpoint and export were copied to `isolated_ckpt/` and the run directory hidden. Both loaded on CPU and gave `(1, 16, 10)` finite outputs on a real validation window, and full vs export outputs were bitwise identical. `right_poses` gave proper rotations (max |R Rᵀ − I| = 7e-16), and a NaN proprio was rejected.

## 6. Remaining limitations

1. **Inherited rotation helper.** `tinydp.dataset.utils.rot_6d_to_rot_mat` (pinned submodule) is still invalid for non-unit, collinear and zero 6-D rows, so Codex's three tests of it still fail. The repair layer never calls it (tested); physical poses must go through `rotation.rot6d_rows_to_matrix` / `R2R2RPolicy.right_poses`. TinyDP-internal uses (`rot_6d_to_quat`, `rot_6d_to_euler`) are outside the training path of this protocol.
2. **Image integrity scope.** Launch verification is byte-level against a manifest built once. The decode, mode and resolution certificate applies because the bytes are identical. A deliberate collision of sha256 is out of scope.
3. **M6** (≈ 2-row image lag) is unchanged by design.
4. **GPU reproducibility.** GPU runs are not bitwise reproducible (`cudnn.benchmark`, as released). Exact continuation is proven on CPU only.
5. **Skip policy constants** (20 consecutive / 1,000 total) are engineering choices, not tuned. No real overflow has been observed in any GPU run (0 skips in all v2 and v3 sessions).
6. **Scientific limits** are unchanged (CODEX_FINAL_AUDIT §6): right-arm, absolute-action, released-pairing variant; no closed-loop evaluation; one human demonstration source; no claim of convergence, success rate or robot readiness.
7. **Self-assessment.** The grades below are mine; the independent re-audit decides.

## 7. Grades and decision

| Issue | `43ab363` (Codex) | Now (self-assessed) |
|---|---|---|
| M1 normalization | PARTIAL (F2) | PASS (finite-input contract enforced) |
| M2 accumulation / scheduler / EMA | FAIL (F1) | PASS (successful-step accounting; CPU and CUDA skip tests) |
| M3 checkpoint / resume | PARTIAL (F7, F3) | PASS on CPU (bitwise, incl. resume after a skip); GPU: state exact, numerics at the nondeterminism level |
| M4 manifest / launch | FAIL (F3, F4) | PASS |
| M5 inference | PARTIAL (F8) | PASS for the offline contract; inherited helper limitation (§6.1) |
| M6 timing | PARTIAL | PARTIAL, by design (unchanged) |
| M7 lifecycle / protocol | PARTIAL (F1, F5, F6) | PASS |

**Decision: no known blocker remains; CONDITIONAL GO.** Full TinyDP training under `baseline_v3` (one run, one GPU, `train_r2r2r_dp.py`) may start **only after the independent re-audit of this branch confirms these repairs**. No full training was launched in this task.
