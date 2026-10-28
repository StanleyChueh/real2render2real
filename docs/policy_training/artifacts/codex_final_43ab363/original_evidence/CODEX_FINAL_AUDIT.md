# Independent final audit: R2R2R TinyDP M1–M7 repairs

**Decision: NO-GO for full TinyDP training at the audited commit.**

Implementation: **`43ab363058c4c9a5a45207b1414315320ef8438d`**  
Reference: **`2a0536689e81b1e01876c8cdf0550fa9fae20227`**  
Reviewer: Codex. Review date: 2026-10-27.

The reported suite was independently reproduced: **89 passed, 1,442 warnings, 31.91 seconds, exit 0**. An additional independent suite produced **6 passed, 22 failed, 793 warnings, 22.78 seconds, exit 1**. These 22 failures are adverse-contract test cases grouped into eight MAJOR findings below; they are not 22 independent defects in the production dataset. No tests are marked xfail or converted into successes merely because they demonstrate a defect.

The normal finite-data path is substantially improved: normalization coverage, normal accumulation, scalar EMA, ordinary RNG isolation, cross-epoch CPU resume and relocated offline inference have positive execution evidence. Nevertheless, a real CPU GradScaler skip caused the committed driver to report one completed update and `finished=true` after **zero optimizer steps**, advancing scheduler and EMA once. That directly violates the successful-update training budget. Missing EMA checkpoint parameters, altered dataset ordering and several non-finite/invalid inputs are also accepted.

No CRITICAL defect was established in the current source artifacts or split membership. The MAJOR defects remain sufficient for NO-GO. This report does not infer that overflow, dataset corruption or checkpoint corruption already occurred in Claude's recorded GPU runs.

## 1. Isolation, scope and preservation

Created detached worktree `/tmp/r2r2r-codex-final-43ab363` at the exact implementation SHA. TinyDP was cloned **locally**, without hardlinks or downloads, into that worktree and checked out at pinned commit `76d8f9cd14affe0066df3dbb38759b086613f617`. Imported TinyDP and repair modules were verified to resolve inside the isolated worktree. The existing `.venv-tinydp` interpreter/packages were used without installation or modification.

All fixtures, logs and this report are outside Claude's working directory. Implementation files were not edited. Production datasets were not regenerated. The test suites create and mutate small synthetic fixtures only. The native GradScaler overflow probe uses CPU tensors; no CUDA training or expensive GPU experiment was launched.

The direct committed range contains **44 changed files, 4,812 insertions and 5 deletions**, confined to `policy_training/` and `docs/policy_training/`. The implementation adds an external training driver rather than editing the upstream TinyDP training loop. Review and tests target that new entry point as well as the inherited helpers it invokes.

Executed environment: Python **3.10.15**, PyTorch **2.5.1+cu121**, torchvision **0.20.1+cu121**, NumPy **1.26.4**, SciPy **1.14.1**, h5py **3.16.0**, Zarr **2.12.0**, pytest **9.1.1**. CUDA was hidden from all audit test processes. These are observed installed versions, not newly installed requirements.

Executed preservation checks:

| Check | Observed result |
|---|---|
| Direct reference → implementation diff over `scripts/`, `real2render2real/`, `dependencies/`, `pyproject.toml`, `uv.lock`, `env_data_gen.sh`, `.gitmodules` | Exit 0, no differences. Generator, upstream source and gitlinks preserved. |
| Main `repro/baseline-datagen` | Still `334e83d4d1be7a4a5e78768cd5344051d4a892fa`. |
| Main `repro/dp-training` | Still the reference `2a0536689e81b1e01876c8cdf0550fa9fae20227`. |
| Main checkout status before/after | Only pre-existing `m dependencies/trajgen`; its two `trajgen.egg-info` metadata modifications were neither changed nor reset. |
| Recursive original submodule status | At pinned commits; TinyDP clean. Isolated TinyDP checkout also clean. |
| V1 manifest and five V1 statistics files | No changes between reference and implementation. |
| Numerical source hashes | All **1,825** source episodes match manifest content hashes and have finite arrays. |
| Prepared views | **3,125** entries and **9,375** sidecars match source arrays/dtypes and symlink destinations; no identity errors. |
| Source fingerprint before/after tests | Both equal the preparation fingerprint below. |
| Isolated tracked implementation after tests | Clean, exact audited SHA. |

```text
Source fingerprint:
dd23bd44ce626485a9f3ec8f78ce3c75cfeadb90785efdcf456c8f997c2f2bcd

V1 manifest SHA-256:
2c58b6f975103c085ed09ec852fa9bcbbc3d22f30d3d77386dcc4678899b1031
```

The historical fingerprint covers paths, sizes, mtimes and HDF5 bytes; it is not a full JPEG-content hash. Therefore source preservation is strongly corroborated within that coverage, not independently authenticated byte-for-byte for every JPEG. No production image was modified by this audit.

Evidence: [real artifact verification](real_artifact_verification.json), [preservation after tests](preservation_after_tests.json).

## 2. M1–M7 disposition

PASS means the stated contract was independently executed successfully; PARTIAL means the original failure is improved/resolved on the normal path but additional requirements or limitations remain; FAIL means a required repair contract is violated by an executed counterexample.

| Issue | Status | Independent evidence and remaining gap |
|---|---|---|
| **M1: normalization** | **PARTIAL** | Odd-row failure repaired, including actual row-15 regression. All five real normalizers match independent SciPy/source-array extrema exactly; all supervised targets covered. Non-finite input can still be sanitized/certified as valid: F2. |
| **M2: accumulation, scheduler, EMA** | **FAIL** | Normal large-batch/accumulated AdamW agrees and scalar EMA oracle passes. Actual GradScaler skip still advances successful-update counter, scheduler and EMA: F1. |
| **M3: checkpoint/resume** | **PARTIAL** | Actual entry-point CPU runs resume bitwise across an epoch boundary, switching 0 → 2 workers. Missing EMA tensor is silently accepted: F7. Changed view ordering also passes launch checks: F3. CUDA resume was not freshly exercised. |
| **M4: manifest/launch enforcement** | **FAIL** | Reported overlap, wrong-subset, stale-sidecar and original unsafe-action cases pass their rejection tests. Wrong manifest metadata, changed view ordering, corrupt JPEG content and false action-layout metadata can still pass preflight: F3/F4. |
| **M5: inference artifact** | **PARTIAL** | EMA/raw choice, denormalization and relocated standalone inference work. Non-finite input yields non-finite actions without rejection; inherited physical rotation conversion remains invalid for degenerate/non-unit vectors: F8. |
| **M6: timing** | **PARTIAL** | Declared `released_pairing` is preserved and both committed delayed-image tests pass. Physical image/state synchronization is deliberately not repaired or established. |
| **M7: scientific lifecycle/protocol** | **PARTIAL** | Frozen external driver executes real bounded CPU lifecycle and successful normal resume. Unsafe numeric protocol values, failed-validation state handling and skipped-update budgets remain: F1/F5/F6. Full-scale GPU behavior remains pending independent verification. |

This grade does not assume earlier work-in-progress observations remain true. In particular, the previous item-RNG clobbering is fixed: the committed tests and the independent ordinary-validation test preserve Python, NumPy and Torch RNG state.

## 3. Execution record

The first sandboxed executions stalled when spawned DataLoader workers were reached. Those audit processes were stopped, and both full suites were rerun outside the sandbox with CUDA hidden. Authoritative completed results are the host-permission reruns, not the stalled logs.

| Execution | Result |
|---|---|
| Committed `policy_training/tests` | **89 passed**, no skips reported, exit 0. |
| Independent `test_independent_final.py` | **6 passed / 22 failed**, exit 1. Failures assert desired behavior and are not implementation edits. |
| Source-data/view/normalizer verifier | Completed, no source/view identity errors; all five actual launch checks pass for the untouched prepared roots. |
| Released M3 resume reproduction | Re-executed CPU toy orchestration of unchanged TinyDP; details below. |

Passed independent numerical/lifecycle checks:

- **Accumulation equivalence:** deterministic float64 AdamW, large batch versus four equal micro-batches, parameter and optimizer-state tolerance `1e-10`. Both final scalar parameters equal **0.7093699989981614**, with one optimizer step each.
- **EMA:** initial zero, decay 0.5, parameters 10 then 20 → EMA **5**, then **12.5**.
- **Validation RNG/weights:** ordinary EMA validation preserves Python, NumPy and Torch RNG, raw model state and training mode. Observed loss **1.5453885793685913**, four windows.
- **Partial validation batches:** batch sizes 3 and 1 with batch losses 1 and 9 → raw per-window mean **3.0**, not the unweighted mean 5.0.
- **Cross-epoch resume:** nine real CPU updates, accumulation four, 30 micro-batches/epoch. Interrupted after u7, u7 checkpoint removed to resume from durable u6, resumed with two spawn workers instead of zero. Model, EMA, optimizer and scheduler are **bitwise identical** to uninterrupted execution; both finish at **36 consumed micro-batches**, **epoch 1 / batch 6**.
- **Inference portability:** copied EMA artifact to another directory and made all synthetic source, view, statistics and manifest paths unavailable. Seeded inference still loads independently and returns finite **`(1,16,10)`** right-arm actions.

Original failure-case coverage was also retained: the committed suite reproduces the V1 odd-row target **5.0**, released-loop six-micro-batch accounting (one optimizer update versus six scheduler/EMA advances), inherited wrapper import failure, original launch rejection cases, and delayed-image filename pairing. Re-running `repro_released_resume.py` reproduces epoch-4 resume invoking epochs **0–5**, with saved EMA **3** instead of **9**; normal v2 continuation fixes this original case.

Evidence: [89-test log](reported89_pytest.log), [independent log](independent_pytest.log), [per-test results](adversarial_results.json), [completed-run observations](observations_host.json), [suite exit records](suite_executions.json), [released resume log](original_resume_counterexample.log).

### Independent real normalization verification

The independent oracle reads source HDF5 arrays and uses SciPy rotation matrices directly; it does not call v2's indexing, conversion or min/max helpers. Bounds were compared against the declared conservative loader-target support, rows **2–158**. Supervised coverage was checked on rows **8–158**, because the loss uses only each window's final observation chunk.

| Train N | Supervised `(episode,row)` pairs checked | Outside bounds at tolerance `1e-6` | Max normalized absolute value | Min/max discrepancy from independent oracle |
|---:|---:|---:|---:|---:|
| 50 | 7,550 | 0 | 1.0 | 0.0 / 0.0 |
| 100 | 15,100 | 0 | 1.0 | 0.0 / 0.0 |
| 150 | 22,650 | 0 | 1.0 | 0.0 / 0.0 |
| 1,000 | 151,000 | 0 | 1.0 | 0.0 / 0.0 |
| 1,461 | 220,611 | 0 | 1.0 | 0.0 / 0.0 |

Original n150 row 15 / right `r00`: **V1 −1.0807001295 → V2 −1.0**. Train membership remains 1,461, validation 182 and test 182; prepared nested subsets are preserved. All actual prepared split roots passed the new launch checks. These positive results do not imply that invalid alternative inputs are rejected.

## 4. Findings and correction instructions

### F1 — MAJOR, blocks training: skipped AMP steps consume the successful-update budget

Source: [driver.py](../r2r2r-codex-final-43ab363/policy_training/r2r2r_dp/driver.py#L284), especially lines 284–289.

Test: `test_amp_skipped_step_does_not_advance_successful_update`.

Executed the committed `Trainer.train` method with a scalar model, unchanged TinyDP NativeScaler wrapper containing a **real `torch.amp.GradScaler('cpu')`**, and a gradient hook injecting Inf for the first four-micro-batch attempt only. The next attempt would be finite. This is an actual skipped optimizer step, not a mock that merely sets a Boolean.

```text
optimizer.step calls:       0
scheduler steps:            1
EMA updates:                1
reported successful update: 1
consumed micro-batches:     4
GradScaler scale:           65536 -> 32768
finished:                  true
```

The method stops before trying the subsequent finite group. Therefore “100,000 optimizer updates” can mean fewer successful optimizer calls. This probe does not prove CUDA-specific execution was tested, but directly exercises the committed accounting branch with native GradScaler skip behavior.

**Correction for Claude Code:** separate attempted groups, consumed micro-batches, skipped groups and successful updates. On overflow, clear/retry according to a documented policy and do not advance scheduler, EMA, successful-update count or success-based validation/checkpoint cadence. Preserve consumed data position. Add a bounded failure/retry policy to avoid endless overflow loops. Update invariants: with skipped attempts, `micro_batches == accumulation * successful_updates` is no longer generally valid; use attempted groups too. Preserve zero-learning-rate optimizer calls as successful calls when they actually execute.

### F2 — MAJOR: non-finite normalization can be hidden by coverage checks

Source: [normalizer_v2.py](../r2r2r-codex-final-43ab363/policy_training/r2r2r_dp/normalizer_v2.py#L69), `compute_bounds`, `normalized`, `coverage`.

Tests: `test_normalization_rejects_nonfinite[nan/inf/neg_inf]`, `test_bound_builder_cannot_certify_nan_data`.

Observed normalization outputs:

```text
NaN  -> 0
+Inf -> 1.7976931348623157e308
-Inf -> -1.7976931348623157e308
```

Injecting NaN into one synthetic training sidecar produces NaN bounds, yet `coverage` reports **0 out-of-bounds rows / 168 checked**, maximum absolute normalized value **1.0**. This is a helper/negative-input failure, not a claim that the real finite dataset contains NaNs. The pinned production manifest protects against ordinary sidecar mutation; the statistics builder still must not certify invalid data.

**Correction:** reject non-finite arrays and bounds before reductions or scaling, validate shapes/range order, and distinguish finite constant dimensions from invalid values. Constant-dimension handling may retain the documented compatibility mapping; do not use blanket sanitization to conceal unrelated NaNs. Emit strict JSON with `allow_nan=False`. Require coverage to assert finite inputs/intermediates separately from the unit-box predicate.

### F3 — MAJOR, affects continuation/provenance: incomplete metadata and ordering validation

Source: [launch_checks.py](../r2r2r-codex-final-43ab363/policy_training/r2r2r_dp/launch_checks.py#L35), [reference manifest checks](../r2r2r-codex-final-43ab363/policy_training/r2r2r_dp/manifest.py#L101).

Tests: `test_manifest_metadata_rejected_even_when_re_pinned[version/counts/salt]`, `test_changed_view_order_rejected`, `test_wrong_action_layout_rejected`.

Executed cases accepted by preflight:

- Manifest version **999**, counts **999/999/999**, or inconsistent salt, after updating the protocol pin and statistics' manifest hash so that the test reaches semantic validation. Ordinary unrepinned byte tampering remains correctly rejected.
- Renaming a training view directory changes the actual instantiated TinyDP sample order, but launch succeeds with unchanged configuration hash and counts.
- Reversing the declared action layout without changing numerical bounds or its stale `config_sha256` is accepted.

The actual production manifest is valid and the prepared splits remain disjoint. These are failures of the advertised accepted-input/provenance contract. In particular, matching membership and `batches_per_epoch` cannot guarantee the same next sample on resume if view order changes.

**Correction:** validate manifest version, counts, unique ranks, supported schema, positive subset sizes and the declared split-order policy using explicit exceptions. Enforce canonical view directory names from split/rank or order the loader canonically by UID/rank. Store and verify a sample-order digest for continuation. Validate the normalizer layout and recompute its declared configuration digest, rather than checking only selected fields. Preserve source-root containment and exact source destinations. Keep critical validation effective under Python optimization mode.

### F4 — MAJOR: JPEG content is not enforced at launch

Source: [launch_checks.py](../r2r2r-codex-final-43ab363/policy_training/r2r2r_dp/launch_checks.py#L90).

Test: `test_corrupt_jpeg_rejected_before_model`.

Replaced one **synthetic fixture** `camera_0/rgb/0007.jpg` with `b'not a jpeg'`, retaining its filename. `check_launch` accepted it. The current check verifies frame names, not image content, resolution or decodability. Prior full decoding plus an unbound, mutable source directory does not prove current image validity.

**Correction:** bind image identity to a verified immutable dataset snapshot or an external versioned image-integrity manifest; verify that identity at launch. Full decoding/content hashing can be performed once and reused under a justified immutability/cache contract rather than repeated unnecessarily. At minimum, retain the prior validation identity and explicitly invalidate it when image content changes. Add corrupted-content and wrong-resolution cases in addition to missing-frame tests. Do not rewrite production images or the original V1 manifest to repair this.

### F5 — MAJOR: invalid numerical protocol values are accepted

Source: [protocol.py](../r2r2r-codex-final-43ab363/policy_training/r2r2r_dp/protocol.py#L53), `_set`, `validate_protocol`.

Tests: `test_protocol_invalid_numerics_rejected[negative_clip/fractional_batch/nan_lr/bool_updates]`.

`resolve` accepts gradient clipping **−1**, batch size **2.5**, learning rate **NaN**, and Boolean `total_updates=true` with zero warm-up. Negative clipping is particularly serious: it is an invalid clipping contract and may reverse gradient direction rather than provide the intended positive norm bound. Later optimizer/shape errors are not equivalent to validated preflight.

**Correction:** implement field-specific finite/type/domain checks. Integer fields must be integers excluding Boolean values; sample counts and dimensions must be positive; worker counts nonnegative; learning rate/weight decay/noise/clip thresholds finite with defined valid ranges. Validate seeds, UNet/horizon shape compatibility and supported camera count. Reject invalid protocols before dataset/model allocation, and use strict JSON. Preserve supported zero-valued settings only where explicitly valid, e.g. zero workers/warm-up/noise.

### F6 — MAJOR: failed/non-finite validation is not transactional

Source: [driver.py](../r2r2r-codex-final-43ab363/policy_training/r2r2r_dp/driver.py#L170).

Tests: `test_validation_exception_restores_weights_and_mode`, `test_validation_nonfinite_loss_rejected`.

An injected exception during EMA validation leaves **raw weights unrestored** and **`model.training=false`**. A model returning NaN during validation is accepted and returns `loss=NaN`. Ordinary validation passes the independent full RNG/weight-restoration check; the exception/non-finite paths do not.

**Correction:** snapshot prior mode/state and restore them in `finally`, including explicit worker cleanup where necessary. Check each validation loss and the final aggregate for finiteness, handle empty evaluation explicitly, and propagate failure clearly. Prevent NaN monitoring records from being treated as successful validation. The analogous apply-EMA/export restore pattern should use the same transactional discipline; that export-exception case was not independently injected here.

### F7 — MAJOR, blocks robust resume: incomplete EMA checkpoint state silently loads

Source: [driver.py](../r2r2r-codex-final-43ab363/policy_training/r2r2r_dp/driver.py#L238), inherited `ModelEMA.load_state_dict`.

Test: `test_checkpoint_missing_ema_parameter_rejected`.

Deleted **`vision_encoder.conv1.weight`** from a real small-model full checkpoint's EMA dictionary. `Trainer.load_full` accepted it. The inherited EMA helper silently ignores absent entries, leaving initialization-era EMA values for missing parameters. Successful normal-checkpoint resume does not cover this corruption case.

**Correction:** validate the exact expected EMA key set, shapes, dtypes and finite values before mutating runtime state; reject missing/extra tensors. Validate model/optimizer/scheduler/scaler/counter consistency and consumed data position. Add negative checkpoint tests, and keep legacy v1 weight-only loading distinct from exact v2 continuation. Do not relax strictness merely to recover an incomplete EMA without a documented migration.

### F8 — MAJOR, inference/pose boundary: non-finite actions and unsafe inherited rotation conversion

Sources: [inference.py](../r2r2r-codex-final-43ab363/policy_training/r2r2r_dp/inference.py#L82), [pinned TinyDP rotation helpers](../r2r2r-codex-final-43ab363/dependencies/tinydp/tinydp/dataset/utils.py#L19).

Tests: `test_inference_nonfinite_input_rejected`, `test_inherited_physical_rotation_conversion_is_valid_or_rejects[nonunit/collinear/zero]`.

One NaN in proprio is accepted; inference returns non-finite normalized and denormalized actions without raising. Separately, the unchanged inherited helper returns:

| Input 6D rows | Result |
|---|---|
| `[2,0,0, 0,3,0]` | Finite but non-orthonormal matrix, determinant **36**. |
| `[1,0,0, 1,0,0]` | Non-finite matrix. |
| `[0,0,0, 0,0,0]` | Zero matrix, determinant **0**. |

The new policy intentionally returns a 6D representation, not a physical rotation matrix. Thus the rotation failures are an **inherited downstream conversion limitation**, not proof that the new offline model must project its training outputs or that its valid 10D artifact fails to relocate. They do prevent claiming safe physical-pose conversion through the existing helper.

**Correction:** reject non-finite inference inputs, bounds, model state and outputs; reject reversed bounds while retaining explicit finite constant-dimension behavior. Validate complete observation shape and denoising-step range. Provide an external row-based 6D-to-SO(3) conversion when physical poses are required: normalize/orthogonalize consistently, guard small/collinear vectors, verify `R^T R≈I` and `det(R)≈+1`, and explicitly reject degenerate inputs. Do not silently change the upstream model or its 6D loss to implement that boundary fix.

## 5. CRITICAL / MAJOR / MINOR classification

- **CRITICAL:** none established in the inspected production data or current split artifacts. No claim is made that a historical training result was invalidated; full production training has not been independently executed.
- **MAJOR:** F1–F8. F1 directly falsifies the successful-update stopping contract; F3/F7 undermine exact continuation on accepted altered/corrupt inputs. The remaining issues affect numerical robustness, input provenance, validation recovery and inference correctness.
- **MINOR documentation correction:** revise the repair report's unqualified M1–M5/M7 PASS/readiness language after addressing these counterexamples. Retain the correct distinction between changing epoch batch order and fixed micro-batch composition/tail exclusion. Do not describe a future repaired invariant as `micro_batches=accum*successful_updates` when skipped attempts consume data.

## 6. Remaining scientific and hardware limits

1. The profile is a **right-arm-only, absolute-action, released-pairing variant**, not an exact 20D Table-8 reproduction. Its normalization and training-loop corrections are disclosed deviations.
2. M6 preserves filenames/indexing and the approximate two-row render delay; it does not establish synchronized sensing or matching real-camera latency. Passed timing fixtures are not proof of universal physical alignment.
3. Held-out synthetic trajectories share a human demonstration, assets and some generation-batch factors. Split disjointness supports synthetic held-out evaluation, not independent real-world generalization.
4. The 41 joint-jump episodes remain; no new claim of smooth physical feasibility, collision-free behavior or grasp success is made.
5. Finite denoising losses and portable offline inference do not establish convergence, closed-loop success, robot readiness or reproduction of reported policy success rates.
6. Fresh CUDA training, CUDA AMP overflow and GPU continuation tests remain **pending**, not passed. The sandbox GPU query failed; a host read-only query succeeded and showed both RTX 4090s at 0% utilization (447 / 383 MiB used). GPUs were not claimed occupied. No new GPU experiment was needed to establish the CPU blockers, and none was launched.
7. Committed historical GPU lifecycle evidence was inspected as historical evidence only. Its A/B comparison already reports non-bitwise GPU weights; it is not substituted for the independent deterministic CPU continuation check.
8. Full-scale resource stability, long-run optimizer behavior and projected 100k-update duration were not tested. Tests use the released model with small CPU-test widths/resolution or scalar numerical harnesses, as identified above.

## 7. Reproduction commands and evidence

These commands run the audited isolated files; use the existing interpreter and do not reinstall dependencies. Spawn-worker tests need the host process/IPC permissions that were unavailable in the restricted sandbox. The completed reruns used those permissions.

```bash
AUDIT=/tmp/r2r2r-codex-final-43ab363
EVID=/tmp/r2r2r-codex-final-evidence
PY=/home/stanley/Stanley_ws/real2render2real/.venv-tinydp/bin/python
export GIT_OPTIONAL_LOCKS=0
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$AUDIT/dependencies/tinydp:$AUDIT/policy_training:$AUDIT/policy_training/tests"
cd "$AUDIT"

git rev-parse HEAD
git diff --exit-code 2a0536689e81b1e01876c8cdf0550fa9fae20227 \
  43ab363058c4c9a5a45207b1414315320ef8438d -- \
  scripts real2render2real dependencies pyproject.toml uv.lock env_data_gen.sh .gitmodules

"$PY" -B -m pytest policy_training/tests -q -p no:cacheprovider \
  -o faulthandler_timeout=45 --basetemp=/tmp/r2r2r-codex-repeat-89

"$PY" -B -m pytest "$EVID/test_independent_final.py" -q -p no:cacheprovider \
  -o faulthandler_timeout=45 --basetemp=/tmp/r2r2r-codex-repeat-independent

"$PY" -B "$EVID/verify_real_artifacts.py"
"$PY" -B policy_training/repro_released_resume.py
```

For a minimal individual numerical reproduction, add a selector such as:

```bash
"$PY" -B -m pytest "$EVID/test_independent_final.py" -q -p no:cacheprovider \
  -k amp_skipped_step
```

The independent suite source is [test_independent_final.py](test_independent_final.py), with [reporting hook](conftest.py). Results are retained in [adversarial_results.json](adversarial_results.json). [observations_host.json](observations_host.json) contains the 28 completed host-run observations; raw `observations.jsonl` also includes the earlier sandbox attempt. Non-finite values are string-encoded in the curated JSON for standards-compliant evidence storage.

## 8. Conditions for reconsidering NO-GO

Claude Code should implement and commit the corrections in the external repair layer while preserving generator/source data/submodule invariants. At minimum, fix F1 successful-step accounting, F2 non-finite checks, F3 semantic/order enforcement and F7 strict EMA restoration, then address validation recovery, numeric protocol gates and inference input/output validation. Satisfy the declared image-integrity and physical-pose boundaries without silently changing the baseline.

Rerun all 89 existing tests and the independent adverse-contract cases against the new exact SHA. Require bitwise CPU cross-epoch continuation and portable inference to remain passing. Only then consider fresh bounded GPU lifecycle verification and a new GO decision. **Do not launch full training from this audit; no training launch is approved by the present NO-GO verdict.**
