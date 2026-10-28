# Independent final re-audit — TinyDP protocol v3

**Decision: NO-GO.** Exact target: `986a3fab059ff982d6fbc23b59875e1bae7857cf`.

The original eight findings are substantially repaired. The complete available CPU suite passes, and 19 of the original 22 failing adversarial cases now pass. The remaining three exercise the deliberately unchanged upstream rotation converter; the external converter and inference boundary pass their new tests. Those three failures are an inherited limitation, not a reason to reject the repaired training path.

The remaining release blocker is **N1: incomplete optimizer/scheduler/RNG checkpoint validation**. An independently corrupted checkpoint with one entire Adam parameter state deleted is accepted, resumes, and reports `finished=true`, then writes a checkpoint that its own validator rejects because Adam steps are `[1,4]` while the completed update count is 4. This violates the claimed strict continuation contract. Ordinary intact-checkpoint CPU continuation is bitwise reproducible; no existing production checkpoint corruption is alleged.

## Commit identity, isolation and preservation

All four full SHAs exist and each is an ancestor of the next (executed `git merge-base --is-ancestor`, all exit 0):

1. Preparation reference: `2a0536689e81b1e01876c8cdf0550fa9fae20227`.
2. Rejected protocol v2: `43ab363058c4c9a5a45207b1414315320ef8438d`.
3. Protocol v3 implementation: `966d45980bd7fd2654fd9f79c62c0dd844a181d5`.
4. Audited target: `986a3fab059ff982d6fbc23b59875e1bae7857cf`.

Direct v2-to-target diff: 66 files, 14,154 insertions, 195 deletions; confined to `policy_training/` and `docs/policy_training/`, including retained original evidence and the image-integrity certificate. The final commit adds verification documentation/evidence over the implementation commit.

Detached worktree: `/tmp/r2r2r-v3-audit-worktree`. TinyDP was cloned locally without hardlinks/downloads into the isolated worktree, at `76d8f9cd14affe0066df3dbb38759b086613f617`. The existing `.venv-tinydp` interpreter is reused through an untracked symlink. No packages were installed. Test imports use isolated `policy_training/` and isolated TinyDP. The mutation runs import each scratch copy through its local test conftest; only isolated TinyDP is on their inherited PYTHONPATH. No implementation edits, source-data writes, trajectory generation, GPU training, pushes or merges were performed.

Executed preservation evidence ([preservation.json](preservation.json), [real_artifact_verification.json](real_artifact_verification.json)):

- Preparation-to-target protected-path diff is empty: `scripts/`, `real2render2real/`, `dependencies/`, `pyproject.toml`, `uv.lock`, `env_data_gen.sh`, `.gitmodules`. Gitlinks unchanged; recursive submodule SHAs recorded.
- `repro/baseline-datagen` remains `334e83d4d1be7a4a5e78768cd5344051d4a892fa`.
- Manifest and original statistics unchanged from preparation; v2 statistics unchanged from rejected v2 commit.
- All **1,825** source numerical episode hashes and finite arrays verified. **3,125** prepared view entries / **9,375** sidecars match source arrays, dtype and symlink destinations; zero errors.
- Independently recomputed SciPy/source-array bounds match every normalizer exactly. Supervised right-target counts for N=50/100/150/1000/1461: **7,550 / 15,100 / 22,650 / 151,000 / 220,611**, all zero out of bounds. Same 182 validation and 182 test trajectories across subsets.
- Source fingerprint remains `dd23bd44ce626485a9f3ec8f78ce3c75cfeadb90785efdcf456c8f997c2f2bcd`. This historical fingerprint covers HDF5 bytes and file metadata, not every JPEG byte.
- Fresh v3 launch checks independently verify the pinned image digests over the prepared train/val/test roots for all five N values. All pass: **4.3 / 4.7 / 5.6 / 18.9 / 25.3 seconds**, respectively. This verifies current bytes against the certificate; it is not a fresh full JPEG-decode certification.

## F1–F8 findings

RESOLVED applies to the original defect and its tested repair boundary. PARTIAL denotes remaining requirements in that area, even where the original specific counterexample is fixed.

| Finding | Original defect | New implementation | Independent test | Result | Evidence |
|---|---|---|---|---|---|
| F1 | GradScaler skip advances successful budget, scheduler, EMA | Counts actual calls reaching optimizer.step; separate attempts/skips/successes; bounded retry | Original real CPU GradScaler counterexample; normal/overflow regression, zero-LR, persistent overflow and resume-after-skip cases | **RESOLVED** | Original observations; full suite; mutation logs |
| F2 | NaN normalized to zero and invalid bounds certified | Finite/bounds checks before scaling/reductions; strict JSON | Original NaN/±Inf cases and NaN sidecar; constant-dimension regression; real-data independent oracle | **RESOLVED** | Original suite; full suite; real artifact JSON |
| F3 | Repinned bad metadata, altered view order, false layout accepted | Explicit manifest semantics, canonical names, symlink restrictions, full stats config/digest, sample-order digest on resume | Original metadata/order/layout tests; negative aliases, ranks/counts/salt, consistent-but-wrong layout digest, optimized Python checks | **RESOLVED** | Original suite; full suite; fresh all-N launches |
| F4 | Invalid JPEG accepted by launch | Versioned pinned, manifest-bound image certificate; recompute camera digests | Original corrupt JPEG; invalid size/missing/truncated/grayscale tests; fresh real-data hashes | **RESOLVED** | Original suite; full suite; real-launch.json |
| F5 | Negative clipping, fractional batch, NaN LR, bool counts accepted | Field-specific domains before model allocation; strict JSON | Original four cases; expanded numeric/config tests; launch-first entrypoint tests | **RESOLVED** | Original suite; full suite |
| F6 | Validation exceptions leave EMA applied/eval mode; NaN loss accepted | try/finally state/mode/worker restoration; finite losses; transactional export | Original exception/NaN cases; both prior modes and export failure; ordinary RNG preservation | **RESOLVED** | Original observations; full suite |
| F7 | Missing EMA tensor silently loads | Exact model/EMA keys, shapes, dtype, finite validation before load; added state/counter checks | Original missing EMA rejection and all new 17 checkpoint cases pass; five additional state-corruption probes fail desired rejection; missing entire Adam state silently resets | **PARTIAL** | Full suite; extra.log; checkpoint-effect/results.json; N1 |
| F8 | Non-finite inference accepted; unsafe physical rotation conversion | Finite/shape checks; portable artifact; external row-based SO(3) converter used by right_poses | Original NaN and relocation tests; new converter nonunit/collinear/zero plus SciPy round-trip tests; new inference safety tests | **RESOLVED in repair layer** | Original suite; full suite; unchanged upstream helper still fails 3 original tests |

Original AMP observation now: optimizer calls **1**, scheduler **1**, EMA **1**, successful updates **1**, micro-batches **8**, attempts **2**, skips **1**, scale **32768**, finished true. The first four-microbatch group overflowed; the second succeeded. Adam returning None is handled correctly. Large-batch versus accumulated float64 AdamW parameter is **0.7093699989981614** on both paths; optimizer moments match at tolerance 1e-10. EMA scalar oracle ends at **12.5**.

Ordinary validation preserves Python/NumPy/Torch RNG, raw weights and mode. Exception restoration now succeeds. Partial validation batches give weighted mean **3.0**. Original relocated inference works after all external source/view/statistics/manifest paths are unavailable, yielding finite `(1,16,10)` actions. Original CPU cross-epoch resume is bitwise equal after 9 updates / 36 micro-batches, epoch 1 batch 6, including switching 0 to 2 workers.

## M1–M7 status

| Issue | Grade | Evidence / limitation |
|---|---|---|
| M1 normalization | **PASS** | All target support covered, finite rejection, train-only extrema verified |
| M2 optimizer/scheduler/EMA/accumulation | **PASS** | Real overflow, zero-LR calls, retry budget, normal numerical equivalence |
| M3 checkpoint/resume | **PARTIAL** | Intact deterministic CPU resume and strict EMA pass; N1 blocks complete strict-state contract |
| M4 manifest/launch | **PASS** | Original counterexamples rejected; all-N real launch and image identity verified |
| M5 inference | **PASS** | Relocation, weight selection, denormalization, finite rejection and safe external pose conversion |
| M6 timing | **PASS for declared released_pairing** | Timing fixtures pass; known approximately two-row lag is intentional, preserved and documented. Physical sensor synchronization remains unestablished. |
| M7 lifecycle/scientific protocol | **PARTIAL** | Actual CPU entrypoint lifecycle and normal resume pass; N1 and fresh GPU validation outstanding |

## Executed tests

| Execution | Observed result |
|---|---|
| Full available `policy_training/tests` | **194 passed**, 3,107 warnings, **61.30 s**, exit 0. Includes the original suite with updated fixtures and new repair regressions; not a separate unchanged 89-test run. |
| Original independent adversarial suite (only evidence directory redirected) | **25 passed / 3 failed**, 792 warnings, **23.23 s**, exit 1. Same three unchanged upstream rotation-helper failures. Thus 19/22 prior failures now pass, all six former passes remain passing. |
| Additional independent checkpoint rejection suite | **0 passed / 5 failed**, 83 warnings, **5.99 s**, exit 1; failures assert required rejection, not expected-failure markers. |
| Independent replay of mutation control | **117 passed**, 1,676 warnings, **46.20 s**, exit 0 on the unmodified scratch copy |
| Independent replay of all 48 committed mutation recipes | **48/48 produced pytest exit 1**; zero timeouts, collection, syntax or import errors. All exact replacements applied once. Some detections concern diagnostics while another check still rejects the bad input; see N2. |
| Additional actual continuation probes | Valid control completes u4; missing Adam state also completes u4 but creates self-invalid checkpoint; wrong moment shape / invalid RNG / NaN LR raise after load accepted. |
| Fresh real-data launch checks | **5/5 passed**; current image identity, source/sidecar identity, split and normalizer gates |
| Fresh independent numerical/source verifier | Completed; zero identity errors and zero bound discrepancies |

Logs: [full-suite.log](full-suite.log), [original/pytest.log](original/pytest.log), [original/observations.jsonl](original/observations.jsonl), [extra.log](extra.log), [checkpoint_effect.log](checkpoint_effect.log), [real-launch.json](real-launch.json), [verify_data.log](verify_data.log).

Mutation evidence: [mutation_results.json](mutation_results.json), [failure review with assertion/exception extracts](mutation_failure_review.json), [mutation runner](run_mutations.py), and `mutation-00.log` through `mutation-48.log` in this report's directory. The runner adds full log retention to the same exact mutation recipes; production code is never changed. Its process exits 0 after completing the experiment, while each mutant's pytest exits 1; these are different result levels.

Verified source imports: [imports.json](imports.json), including isolated `r2r2r_dp.driver` and `tinydp.policy.model` paths. Production checkout after all tests still shows only pre-existing `m dependencies/trajgen`; isolated tracked source and isolated TinyDP remain clean. The worktree has only an untracked interpreter symlink. Existing source datasets, checkpoints and evidence were never opened for writing.

The integrated tests call `train_r2r2r_dp.main` and exercise configuration resolution, launch validation, model construction, actual forward/backward and Adam updates, scheduler/EMA, validation, checkpoint save, resume, export, inference reload and termination. They use small CPU image/UNet dimensions, not a full-width production GPU workload. Historical committed CUDA probes/lifecycle reports were inspected as prior evidence only; no fresh CUDA test was executed and no GPU availability claim is made.

## New findings and actionable corrections

### N1 — MAJOR: accepted checkpoint can silently reset Adam state and report completion

Source: isolated `policy_training/r2r2r_dp/driver.py`, `checkpoint_problems` and `load_full`, particularly optimizer validation around lines 337–354. The check compares group lengths and the **set** of step counts among existing entries. It does not require state for every parameter, required Adam moment keys/shapes, finite optimizer hyperparameters, complete scheduler state or valid RNG payloads.

Smallest counterexample: train the small deterministic fixture to durable u3, delete one key from `checkpoint['optimizer']['state']`, save to a new temporary file, load and continue to u4. `checkpoint_problems` returns `[]`; `load_full` accepts. Adam initializes the missing parameter's state anew and the trainer reports finished. Model weights differ from intact continuation by a maximum absolute **4.839897155761719e-05**. The resulting checkpoint is then rejected by its own check: **`optimizer step counts [1, 4] != update 4`**. The intact control finishes and has no checkpoint problems.

Other executed mutations accepted by `load_full`: `lr=NaN`, missing `exp_avg`, `exp_avg` shape `(2,3)`, scheduler `_last_lr=NaN`, Torch RNG ByteTensor length 1. Actual continuation of the malformed moment raises a tensor-size RuntimeError; malformed RNG raises expected-size 5048/5056 versus 1. NaN LR fails strict metric JSON after the optimizer path has already executed. These are negative-input failures, not evidence that normal saved checkpoints are corrupt.

**Correction:** map serialized optimizer parameter IDs to the expected ordered model parameters/groups; require exact expected initialized state coverage for this fixed model after u>0, required Adam fields, compatible moments and scalar integral steps equal to u. Validate finite/domain-correct group hyperparameters and agreement with checkpoint scheduler/current schedule, including last LR and base LR consistency. Validate scheduler/scaler key completeness and typed/domain-correct values. Validate RNG payloads using temporary/local generators or save/restore guards before mutating live runtime state. Make loading transactional on any validation or deserialization failure. Require all independent negative cases to reject before changing model/EMA/optimizer/counters, retain bitwise normal/skip-resume tests, and optionally validate generated full-checkpoint invariants before publishing a durable file.

Reproductions: [test_extra.py](test_extra.py) and [checkpoint_effect.py](checkpoint_effect.py), with [observed continuation results](checkpoint-effect/results.json).

### N2 — MINOR: mutation success predicate is too broad

The committed `tools/mutation_check.py` treats **any nonzero pytest exit** and even a **900-second timeout** as successful detection. A syntax/import/setup error can therefore count without reaching the intended assertion. The control run is valuable but cannot exclude a mutation-specific unrelated failure. Retain per-mutant tracebacks, reject collection/setup failures and timeouts as inconclusive, and identify the intended test/exception/assertion. Mutation detections measure sensitivity to these particular edits, not independent scientific correctness properties.

Fresh tracebacks also demonstrate diagnostic-only detections. Mutant 12 removes the overlap-specific diagnostic, but other split/content checks still reject overlap; its test fails because the error lacks `overlaps`. Mutant 14 still rejects wrong-subset statistics via recomputed bounds/counts but loses the expected provenance diagnostic. Mutant 32 still rejects the changed layout through its inconsistent digest; the first parameterized test stops on a missing `config=` diagnostic before reaching the consistently rehashed layout case. Mutant 40 still rejects NaN through the finite clipping check; the test fails because the message contains lowercase `nan` instead of the expected `NaN`. These detections do not independently prove that the mutation made unsafe input acceptable. Mutant 20's None-statistics AttributeError is caused by the intended deletion of embedded statistics, not an unrelated import failure. Full retained logs allow these distinctions.

**CRITICAL:** none established in current production data or intact-checkpoint normal CPU training. **MAJOR:** N1. **MINOR:** N2. Deliberate baseline timing and the unused unsafe upstream rotation helper are documented limitations, not newly introduced training failures.

## Reproduction commands

```bash
cd /tmp/r2r2r-v3-audit-worktree
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD/dependencies/tinydp:$PWD/policy_training:$PWD/policy_training/tests"
PY=/home/stanley/Stanley_ws/real2render2real/.venv-tinydp/bin/python
E=/tmp/r2r2r-codex-reaudit-986a3fa
"$PY" -B -m pytest policy_training/tests -q -p no:cacheprovider
bash policy_training/audit/run_codex_final_suite.sh "$E/original-repeat"
"$PY" -B -m pytest "$E/test_extra.py" -q -s -p no:cacheprovider
# The continuation probe creates a fresh output directory: change its 'out' path before repeating.
"$PY" -B "$E/checkpoint_effect.py"
"$PY" -B policy_training/tools/launch_check_all.py policy_training/configs/baseline_v3.json "$E/real-launch-repeat.json"
"$PY" -B "$E/verify_data.py"
# The independent mutation runner makes scratch copies and retains complete pytest failure logs.
python3 -B "$E/run_mutations.py"
```

Spawn-worker tests were executed with host IPC permissions because sandboxed worker startup previously stalled; CUDA remains disabled. Repeat commands write only fresh temporary outputs; choose new basetemp/output paths to preserve this evidence.

## Scientific limits and release decision

This is a corrected external training protocol, not algorithmically identical to released TinyDP training. Train-target normalization coverage, optimizer-update schedule/EMA/accumulation and resume behavior intentionally differ. The policy is a right-only 10D absolute-action variant with 20D proprio, not an exact full bimanual paper reproduction. Fixed microbatch composition/tail exclusion, train-and-val proprio noise and released image pairing remain disclosed inherited choices. Synthetic trajectory disjointness does not establish independent human/scene or real-world generalization. No convergence, robot success or calibrated physical performance is established by finite CPU losses.

**NO-GO for full training at this SHA.** Correct N1, rerun the corruption cases and CPU lifecycle against the next exact commit, and retain the passing original failure cases. Then perform a bounded full-width v3 GPU lifecycle/overflow/resume/inference check on an idle GPU, without interfering with COAST. Fresh CUDA evidence is pending. Only after those checks should a full baseline run be reconsidered; no training is started or approved by this report.
