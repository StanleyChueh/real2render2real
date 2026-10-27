# Bounded GPU lifecycle test — protocol v2 (2026-10-27)

**Scope:** the real entry point (`policy_training/train_r2r2r_dp.py`), the full-size TinyDP model, real data, the N = 50 training subset and at most 100 optimizer updates. **This is not a training result:** losses after ≤ 100 updates say nothing about policy quality, and no 100,000-update run was started.

**Machine:**
- GPU: 2× RTX 4090 24 GB. Only GPU 0 was used (`CUDA_VISIBLE_DEVICES=0`, and the driver refuses more than one visible device).
- Contention: before every run, both GPUs held only desktop/NX processes (GPU 0: 18–447 MiB, 0 % utilization), and no COAST or other experiment processes were running.
- GPU 1 stayed at its desktop baseline throughout (≤ 390 MiB).

**Evidence:** the summaries are committed in `docs/policy_training/artifacts/lifecycle_v2/` (`resources_u20.json`, `resources_u100.json`, `compare_u20_A_vs_B.json`, `inference_check_u20_A.json`). The raw evidence (outside git) is in `~/Stanley_ws/r2r2r_policy_runs/lifecycle_v2/`:
- per-session stdout, `/usr/bin/time -v` output, nvidia-smi samples every 0.5 s, host-memory samples every 1 s and exit codes in `logs/`;
- per-run `metrics.jsonl`, `run_config.json` and checkpoints;
- a superseded first run A (`superseded_u20_A_pre_timing`, identical except that validation/checkpoint timings were not yet logged; exit 0).

**Command template:**

```bash
R=~/Stanley_ws/r2r2r_policy_runs/lifecycle_v2
policy_training/tools/run_lifecycle.sh <tag> --protocol policy_training/configs/lifecycle_u20_v2.json --train-size 50 --output-dir $R/<run> [--stop-after-updates 10 | --resume]
# tools/run_lifecycle.sh = CUDA_VISIBLE_DEVICES=0 /usr/bin/time -v .venv-tinydp/bin/python policy_training/train_r2r2r_dp.py "$@"
#                    + nvidia-smi / free samplers (stopped by PID afterwards)
```

The lifecycle protocols are identical to `baseline_v2.json` in data, model and optimizer (a test enforces this). Only the schedule is shorter.
- **`lifecycle_u20_v2`:** 20 updates, 5 warm-up updates, subset validation every 5 updates (validation-window stride 60, i.e. 546 windows), full validation and the final checkpoint at update 20, checkpoints and exports every 10.
- **`lifecycle_u100_v2`:** 100 updates, subset validation every 25, checkpoints and exports every 50.

## 1. Checklist

| # | Requirement | Result | Evidence |
|---|---|---|---|
| 1 | Actual training entry point | PASS | `train_r2r2r_dp.py`: protocol → launch checks (train 50 / val 182 / test 182, v2 statistics sha256 `f9c0bad1…`) → datasets → model |
| 2 | Train dataloader + forward/backward | PASS | 378 micro-batches per epoch (16 windows × 4 obs × 2 cameras at 448 px); 4 spawn workers |
| 3 | Multiple optimizer updates | PASS | 20 (A), 10 + 10 (B), 100 (§3); 0 GradScaler-skipped steps |
| 4 | EMA and scheduler updates | PASS | lr per update follows cosine with 5 warm-up updates (update 1 at lr 0, as released); one EMA update per optimizer update (CPU-tested; checkpoint EMA ≠ raw weights) |
| 5 | Deterministic bounded validation | PASS | subset validation 546 windows, EMA weights, fixed seed, 17.2 s; full validation 22,022 windows, 375 s |
| 6 | Checkpoint writing | PASS | full checkpoint 2,382,746,586 B in 1.2–1.3 s; EMA export 595,669,140 B in 0.5 s; atomic rename |
| 7 | Checkpoint reload + inference | PASS | `policy_training/check_checkpoint_inference.py` on CPU, with the run directory renamed away during the check: full checkpoint and export load in 0.7 s; one real val window → (1, 16, 10), finite, 100 DDPM steps in 1.1 s; **full vs export bitwise identical** |
| 8 | Resume | PASS (within GPU nondeterminism) | §2 |
| 9 | Clean shutdown | PASS | exit code 0 for every session; no traceback in any stdout; no `train_r2r2r_dp` process left; GPU 0 back to 18 MiB |

## 2. Resume on GPU: run A (20 uninterrupted) vs run B (stop after 10, `--resume` to 20)

| Quantity | Result |
|---|---|
| lr used at each of the 20 updates | **identical** |
| scheduler state, data position (epoch, batch), update and micro-batch counters at u10 and u20 | **identical** |
| config hash, validation windows | identical |
| train loss \|A − B\|, updates 1–10 (**both uninterrupted**) | max 2.2e-4 |
| train loss \|A − B\|, updates 11–20 (B resumed) | max 1.7e-4 |
| model weights max \|A − B\| at u10 (before any resume) / u20 | 3.9e-4 / 4.8e-4 (mean \|w\| ≈ 9.1e-3) |
| EMA weights max \|A − B\| at u10 / u20 | 3.6e-7 / 8.3e-7 |
| full validation loss at u20 | A 1.723183, B 1.723215 |
| metrics log | B's log truncated to the u10 checkpoint, pre-resume copy kept (`metrics.jsonl.before_resume_0`), `resume_history.jsonl` records the checkpoint used |

Comparison: `policy_training/tools/compare_runs.py $R/u20_A $R/u20_B OUT.json`.

**Interpretation:** two fresh GPU runs with identical code and seeds already differ at update 1. That is kernel nondeterminism under `cudnn.benchmark=True`, which the released code sets and this protocol keeps. The differences after resuming are of the same size as before it, so resume adds no detectable error on GPU. Exact resume state is proven on CPU, where interrupted + resumed training equals uninterrupted training **bitwise** (`test_driver.py::test_interrupted_resume_equals_uninterrupted`, with a different worker count after resuming). **Bitwise GPU reproducibility is not claimed.**

## 3. Resources and throughput

| Session | Updates | Exit | Wall | PyTorch peak allocated | GPU 0 peak (nvidia-smi) | GPU 0 mean util | Host used (baseline → peak) | Main-process max RSS |
|---|---|---|---|---|---|---|---|---|
| A (u20) | 20 | 0 | 491.6 s | 15.42 GiB | 22,077 MiB | 31 % | 5.7 → 14.3 GB | 8.5 GiB |
| B1 (stop at 10) | 10 | 0 | 66.4 s | 15.42 GiB | 21,863 MiB | 31 % | 6.3 → 13.7 GB | 7.8 GiB |
| B2 (resume to 20) | 10 | 0 | 459.8 s | 15.43 GiB | 21,763 MiB | 30 % | 6.3 → 15.7 GB | 9.1 GiB |
| u100, 4 workers (protocol) | 100 | 0 | 670.3 s | 15.43 GiB | 23,517 MiB | 33 % | 5.1 → 14.0 GB | 8.5 GiB |
| u100, 8 workers (`--num-workers 8`) | 100 | 0 | 461.7 s | 15.42 GiB | 21,659 MiB | 49 % | 5.3 → 21.1 GB | 15.5 GiB |

Timing (run A, steady state, updates 2–20; 1 update = 4 micro-batches of 16):

| Event | Time |
|---|---|
| optimizer update | mean 1.44 s (0.88–2.04 s); mean data wait 0.62 s |
| first update (worker start-up) | 5.5 s |
| subset validation, 546 windows / 35 batches | 17.2 s |
| full validation, 22,022 windows / 1,377 batches | 375 s |
| full checkpoint / EMA export | 1.3 s / 0.5 s |

At 4 workers the loop is **data-loading bound**: pure compute is ≈ 0.88 s per update, and every other update waits ≈ 1.2 s for JPEG decoding.

GPU headroom is ≈ 2.4 GB at batch 16 (22.1 of 24.6 GB on the device).

## 4. Sustained throughput (100-update runs) and full-run projection

| | 4 workers (frozen protocol) | 8 workers (operational override) |
|---|---|---|
| update time, updates 2–100 (mean / median) | 1.94 s / 2.05 s | 1.11 s / 0.98 s |
| mean data wait per update | 1.23 s | 0.19 s |
| updates per hour (training only) | 1,857 | 3,255 |
| subset validation, 546 windows | 17.3 s | 20.7 s |
| full validation, 22,022 windows | 389 s | 241 s |
| epoch crossing | update 95 spans epoch 0 → 1 (micro-batch 378); positions logged per update | same |
| GradScaler-skipped steps | 0 | 0 |
| lr per update, 4 vs 8 workers | identical | |
| full-validation loss at u100, 4 vs 8 workers | 1.511848 vs 1.511843 | |
| train loss \|4w − 8w\| (max over 100 updates) | 7.3e-3 (GPU nondeterminism accumulating; the data stream itself is identical, see the CPU tests) | |

The 20-update run's mean (1.44 s) was optimistic because the prefetch buffer was still full; the 100-update figures are the sustained rates.

**Projection for one `baseline_v2` run** (100,000 updates, any N; an estimate from 100 updates, not a measured long run):

| | 4 workers | 8 workers |
|---|---|---|
| training | ≈ 53.9 h | ≈ 30.7 h |
| subset validation, 80 batches × 100 times (scaled from 35 batches) | ≈ 1.1 h | ≈ 1.3 h |
| full validation × 4 | ≈ 0.4 h | ≈ 0.3 h |
| checkpoints + exports | < 0.1 h | < 0.1 h |
| **total** | **≈ 55 h** | **≈ 32 h** |
| disk (2 full checkpoints kept + 10 exports + logs) | ≈ 10.8 GB | same |

The data stream is invariant to worker count (bitwise on CPU), so `--num-workers 8` does not change what is trained. It is recorded in `run_config.json` → `runtime` (code added after these lifecycle runs; this probe's override is visible in its command line and its `run_lifecycle.sh` log). Host RAM at 8 workers peaked at 21 GB of 62 GB.

## 5. Observations relevant to the full run

- **EMA warm-up:** with EMA decay 0.9999 per update, the EMA weights after 100 updates are still ≈ 99 % the initial weights. The EMA validation loss (1.51 at u100) is therefore far above the raw training loss (0.31, mean of the last 10 updates), as expected. EMA validation becomes informative only after ~10⁴ updates (the EMA time constant).
- **GPU memory:** the PyTorch allocation is constant (15.4 GiB). The device total (21.7–23.5 GB) includes allocator cache and validation-shape fragmentation. No out-of-memory occurred, but headroom on a 24 GB card is about 1–3 GB. Nothing else should be placed on GPU 0 during a run.
- **Teardown:** the spawn workers exited with the process every time; the "worker killed by signal: Aborted" message from the old smoke test did not appear.
- **Code identity:** the u20 A/B and u100 runs used the repair code before the final commit (`repo_dirty: true` in their `run_config.json`; `code_sha256` per file recorded). The only later change is the `runtime` record in `train_r2r2r_dp.py` (no effect on training).
