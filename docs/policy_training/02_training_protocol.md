# 02 — TinyDP training protocol (proposed; NOT launched)

Status: **prepared for independent audit.** No full training run has been started. Single GPU (GPU 0) only. WandB disabled. No uploads.

## 1. Entry point

`policy_training/train_tinydp.py` executes TinyDP's **unmodified** `dependencies/tinydp/script/train.py` (sha256 recorded per run in `r2r2r_provenance.json`). The only runtime substitution is the dataset class:

- `tinydp.dataset.image_dataset_sim.SequenceDataset` → `r2r2r_dp.tinydp_bridge.ManifestSimDataset`, a subclass that does not override any loading or indexing method. It only sets `dataset_root = --train-root, train_split = 1.0` for the train split and `dataset_root = --eval-root, train_split = 0.0` for val, and forces `action_statistics = --action-statistics`.
- `glob` inside that module is replaced by a sorted glob, so window order is reproducible. Membership is unaffected.

The wrapper rejects: missing `--dataset-cfg.is-sim-data`; `--logging-cfg.log-name` (which triggers `wandb.init`); `--dataset-cfg.dataset-root`, `action-statistics`, `data-subsample-*` and `balance-data` (all controlled by the manifest); and a stale `output_dir/action_statistics.json` that differs from the given file. It sets `WANDB_MODE=disabled`.

## 2. Proposed baseline command (one dataset size N)

```bash
cd ~/Stanley_ws/real2render2real
N=1461; EPOCHS=37   # see §4 for EPOCHS per N
V=~/Stanley_ws/r2r2r_policy_runs/dataset_v1/views
CUDA_VISIBLE_DEVICES=0 .venv-tinydp/bin/python policy_training/train_tinydp.py \
  --train-root $V/train_subsets/n$(printf %04d $N) --eval-root $V/splits/val \
  --action-statistics docs/policy_training/artifacts/action_statistics_train_n$(printf %04d $N).json -- \
  --dataset-cfg.is-sim-data \
  --model-cfg.policy-cfg.pred-right-only \
  --shared-cfg.batch-size 16 --trainer-cfg.accum-iter 4 \
  --trainer-cfg.num-workers 4 \
  --optimizer-cfg.lr 2e-4 --optimizer-cfg.weight-decay 0.09 \
  --trainer-cfg.epochs $EPOCHS \
  --logging-cfg.output-dir ~/Stanley_ws/r2r2r_policy_runs/train/dp_n$(printf %04d $N)_seed0
```

The test split (`views/splits/test`) is never passed to training.

## 3. Configuration vs TinyDP defaults and the paper

| Setting | TinyDP default | Paper (§4.3, Table 8) | Proposed | Note |
|---|---|---|---|---|
| Dataset class | `image_dataset_sim` | — | same class, manifest roots | §1 |
| `--dataset-cfg.is-sim-data` | off | sim data | **on** | required by the schema (01 §3) |
| Arms predicted | both (20-D) | Table 8 lists a 20-D action space | **right only (10-D)**, `pred-right-only` | **Deviation from the paper's Table 8; open decision D1.** Verified: left gripper never set, left EE moves ≤ 6 mm per episode, generator hardcodes `left_hand_envs = zeros`. In 20-D mode the left gripper target is a constant 0 (NaN → 0) and left EE targets are min–max-stretched jitter. The README recommends this flag for one-arm data. Proprio stays 20-D (both arms) either way. |
| Action type | absolute EE (`use_delta_action=False`) | text: absolute EE poses; Table 8: "delta EEF xyz" | **absolute** (TinyDP default; the authors' own `train.py` header uses `--shared-cfg.no-use-delta-action`) | **Open decision D2:** the paper is internally inconsistent |
| Rotation | 6-D (rows of R) | 6-D | same | — |
| Obs horizon / action horizon | 4 / 16 | 4 / 16 | same | — |
| Subsample steps | 2 | not reported | 2 | — |
| Image size / cameras | 448 / 2 | 448 / third-person RGB | same | — |
| Vision encoder | ResNet-18, GroupNorm, **random init** | trained from scratch | same | — |
| Noise-pred net | ConditionalUnet1D (256,256,512,1024), DDPM 100 steps, squaredcos_cap_v2, ε-prediction, gripper loss weight 5 | not detailed | same | 148.9 M parameters |
| Batch size | 64 per GPU | 64 | **16 × accum 4 = 64 effective** | **Deviation:** 64 and 32 per step OOM in the forward pass on a 24 GB 4090 (03 §3). GroupNorm has no batch statistics, so the accumulated gradient equals the 64-batch gradient (up to independent noise/timestep draws). |
| Optimiser | AdamW, lr 1e-4, wd 0.01 | AdamW (0.9, 0.999), base lr 2e-4, wd 0.09 | **lr 2e-4, wd 0.09** (CLI flags) | **Open decision D3:** paper values chosen as the reproduction target. TinyDP's `blr` is unused when `lr` is set. |
| LR schedule | cosine, 500 warm-up **scheduler steps** (hardcoded) | cosine, 500 warm-up steps | unchanged code | **Side effect of accumulation:** TinyDP steps the scheduler per micro-batch, so warm-up = 500 micro-batches = **125 optimiser steps**. The cosine spans the whole run either way. |
| EMA | decay 0.9999, updated every loop iteration | not reported | unchanged code | **Side effect of accumulation:** EMA updates per micro-batch, giving ≈ 0.9996 per optimiser step. |
| Gradient clip | 1.0 | not reported | same | — |
| Mixed precision | bf16 autocast + GradScaler | not reported | same | — |
| Training length | `epochs` (default 100) | **100,000 steps** | epochs chosen per N for ≈ 100k optimiser steps (§4) | — |
| Proprio noise / vision aug | 0.003 / colour jitter (b 0.1, c 0.2, s 0.1, h 0.1) on train | not reported | same | Proprio noise is also applied to val (TinyDP behaviour) |
| Dataloader workers | 20 per loader (2 loaders) | — | **4** | **Deviation:** 20 workers × prefetch 2 × 1.23 GB batches exhausted 62 GB host RAM (kernel OOM-killed a worker, 03 §3). Data content is unaffected. |
| Sampler shuffling | `VideoSampler(seed=42)`; `set_epoch` never called | — | unchanged | Batch order repeats every epoch (upstream behaviour, recorded) |
| Validation | full val pass after **every** epoch | not reported | see blocker B1 | — |
| Checkpoints | every 20 epochs + last, with optimiser + EMA (~2.4 GB each, estimated) | selection not reported | see B2 | — |
| Seeds | 0 (torch/numpy), sampler 42, split salt `r2r2r-dp-v1` | — | seed 0; more seeds recommended | — |
| Hardware | — | 1× GH200, ~3 h | 1× RTX 4090 (GPU 0) | measured 1.28 s per optimiser step incl. 3 micro-batch loads (single step; sustained throughput unmeasured) → ~36 h per 100k steps (estimate) |

## 4. Dataset-size experiments (val/test fixed for all)

Items per episode = 484. Micro-batch = 16 × 4 obs steps. Optimiser step = 4 micro-batches.

| N (train) | micro-batches / epoch | optimiser steps / epoch | epochs for ≈100k steps | flagged episodes in train | val micro-batches if validating every epoch |
|---|---|---|---|---|---|
| 50 | 378 | 94 | 1,064 | 2 | 1,464,064 |
| 100 | 756 | 189 | 530 | 4 | 729,280 |
| 150 | 1,134 | 283 | 354 | 5 | 487,104 |
| 1,000 | 7,562 | 1,890 | 53 | 21 | 72,928 |
| 1,461 | 11,048 | 2,762 | 37 | 33 | 50,912 |

Val split: 182 episodes = 88,088 items = 1,376 micro-batches per pass. Normalisation always uses that subset's own train-only file.

## 5. Blockers and open decisions (for the independent audit)

| ID | Issue | Proposed resolution (not implemented) |
|---|---|---|
| **B1** | TinyDP validates on the full val split after every epoch. With fixed val and a fixed step budget, small N would spend ≫ training compute on validation (N=50: 1.46 M val vs 0.40 M train micro-batches). | Either (a) a documented training-loop deviation validating every K optimiser steps (e.g. every 2,762, matching N=1461's epoch), or (b) a fixed deterministic val **sub-sample** of windows, identical for all N. Requires approval. |
| **B2** | Checkpoint volume: `save_every=20` epochs → 54 checkpoints for N=50 (≈ 130 GB, estimated). | Set `--shared-cfg.save-every` per N so checkpoints fall every ≈ 10k optimiser steps (N=50: 106; N=1461: 4). |
| **B3** | Compute: ≈ 36 h per 100k-step run on one 4090 (estimated from one step); 5 sizes × ≥ 1 seed ≈ 1 week. | Measure sustained steps/s in a bounded run (e.g. 500 steps) after audit; consider fewer sizes first. |
| D1 | Right-only (10-D) vs paper Table 8's 20-D action | Baseline right-only; optional 20-D replicate |
| D2 | Absolute vs delta EE actions (paper inconsistent) | Baseline absolute (TinyDP default, authors' command line); delta as a labelled variant |
| D3 | Optimiser: paper (2e-4 / 0.09) vs TinyDP defaults (1e-4 / 0.01) | Baseline = paper values |
| D4 | 41 flagged episodes define the rotation normalisation range (01 §6) | Keep in baseline; flagged-excluded ablation labelled as a deviation |
| D5 | Image ≈ 2 steps behind state (01 §5.3) | Keep (authors' pairing); study later |

## 6. Evaluation (later; not part of this stage)

Offline: val-split denoising loss (EMA weights, as in `train.py`) and action MSE on the held-out **test** split, computed once per final checkpoint. Closed-loop success needs a simulator rollout harness or the real YuMi. Neither exists in this branch, and **no policy success claim** will be made from losses.
