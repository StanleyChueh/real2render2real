"""One-batch dataloader + forward/backward smoke test with the exact objects TinyDP's train.py builds.

    CUDA_VISIBLE_DEVICES=0 python policy_training/smoke_test.py --train-root ... --eval-root ... \
        --action-statistics ... --out report.json -- <same TinyDP args as training>

Builds datasets, VideoSampler, CollateFunction, MultiEpochsDataLoader, DiffusionPolicy, AdamW and the
loss scaler as in dependencies/tinydp/script/train.py, then runs one optimisation step (bf16 autocast,
clip_grad=1.0) on a train batch, one no-grad loss on a val batch and one short denoising call.
Exits non-zero if any check fails. No checkpoint is written.
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    argv = sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--action-statistics", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv[: argv.index("--")])
    tinydp_args = argv[argv.index("--") + 1:]
    os.environ["WANDB_MODE"] = "disabled"

    import numpy as np
    import torch
    import tyro
    from diffusers.optimization import get_scheduler
    import tinydp.util.misc as misc
    from tinydp.dataset.image_dataset import CollateFunction, VideoSampler
    from tinydp.dataset.utils import aug_vision_transform, default_vision_transform
    from tinydp.policy.model import DiffusionPolicy
    from tinydp.util.args import ExperimentConfig
    from tinydp.util.misc import MultiEpochsDataLoader, NativeScalerWithGradNormCount
    from r2r2r_dp import tinydp_bridge as tb

    torch.multiprocessing.set_start_method("spawn", force=True)  # as train.py
    args = tyro.cli(ExperimentConfig, args=tinydp_args)
    assert args.dataset_cfg.is_sim_data
    os.makedirs(args.logging_cfg.output_dir, exist_ok=True)
    report = {"tinydp_args": tinydp_args, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "checks": {}}
    checks = report["checks"]

    def check(name, ok, detail=None):
        checks[name] = {"ok": bool(ok), **({"detail": detail} if detail is not None else {})}

    torch.manual_seed(args.shared_cfg.seed)
    np.random.seed(args.shared_cfg.seed)
    device = torch.device("cuda")
    report["device"] = torch.cuda.get_device_name(0)
    report["visible_device_count"] = torch.cuda.device_count()

    tb.install_ordered_glob()
    cls = tb.make_manifest_dataset_class(a.train_root, a.eval_root, a.action_statistics)
    res = args.shared_cfg.image_size
    t0 = time.time()
    ds_train = cls(args.dataset_cfg, args.shared_cfg, args.logging_cfg, aug_vision_transform(resolution=res), split="train")
    ds_val = cls(args.dataset_cfg, args.shared_cfg, args.logging_cfg, default_vision_transform(resolution=res), split="val")
    report["dataset_init_s"] = round(time.time() - t0, 1)
    report["n_train_files"], report["n_val_files"] = len(ds_train.file2length), len(ds_val.file2length)
    report["n_train_items"], report["n_val_items"] = len(ds_train), len(ds_val)
    check("train_val_files_disjoint", not (set(map(os.path.realpath, ds_train.file2length)) & set(map(os.path.realpath, ds_val.file2length))))
    stats = json.load(open(a.action_statistics))
    check("dataset_uses_given_statistics", np.allclose(ds_train.stats["max"].numpy(), stats["max_action"]) and np.allclose(ds_val.stats["min"].numpy(), stats["min_action"]))

    B, L = args.shared_cfg.batch_size, args.shared_cfg.seq_length
    pcfg = args.model_cfg.policy_cfg
    collate = CollateFunction(L, pred_left_only=pcfg.pred_left_only, pred_right_only=pcfg.pred_right_only)
    loaders = {}
    for name, ds in (("train", ds_train), ("val", ds_val)):
        sampler = VideoSampler(ds, batch_size=B, sequence_length=L, num_replicas=1, rank=0)
        report[f"{name}_batches_per_epoch"] = len(sampler)
        loaders[name] = MultiEpochsDataLoader(ds, batch_sampler=sampler, num_workers=args.trainer_cfg.num_workers,
                                              collate_fn=collate, pin_memory=args.trainer_cfg.pin_memory, prefetch_factor=2)

    t0 = time.time()
    batch = next(iter(loaders["train"]))
    report["first_train_batch_s"] = round(time.time() - t0, 1)
    shapes = {k: list(v.shape) for k, v in batch.items()}
    report["train_batch_shapes"] = shapes
    check("observation_shape", shapes["observation"] == [B, L, args.shared_cfg.num_cameras, 3, res, res], shapes["observation"])
    check("proprio_shape", shapes["proprio"] == [B, L, 20], shapes["proprio"])
    check("action_shape", shapes["action"] == [B, L, args.shared_cfg.num_pred_steps, 20], shapes["action"])
    check("batch_finite", all(torch.isfinite(v).all().item() for v in batch.values()))
    act = batch["action"]
    right = act[..., 10:20] if pcfg.pred_right_only else act
    check("scaled_actions_in_unit_box", right.abs().max().item() <= 1.0 + 1e-5, float(right.abs().max()))
    report["right_gripper_target_fraction_closed"] = float((act[..., 19] > 0).float().mean())

    model = DiffusionPolicy(shared_cfg=args.shared_cfg, model_cfg=args.model_cfg).to(device)
    report["model"] = repr(model)
    report["n_parameters"] = sum(p.numel() for p in model.parameters())
    check("action_dim", model.action_dim == (10 if (pcfg.pred_right_only or pcfg.pred_left_only) else 20), model.action_dim)
    param_groups = misc.add_weight_decay(model, args.optimizer_cfg.weight_decay)
    optimizer = torch.optim.AdamW(params=param_groups, lr=args.optimizer_cfg.lr)
    loss_scaler = NativeScalerWithGradNormCount()
    lr_scheduler = get_scheduler(name="cosine", optimizer=optimizer, num_warmup_steps=500,
                                 num_training_steps=report["train_batches_per_epoch"] * args.trainer_cfg.epochs)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    torch.cuda.reset_peak_memory_stats()
    model.train()
    optimizer.zero_grad()
    accum = args.trainer_cfg.accum_iter
    it = iter(loaders["train"])
    losses = []
    t0 = time.time()
    for step in range(accum):  # one optimiser step = accum micro-batches, as engine.train_one_epoch
        mb = batch if step == 0 else next(it)
        mb = {k: v.to(device, non_blocking=True) for k, v in mb.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(mb)
        losses.append(float(loss.item()))
        loss = loss / accum
        grad_norm = loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(step + 1) % accum == 0, clip_grad=1.0)
        lr_scheduler.step()
        if (step + 1) % accum == 0:
            optimizer.zero_grad()
    torch.cuda.synchronize()
    report["accum_iter"] = accum
    report["effective_batch_size"] = B * accum
    report["train_step_s"] = round(time.time() - t0, 2)
    report["train_micro_losses"] = losses
    loss = torch.tensor(np.mean(losses))
    report["train_loss"] = float(loss.item())
    report["grad_norm_before_clip"] = float(grad_norm)
    report["peak_gpu_mem_GiB_train_step"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    check("train_loss_finite", math.isfinite(loss.item()), float(loss.item()))
    check("grad_norm_finite", math.isfinite(float(grad_norm)), float(grad_norm))
    changed = sum(not torch.equal(before[n], p.detach()) for n, p in model.named_parameters())
    check("parameters_updated", changed > 0, f"{changed}/{len(before)} tensors changed")
    check("parameters_finite", all(torch.isfinite(p).all().item() for p in model.parameters()))

    vbatch = next(iter(loaders["val"]))
    vbatch = {k: v.to(device, non_blocking=True) for k, v in vbatch.items()}
    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        vloss = model(vbatch)
        small = {"observation": vbatch["observation"][:2], "proprio": vbatch["proprio"][:2]}
        model.noise_scheduler.set_timesteps(10)  # short denoising chain for the smoke test only
        pred = model.forward_inference(small)
    report["val_loss"] = float(vloss.item())
    check("val_loss_finite", math.isfinite(vloss.item()), float(vloss.item()))
    check("inference_shape", list(pred.shape) == [2, args.shared_cfg.num_pred_steps, model.action_dim], list(pred.shape))
    check("inference_finite", torch.isfinite(pred).all().item())
    report["peak_gpu_mem_GiB_total"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    report["all_ok"] = all(c["ok"] for c in checks.values())
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "model"}, indent=1))
    sys.exit(0 if report["all_ok"] else 1)


if __name__ == "__main__":
    main()
