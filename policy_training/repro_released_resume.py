"""Reproduce audit M3 on TinyDP's released script/train.py main() (CPU, toy model/data, real orchestration).

    .venv-tinydp/bin/python -I policy_training/repro_released_resume.py

Checkpoint at epoch 4 with model weight 3 and EMA weight 9, trainer.epochs = 6. A faithful resume would
train epoch 5 only and keep the EMA at 9 before updating it. Prints the observed behaviour as JSON.
Datasets, sampler, model and the epoch function are replaced by toys; resume_from_ckpt, the scheduler,
ModelEMA and the epoch/checkpoint orchestration are the released code.
"""
import importlib.util
import json
import os
import sys
import tempfile
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PY = os.path.join(REPO, "dependencies", "tinydp", "script", "train.py")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["WANDB_MODE"] = "disabled"

import torch  # noqa: E402


class Toy(torch.nn.Module):
    def __init__(self, *a, **k):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0]))


class ToyData(torch.utils.data.Dataset):
    def __init__(self, *a, **k):
        pass

    def __len__(self):
        return 8

    def __getitem__(self, i):
        return {"x": torch.tensor([float(i)])}


def main():
    spec = importlib.util.spec_from_file_location("released_train", TRAIN_PY)
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    calls = []
    train.DiffusionPolicy = Toy
    train.SimSequenceDataset = ToyData
    train.VideoSampler = lambda ds, batch_size, sequence_length, **k: [[0, 1], [2, 3]]
    train.CollateFunction = lambda *a, **k: (lambda b: b)
    train.MultiEpochsDataLoader = lambda ds, **k: [[0], [1]]
    train.misc.init_distributed_mode = lambda args: setattr(args, "distributed", False)

    def fake_epoch(model, data_loader, optimizer, lr_scheduler, device, epoch, loss_scaler, log_writer=None,
                   validate=False, args=None, ema=None):
        if not validate:
            calls.append(epoch)
        return {"loss": 0.0}
    train.train_one_epoch = fake_epoch

    from r2r2r_dp.protocol import default_experiment_config
    with tempfile.TemporaryDirectory() as d:
        ck = os.path.join(d, "checkpoint_4.pt")
        m = Toy()
        with torch.no_grad():
            m.w.fill_(3.0)
        opt = torch.optim.AdamW(train.misc.add_weight_decay(m, 0.01), lr=1e-4)  # same groups as train.py
        torch.save({"model": m.state_dict(), "optimizer": opt.state_dict(), "epoch": 4, "ema": {"w": torch.tensor([9.0])}}, ck)
        args = default_experiment_config()
        args.device = "cpu"
        args.dataset_cfg.is_sim_data = True
        args.trainer_cfg.epochs = 6
        args.trainer_cfg.num_workers = 0
        args.shared_cfg.resume = ck
        args.shared_cfg.save_every = 1000
        args.logging_cfg.output_dir = os.path.join(d, "out")
        args.logging_cfg.log_dir = None
        train.main(args)
        saved = torch.load(os.path.join(d, "out", "checkpoint_5.pt"), weights_only=False)
        out = {"train_epochs_called": calls, "expected_epochs": [5],
               "args_start_epoch_after_resume": args.shared_cfg.start_epoch,
               "saved_checkpoint_keys": sorted(saved), "saved_ema_w": float(saved["ema"]["w"]),
               "checkpoint_ema_w": 9.0, "saved_model_w": float(saved["model"]["w"])}
    print(json.dumps(out))


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
