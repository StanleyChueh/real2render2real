"""Compare two lifecycle runs (usage: compare_runs.py RUN_A RUN_B OUT.json): checkpoint tensors at u10/u20 and metrics."""
import json, sys
import torch

A, B, out = sys.argv[1:4]


def recs(d):
    drop = ("update_s", "data_wait_s", "seconds", "path")
    return [{k: v for k, v in json.loads(l).items() if k not in drop} for l in open(f"{d}/metrics.jsonl")]


def maxdiff(x, y):
    worst, n_diff, n = 0.0, 0, 0
    for k in x:
        a, b = x[k].float(), y[k].float()
        d = float((a - b).abs().max()) if a.numel() else 0.0
        worst = max(worst, d)
        n_diff += int(not torch.equal(x[k], y[k]))
        n += 1
    return {"tensors": n, "tensors_not_bitwise_equal": n_diff, "max_abs_diff": worst}


rep = {}
for u in (10, 20):
    a = torch.load(f"{A}/ckpt_full_u{u:07d}.pt", map_location="cpu", weights_only=False)
    b = torch.load(f"{B}/ckpt_full_u{u:07d}.pt", map_location="cpu", weights_only=False)
    oa = {f"{i}.{k}": v for i, s in a["optimizer"]["state"].items() for k, v in s.items()}
    ob = {f"{i}.{k}": v for i, s in b["optimizer"]["state"].items() for k, v in s.items()}
    rep[f"u{u}"] = {"model": maxdiff(a["model"], b["model"]), "ema": maxdiff(a["ema"], b["ema"]),
                    "optimizer": maxdiff(oa, ob), "scheduler_equal": a["scheduler"] == b["scheduler"],
                    "data_position_equal": a["data_position"] == b["data_position"],
                    "update": [a["update"], b["update"]], "micro_batches": [a["micro_batches"], b["micro_batches"]],
                    "config_sha256_equal": a["run"]["config_sha256"] == b["run"]["config_sha256"],
                    "model_abs_mean": float(sum(v.float().abs().sum() for v in a["model"].values()) /
                                            sum(v.numel() for v in a["model"].values()))}
ra, rb = recs(A), recs(B)
rows = []
for x, y in zip(ra, rb):
    assert (x["type"], x["update"]) == (y["type"], y["update"]), (x, y)
    if x["type"] == "train":
        rows.append({"update": x["update"], "lr_equal": x["lr_used"] == y["lr_used"],
                     "loss_A": x["loss_raw_mean"], "loss_B": y["loss_raw_mean"],
                     "loss_abs_diff": abs(x["loss_raw_mean"] - y["loss_raw_mean"]),
                     "grad_A": x["grad_norm_before_clip"], "grad_B": y["grad_norm_before_clip"],
                     "position_equal": (x["epoch"], x["batch_in_epoch"], x["micro_batches"]) == (y["epoch"], y["batch_in_epoch"], y["micro_batches"])})
    elif x["type"].startswith("val"):
        rows.append({"update": x["update"], "type": x["type"], "loss_A": x["loss"], "loss_B": y["loss"],
                     "windows_equal": x["windows"] == y["windows"]})
rep["records_aligned"] = len(ra) == len(rb)
rep["records"] = rows
json.dump(rep, open(out, "w"), indent=1)
for k in ("u10", "u20"):
    print(k, json.dumps(rep[k]))
for r in rows:
    print(r)
