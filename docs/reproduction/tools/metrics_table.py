"""Print a markdown table of metrics.json (+ dataset_validation.json) for run directories.

usage: python docs/reproduction/tools/metrics_table.py RUN_DIR [RUN_DIR ...]
"""
import json
import os
import sys


def fmt(x, nd=1):
    if x is None:
        return "–"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def row(run_dir):
    m = json.load(open(os.path.join(run_dir, "metrics.json")))[0]
    vpath = os.path.join(run_dir, "dataset_validation.json")
    v = json.load(open(vpath))["summary"] if os.path.exists(vpath) else {}
    env = open(os.path.join(run_dir, "env.txt")).read() if os.path.exists(os.path.join(run_dir, "env.txt")) else ""
    client_txt = os.path.join(run_dir, "client.txt")
    client = "null" if os.path.exists(client_txt) and "client=null" in open(client_txt).read() else "chrome"
    g0, g1 = m["gpu"].get("gpu0", {}), m["gpu"].get("gpu1", {})
    return "| " + " | ".join([
        m["run"], str(m["num_envs"]),
        "GPU0" if "CUDA_VISIBLE_DEVICES=0" in env else "default (render+JAX on GPU1)",
        "lazy" if "PREALLOCATE=false" in env else "25 %",
        client,
        fmt(m["wall_s"], 0), str(m["n_batches"]),
        f'{m["successful_traj"]}/{m["attempted_traj"]}',
        fmt(m["success_rate"] * 100 if m["success_rate"] is not None else None),
        fmt(m.get("init_to_viser_s")), fmt(m.get("t_first_batch_end_s")),
        fmt(m.get("batch_time_s_median")), fmt(m.get("time_per_attempted_traj_s"), 2),
        fmt(m.get("steady_throughput_succ_traj_per_hour"), 0),
        fmt(g0.get("mem_peak_MiB", 0) / 1024, 1), fmt(g1.get("mem_peak_delta_MiB", 0) / 1024, 1),
        fmt(g0.get("util_mean_steady_pct"), 0),
        f'{v.get("n_valid", "–")}/{v.get("n_success_dirs", "–")}',
    ]) + " |"


HEAD = ("| run | envs | GPU placement | JAX alloc | client | wall [s] | batches | succ/attempted | success % | "
        "init→viser [s] | 1st batch [s] | batch median [s] | s / attempted traj | succ traj / h | GPU0 peak [GiB] | "
        "GPU1 Δ [GiB] | GPU0 util % | valid/saved |")

if __name__ == "__main__":
    print(HEAD)
    print("|" + "---|" * (HEAD.count("|") - 1))
    for d in sys.argv[1:]:
        print(row(d))
