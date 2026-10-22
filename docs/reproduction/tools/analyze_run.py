"""Parse a launch_run.sh run directory (run.log, gpu.csv, t_start/t_end) into metrics.

usage: python docs/reproduction/tools/analyze_run.py RUN_DIR [RUN_DIR ...] [--json out.json]

run.log lines are expected to be prefixed with a unix timestamp (tsline.py). For untimestamped
logs (run02) batch times fall back to the timestamps encoded in the success directory names.
"""
import argparse
import csv
import json
import os
import re
from datetime import datetime

import numpy as np

SUCC_RE = re.compile(r"Success Envs: tensor\(\[([^\]]*)\]")
MOVED_RE = re.compile(r"successes/env_\d+_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})")


def parse_log(path):
    batches, viser_t, moved = [], None, []
    timestamped = None
    pending = None  # torch wraps long bool tensors over several lines (num_envs >= 12)
    with open(path, errors="replace") as f:
        for line in f:
            ts = None
            head = line.split(" ", 1)[0]
            try:
                ts = float(head)
                timestamped = True
            except ValueError:
                timestamped = timestamped or False
            if viser_t is None and "viser (listening" in line:
                viser_t = ts
            if pending is not None:
                body = line.split(" ", 1)[1] if timestamped and " " in line else line
                pending = (pending[0], pending[1] + " " + body.strip())
                if "]" not in body:
                    continue
                ts, line = pending
                pending = None
            elif "Success Envs: tensor([" in line and "]" not in line.split("tensor([", 1)[1]:
                pending = (ts, line.rstrip("\n"))
                continue
            m = SUCC_RE.search(line)
            if m:
                flags = [s.strip() == "True" for s in m.group(1).split(",") if s.strip()]
                batches.append((ts, flags))
            m = MOVED_RE.search(line)
            if m:
                moved.append(datetime.strptime(m.group(1), "%Y_%m_%d_%H_%M_%S").timestamp())
    return batches, viser_t, moved, bool(timestamped)


def gpu_stats(path, t0, t_steady):
    rows = []
    with open(path) as f:
        for r in csv.reader(f):
            if len(r) < 5:
                continue
            try:
                t = datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
                rows.append((t, int(r[1]), float(r[2]), float(r[3]), float(r[4])))
            except ValueError:
                continue
    out = {}
    for g in sorted({r[1] for r in rows}):
        rg = [r for r in rows if r[1] == g]
        base = rg[0][2]
        steady = [r for r in rg if t_steady is not None and r[0] >= t_steady]
        out[f"gpu{g}"] = {
            "mem_baseline_MiB": base,
            "mem_peak_MiB": max(r[2] for r in rg),
            "mem_peak_delta_MiB": max(r[2] for r in rg) - base,
            "util_mean_steady_pct": float(np.mean([r[3] for r in steady])) if steady else None,
            "power_mean_steady_W": float(np.mean([r[4] for r in steady])) if steady else None,
        }
    return out


def analyze(run_dir):
    t_start = int(open(os.path.join(run_dir, "t_start")).read())
    t_end = int(open(os.path.join(run_dir, "t_end")).read()) if os.path.exists(os.path.join(run_dir, "t_end")) else None
    batches, viser_t, moved, timestamped = parse_log(os.path.join(run_dir, "run.log"))
    n_envs = len(batches[0][1]) if batches else 0
    env_txt = os.path.join(run_dir, "env.txt")
    if not n_envs:  # run produced no batch (e.g. OOM): fall back to the requested num_envs
        req = [l.split("=", 1)[1] for l in open(env_txt) if l.startswith("R2R2R_NUM_ENVS=")] if os.path.exists(env_txt) else []
        n_envs = int(req[0]) if req else 2
    # first "Success Envs" is printed at the end of the first rollout batch
    flags = np.array([b[1] for b in batches]) if batches else np.zeros((0, 0))
    res = {
        "run": os.path.basename(run_dir.rstrip("/")),
        "num_envs": n_envs,
        "wall_s": (t_end - t_start) if t_end else None,
        "n_batches": len(batches),
        "attempted_traj": int(flags.size),
        "successful_traj": int(flags.sum()),
        "success_rate": float(flags.mean()) if flags.size else None,
    }
    if timestamped and batches:
        bt = np.array([b[0] for b in batches])
        res["init_to_viser_s"] = (viser_t - t_start) if viser_t else None
        res["t_first_batch_end_s"] = float(bt[0] - t_start)
        dts = np.diff(bt)
    else:
        mv = np.unique(np.array(moved))
        dts = np.diff(mv) if len(mv) > 1 else np.array([])
        bt = mv
    if len(dts):
        steady_dur = float(bt[-1] - bt[0])
        n_steady = int(flags[1:].size)
        n_succ_steady = int(flags[1:].sum())
        res.update({
            "batch_time_s_median": float(np.median(dts)),
            "batch_time_s_mean": float(np.mean(dts)),
            "batch_time_s_p95": float(np.percentile(dts, 95)),
            "time_per_attempted_traj_s": float(np.median(dts) / max(n_envs, 1)),
            "steady_throughput_succ_traj_per_hour": n_succ_steady / steady_dur * 3600 if steady_dur > 0 else None,
            "steady_throughput_attempted_traj_per_hour": n_steady / steady_dur * 3600 if steady_dur > 0 else None,
        })
        t_steady = float(bt[0])
    else:
        t_steady = None
    res["gpu"] = gpu_stats(os.path.join(run_dir, "gpu.csv"), t_start, t_steady)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    results = [analyze(d) for d in a.run_dirs]
    print(json.dumps(results, indent=2))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
