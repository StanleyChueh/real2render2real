"""Reproduction figures (PNG) from run directories. Read-only w.r.t. data.

usage:
  python docs/reproduction/tools/make_figures.py --out FIG_DIR \
      --dataset RUN_DIR/output_data/yumi_coffee_maker \
      --runs RUN_DIR [RUN_DIR ...]          # metrics.json + gpu.csv per run
"""
import argparse
import csv
import glob
import json
import os
from datetime import datetime

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# reference categorical palette (dataviz skill, light mode), fixed order
C = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "lines.linewidth": 2,
})


def episodes(ds):
    return sorted(glob.glob(os.path.join(ds, "successes", "*")))


def fig_samples(ds, out, n_eps=3, ts=(0, 55, 95, 160)):
    eps = episodes(ds)
    pick = [eps[i] for i in np.linspace(0, len(eps) - 1, n_eps).astype(int)]
    cams = sorted(os.path.basename(c) for c in glob.glob(os.path.join(pick[0], "camera_*")))
    fig, axes = plt.subplots(n_eps * len(cams), len(ts), figsize=(3.2 * len(ts), 1.9 * n_eps * len(cams)))
    for i, ep in enumerate(pick):
        for j, cam in enumerate(cams):
            frames = sorted(glob.glob(os.path.join(ep, cam, "rgb", "*")))
            for k, t in enumerate(ts):
                ax = axes[i * len(cams) + j, k]
                im = cv2.imread(frames[min(t, len(frames) - 1)])[:, :, ::-1]
                ax.imshow(im)
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                for s in ax.spines.values():
                    s.set_visible(False)
                if i == 0 and j == 0:
                    ax.set_title(f"t = {t}")
                if k == 0:
                    ax.set_ylabel(f"{os.path.basename(ep)[:5]}…{os.path.basename(ep)[-8:]}\n{cam}", fontsize=8, color=INK2)
    fig.suptitle("Rendered RGB samples — YuMi coffee maker (2 calibrated cameras, 1280×720)", color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "samples_rgb.png"), dpi=110)
    plt.close(fig)


def load_robot(ep):
    with h5py.File(os.path.join(ep, "robot_data", "robot_data.h5"), "r") as f:
        return f["joint_angles"][:], f["ee_poses"][:], f["gripper_binary_cmd"][:].astype(float)


def fig_trajectories(ds, out, n=60):
    eps = episodes(ds)
    eps = [eps[i] for i in np.linspace(0, len(eps) - 1, min(n, len(eps))).astype(int)]
    data = [load_robot(e) for e in eps]
    # ee_poses: 14 = [left xyz + wxyz, right xyz + wxyz] (layout checked in 02_generation_results.md)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for name, sl, col, ax in (("left EE", slice(0, 3), C[0], axes[0]), ("right EE", slice(7, 10), C[1], axes[0])):
        for _, ee, _ in data:
            ax.plot(ee[:, sl][:, 0], ee[:, sl][:, 1], color=col, alpha=0.25, linewidth=1)
        ax.plot([], [], color=col, label=name)
    axes[0].set_title(f"EE paths, top view (x–y), {len(data)} episodes")
    axes[0].set_xlabel("x [m]"); axes[0].set_ylabel("y [m]"); axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend(frameon=False)
    for name, sl, col in (("left EE", slice(0, 3), C[0]), ("right EE", slice(7, 10), C[1])):
        for _, ee, _ in data:
            axes[1].plot(ee[:, sl][:, 2], color=col, alpha=0.25, linewidth=1)
        axes[1].plot([], [], color=col, label=name)
    axes[1].set_title("EE height over time"); axes[1].set_xlabel("step"); axes[1].set_ylabel("z [m]")
    axes[1].legend(frameon=False)
    g = np.stack([d[2][:, :] for d in data if d[2].shape == data[0][2].shape])
    for k, side in enumerate(("left", "right")[: g.shape[2]]):
        axes[2].plot(g[:, :, k].mean(0), color=C[k], label=f"{side} gripper closed (fraction of episodes)")
    axes[2].set_title("Gripper binary command (episode mean)"); axes[2].set_xlabel("step"); axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "trajectories.png"), dpi=110)
    plt.close(fig)

    ja = np.stack([d[0] for d in data if d[0].shape == data[0][0].shape])
    names = open(os.path.join(eps[0], "robot_data", "joint_names.txt")).read().split("\n")
    fig, axes = plt.subplots(2, 7, figsize=(16, 5), sharex=True)
    for j in range(14):
        ax = axes.flat[j]
        lo, med, hi = np.percentile(ja[:, :, j], [5, 50, 95], axis=0)
        ax.fill_between(np.arange(len(med)), lo, hi, color=C[0], alpha=0.2, linewidth=0)
        ax.plot(med, color=C[0])
        ax.set_title(names[j] if j < len(names) else f"joint {j}", fontsize=9)
    fig.suptitle(f"Arm joint angles [rad]: median and 5–95% band over {ja.shape[0]} episodes", color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "joint_angles.png"), dpi=110)
    plt.close(fig)


def read_gpu(run):
    rows = []
    with open(os.path.join(run, "gpu.csv")) as f:
        for r in csv.reader(f):
            try:
                rows.append((datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp(), int(r[1]), float(r[2]), float(r[3])))
            except (ValueError, IndexError):
                pass
    return np.array(rows)


def fig_gpu_timeline(runs, out):
    fig, axes = plt.subplots(len(runs), 1, figsize=(10, 2.6 * len(runs)), sharex=True, squeeze=False)
    for ax, run in zip(axes[:, 0], runs):
        a = read_gpu(run)
        t0 = int(open(os.path.join(run, "t_start")).read())
        for g in (0, 1):
            s = a[a[:, 1] == g]
            ax.plot((s[:, 0] - t0) / 60, s[:, 2] / 1024, color=C[g], label=f"GPU {g}")
        ax.set_ylabel("memory [GiB]"); ax.set_ylim(0, 24.5)
        ax.set_title(os.path.basename(run.rstrip("/")))
        ax.legend(frameon=False, loc="upper right")
    axes[-1, 0].set_xlabel("minutes since launch")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "gpu_memory_timeline.png"), dpi=110)
    plt.close(fig)


def fig_scaling(groups, out):
    """groups: list of (label, [metrics dicts]); one line series per group (max 3, fixed palette order)."""
    panels = [
        ("steady_throughput_succ_traj_per_hour", "successful trajectories / hour", None),
        ("batch_time_s_median", "median batch time [s]", None),
        ("gpu0_peak", "GPU 0 peak memory [GiB]", (0, 24.5)),
        ("success_rate", "success rate (feasibility filter)", (0, 1)),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.8))
    for ax, (key, label, ylim) in zip(axes, panels):
        for gi, (glabel, ms) in enumerate(groups):
            ms = sorted(ms, key=lambda r: r["num_envs"])
            ok = [r for r in ms if r["n_batches"] > 0]
            xs = [r["num_envs"] for r in ok]
            ys = [r["gpu"]["gpu0"]["mem_peak_MiB"] / 1024 if key == "gpu0_peak" else r.get(key) for r in ok]
            ax.plot(xs, ys, color=C[gi], marker="o", markersize=7, markeredgecolor=SURF, markeredgewidth=2, label=glabel)
            if key == "gpu0_peak":
                for r in ms:
                    if r["n_batches"] == 0:  # failed (OOM): mark at the memory it reached
                        ax.plot([r["num_envs"]], [r["gpu"]["gpu0"]["mem_peak_MiB"] / 1024], marker="x", markersize=9,
                                color=C[gi], linestyle="none")
                        ax.annotate("OOM", (r["num_envs"], r["gpu"]["gpu0"]["mem_peak_MiB"] / 1024), xytext=(-24, -4),
                                    textcoords="offset points", fontsize=8, color=INK2)
        ax.set_title(label, fontsize=10); ax.set_xlabel("num_envs")
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Scaling parallel environments on one RTX 4090 (GPU 0 only)", color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "scaling_runtime_gpu.png"), dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset")
    ap.add_argument("--scaling", nargs="*", default=[], help='"label=run_dir,run_dir,..." (one per series)')
    ap.add_argument("--timeline-runs", nargs="*", default=[])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.dataset:
        fig_samples(a.dataset, a.out)
        fig_trajectories(a.dataset, a.out)
    if a.timeline_runs:
        fig_gpu_timeline(a.timeline_runs, a.out)
    if a.scaling:
        groups = []
        for spec in a.scaling:
            label, runs = spec.split("=", 1)
            groups.append((label, [json.load(open(os.path.join(r, "metrics.json")))[0] for r in runs.split(",")]))
        fig_scaling(groups, a.out)


if __name__ == "__main__":
    main()
