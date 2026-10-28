"""Prepare R2R2R rollouts for TinyDP training without touching the source episodes.

    python policy_training/prepare_dataset.py manifest --source-root SRC --out MANIFEST
    python policy_training/prepare_dataset.py trial    --manifest MANIFEST --n 2 --out TRIAL_ROOT
    python policy_training/prepare_dataset.py views    --manifest MANIFEST --out-root VIEWS
    python policy_training/prepare_dataset.py stats    --manifest MANIFEST --views-root VIEWS --out-dir STATS
    python policy_training/prepare_dataset.py analyze  --manifest MANIFEST --views-root VIEWS --out REPORT
    python policy_training/prepare_dataset.py image-integrity --manifest MANIFEST --out FILE [--workers 16]

Every command exits non-zero on any failed check.
"""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2r2r_dp import adapter, analysis, manifest as mf, verify  # noqa: E402
from r2r2r_dp.schema import YUMI_COFFEE_MAKER, load_episode_arrays, validate_episode  # noqa: E402

TASK_GLOB = os.path.join("run0*", "output_data", "yumi_coffee_maker", "successes", "*")
STATS_KW = dict(num_pred_steps=16, subsample_steps=2)  # TinyDP defaults (SharedConfig / DatasetConfig)


def _fail(msg):
    print(f"FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def _source_dirs(m):
    return [os.path.join(m["source_root"], e["rel_path"]) for e in m["episodes"]]


def cmd_manifest(a):
    eps = sorted(glob.glob(os.path.join(a.source_root, TASK_GLOB)))
    if not eps:
        _fail("no episodes found")
    with ProcessPoolExecutor(16) as ex:
        recs = list(ex.map(validate_episode, eps, chunksize=8))
    bad = [r for r in recs if r["errors"]]
    if bad:
        _fail(f"{len(bad)} invalid episodes, e.g. {bad[0]['episode']}: {bad[0]['errors'][:2]}")
    m = mf.build_manifest(eps, a.source_root, salt=a.salt, n_val=a.n_val, n_test=a.n_test,
                          extra={"task": "yumi_coffee_maker", "episode_schema": "policy_training/r2r2r_dp/schema.py:YUMI_COFFEE_MAKER"})
    mf.save_manifest(m, a.out)
    runs = sorted({e["uid"].split("/")[0] for e in m["episodes"]})
    per_run = {r: {s: sum(e["split"] == s and e["uid"].startswith(r + "/") for e in m["episodes"]) for s in ("train", "val", "test")} for r in runs}
    print(json.dumps({"counts": m["counts"], "subsets": {k: len(v) for k, v in m["train_subsets"].items()}, "per_run": per_run}, indent=1))


def _load_and_iterate(view_root, n_items=8):
    """Instantiate TinyDP's sim dataset on the view (train-only stats) and fetch a few items."""
    import tempfile
    import tyro
    from tinydp.util.args import ExperimentConfig
    from tinydp.dataset.utils import default_vision_transform
    from r2r2r_dp import tinydp_bridge as tb
    args = tyro.cli(ExperimentConfig, args=["--dataset-cfg.dataset-root", view_root, "--dataset-cfg.is-sim-data"])
    out = tempfile.mkdtemp(prefix="r2r2r_trial_")
    stats = os.path.join(out, "stats.json")
    tb.compute_action_statistics_tinydp(view_root, args.shared_cfg, args.dataset_cfg, stats)
    args.logging_cfg.output_dir = out
    tb.install_ordered_glob()
    cls = tb.make_manifest_dataset_class(view_root, view_root, stats)
    ds = cls(args.dataset_cfg, args.shared_cfg, args.logging_cfg, default_vision_transform(args.shared_cfg.image_size), split="train")
    shapes = None
    for i in np.linspace(0, len(ds) - 1, n_items).astype(int):
        item = ds[int(i)]
        for k, v in item.items():
            if not np.isfinite(v.numpy()).all():
                _fail(f"non-finite {k} in item {i}")
        shapes = {k: list(v.shape) for k, v in item.items()}
    return len(ds), shapes


def cmd_trial(a):
    m = mf.load_manifest(a.manifest)
    entries = [e for e in m["episodes"] if e["split"] == "train"][: a.n]
    src_dirs = [os.path.join(m["source_root"], e["rel_path"]) for e in entries]
    before = verify.tree_fingerprint(src_dirs)
    adapter.build_view(a.out, entries, m["source_root"])
    problems = verify.verify_view(a.out, m["source_root"], entries)
    n_items, shapes = _load_and_iterate(a.out)
    after = verify.tree_fingerprint(src_dirs)
    if before != after:
        problems.append("source episodes changed")
    print(json.dumps({"n_episodes": len(entries), "problems": problems, "tinydp_items": n_items, "item_shapes": shapes,
                      "source_fingerprint_unchanged": before == after}, indent=1))
    if problems:
        _fail("trial verification")


def cmd_views(a):
    m = mf.load_manifest(a.manifest)
    src_dirs = _source_dirs(m)
    before = verify.tree_fingerprint(src_dirs)
    roots = adapter.build_all_views(m, a.out_root)
    eps = {e["uid"]: e for e in m["episodes"]}
    report = {}
    for key, root in roots.items():
        if key in ("val", "test"):
            entries = [e for e in m["episodes"] if e["split"] == key]
        else:
            entries = [eps[u] for u in m["train_subsets"][str(int(key.split("_n")[1]))]]
        problems = verify.verify_view(root, m["source_root"], entries)
        report[key] = {"root": root, "n": len(entries), "problems": problems[:5], "n_problems": len(problems)}
    after = verify.tree_fingerprint(src_dirs)
    report["source_fingerprint_unchanged"] = before == after
    report["source_fingerprint"] = after
    with open(os.path.join(a.out_root, "views.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    if before != after or any(v.get("n_problems") for v in report.values() if isinstance(v, dict)):
        _fail("view verification")


def cmd_stats(a):
    import tyro
    from tinydp.util.args import ExperimentConfig
    from r2r2r_dp import tinydp_bridge as tb
    m = mf.load_manifest(a.manifest)
    os.makedirs(a.out_dir, exist_ok=True)
    args = tyro.cli(ExperimentConfig, args=["--dataset-cfg.is-sim-data"])
    summary = {}
    for n in sorted(m["train_subsets"], key=int):
        root = os.path.join(a.views_root, "train_subsets", f"n{int(n):04d}")
        out = os.path.join(a.out_dir, f"action_statistics_train_n{int(n):04d}.json")
        if os.path.exists(out):
            _fail(f"{out} exists")
        s = tb.compute_action_statistics_tinydp(root, args.shared_cfg, args.dataset_cfg, out)
        ref = tb.compute_action_statistics_numpy(root, **STATS_KW)
        ok = np.allclose(s["min_action"], ref["min_action"], atol=1e-6) and np.allclose(s["max_action"], ref["max_action"], atol=1e-6)
        if not ok:
            _fail(f"TinyDP statistics for n={n} disagree with the independent computation")
        summary[n] = {"file": out, "shape": s["shape"], "matches_independent": ok}
    print(json.dumps(summary, indent=1))


def cmd_stats_v2(a):
    """Audit M1: v2 train-only bounds over every loader-returned target; v1 files are only read."""
    from r2r2r_dp import normalizer_v2 as nv
    m = mf.load_manifest(a.manifest)
    msha = nv.file_sha256(a.manifest)
    os.makedirs(a.out_dir, exist_ok=True)
    s, L, P = STATS_KW["subsample_steps"], 4, STATS_KW["num_pred_steps"]
    val_h5s = nv.view_episode_h5s(os.path.join(a.views_root, "splits", "val"))
    summary = {"manifest_sha256": msha, "subsets": {}}
    for n in sorted(m["train_subsets"], key=int):
        tag = f"n{int(n):04d}"
        root = os.path.join(a.views_root, "train_subsets", tag)
        out = os.path.join(a.out_dir, f"action_statistics_v2_train_{tag}.json")
        if os.path.exists(out):
            _fail(f"{out} exists")
        st = nv.build_statistics(root, tag, m["train_subsets"][n], msha, s, L, P)
        h5s = nv.view_episode_h5s(root)
        cov2 = nv.coverage(h5s, st, s, L, P)
        if cov2["rows_out_of_bounds"]:
            _fail(f"{tag}: v2 bounds do not cover the training targets: {cov2}")
        v1 = json.load(open(os.path.join(a.v1_dir, f"action_statistics_train_{tag}.json")))
        nv.dump_statistics(st, out)
        lo1, hi1 = np.array(v1["min_action"]), np.array(v1["max_action"])
        summary["subsets"][tag] = {
            "file": os.path.relpath(out), "sha256": nv.file_sha256(out),
            "v2_train_coverage_right": cov2, "v1_train_coverage_right": nv.coverage(h5s, v1, s, L, P),
            "v2_minus_v1_min": (np.array(st["min_action"]) - lo1).tolist(),
            "v2_minus_v1_max": (np.array(st["max_action"]) - hi1).tolist(),
            "val_targets_outside_v2_train_bounds_right": nv.coverage(val_h5s, st, s, L, P),
        }
    with open(os.path.join(a.out_dir, "normalizer_v2_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, allow_nan=False)
        f.write("\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if "coverage" in kk or kk.startswith("val")}
                      for k, v in summary["subsets"].items()}, indent=1))


def _profile(ep_dir):
    arr = load_episode_arrays(ep_dir)
    p = analysis.joint_jump_profile(arr["joint_angles"], arr["ee_poses"])
    p["row_order_violations_ee"] = len(analysis.row_order_violations(arr["ee_poses"][:, 7:10]))
    # same detector on joints: without an EE violation in the same row pair this is IK chatter, not reordering
    p["joint_zigzag_steps"] = len(analysis.row_order_violations(arr["joint_angles"][:, :14], min_abs=1e-2))
    return p


def _lag_at_jump(args):
    ep_dir, t = args
    out = {}
    for cam in ("camera_0", "camera_1"):
        en = analysis.image_motion_energy(ep_dir, cam, YUMI_COFFEE_MAKER.num_steps)
        w = np.arange(max(0, t - 3), min(len(en), t + 6))
        out[cam] = int(w[np.argmax(en[w])] - t)
    return out


def _lag_xcorr(ep_dir):
    arr = load_episode_arrays(ep_dir)
    speed = np.linalg.norm(np.diff(arr["ee_poses"][:, 7:10].astype(np.float64), axis=0), axis=1)
    return {cam: analysis.best_lag(speed, analysis.image_motion_energy(ep_dir, cam, YUMI_COFFEE_MAKER.num_steps))["lag"]
            for cam in ("camera_0", "camera_1")}


def cmd_analyze(a):
    from collections import Counter
    from r2r2r_dp import tinydp_bridge as tb
    m = mf.load_manifest(a.manifest)
    dirs = _source_dirs(m)
    with ProcessPoolExecutor(16) as ex:
        profs = list(ex.map(_profile, dirs, chunksize=16))
    for e, p in zip(m["episodes"], profs):
        p["uid"], p["split"] = e["uid"], e["split"]
    flagged = [p for p in profs if p["flagged"]]
    rep = {"n_episodes": len(profs), "n_flagged_gt_0.3rad": len(flagged),
           "flagged_by_split": dict(Counter(p["split"] for p in flagged)),
           "flagged_uids": sorted(p["uid"] for p in flagged),
           "flagged_step_hist": dict(Counter(p["step"] for p in flagged)),
           "flagged_joint_hist": dict(Counter(YUMI_COFFEE_MAKER.joint_names[p["joint_index"]] for p in flagged)),
           "row_order_violations_episodes_ee": sum(1 for p in profs if p["row_order_violations_ee"]),
           "joint_zigzag_episodes": sorted(p["uid"] for p in profs if p["joint_zigzag_steps"])}
    for k in ("right_ee_step_mm_at_jump", "right_ee_rot_deg_at_jump", "right_ee_step_mm_max", "right_ee_rot_deg_max", "max_joint_step"):
        for name, grp in (("flagged", flagged), ("unflagged", [p for p in profs if not p["flagged"]])):
            v = np.array([p[k] for p in grp])
            rep[f"{k}_{name}_median_p99_max"] = [float(np.median(v)), float(np.percentile(v, 99)), float(v.max())]
    # effect on train-only normalisation: full train set with vs without flagged episodes
    train_root = os.path.join(a.views_root, "train_subsets", f"n{m['counts']['train']:04d}")
    flagged_uids = {p["uid"] for p in flagged}
    keep_root = os.path.join(a.views_root, "_analysis_train_without_flagged")
    if not os.path.exists(keep_root):
        os.makedirs(keep_root)
        for ep in sorted(os.listdir(train_root)):
            if json.load(open(os.path.join(train_root, ep, "source.json")))["uid"] not in flagged_uids:
                os.symlink(os.path.join(train_root, ep), os.path.join(keep_root, ep))
    s_all = tb.compute_action_statistics_numpy(train_root, **STATS_KW)
    s_keep = tb.compute_action_statistics_numpy(keep_root, **STATS_KW)
    rng_all = np.array(s_all["max_action"]) - np.array(s_all["min_action"])
    rng_keep = np.array(s_keep["max_action"]) - np.array(s_keep["min_action"])
    rep["stats_range_ratio_with_vs_without_flagged"] = np.round(rng_all / np.where(rng_keep > 0, rng_keep, np.nan), 6).tolist()
    rep["stats_max_abs_diff_with_vs_without_flagged"] = float(np.nanmax(np.abs(np.r_[np.subtract(s_all["min_action"], s_keep["min_action"]), np.subtract(s_all["max_action"], s_keep["max_action"])])))
    # image/state timing: sharp events (flagged joint jumps) and cross-correlation on a deterministic sample
    with ProcessPoolExecutor(16) as ex:
        jl = list(ex.map(_lag_at_jump, [(os.path.join(m["source_root"], next(e["rel_path"] for e in m["episodes"] if e["uid"] == p["uid"])), p["step"]) for p in flagged]))
        sample = dirs[:: max(1, len(dirs) // 60)]
        xl = list(ex.map(_lag_xcorr, sample))
    rep["image_lag_at_joint_jumps_frames"] = {cam: dict(Counter(r[cam] for r in jl)) for cam in ("camera_0", "camera_1")}
    rep["image_lag_xcorr_frames"] = {cam: dict(Counter(r[cam] for r in xl)) for cam in ("camera_0", "camera_1")}
    rep["image_lag_xcorr_n"] = len(xl)
    # left arm is passive: EE motion range per episode
    left_range = []
    for d in dirs[:: max(1, len(dirs) // 200)]:
        ee = load_episode_arrays(d)["ee_poses"][:, 0:3].astype(np.float64)
        left_range.append(float(np.linalg.norm(ee.max(0) - ee.min(0)) * 1e3))
    rep["left_ee_range_mm_per_episode_median_max"] = [float(np.median(left_range)), float(np.max(left_range))]
    with open(a.out, "w") as f:
        json.dump({"summary": rep, "episodes": profs}, f, indent=1)
    print(json.dumps(rep, indent=1))


def cmd_image_integrity(a):
    """Codex final audit F4: decode every source frame once (read-only) and record per-camera byte digests."""
    from r2r2r_dp import image_integrity as ii
    from r2r2r_dp.normalizer_v2 import file_sha256
    m = mf.load_manifest(a.manifest)
    try:
        rec = ii.build_image_integrity(m, file_sha256(a.manifest), YUMI_COFFEE_MAKER.image_size, workers=a.workers)
    except ii.ImageIntegrityError as e:
        _fail(str(e))
    ii.dump_image_integrity(rec, a.out)
    print(json.dumps({"out": a.out, "sha256": file_sha256(a.out), "episodes": len(rec["episodes"]),
                      "frames": 2 * sum(e["num_steps"] for e in m["episodes"]), "image_size": rec["image_size"]}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("manifest"); p.add_argument("--source-root", required=True); p.add_argument("--out", required=True)
    p.add_argument("--salt", default="r2r2r-dp-v1"); p.add_argument("--n-val", type=int, default=182); p.add_argument("--n-test", type=int, default=182)
    p = sp.add_parser("trial"); p.add_argument("--manifest", required=True); p.add_argument("--n", type=int, required=True); p.add_argument("--out", required=True)
    p = sp.add_parser("views"); p.add_argument("--manifest", required=True); p.add_argument("--out-root", required=True)
    p = sp.add_parser("stats"); p.add_argument("--manifest", required=True); p.add_argument("--views-root", required=True); p.add_argument("--out-dir", required=True)
    p = sp.add_parser("stats-v2"); p.add_argument("--manifest", required=True); p.add_argument("--views-root", required=True); p.add_argument("--v1-dir", required=True); p.add_argument("--out-dir", required=True)
    p = sp.add_parser("analyze"); p.add_argument("--manifest", required=True); p.add_argument("--views-root", required=True); p.add_argument("--out", required=True)
    p = sp.add_parser("image-integrity"); p.add_argument("--manifest", required=True); p.add_argument("--out", required=True); p.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    {"manifest": cmd_manifest, "trial": cmd_trial, "views": cmd_views, "stats": cmd_stats, "stats-v2": cmd_stats_v2, "analyze": cmd_analyze,
     "image-integrity": cmd_image_integrity}[a.cmd](a)


if __name__ == "__main__":
    main()
