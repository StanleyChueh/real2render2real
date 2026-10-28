"""Invariants after the M1-M7 and Codex-final (F1-F8) repairs (read-only).

    .venv-tinydp/bin/python policy_training/verify_repair_invariants.py --base 2a05366 \
        --checkpoint RUN/ckpt_full_u0000100.pt --out report.json

Checks: protected repository paths equal the data-generation baseline; submodules clean and at their pins;
v1 statistics and manifest unchanged; source-episode fingerprint equals the preparation record; no view file
modified since the given date; all train sizes still pass the launch checks (incl. image integrity);
checkpoint counters: Adam steps == scheduler steps == successful updates, attempted == successful +
skipped, micro-batches == accum x attempted == epoch x batches_per_epoch + batch. Legacy v2 checkpoints
(no skip counters) are checked with attempted == successful.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

SOURCE_FINGERPRINT = "dd23bd44ce626485a9f3ec8f78ce3c75cfeadb90785efdcf456c8f997c2f2bcd"
PROTECTED = ["scripts", "real2render2real", "dependencies", "pyproject.toml", "uv.lock", "env_data_gen.sh"]


def git(*a, cwd=REPO):
    r = subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="commit the repair branch started from")
    ap.add_argument("--baseline-branch", default="repro/baseline-datagen")
    ap.add_argument("--checkpoint", action="append", default=[])
    ap.add_argument("--views-modified-since", default="2026-10-24")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rep = {}
    rc, out = git("diff", "--stat", "--ignore-submodules=dirty", a.baseline_branch, "--", *PROTECTED)
    rep["protected_paths_equal_baseline_branch"] = {"ok": rc == 0 and out == "", "diff": out}
    # pre-existing, documented since the data-generation audit: generated trajgen.egg-info metadata only
    rc, out = git("status", "--porcelain", cwd=os.path.join(REPO, "dependencies", "trajgen"))
    rep["trajgen_worktree_drift"] = {"ok": all(l.split()[-1].startswith("trajgen.egg-info/") for l in out.splitlines()),
                                     "status": out}
    rc, out = git("submodule", "status", "--recursive")
    dirty = [l for l in out.splitlines() if l[:1] in "+-U"]
    rep["submodules_at_pins"] = {"ok": not dirty, "not_at_pin": dirty}
    rc, out = git("status", "--porcelain", cwd=os.path.join(REPO, "dependencies", "tinydp"))
    rep["tinydp_worktree_clean"] = {"ok": out == "", "status": out}
    rc, base_sha = git("rev-parse", a.baseline_branch)
    rep["baseline_branch_head"] = base_sha
    v1 = ["docs/policy_training/artifacts/manifest_v1.json"] + sorted(
        os.path.relpath(p, REPO) for p in glob.glob(os.path.join(REPO, "docs/policy_training/artifacts/action_statistics_train_n*.json")))
    rc, out = git("diff", "--stat", a.base, "--", *v1)
    rep["v1_statistics_and_manifest_unchanged"] = {"ok": rc == 0 and out == "", "files": v1, "diff": out}

    from r2r2r_dp import verify
    from r2r2r_dp.manifest import load_manifest
    m = load_manifest(os.path.join(REPO, "docs/policy_training/artifacts/manifest_v1.json"))
    t = time.time()
    fp = verify.tree_fingerprint([os.path.join(m["source_root"], e["rel_path"]) for e in m["episodes"]])
    rep["source_fingerprint"] = {"ok": fp == SOURCE_FINGERPRINT, "value": fp, "seconds": round(time.time() - t, 1)}
    views = os.path.expanduser("~/Stanley_ws/r2r2r_policy_runs/dataset_v1/views")
    newer = subprocess.run(["find", views, "-newermt", a.views_modified_since, "-not", "-type", "d"],
                           capture_output=True, text=True).stdout.split()
    rep["views_unmodified"] = {"ok": not newer, "files_newer_than": a.views_modified_since, "examples": newer[:5]}

    from r2r2r_dp.launch_checks import check_launch
    from r2r2r_dp.protocol import load_protocol, resolve
    p = load_protocol(os.path.join(HERE, "configs", "baseline_v3.json"))
    lc = {}
    for n in (50, 100, 150, 1000, 1461):
        try:
            lc[n] = {"ok": True, **check_launch(resolve(p, n, f"/nonexistent/probe_n{n}"))}
        except Exception as e:  # report, do not hide
            lc[n] = {"ok": False, "error": str(e)[:2000]}
    rep["launch_checks_all_sizes"] = {"ok": all(v["ok"] for v in lc.values()), "per_size": lc}

    import torch
    ck = {}
    for path in a.checkpoint:
        st = torch.load(path, map_location="cpu", weights_only=False)
        steps = sorted({int(s["step"]) for s in st["optimizer"]["state"].values()})
        accum = st["run"]["protocol"]["schedule"]["accum_micro_batches"]
        att, sk = st.get("attempted_updates", st["update"]), st.get("skipped_updates", 0)
        dp = st["data_position"]
        ck[path] = {"format": st["format"], "update": st["update"], "attempted_updates": att, "skipped_updates": sk,
                    "micro_batches": st["micro_batches"], "adam_steps": steps,
                    "scheduler_last_epoch": st["scheduler"]["last_epoch"], "data_position": dp,
                    "ok": steps == [st["update"]] and st["scheduler"]["last_epoch"] == st["update"]
                          and att == st["update"] + sk and st["micro_batches"] == accum * att
                          and st["micro_batches"] == dp["epoch"] * dp["batches_per_epoch"] + dp["batch"]}
    rep["checkpoint_counters"] = {"ok": all(v["ok"] for v in ck.values()), "per_checkpoint": ck}
    rep["all_ok"] = all(v["ok"] for v in rep.values() if isinstance(v, dict) and "ok" in v)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1, allow_nan=False)
    print(json.dumps({k: (v["ok"] if isinstance(v, dict) and "ok" in v else v) for k, v in rep.items()}, indent=1))
    sys.exit(0 if rep["all_ok"] else 1)


if __name__ == "__main__":
    main()
