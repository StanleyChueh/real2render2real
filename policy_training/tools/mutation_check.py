"""Phase 4 (usage: .venv-tinydp/bin/python policy_training/tools/mutation_check.py OUT.json): re-introduce each audited defect into a scratch copy of policy_training and confirm a test fails.

Works on a copy (never edits the repository). Each mutation is a single exact string replacement.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = os.path.join(REPO, ".venv-tinydp", "bin", "python")

MUTATIONS = [
    ("M1 even-row-only normalizer", "r2r2r_dp/normalizer_v2.py",
     "        rows = released_target_rows(len(a), s, L, P)\n        mins.append",
     "        rows = np.arange(s, len(a), s)\n        mins.append",
     "tests/test_normalizer_v2.py::test_odd_row_spike_v1_out_of_bounds_v2_covered"),
    ("M2 scheduler per micro-batch", "r2r2r_dp/driver.py",
     "                self._advance_position()\n",
     "                self._advance_position()\n                self.scheduler.step() if m < accum - 1 else None\n",
     "tests/test_driver.py::test_update_semantics"),
    ("M2 EMA per micro-batch", "r2r2r_dp/driver.py",
     "                self._advance_position()\n",
     "                self._advance_position()\n                self.ema.update(self.model) if m < accum - 1 else None\n",
     "tests/test_driver.py::test_update_semantics"),
    ("M2 loss logged divided by accum", "r2r2r_dp/driver.py",
     "                raws.append(raw)\n", "                raws.append(raw / accum)\n",
     "tests/test_driver.py::test_update_semantics"),
    ("M2 drop partial accumulation at epoch end", "r2r2r_dp/driver.py",
     "            for batch in list(iter(self.sampler))[b:]:\n",
     "            for batch in list(iter(self.sampler))[b:len(self.sampler) // 4 * 4]:\n",
     "tests/test_driver.py::test_update_semantics"),
    ("M2 no set_epoch", "r2r2r_dp/driver.py",
     "            self.sampler.set_epoch(e)\n", "            self.sampler.set_epoch(0)\n",
     "tests/test_driver.py::test_update_semantics"),
    ("M3 scheduler state not restored", "r2r2r_dp/driver.py",
     "        self.scheduler.load_state_dict(st[\"scheduler\"])\n", "",
     "tests/test_driver.py::test_interrupted_resume_equals_uninterrupted"),
    ("M3 EMA not restored", "r2r2r_dp/driver.py",
     "        self.ema.load_state_dict({k: v.to(self.device) for k, v in st[\"ema\"].items()})\n", "",
     "tests/test_driver.py::test_interrupted_resume_equals_uninterrupted"),
    ("M3 RNG not restored", "r2r2r_dp/driver.py",
     "            _set_rng_state(self._resume_rng, self.device)\n", "            pass\n",
     "tests/test_driver.py::test_interrupted_resume_equals_uninterrupted"),
    ("M3 data position not restored", "r2r2r_dp/driver.py",
     "        self.epoch, self.batch_in_epoch = st[\"data_position\"][\"epoch\"], st[\"data_position\"][\"batch\"]\n", "",
     "tests/test_driver.py::test_interrupted_resume_equals_uninterrupted"),
    ("M3 item seeding clobbers training RNG", "r2r2r_dp/driver.py",
     "                random.setstate(py)\n                np.random.set_state(npst)\n", "                pass\n",
     "tests/test_driver.py::test_item_seeding_is_keyed_and_leaves_caller_rng_untouched"),
    ("M4 no root overlap check", "r2r2r_dp/launch_checks.py",
     "                problems.append(f\"{a} root {roots[a]} overlaps {b} root {roots[b]}\")", "                pass",
     "tests/test_launch_checks.py::test_train_eval_overlap_rejected"),
    ("M4 no sidecar hash check", "r2r2r_dp/launch_checks.py",
     "        if len(side) == len(SIDE_CARS) and content_hash(side) != m[\"content_sha256\"]:",
     "        if False:", "tests/test_launch_checks.py::test_mutated_sidecar_rejected"),
    ("M4 no statistics provenance check", "r2r2r_dp/launch_checks.py",
     "        if got != exp:\n            problems.append", "        if False:\n            problems.append",
     "tests/test_launch_checks.py::test_wrong_subset_statistics_rejected"),
    ("M4 no numeric statistics recomputation", "r2r2r_dp/launch_checks.py",
     "    if recompute and not problems", "    if False and recompute and not problems",
     "tests/test_launch_checks.py::test_tampered_statistics_rejected"),
    ("M4 accept delta actions", "r2r2r_dp/protocol.py",
     "\"shared_cfg.use_delta_action\": False, \"model_cfg", "\"model_cfg",
     "tests/test_launch_checks.py::test_absolute_statistics_with_delta_actions_rejected"),
    ("M4 accept arbitrary TinyDP overrides", "r2r2r_dp/protocol.py",
     "    bad = set(t) - TINYDP_ALLOWED\n", "    bad = set()\n", "tests/test_launch_checks.py::test_protocol_rejections"),
    ("M4 output dir reuse", "r2r2r_dp/launch_checks.py",
     "    if not allow_existing_output and os.path.exists(out) and os.listdir(out):",
     "    if False:", "tests/test_launch_checks.py::test_nonempty_output_dir_rejected"),
    ("M5 inference uses raw weights by default", "r2r2r_dp/inference.py",
     "        if weights == \"ema\" and ckpt[\"kind\"] == \"full\":", "        if False:",
     "tests/test_inference.py"),
    ("M5 statistics not embedded in checkpoint", "r2r2r_dp/driver.py",
     "                \"action_statistics\": self.stats_json}", "                \"action_statistics\": None}",
     "tests/test_inference.py"),
    ("M5 denormalise with left-arm statistics", "r2r2r_dp/inference.py",
     "        self.right_min, self.right_max = lo[10:20], hi[10:20]", "        self.right_min, self.right_max = lo[0:10], hi[0:10]",
     "tests/test_inference.py"),
]


def link_repo(d):
    """The copy needs the (unmodified) submodules for provenance and docs/ for the real-data test."""
    for name in ("dependencies", "docs"):
        os.symlink(os.path.join(REPO, name), os.path.join(d, name))


def control():
    """All targeted tests must pass on the unmutated copy, so a failure can only come from the mutation."""
    tests = sorted({m[4] for m in MUTATIONS})
    with tempfile.TemporaryDirectory() as d:
        work = os.path.join(d, "policy_training")
        shutil.copytree(os.path.join(REPO, "policy_training"), work, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        link_repo(d)
        r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests], cwd=work,
                           capture_output=True, text=True, timeout=1800)
        tail = [l for l in r.stdout.splitlines() if "passed" in l or "failed" in l][-1:]
        return {"control_exit": r.returncode, "summary": tail[0] if tail else "", "tests": tests}


def main():
    ctl = control()
    print(json.dumps(ctl), flush=True)
    if ctl["control_exit"] != 0:
        sys.exit("control run failed; mutation results would be meaningless")
    results = []
    for name, rel, old, new, test in MUTATIONS:
        with tempfile.TemporaryDirectory() as d:
            work = os.path.join(d, "policy_training")
            shutil.copytree(os.path.join(REPO, "policy_training"), work,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            link_repo(d)
            path = os.path.join(work, rel)
            src = open(path).read()
            if src.count(old) != 1:
                results.append({"mutation": name, "status": "MUTATION NOT APPLICABLE", "count": src.count(old)})
                continue
            open(path, "w").write(src.replace(old, new))
            r = subprocess.run([PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", test], cwd=work,
                               capture_output=True, text=True, timeout=900)
            tail = [l for l in r.stdout.splitlines() if "passed" in l or "failed" in l or "error" in l.lower()][-1:]
            results.append({"mutation": name, "test": test, "pytest_exit": r.returncode,
                            "detected": r.returncode != 0, "summary": tail[0] if tail else ""})
        print(json.dumps(results[-1]), flush=True)
    json.dump({"control": ctl, "mutations": results}, open(sys.argv[1], "w"), indent=1)
    print("all detected:", all(x.get("detected") for x in results))


if __name__ == "__main__":
    main()
