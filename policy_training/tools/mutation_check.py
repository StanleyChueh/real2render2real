"""Re-introduce each audited defect (M1-M5 and Codex final F1-F8) into a scratch copy of policy_training
and confirm its regression test fails.  Usage: .venv-tinydp/bin/python policy_training/tools/mutation_check.py OUT.json

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
     "        rows = released_target_rows(len(a), s, L, P)\n        if len(rows) == 0",
     "        rows = np.arange(s, len(a), s)\n        if len(rows) == 0",
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
     "        self.ema.load_state_dict({k: v.to(self.device) for k, v in st[\"ema\"].items()})  # exact key set checked\n", "",
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
    # ---- Codex final audit (43ab363) F1-F8: undo one repair at a time ----
    ("F1 skipped GradScaler step counted as an update", "r2r2r_dp/driver.py",
     "            steps = self.optimizer_calls - calls_before\n", "            steps = 1\n",
     "tests/test_codex_final_training.py::test_skipped_step_advances_nothing_but_data"),
    ("F1 no bound on consecutive skipped steps", "r2r2r_dp/driver.py",
     "                if (self.consecutive_skipped >= MAX_CONSECUTIVE_SKIPPED_UPDATES\n",
     "                if (False\n", "tests/test_codex_final_training.py::test_persistent_overflow_aborts_without_counting"),
    ("F1 skip counters not restored on resume", "r2r2r_dp/driver.py",
     "        self.attempted_updates, self.skipped_updates = st[\"attempted_updates\"], st[\"skipped_updates\"]\n", "",
     "tests/test_codex_final_training.py::test_resume_after_skip_equals_uninterrupted"),
    ("F2 normalized() sanitizes non-finite actions", "r2r2r_dp/normalizer_v2.py",
     "    a = require_finite(\"actions\", a)\n", "    a = np.asarray(a, dtype=np.float64)\n",
     "tests/test_codex_final_inputs.py::test_normalized_rejects_nonfinite_actions_and_bounds"),
    ("F2 statistics built from non-finite sidecars (all finite checks off)", "r2r2r_dp/normalizer_v2.py",
     "    if not np.isfinite(a).all():\n        bad = np.argwhere", "    if False:\n        bad = np.argwhere",
     "tests/test_codex_final_inputs.py::test_nan_sidecar_cannot_be_certified"),
    ("F3 manifest counts not checked", "r2r2r_dp/manifest.py",
     "    if counts != actual:\n", "    if False:\n", "tests/test_codex_final_inputs.py::test_manifest_semantics"),
    ("F3 manifest salt order not checked", "r2r2r_dp/manifest.py",
     "        if (e[\"split\"], e[\"split_rank\"]) != want:\n", "        if False:\n", "tests/test_codex_final_inputs.py::test_manifest_semantics"),
    ("F3 manifest version not checked", "r2r2r_dp/manifest.py",
     "    if not _is_int(m[\"version\"]) or m[\"version\"] != MANIFEST_VERSION:\n", "    if False:\n",
     "tests/test_codex_final_inputs.py::test_manifest_semantics"),
    ("F3 view directory names not canonical", "r2r2r_dp/launch_checks.py",
     "        if parts[0] != canonical:\n", "        if False:\n", "tests/test_codex_final_inputs.py::test_renamed_view_rejected"),
    ("F3 symlinked episode aliases accepted", "r2r2r_dp/launch_checks.py",
     "        if os.path.islink(ep) or os.path.islink(os.path.join(ep, \"robot_data\")):\n", "        if False:\n",
     "tests/test_codex_final_inputs.py::test_symlinked_episode_alias_rejected"),
    ("F3 statistics config (layout) not compared", "r2r2r_dp/launch_checks.py",
     "        \"config\": (st[\"config\"], statistics_config(s, L, P)),\n", "",
     "tests/test_codex_final_inputs.py::test_statistics_declared_config_enforced"),
    ("F3 statistics config digest not recomputed", "r2r2r_dp/launch_checks.py",
     "        \"config_sha256\": (st[\"config_sha256\"], config_sha256(st[\"config\"])),\n", "",
     "tests/test_codex_final_inputs.py::test_statistics_declared_config_enforced"),
    ("F3 checkpoint sample order not compared", "r2r2r_dp/driver.py",
     "        if st[\"train_order_sha256\"] != self.train_order_sha256:\n", "        if False:\n",
     "tests/test_codex_final_training.py::test_corrupt_checkpoint_rejected_before_any_state_changes"),
    ("F4 frame bytes not verified at launch", "r2r2r_dp/launch_checks.py",
     "    problems.extend(verify_episodes(eps, ii, workers))\n", "",
     "tests/test_codex_final_inputs.py::test_corrupt_wrong_size_and_missing_frames_rejected"),
    ("F4 builder does not check resolution", "r2r2r_dp/image_integrity.py",
     "        if fmt != \"JPEG\" or mode != \"RGB\" or tuple(got) != tuple(size):\n", "        if False:\n",
     "tests/test_codex_final_inputs.py::test_builder_decodes_and_checks_resolution"),
    ("F5 negative/zero grad_clip accepted", "r2r2r_dp/protocol.py",
     "    if not _is_real(s[\"grad_clip\"], 0, lo_open=True):\n", "    if False:\n",
     "tests/test_codex_final_inputs.py::test_invalid_protocol_values_rejected"),
    ("F5 booleans accepted as integers", "r2r2r_dp/protocol.py",
     "    return type(v) is int and (lo is None", "    return isinstance(v, int) and (lo is None",
     "tests/test_codex_final_inputs.py::test_invalid_protocol_values_rejected"),
    ("F5 resolved TinyDP values not validated", "r2r2r_dp/protocol.py",
     "    validate_resolved_tinydp(cfg)\n", "", "tests/test_codex_final_inputs.py::test_invalid_protocol_values_rejected"),
    ("F5 NaN literals accepted in protocol files", "r2r2r_dp/protocol.py",
     "        p = json.load(f, parse_constant=_reject_constant)\n", "        p = json.load(f)\n",
     "tests/test_codex_final_inputs.py::test_protocol_file_is_strict_json"),
    ("F6 validation not transactional", "r2r2r_dp/driver.py",
     "        finally:\n            if raw is not None:\n                self.model.load_state_dict(raw)\n            self.model.train(was_training)\n",
     "        except BaseException:\n            raise\n        if raw is not None:\n            self.model.load_state_dict(raw)\n        self.model.train(was_training)\n",
     "tests/test_codex_final_training.py::test_validation_failure_restores_weights_and_mode"),
    ("F6 non-finite validation loss accepted", "r2r2r_dp/driver.py",
     "                            if not math.isfinite(loss):\n", "                            if False:\n",
     "tests/test_codex_final_training.py::test_nonfinite_and_empty_validation_raise"),
    ("F6 non-finite EMA export accepted", "r2r2r_dp/driver.py",
     "        if bad:\n            raise FloatingPointError(f\"EMA export", "        if False:\n            raise FloatingPointError(f\"EMA export",
     "tests/test_codex_final_training.py::test_export_failure_restores_raw_weights"),
    ("F7 EMA key set/shape/finiteness not checked", "r2r2r_dp/driver.py",
     "        p += _tensor_dict_problems(\"ema\", st[\"ema\"], self.ema.ema)\n", "",
     "tests/test_codex_final_training.py::test_corrupt_checkpoint_rejected_before_any_state_changes"),
    ("F7 counters/data position not checked", "r2r2r_dp/driver.py",
     "        if mb != accum * att:\n", "        if False:\n",
     "tests/test_codex_final_training.py::test_corrupt_checkpoint_rejected_before_any_state_changes"),
    ("F8 non-finite inference inputs accepted", "r2r2r_dp/inference.py",
     "            if not x.is_floating_point() or not torch.isfinite(x).all():\n", "            if False:\n",
     "tests/test_codex_final_inputs.py::test_inference_rejects_invalid_inputs"),
    ("F8 non-finite policy output returned", "r2r2r_dp/inference.py",
     "        if not (torch.isfinite(z).all() and torch.isfinite(action).all()):\n", "        if False:\n",
     "tests/test_codex_final_inputs.py::test_nonfinite_output_and_weights_rejected"),
    ("F8 collinear rot6d rows accepted", "r2r2r_dp/rotation.py",
     "        raise ValueError(\"rot6d rows are zero or (nearly) collinear\")\n", "        pass\n",
     "tests/test_codex_final_inputs.py::test_rotation_converter_cases"),
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
            try:
                r = subprocess.run([PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", test], cwd=work,
                                   capture_output=True, text=True, timeout=900)
            except subprocess.TimeoutExpired:  # the test did not pass within the limit: counted as detected
                results.append({"mutation": name, "test": test, "pytest_exit": None, "detected": True,
                                "summary": "timeout after 900 s"})
            else:
                tail = [l for l in r.stdout.splitlines() if "passed" in l or "failed" in l or "error" in l.lower()][-1:]
                results.append({"mutation": name, "test": test, "pytest_exit": r.returncode,
                                "detected": r.returncode != 0, "summary": tail[0] if tail else ""})
        print(json.dumps(results[-1]), flush=True)
    json.dump({"control": ctl, "mutations": results}, open(sys.argv[1], "w"), indent=1)
    print("all detected:", all(x.get("detected") for x in results))


if __name__ == "__main__":
    main()
