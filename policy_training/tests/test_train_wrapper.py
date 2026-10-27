"""train_tinydp.py argument guards and dataset substitution (does not start training)."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import train_tinydp  # noqa: E402


@pytest.fixture
def roots(tmp_path):
    for d in ("train", "val"):
        (tmp_path / d).mkdir()
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"shape": [20], "min_action": [0.0] * 20, "max_action": [1.0] * 20}))
    return tmp_path, ["--train-root", str(tmp_path / "train"), "--eval-root", str(tmp_path / "val"), "--action-statistics", str(stats)]


@pytest.mark.parametrize("extra", [
    [],  # missing is-sim-data
    ["--dataset-cfg.is-sim-data", "--logging-cfg.log-name", "x"],  # would start WandB
    ["--dataset-cfg.is-sim-data", "--dataset-cfg.dataset-root", "/elsewhere"],
    ["--dataset-cfg.is-sim-data", "--dataset-cfg.data-subsample-num-traj=50"],
    ["--dataset-cfg.is-sim-data", "--dataset-cfg.action-statistics", "/x.json"],
])
def test_rejects_unsafe_arguments(roots, extra):
    tmp, ours = roots
    with pytest.raises(SystemExit):
        train_tinydp.prepare(ours + ["--", *extra, "--logging-cfg.output-dir", str(tmp / "out")])


def test_installs_manifest_dataset_and_records_provenance(roots, monkeypatch):
    tmp, ours = roots
    monkeypatch.delenv("WANDB_MODE", raising=False)
    args = train_tinydp.prepare(ours + ["--", "--dataset-cfg.is-sim-data", "--logging-cfg.output-dir", str(tmp / "out")])
    import tinydp.dataset.image_dataset_sim as sim
    from r2r2r_dp.tinydp_bridge import ManifestSimDataset
    assert sim.SequenceDataset is ManifestSimDataset
    assert ManifestSimDataset._config["train_root"].endswith("train") and ManifestSimDataset._config["eval_root"].endswith("val")
    assert os.environ["WANDB_MODE"] == "disabled"
    prov = json.load(open(tmp / "out" / "r2r2r_provenance.json"))
    assert prov["tinydp_args"] == args and len(prov["train_py_sha256"]) == 64


def test_legacy_launcher_requires_explicit_flag(roots, monkeypatch):
    tmp, ours = roots
    monkeypatch.setattr(train_tinydp.runpy, "run_path", lambda *a, **k: pytest.fail("must not start training"))
    with pytest.raises(SystemExit, match="legacy v1 launcher"):
        train_tinydp.main(ours + ["--", "--dataset-cfg.is-sim-data", "--logging-cfg.output-dir", str(tmp / "out")])
