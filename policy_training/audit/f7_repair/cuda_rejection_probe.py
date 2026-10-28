"""CUDA counterpart of the F7 rejection cases (usage: CUDA_VISIBLE_DEVICES=0 .venv-tinydp/bin/python
policy_training/audit/f7_repair/cuda_rejection_probe.py OUT.json). Synthetic fixture, small model, bf16 autocast,
real GradScaler('cuda'). Trains to u2 on CUDA, then checks that a CUDA Trainer accepts the intact checkpoint
and rejects corrupted CUDA RNG / Adam / scaler state before changing any state. Exit 1 on any unexpected result."""
import copy, json, os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
PT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [PT, os.path.join(PT, "tests")]
import torch  # noqa: E402
from r2r2r_dp import driver  # noqa: E402
from r2r2r_dp.protocol import resolve  # noqa: E402
from run_fixtures import make_run_fixture  # noqa: E402


def trainer(fx, out):
    r = resolve(fx["protocol"], 3, out)
    driver.write_run_files(r)
    return driver.Trainer(r, device="cuda")


def main():
    assert torch.cuda.device_count() == 1, "set CUDA_VISIBLE_DEVICES to exactly one GPU"
    rep = {"device": torch.cuda.get_device_name(0), "cases": {}}
    with tempfile.TemporaryDirectory() as d:
        fx = make_run_fixture(os.path.join(d, "fx"), total_updates=4, warmup_updates=2, val_every_updates=100,
                              full_val_every_updates=100, full_ckpt_every_updates=2, inference_export_every_updates=100)
        t = trainer(fx, os.path.join(d, "a"))
        t.train(stop_after=2)
        st = torch.load(os.path.join(t.out, "ckpt_full_u0000002.pt"), map_location="cpu", weights_only=False)
        rep["scaler_enabled"], rep["rng_keys"] = t.scaler._scaler.is_enabled(), sorted(st["rng"])
        k0 = next(iter(st["optimizer"]["state"]))
        cases = {"valid": lambda s: None,
                 "cuda_rng_missing": lambda s: s["rng"].pop("cuda"),
                 "cuda_rng_length": lambda s: s["rng"].__setitem__("cuda", torch.zeros(3, dtype=torch.uint8)),
                 "missing_one_adam_state": lambda s: s["optimizer"]["state"].pop(k0),
                 "scaler_empty": lambda s: s.__setitem__("scaler", {}),
                 "scaler_nan_scale": lambda s: s["scaler"].__setitem__("scale", float("nan"))}
        for name, f in cases.items():
            s = copy.deepcopy(st)
            f(s)
            p = os.path.join(d, name + ".pt")
            torch.save(s, p)
            u = trainer(fx, os.path.join(d, "load_" + name))
            try:
                u.load_full(p)
                rec = {"result": "applied", "update": u.update}
            except driver.CheckpointError as e:
                rec = {"result": "rejected", "error": str(e).split(": ", 1)[1][:300],
                       "state_unchanged": u.update == 0 and not u.optimizer.state}
            rep["cases"][name] = rec
    ok = rep["cases"]["valid"] == {"result": "applied", "update": 2} and all(
        r["result"] == "rejected" and r["state_unchanged"] for n, r in rep["cases"].items() if n != "valid")
    rep["all_ok"] = ok
    with open(sys.argv[1], "w") as f:
        json.dump(rep, f, indent=1, allow_nan=False)
    print(json.dumps(rep, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
