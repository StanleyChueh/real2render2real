"""Which F7 corruption cases does a validator accept? Run with the policy_training tree under test on sys.path."""
import json, os, sys, tempfile
import torch
from r2r2r_dp import driver
from r2r2r_dp.protocol import resolve
import test_f7_checkpoint_state as T
from run_fixtures import make_run_fixture
with tempfile.TemporaryDirectory() as d:
    fx = make_run_fixture(os.path.join(d, "fx"), total_updates=4, warmup_updates=2, val_every_updates=100,
                          full_val_every_updates=100, full_ckpt_every_updates=1, keep_full_checkpoints=5,
                          inference_export_every_updates=100)
    t = T._trainer(fx, os.path.join(d, "a")); t.train(stop_after=3)
    u3 = torch.load(os.path.join(t.out, "ckpt_full_u0000003.pt"), weights_only=False)
    v = T._trainer(fx, os.path.join(d, "v"))
    res = {}
    for c in T.CASES:
        try:
            bad = T._corrupt(u3, c)
        except KeyError as e:
            res[c] = f"n/a ({e} not in this checkpoint format)"; continue
        try:
            p = v.checkpoint_problems(bad)
        except Exception as e:
            res[c] = f"validator crashed: {type(e).__name__}: {str(e)[:80]}"; continue
        res[c] = "ACCEPTED" if p == [] else "rejected: " + p[0][:120]
print(json.dumps({"code": driver.__file__, "accepted": sorted(k for k, r in res.items() if r == "ACCEPTED"),
                  "n_cases": len(res), "results": res}, indent=1))
