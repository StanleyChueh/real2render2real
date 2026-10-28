"""Run the real-data launch checks of a protocol for every train size (read-only).

    .venv-tinydp/bin/python policy_training/tools/launch_check_all.py policy_training/configs/baseline_v3.json OUT.json

Records, per N, the resolved config_sha256, the launch report and the wall time. Exit 1 if any size fails.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from r2r2r_dp.launch_checks import LaunchError, check_launch  # noqa: E402
from r2r2r_dp.protocol import load_protocol, resolve  # noqa: E402


def main():
    proto_path, out = sys.argv[1], sys.argv[2]
    p = load_protocol(proto_path)
    rep, ok = {}, True
    for n in (50, 100, 150, 1000, 1461):
        run = resolve(p, n, f"/nonexistent/launch_probe_n{n:04d}")
        t = time.time()
        try:
            rep[str(n)] = {"config_sha256": run["config_sha256"], **check_launch(run)}
        except LaunchError as e:
            ok = False
            rep[str(n)] = {"config_sha256": run["config_sha256"], "problems": e.problems[:50]}
        rep[str(n)]["seconds"] = round(time.time() - t, 1)
        print(n, rep[str(n)]["seconds"], "ok" if "problems" not in rep[str(n)] else "FAILED", flush=True)
    with open(out, "w") as f:
        json.dump(rep, f, indent=1, allow_nan=False)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
