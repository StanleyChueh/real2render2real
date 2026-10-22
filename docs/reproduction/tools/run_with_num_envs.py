"""Execute the unmodified scripts/run.py with a different num_envs for the active scene config.

run.py hardcodes `YumiCoffeeMakerCfg(num_envs=2, env_spacing=1.5)` and the authors' comment says to
edit it per GPU. To keep run.py byte-identical on disk, this wrapper substitutes exactly that one
literal in memory and executes the result as __main__ with run.py's own __file__ (so relative
data/output paths are unchanged). Nothing else is altered.

usage: R2R2R_NUM_ENVS=10 python docs/reproduction/tools/run_with_num_envs.py [run.py args]
"""
import os
import sys

RUN_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "scripts", "run.py")
RUN_PY = os.path.realpath(RUN_PY)
ORIG = "scene_config = YumiCoffeeMakerCfg(num_envs=2, env_spacing=1.5)"

n = int(os.environ["R2R2R_NUM_ENVS"])
src = open(RUN_PY).read()
assert src.count(ORIG) == 1, "run.py changed; refusing to patch"
src = src.replace(ORIG, f"scene_config = YumiCoffeeMakerCfg(num_envs={n}, env_spacing=1.5)")
print(f"[run_with_num_envs] executing {RUN_PY} with num_envs={n}", flush=True)
sys.argv[0] = RUN_PY
exec(compile(src, RUN_PY, "exec"), {"__name__": "__main__", "__file__": RUN_PY})
