#!/bin/bash
# usage: [CUDA_VISIBLE_DEVICES=..] [R2R2R_NUM_ENVS=N] [CLIENT=null|chrome] launch_run.sh RUN_NAME TIMEOUT_S [run.py args]
# Runs the UNMODIFIED scripts/run.py (or, if R2R2R_NUM_ENVS is set, run.py via the num_envs wrapper)
# with GPU telemetry, timestamped log, automatic headless Chrome viser client, and cleanup of the
# BatchDataLogger workers orphaned at Kit shutdown (they also hold stdout open, hence the
# process substitution instead of a pipe).
set -u
RUN=$1; TO=$2; shift 2
RD=~/Stanley_ws/r2r2r_repro_runs; R=$RD/$RUN; mkdir -p $R
cd ~/Stanley_ws/real2render2real
source .venv/bin/activate
unset CONDA_PREFIX
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
env | grep -E "^(CUDA_VISIBLE_DEVICES|XLA_|R2R2R_)" > $R/env.txt
echo "$@" > $R/args.txt
if [ -n "${R2R2R_NUM_ENVS:-}" ]; then ENTRY=docs/reproduction/tools/run_with_num_envs.py; else ENTRY=scripts/run.py; fi
echo $ENTRY > $R/entry.txt
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits -lms 1000 > $R/gpu.csv &
SMI1=$!
( while true; do echo "== $(date +%s)"; nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader; nvidia-smi pmon -c 1 -s um | grep python; sleep 10; done ) > $R/gpu_procs.log 2>&1 &
SMI2=$!
(
  until grep -q "viser (listening" $R/run.log 2>/dev/null; do sleep 1; done
  PORT=$(grep -m1 -o "listening \*:[0-9]*" $R/run.log | grep -o "[0-9]*$")
  echo "client -> $PORT at $(date +%s)" > $R/client.txt
  echo "client=${CLIENT:-null}" >> $R/client.txt; rm -rf $R/chrome_profile
  if [ "${CLIENT:-null}" = null ]; then exec python docs/reproduction/tools/viser_null_client.py $PORT > $R/client.log 2>&1; fi
  exec google-chrome --headless=new --disable-gpu --no-first-run --user-data-dir=$R/chrome_profile http://localhost:$PORT > $R/chrome.log 2>&1
) &
CLI=$!
date +%s > $R/t_start
timeout --signal=INT --kill-after=60 $TO python $ENTRY "$@" > >(python3 -I ~/Stanley_ws/real2render2real/docs/reproduction/tools/tsline.py > $R/run.log) 2>&1
echo "exit=$?" > $R/exit_code
date +%s > $R/t_end
for p in $(pgrep -f "^python (scripts/run.py|docs/reproduction/tools/run_with_num_envs.py)"); do kill -9 $p; done
kill $SMI1 $SMI2 2>/dev/null
pkill -P $CLI 2>/dev/null; kill $CLI 2>/dev/null
pkill -f -- "--user-data-dir=$R/chrome_profile" 2>/dev/null
sleep 2
exit 0
