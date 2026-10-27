#!/usr/bin/env bash
# usage: run_lifecycle.sh <tag> <train.py args...>   (logs in lifecycle_v2/logs/<tag>.*)
set -u
TAG=$1; shift
REPO=$(cd "$(dirname "$0")/../.." && pwd)
L=${LIFECYCLE_LOGS:-$HOME/Stanley_ws/r2r2r_policy_runs/lifecycle_v2/logs}; mkdir -p $L
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader -lms 500 > $L/$TAG.gpu.csv &
GPID=$!
( while true; do echo "$(date +%s.%N) $(free -m | awk '/Mem:/{print $3, $7}')"; sleep 1; done ) > $L/$TAG.hostmem.txt &
MPID=$!
cd $REPO
START=$(date +%s.%N)
CUDA_VISIBLE_DEVICES=0 /usr/bin/time -v -o $L/$TAG.time.txt .venv-tinydp/bin/python policy_training/train_r2r2r_dp.py "$@" > $L/$TAG.stdout.txt 2>&1
RC=$?
END=$(date +%s.%N)
sleep 3   # let worker teardown finish before the last samples
kill $GPID $MPID
echo "{\"tag\": \"$TAG\", \"exit_code\": $RC, \"wall_s\": $(echo "$END - $START" | bc)}" > $L/$TAG.exit.json
cat $L/$TAG.exit.json
