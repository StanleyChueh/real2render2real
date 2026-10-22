#!/bin/bash
# num_envs scaling on GPU 0, JAX lazy alloc, minimal python viser client (no browser).
cd ~/Stanley_ws/real2render2real
source .venv/bin/activate
for N in 2 5 8 10 12; do
  RUN=run07_env${N}_gpu0_noprealloc_nullclient
  CLIENT=null CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false R2R2R_NUM_ENVS=$N ~/Stanley_ws/real2render2real/docs/reproduction/tools/launch_run.sh $RUN 480
  R=~/Stanley_ws/r2r2r_repro_runs/$RUN
  python docs/reproduction/tools/analyze_run.py $R --json $R/metrics.json > /dev/null 2>&1
  if [ -d output_data ]; then
    python docs/reproduction/tools/validate_dataset.py output_data/yumi_coffee_maker --json $R/dataset_validation.json > /dev/null 2>&1
    mv output_data $R/output_data
  fi
  echo "done $RUN $(cat $R/exit_code) batches=$(grep -c 'Success Envs' $R/run.log) oom=$(grep -c 'OutOfMemoryError' $R/run.log)"
  sleep 15
done
echo ALLDONE
