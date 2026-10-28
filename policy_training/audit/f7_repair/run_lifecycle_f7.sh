#!/usr/bin/env bash
# Bounded real-data lifecycle for the F7 repair (GPU 0, N=50, configs/lifecycle_u20_v3.json, 20 updates):
#   A = 20 uninterrupted updates;  B = stop after 10, then --resume to 20;
#   C = separate stop-after-10 session whose u10 checkpoint then has one parameter's Adam state deleted;
#       its --resume must exit non-zero before any update and leave every file except resume_history.jsonl unchanged.
# usage: run_lifecycle_f7.sh ROOT [STAGES]   (STAGES default "A B C"; logs in ROOT/logs)
set -u
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
ROOT=$1; STAGES=${2:-"A B C"}
mkdir -p "$ROOT/logs"; export LIFECYCLE_LOGS="$ROOT/logs"
cd "$REPO"
P="--protocol policy_training/configs/lifecycle_u20_v3.json --train-size 50 --device cuda"
L=policy_training/tools/run_lifecycle.sh
for s in $STAGES; do case $s in
A) bash $L A $P --output-dir "$ROOT/u20_A" ;;
B) bash $L B1 $P --output-dir "$ROOT/u20_B" --stop-after-updates 10
   bash $L B2 $P --output-dir "$ROOT/u20_B" --resume ;;
C) C="$ROOT/u10_C_corrupt"; [ -e "$C" ] && { echo "$C exists"; exit 2; }
   bash $L C1 $P --output-dir "$C" --stop-after-updates 10 || exit 1
   CUDA_VISIBLE_DEVICES='' .venv-tinydp/bin/python - "$C/ckpt_full_u0000010.pt" <<'PY' || { echo "corruption step failed"; exit 1; }
import sys, torch
p = sys.argv[1]; st = torch.load(p, map_location="cpu", weights_only=False)
k = next(iter(st["optimizer"]["state"])); del st["optimizer"]["state"][k]; torch.save(st, p)
print("deleted Adam state of optimizer parameter id", k, "->", len(st["optimizer"]["state"]), "entries left")
PY
   (cd "$C" && sha256sum * > "$ROOT/logs/C.sha256.before")
   bash $L C2 $P --output-dir "$C" --resume
   (cd "$C" && sha256sum * > "$ROOT/logs/C.sha256.after")
   diff <(grep -v resume_history "$ROOT/logs/C.sha256.before") <(grep -v resume_history "$ROOT/logs/C.sha256.after") \
     && echo "C: all files except resume_history.jsonl unchanged" ;;
esac; done
