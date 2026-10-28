#!/usr/bin/env bash
# Run Codex's independent final-audit suite (vendored verbatim from /tmp/r2r2r-codex-final-evidence,
# see codex_final_43ab363/SHA256SUMS.original) against the current checkout, without touching the
# original evidence directory.
#
#   policy_training/audit/run_codex_final_suite.sh OUT_DIR [pytest args...]
#
# The only transformation is the evidence path: the suite's observe() and its conftest write to
# /tmp/r2r2r-codex-final-evidence, which is replaced by OUT_DIR in a temporary copy. Test bodies,
# assertions and fixtures are unchanged. Exit code = pytest's exit code.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OUT="$(mkdir -p "$1" && cd "$1" && pwd)"; shift
SRC="$HERE/codex_final_43ab363"
(cd "$SRC" && sha256sum -c --quiet <(grep -E ' (test_independent_final|conftest)\.py$' SHA256SUMS.original))
SUITE="$(mktemp -d "$OUT/suite.XXXX")"
for f in test_independent_final.py conftest.py; do
  sed "s#/tmp/r2r2r-codex-final-evidence#$OUT#g" "$SRC/$f" > "$SUITE/$f"
done
diff <(sed "s#/tmp/r2r2r-codex-final-evidence#$OUT#g" "$SRC/test_independent_final.py") "$SUITE/test_independent_final.py"
: > "$OUT/observations.jsonl"
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$REPO/dependencies/tinydp:$REPO/policy_training:$REPO/policy_training/tests"
cd "$REPO"
set +e
"$REPO/.venv-tinydp/bin/python" -B -m pytest "$SUITE/test_independent_final.py" -q -p no:cacheprovider \
  -o faulthandler_timeout=120 --basetemp="$OUT/basetemp" "$@" 2>&1 | tee "$OUT/pytest.log"
rc=${PIPESTATUS[0]}
echo "{\"exit_code\": $rc, \"commit\": \"$(git rev-parse HEAD)\", \"dirty\": $( [ -z "$(git status --porcelain -- policy_training)" ] && echo false || echo true )}" > "$OUT/suite_exit.json"
exit $rc
