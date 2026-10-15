#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
"$PY" -c 'import sys; assert "OccFM" in sys.executable, "conda activate OccFM first"'
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
REFERENCE_DIR="${REFERENCE_DIR:-$ROOT/outputs/p0_f9_shared_history/stc_empirical_refiner500_20261014_194954}"
if [[ "${CLEAN_FULL_RESUME:-0}" == 1 && -z "${CLEAN_FULL_DIR:-}" ]]; then echo 'Set original CLEAN_FULL_DIR to resume'; exit 1; fi
CLEAN_FULL_DIR="${CLEAN_FULL_DIR:-$ROOT/outputs/p0_f9_shared_history/stc_clean_full500_$(date +%Y%m%d_%H%M%S)}"
args=(--reference-workflow "$REFERENCE_DIR" --out-dir "$CLEAN_FULL_DIR")
if [[ "${CLEAN_FULL_RESUME:-0}" == 1 ]]; then args+=(--resume-experiment); fi
echo "CLEAN FULL500: original GTepoch2, no noise, same remaining64. Output: $CLEAN_FULL_DIR"
exec "$PY" -u -m tools.stc_clean_full.run "${args[@]}" "$@"
