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
if [[ "${CLEAN_CONT_RESUME:-0}" == 1 && -z "${CLEAN_CONT_DIR:-}" ]]; then echo 'Set original CLEAN_CONT_DIR to resume'; exit 1; fi
CLEAN_CONT_DIR="${CLEAN_CONT_DIR:-$ROOT/outputs/p0_f9_shared_history/stc_clean_continue1000_$(date +%Y%m%d_%H%M%S)}"
args=(--out-dir "$CLEAN_CONT_DIR")
source_dir="${CLEAN_CONT_SOURCE:-${CLEAN_FULL_DIR:-}}"
if [[ -n "$source_dir" ]]; then args+=(--source-dir "$source_dir"); fi
if [[ "${CLEAN_CONT_RESUME:-0}" == 1 ]]; then args+=(--resume-experiment); fi
echo "Continue CLEAN full500: retain weights/Adam/RNG, next500 groups, NEW cosine; output=$CLEAN_CONT_DIR"
exec "$PY" -u -m tools.stc_clean_continue.run "${args[@]}" "$@"
