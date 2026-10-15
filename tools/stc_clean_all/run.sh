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
if [[ "${CLEAN_ALL_RESUME:-0}" == 1 && -z "${CLEAN_ALL_DIR:-}" ]]; then echo 'Set original CLEAN_ALL_DIR to resume'; exit 1; fi
CLEAN_ALL_DIR="${CLEAN_ALL_DIR:-$ROOT/outputs/p0_f9_shared_history/clean1000_all_$(date +%Y%m%d_%H%M%S)}"
args=(--out-dir "$CLEAN_ALL_DIR")
source_dir="${CLEAN_ALL_SOURCE:-${CLEAN_CONT_DIR:-}}"
if [[ -n "$source_dir" ]]; then args+=(--source-dir "$source_dir"); fi
if [[ "${CLEAN_ALL_RESUME:-0}" == 1 ]]; then args+=(--resume); fi
echo "Frozen cumulative clean1000, ALL4369 OCC+GT/STC+GT, no training. Output=$CLEAN_ALL_DIR"
exec "$PY" -u -m tools.stc_clean_all.run "${args[@]}" "$@"
