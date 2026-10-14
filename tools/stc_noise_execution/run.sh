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
: "${CALIB_RUN_DIR:?Set CALIB_RUN_DIR to the stopped existing paired workflow directory}"
echo "Resume SAME workflow: $CALIB_RUN_DIR; clean complete skipped; original Adam/RNG/500-step cosine preserved."
exec "$PY" -u -m tools.stc_noise_execution.workflow --out-dir "$CALIB_RUN_DIR" "$@"
