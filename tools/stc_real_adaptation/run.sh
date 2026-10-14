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
if [[ "${REAL_STC_RESUME:-0}" == 1 && -z "${REAL_STC_DIR:-}" ]]; then echo 'Set original REAL_STC_DIR'; exit 1; fi
REAL_STC_DIR="${REAL_STC_DIR:-$ROOT/outputs/p0_f9_shared_history/stc_real18_adapt500_$(date +%Y%m%d_%H%M%S)}"
args=(--reference-dir "$REFERENCE_DIR" --out-dir "$REAL_STC_DIR")
if [[ "${REAL_STC_RESUME:-0}" == 1 ]]; then args+=(--resume); fi
echo "REAL STC18 -> GT future / fixed500 / original vs adapted remaining64: $REAL_STC_DIR"
exec "$PY" -u -m tools.stc_real_adaptation.run "${args[@]}" "$@"
