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
if [[ "${MIXED_STC_RESUME:-0}" == 1 && -z "${MIXED_STC_DIR:-}" ]]; then echo 'Set original MIXED_STC_DIR'; exit 1; fi
MIXED_STC_DIR="${MIXED_STC_DIR:-$ROOT/outputs/p0_f9_shared_history/stc_gt_mixed500_$(date +%Y%m%d_%H%M%S)}"
args=(--reference-dir "$REFERENCE_DIR" --out-dir "$MIXED_STC_DIR")
if [[ "${MIXED_STC_RESUME:-0}" == 1 ]]; then args+=(--resume); fi
echo "Expanded REAL STC + clean TRAIN GT / fixed500 / same remaining64: $MIXED_STC_DIR"
exec "$PY" -u -m tools.stc_mixed_adaptation.run "${args[@]}" "$@"
