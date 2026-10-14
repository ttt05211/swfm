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
if [[ "${FULL_NOISE_RESUME:-0}" == 1 && -z "${FULL_NOISE_DIR:-}" ]]; then echo 'Set original FULL_NOISE_DIR to resume'; exit 1; fi
FULL_NOISE_DIR="${FULL_NOISE_DIR:-$ROOT/outputs/p0_f9_shared_history/stc_full_unfreeze500_$(date +%Y%m%d_%H%M%S)}"
args=(--reference-workflow "$REFERENCE_DIR" --out-dir "$FULL_NOISE_DIR")
if [[ "${FULL_NOISE_RESUME:-0}" == 1 ]]; then args+=(--resume-experiment); fi
echo "ORIGINAL GTepoch2, FULL unfreeze, fresh Adam x500, no previous adapted weights. FULL_NOISE_DIR=$FULL_NOISE_DIR"
exec "$PY" -u -m tools.stc_noise_full.run "${args[@]}" "$@"
