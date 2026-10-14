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
if [[ "${SYNTH_EVAL_RESUME:-0}" == 1 && -z "${SYNTH_EVAL_DIR:-}" ]]; then echo 'Set original SYNTH_EVAL_DIR'; exit 1; fi
SYNTH_EVAL_DIR="${SYNTH_EVAL_DIR:-$ROOT/outputs/p0_f9_shared_history/stc_synthetic_holdout64_$(date +%Y%m%d_%H%M%S)}"
args=(--reference-dir "$REFERENCE_DIR" --out-dir "$SYNTH_EVAL_DIR")
if [[ -n "${FULL_NOISE_DIR:-}" ]]; then args+=(--full-dir "$FULL_NOISE_DIR"); fi
if [[ "${SYNTH_EVAL_RESUME:-0}" == 1 ]]; then args+=(--resume); fi
echo "Read-only three-model synthetic holdout evaluation: $SYNTH_EVAL_DIR"
exec "$PY" -u -m tools.stc_noise_holdout.run "${args[@]}" "$@"
