#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
"$PY" -c 'import sys; assert "OccFM" in sys.executable, "conda activate OccFM first"'
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
NOISE_FIT_DIR="${NOISE_FIT_DIR:-$ROOT/outputs/p0_f9_shared_history/stc_noise_fit_$(date +%Y%m%d_%H%M%S)}"
args=(--dataroot "${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
  --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}" --out-dir "$NOISE_FIT_DIR"
  --workers "${NOISE_CPU_WORKERS:-2}")
if [[ -n "${PRIOR_RESULT:-}" ]]; then args+=(--prior-result "$PRIOR_RESULT"); fi
if [[ "${NOISE_FIT_RESUME:-0}" == 1 ]]; then args+=(--resume); fi
echo "Fit actual STC historical errors on frozen DEV64 scenes; remaining labels NOT read; CPU only; NO training. Output: $NOISE_FIT_DIR"
exec "$PY" -u -m tools.stc_noise_profile.run "${args[@]}" "$@"
