#!/usr/bin/env bash
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == OccFM ]] || { echo 'Please conda activate OccFM first' >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${PY:-$(command -v python)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export SWFM_COLUMN_CPU_BACKEND=native SWFM_COLUMN_CPU_BUNDLE=1
export SWFM_LOCAL_FAST_SUPERVISION=0 SWFM_LOCAL_STATIC_ROI=0
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
SOURCE="${WAYMO10_LAYOUT_SOURCE:?Set WAYMO10_LAYOUT_SOURCE to the stopped parallel output}"
OUT="${WAYMO10_LAYOUT_OUT:-$ROOT/outputs/p0_f9_joint_surface_ccr/waymo10_layout_$(date +%Y%m%d_%H%M%S)_$$}"
echo "SPEED ONLY: same v2, 2 processes x2 threads vs4 x1, alternating identical windows; no scientific count updates."
echo "Pause source safely first; source lease blocks concurrent resume. Output: $OUT"
exec "$PY" -u tools/real_motion/benchmark_p0_f9_waymo10_worker_layout.py \
  --continue-from-dir "$SOURCE" --out-dir "$OUT" --config "$ROOT/configs/real_motion_occfm.yaml" \
  --speed-windows "${WAYMO10_LAYOUT_WINDOWS:-64}" --repeats "${WAYMO10_LAYOUT_REPEATS:-2}" "$@"
