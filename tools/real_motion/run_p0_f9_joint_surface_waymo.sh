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
WAYMO_ROOT="${WAYMO_ROOT:-$ROOT/data/waymo}"
RUNS_ROOT="${SURFACE_RUNS_ROOT:-$ROOT/outputs/p0_f9_joint_surface_ccr}"
RUN_DIR="${SURFACE_RUN_DIR:-$RUNS_ROOT/full20_nohup_resume_20261008_190410_787}"
OUT="${WAYMO_OUT:-$RUNS_ROOT/waymo_i2world_2hz_$(date +%Y%m%d_%H%M%S)_$$}"
ARGS=(--waymo-root "$WAYMO_ROOT" --out-dir "$OUT"
      --config "$ROOT/configs/real_motion_occfm.yaml"
      --raw-free-label "${WAYMO_RAW_FREE_LABEL:-23}"
      --expected-scenes "${WAYMO_EXPECTED_SCENES:-202}"
      --max-windows "${WAYMO_MAX_WINDOWS:-0}"
      --frame-cache-mib "${WAYMO_FRAME_CACHE_MIB:-256}" --cpu-workers 4)
if [[ -n "${WAYMO_CHECKPOINT:-}" ]]; then
  ARGS+=(--checkpoint "$WAYMO_CHECKPOINT")
else
  ARGS+=(--runs-root "$RUNS_ROOT" --run-dir "$RUN_DIR")
  if [[ -n "${SURFACE_MEAN_SOURCE:-}" ]]; then ARGS+=(--source-bundle-dir "$SURFACE_MEAN_SOURCE"); fi
fi
if [[ "${WAYMO_RESUME:-0}" == 1 ]]; then
  if [[ -z "${WAYMO_OUT:-}" ]]; then echo 'Resume requires WAYMO_OUT=the original Waymo evaluation directory'; exit 2; fi
  ARGS+=(--resume)
fi
if [[ "${WAYMO_AUDIT_ONLY:-0}" == 1 ]]; then ARGS+=(--audit-only); fi
if [[ "${WAYMO_NO_GRAPHS:-0}" == 1 ]]; then ARGS+=(--no-graphs); fi
if [[ "${WAYMO_PARALLEL_MAJORITY:-1}" == 1 ]]; then ARGS+=(--parallel-majority); fi
echo 'I2-World Waymo 2Hz zero-shot: frozen mean, four histories -> six futures; no training/tuning/downloads.'
echo 'Only official validation NPZ + two trusted metadata PKLs required; nuScenes caches are NOT reused for Waymo.'
echo "Output: $OUT"
exec "$PY" -u tools/real_motion/eval_p0_f9_joint_surface_waymo.py "${ARGS[@]}" "$@"
