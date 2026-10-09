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
OUT="${WAYMO10_V2_OUT:-$RUNS_ROOT/waymo10_fast_v2_$(date +%Y%m%d_%H%M%S)_$$}"
ARGS=(--waymo-root "$WAYMO_ROOT" --out-dir "$OUT" --config "$ROOT/configs/real_motion_occfm.yaml"
      --raw-free-label "${WAYMO_RAW_FREE_LABEL:-23}" --expected-scenes "${WAYMO_EXPECTED_SCENES:-202}"
      --max-windows "${WAYMO10_MAX_WINDOWS:-0}" --frame-cache-mib "${WAYMO10_FRAME_CACHE_MIB:-256}"
      --cpu-workers "${WAYMO10_FAST_WORKERS:-4}" --geometry-cache-mib "${WAYMO10_GEOMETRY_MIB:-1024}"
      --surface-chunk "${WAYMO10_SURFACE_CHUNK:-4096}" --speed-windows "${WAYMO10_SPEED_WINDOWS:-16}")
if [[ -n "${WAYMO_CHECKPOINT:-}" ]]; then
  ARGS+=(--checkpoint "$WAYMO_CHECKPOINT")
else
  ARGS+=(--runs-root "$RUNS_ROOT" --run-dir "${SURFACE_RUN_DIR:-$RUNS_ROOT/full20_nohup_resume_20261008_190410_787}")
  if [[ -n "${SURFACE_MEAN_SOURCE:-}" ]]; then ARGS+=(--source-bundle-dir "$SURFACE_MEAN_SOURCE"); fi
fi
if [[ "${WAYMO10_V2_RESUME:-0}" == 1 ]]; then
  [[ -n "${WAYMO10_V2_OUT:-}" && -z "${WAYMO10_V2_CONTINUE_FROM:-}" ]] || { echo 'Resume needs original FAST output and no CONTINUE_FROM'; exit 2; }
  ARGS+=(--resume)
elif [[ -n "${WAYMO10_V2_CONTINUE_FROM:-}" ]]; then
  ARGS+=(--continue-from-dir "$WAYMO10_V2_CONTINUE_FROM")
fi
if [[ "${WAYMO10_V2_SPEED_ONLY:-0}" == 1 ]]; then ARGS+=(--speed-only); fi
if [[ "${WAYMO10_NO_GRAPHS:-0}" == 1 ]]; then ARGS+=(--no-graphs); fi
if [[ "${WAYMO10_PARALLEL_MAJORITY:-1}" == 1 ]]; then ARGS+=(--parallel-majority); fi
if [[ "${WAYMO10_HISTORY_PREFETCH:-1}" == 0 ]]; then ARGS+=(--no-history-prefetch); fi
echo 'Exact 10Hz acceleration ONLY: exact tube collision scatter/parallel preparation + native same-K16 surface fit.'
echo '2Hz entry unchanged; FOUR histories/SIX outputs/all native anchors/weights/thresholds unchanged.'
echo "New FAST output: $OUT; V1/V2 paired speed check is NOT formal FPS; original evaluation counts kept read-only."
exec "$PY" -u tools/real_motion/eval_p0_f9_joint_surface_waymo_10hz_fast_v2.py "${ARGS[@]}" "$@"
