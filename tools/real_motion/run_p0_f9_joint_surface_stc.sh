#!/usr/bin/env bash
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == OccFM ]] || { echo '先 conda activate OccFM' >&2; exit 2; }
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
: "${STC_ROOT:?请设置解压后的 STC_ROOT（其下直接是 scene-XXXX）}"
PLAN_CACHE="${STC_PLAN_CACHE:-/root/nas/occ/new_code/cache/come_main_table/camera_pred}"
DATAROOT="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
RUNS="${SURFACE_RUNS_ROOT:-$ROOT/outputs/p0_f9_joint_surface_ccr}"
RUN="${SURFACE_RUN_DIR:-$RUNS/full20_nohup_resume_20261008_190410_787}"
POP="${STC_POPULATION:-dev64}"
OUT="${STC_OUT:-$RUNS/stc_four_${POP}_$(date +%Y%m%d_%H%M%S)_$$}"
ARGS=(--dataroot "$DATAROOT" --stc-root "$STC_ROOT" --plan-cache "$PLAN_CACHE"
      --out-dir "$OUT" --population "$POP" --config "$ROOT/configs/real_motion_occfm.yaml"
      --cpu-workers "${STC_CPU_WORKERS:-4}" --frame-cache-mib "${STC_FRAME_CACHE_MIB:-512}")
if [[ "$POP" != all ]]; then
  ARGS+=(--population-manifest "${STC_POPULATION_MANIFEST:-$ROOT/data/p0_f9_v21_dev64_manifest.json}")
fi
if [[ -n "${STC_CHECKPOINT:-}" ]]; then
  ARGS+=(--checkpoint "$STC_CHECKPOINT")
else
  ARGS+=(--runs-root "$RUNS" --run-dir "$RUN")
  if [[ -n "${SURFACE_MEAN_SOURCE:-}" ]]; then ARGS+=(--source-bundle-dir "$SURFACE_MEAN_SOURCE"); fi
fi
if [[ "${STC_RESUME:-0}" == 1 ]]; then
  [[ -n "${STC_OUT:-}" ]] || { echo '续评必须指定原 STC_OUT' >&2; exit 2; }
  ARGS+=(--resume)
fi
if [[ "${STC_AUDIT_ONLY:-0}" == 1 ]]; then ARGS+=(--audit-only); fi
if [[ "${STC_NO_GRAPHS:-0}" == 1 ]]; then ARGS+=(--no-graphs); fi
ARGS+=(--parallel-majority)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export SWFM_COLUMN_CPU_BACKEND=native SWFM_COLUMN_CPU_BUNDLE=1
export SWFM_LOCAL_FAST_SUPERVISION=0 SWFM_LOCAL_STATIC_ROI=0
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
echo '冻结平均网络：四历史→六未来；STC/Occ × GT/Pred ego，全部同人口、无评分mask。'
echo '不重训、不校准、不复用旧Camera latent/GT learned geometry；四设置完成才读未来标签。'
echo "Output: $OUT"
exec "$PY" -u tools/real_motion/eval_p0_f9_joint_surface_stc.py "${ARGS[@]}" "$@"
