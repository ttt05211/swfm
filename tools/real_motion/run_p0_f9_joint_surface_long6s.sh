#!/usr/bin/env bash
# Frozen Surface mean only. Old Local epoch19 entry remains unchanged.
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == OccFM ]] || { echo '请先 conda activate OccFM' >&2; exit 2; }
[[ $# -le 3 ]] || { echo "用法: bash $0 [dev64|dev512|all] [OUTPUT_DIR] [--resume]" >&2; exit 2; }
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
POP="${1:-dev64}"
case "$POP" in dev64|dev512|all) ;; *) echo '人口须为dev64/dev512/all' >&2; exit 2 ;; esac
RUN="${SURFACE_COMPARE_RUN:-$ROOT/outputs/p0_f9_joint_surface_ccr/full20_nohup_resume_20261008_190410_787}"
OUT="${2:-$ROOT/outputs/p0_f9_joint_surface_ccr/long6s_mean_${POP}_$(date +%Y%m%d_%H%M%S)_$$}"
ALIGNMENT="${SURFACE_LONG_ALIGNMENT:-legacy_cache6s}"
DATAROOT="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
E14="${E14:-$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt}"
EXTRA=()
if [[ $# == 3 ]]; then
  [[ "$3" == --resume && -f "$OUT/evaluation_state.json" ]] || { echo '续评须指定原输出目录和--resume' >&2; exit 2; }
  EXTRA+=(--resume)
else
  [[ ! -e "$OUT" ]] || { echo "拒绝覆盖 $OUT；继续评估须显式--resume" >&2; exit 2; }
fi
[[ -z "${SURFACE_MEAN_SOURCE:-}" ]] || EXTRA+=(--source-bundle-dir "$SURFACE_MEAN_SOURCE")
case "${SURFACE_LONG_REDETECT:-1}" in 1) ;; 0) EXTRA+=(--no-redetect-comparison) ;; *) echo 'SURFACE_LONG_REDETECT须为0/1' >&2; exit 2 ;; esac
case "${SURFACE_LONG_GRAPHS:-1}" in 1) ;; 0) EXTRA+=(--no-graphs) ;; *) echo 'SURFACE_LONG_GRAPHS须为0/1' >&2; exit 2 ;; esac
VAL_CACHE="${SURFACE_LONG_VAL_CACHE:-$ROOT/cache/p0_f9_ccr_history_geometry_val_v1}"
if [[ "$VAL_CACHE" != off ]]; then
  [[ -d "$VAL_CACHE" ]] || { echo "[MISSING] $VAL_CACHE；不用缓存可显式SURFACE_LONG_VAL_CACHE=off" >&2; exit 2; }
  EXTRA+=(--val-history-cache "$VAL_CACHE")
fi
for path in "$ROOT/configs/real_motion_occfm.yaml" "$ROOT/data/p0_f9_v18_se2_val_all_4369.pt" \
  "$ROOT/data/p0_f9_v21_dev64_manifest.json" "$E14" "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl" "$RUN/training.json"; do
  [[ -f "$path" ]] || { echo "[MISSING] $path" >&2; exit 2; }
done
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export SWFM_COLUMN_CPU_BACKEND=native SWFM_COLUMN_CPU_BUNDLE=1
export SWFM_LOCAL_FAST_SUPERVISION=0 SWFM_LOCAL_STATIC_ROI=0
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
case "$ALIGNMENT" in
  legacy_cache6s) ;;
  geniedrive_code10s)
    [[ "$POP" == all ]] || { echo 'GenieDrive严格公开代码人口须选all' >&2; exit 2; }
    GENIE_INFO="${GENIEDRIVE_INFO:-$ROOT/data/geniedrive/world-nuscenes_infos_val.pkl}"
    "$PY" -u tools/real_motion/download_geniedrive_eval_info.py --out "$GENIE_INFO"
    EXTRA+=(--geniedrive-info "$GENIE_INFO")
    ;;
  *) echo '未知SURFACE_LONG_ALIGNMENT' >&2; exit 2 ;;
esac
mkdir -p "$(dirname -- "$OUT")"
echo '冻结Surface平均5/6/8/12/14：严格4历史→六帧预测→最后四张预测→再六帧；报告1–6秒。'
echo '轨迹接续为固定主结果，redetect同人口对照；ADD raw0.5 / REMOVEoff；不重训、不重新平均。'
echo "人口=$POP 对齐=$ALIGNMENT；第二段不读取真实未来缓存。输出：$OUT"
"$PY" -u tools/real_motion/eval_p0_f9_joint_surface_long_rollout.py \
  --config "$ROOT/configs/real_motion_occfm.yaml" --run-dir "$RUN" \
  --runs-root "$ROOT/outputs/p0_f9_joint_surface_ccr" --out-dir "$OUT" \
  --dev-cache "$ROOT/data/p0_f9_v18_se2_val_all_4369.pt" \
  --population-manifest "$ROOT/data/p0_f9_v21_dev64_manifest.json" \
  --base-checkpoint "$E14" --dataroot "$DATAROOT" \
  --dev-info "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl" \
  --population "$POP" --population-alignment "$ALIGNMENT" \
  --cpu-workers "${SURFACE_LONG_CPU_WORKERS:-10}" --prefetch-workers 4 \
  --ccr-cpu-execution native_parallel --ccr-cpu-workers 4 --surface-query-workers 4 \
  "${EXTRA[@]}" 2>&1 | tee -a "$OUT.log"
echo "完成：$OUT/summary.txt；详细指标/审计：$OUT/evaluation.json"
