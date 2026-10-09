#!/usr/bin/env bash
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == OccFM ]] || { echo '先 conda activate OccFM' >&2; exit 2; }
[[ $# -eq 0 ]] || { echo '通过 SURFACE_COMPARE_RUN / SURFACE_COMPARE_OUT / SURFACE_COMPARE_RESUME 配置。' >&2; exit 2; }
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
RUN="${SURFACE_COMPARE_RUN:-$ROOT/outputs/p0_f9_joint_surface_ccr/full20_nohup_resume_20261008_190410_787}"
OUT="${SURFACE_COMPARE_OUT:-$ROOT/outputs/p0_f9_joint_surface_ccr/checkpoint_comparison_$(date +%Y%m%d_%H%M%S)_$$}"
DATAROOT="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
E14="${E14:-$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt}"
VAL_HISTORY="${CCR_VAL_CACHE:-$ROOT/cache/p0_f9_ccr_history_geometry_val_v1}"
EXTRA=()
if [[ "${SURFACE_COMPARE_RESUME:-0}" == 1 ]]; then
  [[ -f "$OUT/bundle.json" ]] || { echo "[MISSING] $OUT/bundle.json；必须指定原对照输出目录。" >&2; exit 2; }
  EXTRA+=(--resume)
else
  [[ ! -e "$OUT" && ! -e "$OUT.log" ]] || { echo "拒绝覆盖 $OUT；接续对照请显式 RESUME=1。" >&2; exit 2; }
fi
for FILE in "$RUN/training.json" "$RUN/last.pt" "$E14" "$ROOT/configs/real_motion_occfm.yaml" \
  "$ROOT/data/p0_f9_v18_se2_val_all_4369.pt" "$ROOT/data/p0_f9_v21_dev64_manifest.json" \
  "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl"; do
  [[ -f "$FILE" ]] || { echo "[MISSING] $FILE" >&2; exit 2; }
done
[[ -d "$VAL_HISTORY" ]] || { echo "[MISSING] $VAL_HISTORY；不重建缓存。" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export SWFM_COLUMN_CPU_BACKEND=native SWFM_COLUMN_CPU_BUNDLE=1 SWFM_COLUMN_CPU_HORIZONS=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
echo '一次对照：第6/8/12轮 + 第5/6/8/12/14轮整网等权平均；第20轮复用已有dev512结果。'
echo '四项指标及1/2/3s全部报告；共享历史/几何/Moving评估区域，每个候选运动与CCR重新推理。'
echo '原checkpoint/optimizer/RNG/VAL缓存只读；不训练、不扫阈值、不跑full、不自动选优部署。'
echo "输出 $OUT；Ctrl+C/TERM后用同目录 SURFACE_COMPARE_RESUME=1 接续评估。"
mkdir -p "$(dirname -- "$OUT")"
"$PY" -u tools/real_motion/compare_p0_f9_joint_surface_checkpoints.py \
  --run-dir "$RUN" --runs-root "$ROOT/outputs/p0_f9_joint_surface_ccr" --out-dir "$OUT" \
  --config "$ROOT/configs/real_motion_occfm.yaml" --base-checkpoint "$E14" \
  --dev-cache "$ROOT/data/p0_f9_v18_se2_val_all_4369.pt" \
  --population-manifest "$ROOT/data/p0_f9_v21_dev64_manifest.json" \
  --dataroot "$DATAROOT" --dev-info "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl" \
  --ccr-val-history-cache "$VAL_HISTORY" --ccr-val-history-cache-ram-mib 512 \
  --descriptor-disk-mib 0 --descriptor-ram-mib 0 --frame-cache-mib 2048 \
  --ccr-cpu-execution native_parallel --ccr-cpu-workers 4 --surface-query-workers 4 \
  --ccr-prefetch-workers 4 --cpu-workers 10 --checkpoint-every 8 "${EXTRA[@]}" 2>&1 | tee -a "$OUT.log"
echo "结果 $OUT/summary.txt；完整四项/分类别/时距/分支/场景见 comparison.json；平均权重仅用于评估。"
