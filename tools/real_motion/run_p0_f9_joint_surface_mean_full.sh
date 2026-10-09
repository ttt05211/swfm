#!/usr/bin/env bash
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == OccFM ]] || { echo '先 conda activate OccFM' >&2; exit 2; }
[[ $# -eq 0 ]] || { echo '用 SURFACE_MEAN_SOURCE / SURFACE_MEAN_FULL_OUT / SURFACE_MEAN_FULL_RESUME 配置。' >&2; exit 2; }
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
RUN="${SURFACE_COMPARE_RUN:-$ROOT/outputs/p0_f9_joint_surface_ccr/full20_nohup_resume_20261008_190410_787}"
OUT="${SURFACE_MEAN_FULL_OUT:-$ROOT/outputs/p0_f9_joint_surface_ccr/mean_full4369_$(date +%Y%m%d_%H%M%S)_$$}"
DATAROOT="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
E14="${E14:-$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt}"
VAL_HISTORY="${CCR_VAL_CACHE:-$ROOT/cache/p0_f9_ccr_history_geometry_val_v1}"
EXTRA=()
[[ -z "${SURFACE_MEAN_SOURCE:-}" ]] || EXTRA+=(--source-bundle-dir "$SURFACE_MEAN_SOURCE")
if [[ "${SURFACE_MEAN_FULL_RESUME:-0}" == 1 ]]; then
  [[ -f "$OUT/bundle.json" ]] || { echo "[MISSING] $OUT/bundle.json；请指定原全集评估输出目录。" >&2; exit 2; }
  EXTRA+=(--resume)
else
  [[ ! -e "$OUT" && ! -e "$OUT.log" ]] || { echo "拒绝覆盖 $OUT；续评需 RESUME=1。" >&2; exit 2; }
fi
for FILE in "$RUN/last.pt" "$E14" "$ROOT/configs/real_motion_occfm.yaml" \
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
echo '仅评估已完成DEV512对照的平均模型：full4369；4历史→6未来，ADD raw0.5 / REMOVEoff。'
echo '不重新平均、不重跑其他单轮、不修改原权重；VAL缓存只读，支持Ctrl+C/TERM续评。'
echo "输出 $OUT；自动查找同权重已完成对照，多种权重冲突则停止。"
mkdir -p "$(dirname -- "$OUT")"
"$PY" -u tools/real_motion/eval_p0_f9_joint_surface_mean_full.py \
  --run-dir "$RUN" --runs-root "$ROOT/outputs/p0_f9_joint_surface_ccr" --out-dir "$OUT" \
  --config "$ROOT/configs/real_motion_occfm.yaml" --base-checkpoint "$E14" \
  --dev-cache "$ROOT/data/p0_f9_v18_se2_val_all_4369.pt" \
  --population-manifest "$ROOT/data/p0_f9_v21_dev64_manifest.json" \
  --dataroot "$DATAROOT" --dev-info "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl" \
  --ccr-val-history-cache "$VAL_HISTORY" --ccr-val-history-cache-ram-mib 512 \
  --descriptor-disk-mib 0 --descriptor-ram-mib 0 --frame-cache-mib 2048 \
  --ccr-cpu-execution native_parallel --ccr-cpu-workers 4 --surface-query-workers 4 \
  --ccr-prefetch-workers 4 --cpu-workers 10 --checkpoint-every 8 "${EXTRA[@]}" 2>&1 | tee -a "$OUT.log"
echo "结果 $OUT/summary.txt；完整四指标/时距/分支/类别/场景见 full_validation.json。"
