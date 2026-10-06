#!/usr/bin/env bash
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == OccFM ]] || { echo '请先 conda activate OccFM' >&2; exit 2; }
[[ $# -le 1 ]] || { echo "用法: bash $0 [epoch19 checkpoint]" >&2; exit 2; }
ROOT=/root/nas/occ/swfm
cd "$ROOT"
PY="$(command -v python)"
TEACHER="${1:-$ROOT/outputs/p0_f9_joint_causal_columns/checkpoint_selection_20261004_220555/epoch_0019.pt}"
DATAROOT=/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes
OUT="${CCR_OUT:-$ROOT/outputs/p0_f9_point_ccr/screen20x3_$(date +%Y%m%d_%H%M%S)_$$}"
[[ ! -e "$OUT" && ! -e "$OUT.log" ]] || { echo "拒绝覆盖 $OUT；resume也使用新目录。" >&2; exit 2; }
if pgrep -f '[p]ython.*(train_p0_f9_joint|eval_p0_f9_joint|run_p0_f9_shared_evidence|run_p0_f9_source_repair|train_p0_f9_height_shared_field|train_p0_f9_point_ccr)' >/dev/null; then
  echo '请先安全停止其他训练/评估，避免争抢资源；本脚本不kill进程。' >&2; exit 2
fi
for FILE in "$TEACHER" "$ROOT/configs/real_motion_occfm.yaml" "$ROOT/data/p0_f9_v18_se2_train_full.pt" \
  "$ROOT/data/p0_f9_v18_se2_val_all_4369.pt" "$ROOT/data/p0_f9_v21_dev64_manifest.json" \
  "$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt" \
  "$DATAROOT/nuscenes_infos_train_temporal_v3_scene.pkl" "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl"; do
  [[ -f "$FILE" ]] || { echo "[MISSING] $FILE" >&2; exit 2; }
done
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export SWFM_COLUMN_CPU_BACKEND=native SWFM_COLUMN_CPU_BUNDLE=1 SWFM_COLUMN_CPU_HORIZONS=1
EXTRA=()
if [[ -n "${CCR_RESUME:-}" ]]; then
  [[ -f "$CCR_RESUME" ]] || { echo "[MISSING] $CCR_RESUME" >&2; exit 2; }
  EXTRA+=(--resume "$CCR_RESUME")
fi
if [[ "${CCR_SMOKE:-0}" == 1 ]]; then
  EXTRA+=(--prior-windows 2 --eval-windows 2 --fps-windows 1 --speed-repeats 1 --max-updates 2 --skip-dev512)
fi
echo 'Pointwise CCR：4086 TRAIN窗口 x 3轮；4历史→6未来；冻结epoch19 motion，新头随机初始化，GT-only无KD/AE。'
echo '每轮dev64，最后dev512 + 同卡六帧FPS；固定ADD=.5/REMOVE=.95；不自动扩训、调阈值或部署。'
echo '原几何缓存只读；新固定历史描述子磁盘<=16GiB、RAM<=1GiB，磁盘至少保留2GiB；绝不缓存未来GT/learned features。'
echo "输出：$OUT；Ctrl+C/TERM会在完整更新后保存last.pt；kill -9只能恢复周期断点。"
mkdir -p "$(dirname "$OUT")"
"$PY" -u tools/real_motion/train_p0_f9_point_ccr.py \
  --config "$ROOT/configs/real_motion_occfm.yaml" --checkpoint "$TEACHER" \
  --train-cache "$ROOT/data/p0_f9_v18_se2_train_full.pt" --dev-cache "$ROOT/data/p0_f9_v18_se2_val_all_4369.pt" \
  --population-manifest "$ROOT/data/p0_f9_v21_dev64_manifest.json" \
  --base-checkpoint "$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt" \
  --dataroot "$DATAROOT" --train-info "$DATAROOT/nuscenes_infos_train_temporal_v3_scene.pkl" \
  --dev-info "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl" --out-dir "$OUT" \
  --causal-geometry-cache "$ROOT/outputs/p0_f9_joint_causal_columns/causal_geometry_cache_v1" \
  --descriptor-cache "$ROOT/outputs/p0_f9_point_ccr/fixed_history_cache" \
  --descriptor-disk-mib "${CCR_DISK_MIB:-16384}" --descriptor-ram-mib 1024 \
  --cpu-workers 8 --epochs 3 --train-fraction .2 "${EXTRA[@]}" 2>&1 | tee "$OUT.log"
echo "结果：$OUT/summary.txt；详细：$OUT/screen.json；断点：$OUT/last.pt"
