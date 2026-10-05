#!/usr/bin/env bash
# ONE bounded run; never changes the old checkpoint/full training directories.
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == OccFM ]] || { echo '请先 conda activate OccFM' >&2; exit 2; }
[[ $# -le 1 ]] || { echo "用法: bash $0 [teacher checkpoint]" >&2; exit 2; }
ROOT=/root/nas/occ/swfm
cd "$ROOT"
PY="$(command -v python)"
TEACHER="${1:-$ROOT/outputs/p0_f9_joint_causal_columns/checkpoint_selection_20261004_220555/epoch_0019.pt}"
DATAROOT=/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes
if pgrep -f '[p]ython.*(train_p0_f9_joint|eval_p0_f9_joint|calibrate_p0_f9_joint|run_p0_f9_shared_evidence_pilot)' >/dev/null; then
  echo '请先安全停止其他训练/评估，避免争抢资源；本脚本不会kill进程。' >&2; exit 2
fi
OUT="${SHARED_PILOT_OUT:-$ROOT/outputs/p0_f9_joint_causal_columns/shared_pilot_$(date +%Y%m%d_%H%M%S)_$$}"
[[ ! -e "$OUT" ]] || { echo "拒绝覆盖 $OUT" >&2; exit 2; }
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
if [[ "${SHARED_PILOT_SMOKE:-0}" == 1 ]]; then
  EXTRA+=(--eval-windows 2 --fps-windows 1 --speed-train-windows 4 --speed-repeats 1 --max-updates 2)
else
  EXTRA+=(--eval-windows 64 --fps-windows 6 --speed-train-windows 32 --speed-repeats 2)
fi
if [[ "${SHARED_PILOT_DEV512:-0}" == 1 ]]; then EXTRA+=(--final-dev512); fi
if [[ "${SHARED_PILOT_NO_TRAIN:-0}" == 1 ]]; then EXTRA+=(--no-train); fi
if [[ -n "${SHARED_PILOT_RESUME:-}" ]]; then
  [[ -f "$SHARED_PILOT_RESUME" ]] || { echo "[MISSING] $SHARED_PILOT_RESUME" >&2; exit 2; }
  EXTRA+=(--resume "$SHARED_PILOT_RESUME")
fi
echo "一次完成：epoch19稀疏36-token探针、完整GPU整数/采样检查、真实联合反传测速、六帧FPS、20%人口短程迁移及最终评估。"
echo "固定4历史→6未来；原epoch19和旧optimizer不动；只复用已有因果缓存，磁盘新增缓存配额为0。输出 $OUT"
echo '短程迁移冻结motion；正式联合训练速度单独实测且不冻结；不自动启动15/20轮，不调整阈值，不部署。'
"$PY" -u tools/real_motion/run_p0_f9_shared_evidence_pilot.py \
  --config "$ROOT/configs/real_motion_occfm.yaml" --checkpoint "$TEACHER" \
  --train-cache "$ROOT/data/p0_f9_v18_se2_train_full.pt" --dev-cache "$ROOT/data/p0_f9_v18_se2_val_all_4369.pt" \
  --population-manifest "$ROOT/data/p0_f9_v21_dev64_manifest.json" \
  --base-checkpoint "$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt" \
  --dataroot "$DATAROOT" --train-info "$DATAROOT/nuscenes_infos_train_temporal_v3_scene.pkl" \
  --dev-info "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl" --out-dir "$OUT" \
  --causal-geometry-cache "$ROOT/outputs/p0_f9_joint_causal_columns/causal_geometry_cache_v1" \
  --cpu-workers 8 --train-fraction .2 --train-epochs 1 "${EXTRA[@]}"
echo "全部结果：$OUT/summary.txt；完整数据：$OUT/bundle.json；安全续训断点：$OUT/migration_last.pt"
