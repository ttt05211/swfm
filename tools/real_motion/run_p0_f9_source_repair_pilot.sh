#!/usr/bin/env bash
# One finite experiment; never overwrites the old selected/full-training model.
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == OccFM ]] || { echo '请先 conda activate OccFM' >&2; exit 2; }
[[ $# -le 1 ]] || { echo "用法: bash $0 [epoch19 checkpoint]" >&2; exit 2; }
ROOT=/root/nas/occ/swfm
cd "$ROOT"
PY="$(command -v python)"
TEACHER="${1:-$ROOT/outputs/p0_f9_joint_causal_columns/checkpoint_selection_20261004_220555/epoch_0019.pt}"
DATAROOT=/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes
if pgrep -f '[p]ython.*(train_p0_f9_joint|eval_p0_f9_joint|calibrate_p0_f9_joint|run_p0_f9_shared_evidence_pilot|run_p0_f9_source_repair_pilot)' >/dev/null; then
  echo '请先安全停止其他训练/评估，避免争抢资源；本脚本不会kill进程。' >&2; exit 2
fi
OUT="${SOURCE_REPAIR_OUT:-$ROOT/outputs/p0_f9_joint_causal_columns/source_repair_$(date +%Y%m%d_%H%M%S)_$$}"
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
if [[ "${SOURCE_REPAIR_SMOKE:-0}" == 1 ]]; then
  EXTRA+=(--eval-windows 2 --fps-windows 1 --speed-train-windows 4 --speed-repeats 1 --max-updates 2)
else
  EXTRA+=(--eval-windows 64 --fps-windows 6 --speed-repeats 2)
fi
if [[ "${SOURCE_REPAIR_DEV512:-0}" == 1 ]]; then EXTRA+=(--final-dev512); fi
if [[ -n "${SOURCE_REPAIR_RESUME:-}" ]]; then
  [[ -f "$SOURCE_REPAIR_RESUME" ]] || { echo "[MISSING] $SOURCE_REPAIR_RESUME" >&2; exit 2; }
  EXTRA+=(--resume "$SOURCE_REPAIR_RESUME")
fi
echo '一次完成：真实静态/动态/REMOVE拆分 + 证据域oracle + TRAIN20%一遍迁移 + dev64 + 六帧FPS。'
echo '4历史→6未来；冻结epoch19 motion，不是从头联合训练；旧GEN保持，稀疏refine只ADD。'
echo "原实验/optimizer不动；只读现有因果缓存，新增几何磁盘配额为0。输出 $OUT"
"$PY" -u tools/real_motion/run_p0_f9_source_repair_pilot.py \
  --config "$ROOT/configs/real_motion_occfm.yaml" --checkpoint "$TEACHER" \
  --train-cache "$ROOT/data/p0_f9_v18_se2_train_full.pt" --dev-cache "$ROOT/data/p0_f9_v18_se2_val_all_4369.pt" \
  --population-manifest "$ROOT/data/p0_f9_v21_dev64_manifest.json" \
  --base-checkpoint "$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt" \
  --dataroot "$DATAROOT" --train-info "$DATAROOT/nuscenes_infos_train_temporal_v3_scene.pkl" \
  --dev-info "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl" --out-dir "$OUT" \
  --causal-geometry-cache "$ROOT/outputs/p0_f9_joint_causal_columns/causal_geometry_cache_v1" \
  --cpu-workers 8 --train-fraction .2 "${EXTRA[@]}"
echo "结果：$OUT/summary.txt；详细数据：$OUT/bundle.json；新协议断点：$OUT/migration_last.pt"
cat "$OUT/summary.txt"
