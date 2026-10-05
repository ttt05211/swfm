#!/usr/bin/env bash
# Freeze the selected epoch19; never train, recalibrate or touch original runs.
set -euo pipefail
if [[ "${CONDA_DEFAULT_ENV:-}" != OccFM ]]; then echo '请先 conda activate OccFM' >&2; exit 2; fi
if [[ $# -gt 3 ]]; then echo "用法: bash $0 [dev64|dev512|all] [OUTPUT_DIR] [--resume]" >&2; exit 2; fi
ROOT=/root/nas/occ/swfm
cd "$ROOT"
POPULATION="${1:-dev64}"
case "$POPULATION" in dev64|dev512|all) ;; *) echo 'population须为dev64/dev512/all' >&2; exit 2 ;; esac
CHECKPOINT="$ROOT/outputs/p0_f9_joint_causal_columns/checkpoint_selection_20261004_220555/epoch_0019.pt"
OUT="${2:-$ROOT/outputs/p0_f9_joint_causal_columns/long6s_epoch19_${POPULATION}_$(date +%Y%m%d_%H%M%S)_$$}"
EXTRA=()
if [[ -n "${LONG_HANDOFF_MODES:-}" ]]; then EXTRA+=(--handoff-modes "$LONG_HANDOFF_MODES"); fi
case "${LONG_COMPARE_E14:-0}" in
  0) ;; 1) EXTRA+=(--compare-e14) ;; *) echo 'LONG_COMPARE_E14须为0或1' >&2; exit 2 ;;
esac
if [[ $# == 3 ]]; then
  [[ "$3" == --resume ]] || { echo '第三个参数只允许--resume' >&2; exit 2; }
  EXTRA+=(--resume)
fi
for path in "$CHECKPOINT" "$ROOT/data/p0_f9_v18_se2_val_all_4369.pt" "$ROOT/data/p0_f9_v21_dev64_manifest.json" \
  "$ROOT/configs/real_motion_occfm.yaml" "$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt" \
  /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes/nuscenes_infos_val_temporal_v3_scene.pkl; do
  [[ -f "$path" ]] || { echo "[MISSING] $path" >&2; exit 2; }
done
PY="$(command -v python)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export SWFM_COLUMN_CPU_BACKEND="${SWFM_COLUMN_CPU_BACKEND:-native}" SWFM_COLUMN_CPU_BUNDLE=1
export SWFM_LOCAL_FAST_SUPERVISION=0 SWFM_LOCAL_STATIC_ROI=0
echo "冻结epoch19：严格4历史→预测6帧→取最后4张预测→继续预测6帧；报告1–6s，不训练。"
echo "population=$POPULATION；dev512/all只评完整6秒窗口，不补短序列。输出：$OUT"
"$PY" -u tools/real_motion/eval_p0_f9_joint_zero_shot_long_rollout.py \
  --config "$ROOT/configs/real_motion_occfm.yaml" --checkpoint "$CHECKPOINT" --expected-epoch 19 \
  --dev-cache "$ROOT/data/p0_f9_v18_se2_val_all_4369.pt" \
  --population-manifest "$ROOT/data/p0_f9_v21_dev64_manifest.json" \
  --base-checkpoint "$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt" \
  --dataroot /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes \
  --dev-info /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes/nuscenes_infos_val_temporal_v3_scene.pkl \
  --out-dir "$OUT" --population "$POPULATION" --cpu-workers "${LONG_EVAL_CPU_WORKERS:-8}" \
  --batch-size "${LONG_EVAL_BATCH_SIZE:-256}" --feature-backend "${LONG_FEATURE_BACKEND:-cpu}" "${EXTRA[@]}"
echo "完成：$OUT/summary.txt；完整指标、原始计数与审计：$OUT/evaluation.json"
