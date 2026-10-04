#!/usr/bin/env bash
# Fixed-weight tuning only. Wait for existing GPU evaluation; never signal it.
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == OccFM ]] || { echo '请先 conda activate OccFM' >&2; exit 2; }
[[ $# -le 1 ]] || { echo "用法: bash $0 [checkpoint]" >&2; exit 2; }
ROOT=/root/nas/occ/swfm
cd "$ROOT"
PY="$(command -v python)"
CHECKPOINT="${1:-$ROOT/outputs/p0_f9_joint_causal_columns/checkpoint_selection_20261004_220555/epoch_0019.pt}"
[[ -f "$CHECKPOINT" ]] || { echo "[MISSING] $CHECKPOINT" >&2; exit 2; }
if [[ "${JOINT_THRESHOLD_RESUME:-0}" == 1 && -z "${JOINT_THRESHOLD_OUT:-}" ]]; then
  echo '恢复校准必须设置 JOINT_THRESHOLD_OUT 为原校准输出目录。' >&2; exit 2
fi
while pgrep -f '[p]ython.*(train_p0_f9_joint_causal_columns_full.py|eval_p0_f9_joint_causal_columns.py|eval_p0_f9_joint_checkpoints.py|calibrate_p0_f9_joint_thresholds.py|eval_p0_f9_joint_zero_shot_long_rollout.py)' >/dev/null; do
  echo '已有训练/评估在运行；等待15秒，不中断它、不同时抢GPU。'
  sleep 15
done
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export SWFM_COLUMN_CPU_BACKEND=native SWFM_COLUMN_CPU_BUNDLE=1
export SWFM_LOCAL_FAST_SUPERVISION=0 SWFM_LOCAL_STATIC_ROI=0
OUT="${JOINT_THRESHOLD_OUT:-$ROOT/outputs/p0_f9_joint_causal_columns/threshold_dev512_epoch19_$(date +%Y%m%d_%H%M%S)_$$}"
EXTRA=()
if [[ "${JOINT_THRESHOLD_RESUME:-0}" == 1 ]]; then EXTRA+=(--resume); fi
"$PY" -m pytest -q tests/test_joint_threshold_sweep.py
echo '固定权重dev512校准：一次概率前向，64个joint阈值组合；不训练、不修改正在运行的full结果。'
echo "输出: $OUT"
"$PY" -u tools/real_motion/calibrate_p0_f9_joint_thresholds.py \
  --config "$ROOT/configs/real_motion_occfm.yaml" --checkpoint "$CHECKPOINT" \
  --dev-cache "$ROOT/data/p0_f9_v18_se2_val_all_4369.pt" \
  --population-manifest "$ROOT/data/p0_f9_v21_dev64_manifest.json" \
  --base-checkpoint "$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt" \
  --dataroot /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes \
  --dev-info /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes/nuscenes_infos_val_temporal_v3_scene.pkl \
  --out-dir "$OUT" --cpu-workers 8 --checkpoint-every 8 "${EXTRA[@]}"
echo "结果: $OUT/summary.txt；全部组合: calibration.json。这是调参分数，不是独立测试。"
