#!/usr/bin/env bash
# Bounded read-only real CUDA eval benchmark; never starts a full evaluation.
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == OccFM ]] || { echo '请先 conda activate OccFM' >&2; exit 2; }
[[ $# -le 1 ]] || { echo "用法: bash $0 [CHECKPOINT]" >&2; exit 2; }
ROOT=/root/nas/occ/swfm
cd "$ROOT"
CHECKPOINT="${1:-$ROOT/outputs/p0_f9_joint_causal_columns/checkpoint_selection_20261004_220555/epoch_0019.pt}"
[[ -f "$CHECKPOINT" ]] || { echo "[MISSING] $CHECKPOINT" >&2; exit 2; }
if pgrep -f '[p]ython.*(eval_p0_f9_joint|calibrate_p0_f9_joint|train_p0_f9_joint)' >/dev/null; then
  echo '还有训练/评估/校准进程，测速会相互争抢资源；请等其结束再运行。不会停止任何进程。' >&2
  exit 2
fi
OUT="${JOINT_EVAL_SPEED_OUT:-$ROOT/outputs/p0_f9_joint_causal_columns/eval_speed_$(date +%Y%m%d_%H%M%S)_$$}"
[[ ! -e "$OUT" ]] || { echo "拒绝覆盖 $OUT" >&2; exit 2; }
PY="$(command -v python)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export SWFM_COLUMN_CPU_BACKEND=native SWFM_COLUMN_CPU_BUNDLE=1
export SWFM_LOCAL_FAST_SUPERVISION=0 SWFM_LOCAL_STATIC_ROI=0
SUITE=()
if [[ "${JOINT_EVAL_SPEED_COLUMN_SUITE:-0}" == 1 ]]; then SUITE+=(--speed-column-probability); fi
echo "只计时：同epoch19、同scene-balanced dev64子集、同batch256；三种模式正反序各测一次。column suite=${JOINT_EVAL_SPEED_COLUMN_SUITE:-0} 输出 $OUT"
"$PY" -u tools/real_motion/eval_p0_f9_joint_causal_columns.py \
  --config "$ROOT/configs/real_motion_occfm.yaml" --checkpoint "$CHECKPOINT" \
  --dev-cache "$ROOT/data/p0_f9_v18_se2_val_all_4369.pt" \
  --population-manifest "$ROOT/data/p0_f9_v21_dev64_manifest.json" \
  --base-checkpoint "$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt" \
  --dataroot /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes \
  --dev-info /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes/nuscenes_infos_val_temporal_v3_scene.pkl \
  --population dev64 --fixed-monitor-thresholds --out-dir "$OUT" \
  --cpu-workers 8 --batch-size 256 --column-feature-backend cpu \
  --speed-benchmark --speed-windows "${JOINT_EVAL_SPEED_WINDOWS:-32}" --speed-repeats 2 "${SUITE[@]}"
echo "测速完成：$OUT/summary.txt；分段详细数据：$OUT/speed.json"
