#!/usr/bin/env bash
# One command, independent exact speed comparison, no training/full eval.
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == OccFM ]] || { echo '请先 conda activate OccFM' >&2; exit 2; }
[[ $# -le 1 ]] || { echo "用法: bash $0 [CHECKPOINT]" >&2; exit 2; }
ROOT=/root/nas/occ/swfm
cd "$ROOT"
CHECKPOINT="${1:-$ROOT/outputs/p0_f9_joint_causal_columns/checkpoint_selection_20261004_220555/epoch_0019.pt}"
[[ -f "$CHECKPOINT" ]] || { echo "[MISSING] $CHECKPOINT" >&2; exit 2; }
if pgrep -f '[p]ython.*(eval_p0_f9_joint|calibrate_p0_f9_joint|train_p0_f9_joint)' >/dev/null; then
  echo '测速请勿与训练/评估/校准争抢资源；不会停止已有进程。' >&2; exit 2
fi
PY="$(command -v python)"
OUT="${JOINT_EXECUTION_SPEED_OUT:-$ROOT/outputs/p0_f9_joint_causal_columns/execution_speed_fps_$(date +%Y%m%d_%H%M%S)_$$}"
[[ ! -e "$OUT" ]] || { echo "拒绝覆盖 $OUT" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export SWFM_COLUMN_CPU_BACKEND=native SWFM_COLUMN_CPU_BUNDLE=1
export SWFM_LOCAL_FAST_SUPERVISION=0 SWFM_LOCAL_STATIC_ROI=0
echo "同一epoch19，固定候选阈值0.5/0.5/0.95；不训练、不选ckpt、不跑full。输出 $OUT"
echo '一次完成：eval原版/异步回传/Graph/相同patch复用；正反序两遍；另测真正六帧生成FPS和同卡旧E14。'
"$PY" -u tools/real_motion/eval_p0_f9_joint_causal_columns.py \
  --config "$ROOT/configs/real_motion_occfm.yaml" --checkpoint "$CHECKPOINT" \
  --dev-cache "$ROOT/data/p0_f9_v18_se2_val_all_4369.pt" \
  --population-manifest "$ROOT/data/p0_f9_v21_dev64_manifest.json" \
  --base-checkpoint "$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt" \
  --dataroot /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes \
  --dev-info /root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes/nuscenes_infos_val_temporal_v3_scene.pkl \
  --population dev64 --out-dir "$OUT" --thresholds 0.5 0.5 0.95 \
  --cpu-workers 8 --batch-size 256 --column-feature-backend cpu \
  --speed-benchmark --speed-execution-and-fps \
  --speed-windows "${JOINT_EXECUTION_SPEED_WINDOWS:-32}" --fps-windows "${JOINT_EXECUTION_FPS_WINDOWS:-18}" --speed-repeats 2
echo "结果：$OUT/summary.txt；详细耗时和复用率：$OUT/speed.json"
