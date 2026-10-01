#!/usr/bin/env bash
# One matched-budget comparison. Never stop/overwrite the existing full15 run.
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == OccFM ]] || { echo "请先 conda activate OccFM" >&2; exit 2; }
MODE="${1:-screen}"
case "$MODE" in screen|smoke) ;; *) echo "用法: bash $0 [screen|smoke]" >&2; exit 2 ;; esac
ROOT=/root/nas/occ/swfm
cd "$ROOT"
PY="$(command -v python)"
if pgrep -f '[p]ython.*train_p0_f9_joint_causal_columns' >/dev/null; then
  echo "检测到 joint 训练仍在运行。请先等它完成或安全保存停止，避免争抢 GPU/CPU；本脚本不会停止它。" >&2
  exit 2
fi
DATAROOT=/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes
RUN_DIR="${ADAPTIVE_REFINE_OUT:-$ROOT/outputs/p0_f9_adaptive_refine/pair_${MODE}_$(date +%Y%m%d_%H%M%S)_$(git rev-parse --short HEAD)}"
[[ ! -e "$RUN_DIR" ]] || { echo "拒绝覆盖: $RUN_DIR" >&2; exit 2; }
COMMON=(--config "$ROOT/configs/real_motion_occfm.yaml"
  --train-cache "$ROOT/data/p0_f9_v18_se2_train_full.pt"
  --dev-cache "$ROOT/data/p0_f9_v18_se2_val_all_4369.pt"
  --population-manifest "$ROOT/data/p0_f9_v21_dev64_manifest.json"
  --base-checkpoint "$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt"
  --dataroot "$DATAROOT" --train-info "$DATAROOT/nuscenes_infos_train_temporal_v3_scene.pkl"
  --dev-info "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl"
  --mode "$MODE" --cpu-workers 8 --frame-cache-mib 512 --eval-batch-size 256 --seed 20261002
  --causal-geometry-cache "$RUN_DIR/fixed_geometry_cache")
for path in "$ROOT/configs/real_motion_occfm.yaml" "$ROOT/data/p0_f9_v18_se2_train_full.pt" \
  "$ROOT/data/p0_f9_v18_se2_val_all_4369.pt" "$ROOT/data/p0_f9_v21_dev64_manifest.json" \
  "$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt" \
  "$DATAROOT/nuscenes_infos_train_temporal_v3_scene.pkl" "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl"; do
  [[ -f "$path" ]] || { echo "[MISSING] $path" >&2; exit 2; }
done
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -c 'import torch,sys; print(sys.executable, torch.__version__); assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA/BF16 required"'
"$PY" -m pytest -q tests/test_adaptive_column_context.py tests/test_joint_causal_columns.py
mkdir -p "$RUN_DIR"
echo "两组均为 TRAIN1024 / 1024窗口更新，约1遍；原初始化/候选/loss/校准/评估不变。固定几何缓存上限4GiB，非模型输出缓存。"
"$PY" -u tools/real_motion/train_p0_f9_joint_causal_columns.py "${COMMON[@]}" \
  --out-dir "$RUN_DIR/local" 2>&1 | tee "$RUN_DIR/local.log"
"$PY" -u tools/real_motion/train_p0_f9_joint_causal_columns.py "${COMMON[@]}" \
  --adaptive-refine --out-dir "$RUN_DIR/adaptive" 2>&1 | tee "$RUN_DIR/adaptive.log"
"$PY" -u tools/real_motion/compare_p0_f9_adaptive_refine.py \
  --local "$RUN_DIR/local/summary.json" --adaptive "$RUN_DIR/adaptive/summary.json" --out-dir "$RUN_DIR/report"
echo "完成。请发回: $RUN_DIR/report/combined_summary.txt"
