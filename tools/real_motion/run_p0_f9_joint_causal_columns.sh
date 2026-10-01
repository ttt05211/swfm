#!/usr/bin/env bash
# One bounded random-init joint screen; no full validation or automatic retries.
set -euo pipefail
if [[ "${CONDA_DEFAULT_ENV:-}" != OccFM ]]; then echo "请先 conda activate OccFM" >&2; exit 2; fi
MODE="${1:-screen}"
case "$MODE" in smoke|screen) ;; *) echo "用法：bash $0 [screen|smoke]" >&2; exit 2 ;; esac
JOINT_ROOT=/root/nas/occ/swfm
cd "$JOINT_ROOT"
PY="$(command -v python)"
DATAROOT=/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes
RUN_DIR="${JOINT_COLUMNS_OUT:-$JOINT_ROOT/outputs/p0_f9_joint_causal_columns/${MODE}_$(date +%Y%m%d_%H%M%S)_$(git rev-parse --short HEAD)}"
WORKERS="${JOINT_COLUMNS_CPU_WORKERS:-8}"
CACHE="${JOINT_COLUMNS_FRAME_CACHE_MIB:-256}"
if [[ -e "$RUN_DIR" ]]; then echo "拒绝覆盖：$RUN_DIR" >&2; exit 2; fi
for file in "$JOINT_ROOT/configs/real_motion_occfm.yaml" \
  "$JOINT_ROOT/data/p0_f9_v18_se2_train_full.pt" "$JOINT_ROOT/data/p0_f9_v18_se2_val_all_4369.pt" \
  "$JOINT_ROOT/data/p0_f9_v21_dev64_manifest.json" "$JOINT_ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt" \
  "$DATAROOT/nuscenes_infos_train_temporal_v3_scene.pkl" "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl"; do
  if [[ ! -f "$file" ]]; then echo "[MISSING] $file" >&2; exit 2; fi
done
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$JOINT_ROOT:$JOINT_ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -c 'import torch,sys; print(sys.executable,torch.__version__); assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA/BF16 unavailable"'
"$PY" -m pytest -q tests/test_joint_causal_columns.py tests/test_column_runtime_pipeline.py
mkdir -p "$RUN_DIR"
echo "随机初始化一阶段：TRAIN1024 / 1024 updates，固定dev64监控，最终dev512；不跑full4369。"
echo "包含同初始化V18-only对照；E14只作参考，不加载为初始权重。输出：$RUN_DIR"
"$PY" -u tools/real_motion/train_p0_f9_joint_causal_columns.py \
  --config "$JOINT_ROOT/configs/real_motion_occfm.yaml" \
  --train-cache "$JOINT_ROOT/data/p0_f9_v18_se2_train_full.pt" \
  --dev-cache "$JOINT_ROOT/data/p0_f9_v18_se2_val_all_4369.pt" \
  --population-manifest "$JOINT_ROOT/data/p0_f9_v21_dev64_manifest.json" \
  --base-checkpoint "$JOINT_ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt" \
  --dataroot "$DATAROOT" --train-info "$DATAROOT/nuscenes_infos_train_temporal_v3_scene.pkl" \
  --dev-info "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl" \
  --mode "$MODE" --cpu-workers "$WORKERS" --frame-cache-mib "$CACHE" \
  --eval-batch-size 256 --out-dir "$RUN_DIR/model" 2>&1 | tee "$RUN_DIR/run.log"
echo "完成。发回 $RUN_DIR/model/summary.txt 和 progress.jsonl；不会自动扩训或重试。"
