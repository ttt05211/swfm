#!/usr/bin/env bash
# One bounded TRAIN + TRAIN-only calibration + shared dev64/dev512 four-way screen.
set -euo pipefail
if [[ "${CONDA_DEFAULT_ENV:-}" != OccFM ]]; then
  echo "请先 conda activate OccFM，再用 bash 运行。" >&2; exit 2
fi
MODE="${1:-screen}"
case "$MODE" in smoke|screen) ;; *) echo "用法：bash $0 [smoke|screen]" >&2; exit 2 ;; esac
COLUMN_ROOT=/root/nas/occ/swfm
cd "$COLUMN_ROOT"
PY="$(command -v python)"
BASE_CKPT="$COLUMN_ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt"
TRAIN_V18="$COLUMN_ROOT/data/p0_f9_v18_se2_train_full.pt"
DEV_V18="$COLUMN_ROOT/data/p0_f9_v18_se2_val_all_4369.pt"
MANIFEST64="$COLUMN_ROOT/data/p0_f9_v21_dev64_manifest.json"
DATAROOT=/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes
TRAIN_INFO="$DATAROOT/nuscenes_infos_train_temporal_v3_scene.pkl"
DEV_INFO="$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl"
CONFIG="$COLUMN_ROOT/configs/real_motion_occfm.yaml"
RUN_DIR="${CAUSAL_COLUMNS_OUT:-$COLUMN_ROOT/outputs/p0_f9_causal_columns/${MODE}_$(date +%Y%m%d_%H%M%S)_$(git rev-parse --short HEAD)}"
WORKERS="${CAUSAL_COLUMNS_CPU_WORKERS:-8}"
BATCH="${CAUSAL_COLUMNS_BATCH_SIZE:-256}"
RAM="${CAUSAL_COLUMNS_BANK_MAX_MIB:-4096}"
if [[ -e "$RUN_DIR" ]]; then echo "拒绝覆盖：$RUN_DIR" >&2; exit 2; fi
for file in "$BASE_CKPT" "$TRAIN_V18" "$DEV_V18" "$MANIFEST64" "$TRAIN_INFO" "$DEV_INFO" "$CONFIG"; do
  if [[ ! -f "$file" ]]; then echo "[MISSING] $file" >&2; exit 2; fi
done
if [[ ! -d "$DATAROOT" ]]; then echo "[MISSING] $DATAROOT" >&2; exit 2; fi
if [[ ! -f upstream_occfm/forecast/models/__init__.py ]]; then
  echo "upstream_occfm 未初始化，请按仓库约定使用 Git 代理初始化固定子模块。" >&2; exit 2
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$COLUMN_ROOT:$COLUMN_ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -c 'import sys,torch; print("Python:",sys.executable,"Torch:",torch.__version__); assert torch.cuda.is_available(), "OccFM CUDA 不可用"; assert torch.cuda.is_bf16_supported(), "GPU 不支持 BF16"; print("GPU:",torch.cuda.get_device_name(0))'
"$PY" -m pytest -q tests/test_causal_columns.py
mkdir -p "$RUN_DIR"
echo "===== 生成 + source归属refine；V18冻结；不单独要求服务器smoke ====="
echo "输出：$RUN_DIR；TRAIN-only校准后，同趟输出V18/生成/refine/联合，含固定阈值诊断。"
"$PY" -u tools/real_motion/train_p0_f9_causal_columns.py \
  --config "$CONFIG" --train-cache "$TRAIN_V18" --dev-cache "$DEV_V18" \
  --population-manifest "$MANIFEST64" --base-checkpoint "$BASE_CKPT" \
  --dataroot "$DATAROOT" --train-info "$TRAIN_INFO" --dev-info "$DEV_INFO" \
  --out-dir "$RUN_DIR/model" --mode "$MODE" --cpu-workers "$WORKERS" \
  --batch-size "$BATCH" --bank-max-mib "$RAM" \
  2>&1 | tee "$RUN_DIR/run.log"
echo "完成。请发回 $RUN_DIR/model/summary.txt；全拒绝不算成功，不自动重训或放大。"
