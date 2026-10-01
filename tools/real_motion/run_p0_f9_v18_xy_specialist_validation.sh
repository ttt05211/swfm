#!/usr/bin/env bash
# Evaluate existing checkpoints ONCE; do not retrain or rewrite screen_pass.
set -euo pipefail
if [[ "${CONDA_DEFAULT_ENV:-}" != OccFM ]]; then
  echo "请先 conda activate OccFM，再用 bash 运行。" >&2; exit 2
fi
POPULATION="${1:-full4369}"
case "$POPULATION" in full4369|dev512) ;; *) echo "用法：bash $0 [full4369|dev512]" >&2; exit 2 ;; esac
XY_ROOT=/root/nas/occ/swfm
cd "$XY_ROOT"
PY="$(command -v python)"
BASE_CKPT="$XY_ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt"
DEV_V18="$XY_ROOT/data/p0_f9_v18_se2_val_all_4369.pt"
MANIFEST64="$XY_ROOT/data/p0_f9_v21_dev64_manifest.json"
DATAROOT=/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes
DEV_INFO="$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl"
CONFIG="$XY_ROOT/configs/real_motion_occfm.yaml"
# Exact run directory supplied in the user's latest summary; not a guessed path.
MODEL_DIR="${V18_XYS_EVAL_MODEL_DIR:-$XY_ROOT/outputs/p0_f9_v18_xy_specialist/screen20_20261001_103415_e5a122a/model}"
WORKERS="${V18_XYS_CPU_WORKERS:-8}"
RUN_DIR="${V18_XYS_EVAL_OUT:-$(dirname "$MODEL_DIR")/eval_${POPULATION}_$(date +%Y%m%d_%H%M%S)_$(git rev-parse --short HEAD)}"
if [[ -e "$RUN_DIR" ]]; then echo "拒绝覆盖：$RUN_DIR" >&2; exit 2; fi
for file in "$BASE_CKPT" "$DEV_V18" "$MANIFEST64" "$DEV_INFO" "$CONFIG" \
  "$MODEL_DIR/execution_contract.json" "$MODEL_DIR/summary.json" \
  "$MODEL_DIR/frozen_yaw_xy_best.pt" "$MODEL_DIR/frozen_yaw_xy_last.pt" \
  "$MODEL_DIR/scheduled_gt_yaw_xy_best.pt" "$MODEL_DIR/scheduled_gt_yaw_xy_last.pt"; do
  if [[ ! -f "$file" ]]; then echo "[MISSING] $file" >&2; exit 2; fi
done
if [[ ! -d "$DATAROOT" ]]; then echo "[MISSING] $DATAROOT" >&2; exit 2; fi
if [[ ! -f upstream_occfm/forecast/models/__init__.py ]]; then
  echo "upstream_occfm 子模块未初始化；请通过 Git 代理初始化仓库声明的子模块。" >&2; exit 2
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$XY_ROOT:$XY_ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
echo "===== 验证预检查：不训练、不改checkpoint ====="
"$PY" -c 'import sys,torch; print("Python:",sys.executable,"Torch:",torch.__version__); assert torch.cuda.is_available(), "OccFM CUDA 不可用"; assert torch.cuda.is_bf16_supported(), "GPU 不支持 BF16"; print("GPU:",torch.cuda.get_device_name(0))'
"$PY" -m pytest -q tests/test_v18_xy_specialist.py
mkdir -p "$RUN_DIR"
echo "===== $POPULATION：同趟评估原E14、两组best/last ====="
echo "原模型：$MODEL_DIR；结果：$RUN_DIR/model；只有评估，没有训练。"
"$PY" -u tools/real_motion/eval_p0_f9_v18_xy_specialist.py \
  --config "$CONFIG" --dev-cache "$DEV_V18" --population-manifest "$MANIFEST64" \
  --model-dir "$MODEL_DIR" --base-checkpoint "$BASE_CKPT" --dataroot "$DATAROOT" \
  --dev-info "$DEV_INFO" --out-dir "$RUN_DIR/model" --population "$POPULATION" --cpu-workers "$WORKERS" \
  2>&1 | tee "$RUN_DIR/run.log"
echo "完成。请发回 $RUN_DIR/model/summary.txt；不据此改选best，不自动重训。"
