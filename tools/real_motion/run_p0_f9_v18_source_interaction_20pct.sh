#!/usr/bin/env bash
# ONE paired train/calibrate/dev experiment; no automatic extra experiments.
set -euo pipefail
if [[ "${CONDA_DEFAULT_ENV:-}" != OccFM ]]; then
  echo "请先 conda activate OccFM，再用 bash 运行。" >&2; exit 2
fi
MODE="${1:-screen}"
case "$MODE" in smoke|screen) ;; *) echo "用法：bash $0 [smoke|screen]" >&2; exit 2 ;; esac
SI_ROOT=/root/nas/occ/swfm
cd "$SI_ROOT"
PY="$(command -v python)"
BASE_CKPT="$SI_ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt"
TRAIN_V18="$SI_ROOT/data/p0_f9_v18_se2_train_full.pt"
DEV_V18="$SI_ROOT/data/p0_f9_v18_se2_val_all_4369.pt"
MANIFEST64="$SI_ROOT/data/p0_f9_v21_dev64_manifest.json"
DATAROOT=/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes
TRAIN_INFO="$DATAROOT/nuscenes_infos_train_temporal_v3_scene.pkl"
DEV_INFO="$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl"
CONFIG="$SI_ROOT/configs/real_motion_occfm.yaml"
WORKERS="${V18_SI_CPU_WORKERS:-8}"
SOURCES="${V18_SI_SOURCE_BUDGET:-256}"
WINDOWS="${V18_SI_WINDOW_BUDGET:-8}"
EPOCHS="${V18_SI_EPOCHS:-3}"
RUN_DIR="${V18_SI_OUT:-$SI_ROOT/outputs/p0_f9_v18_source_interaction/${MODE}20_$(date +%Y%m%d_%H%M%S)_$(git rev-parse --short HEAD)}"
if [[ -e "$RUN_DIR" ]]; then echo "拒绝覆盖：$RUN_DIR" >&2; exit 2; fi
for file in "$BASE_CKPT" "$TRAIN_V18" "$DEV_V18" "$MANIFEST64" "$TRAIN_INFO" "$DEV_INFO" "$CONFIG"; do
  if [[ ! -f "$file" ]]; then echo "[MISSING] $file" >&2; exit 2; fi
done
if [[ ! -d "$DATAROOT" ]]; then echo "[MISSING] $DATAROOT" >&2; exit 2; fi
if [[ ! -f upstream_occfm/forecast/models/__init__.py ]]; then
  echo "upstream_occfm 子模块未初始化；请通过 Git 代理初始化仓库声明的子模块。" >&2; exit 2
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$SI_ROOT:$SI_ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
echo "===== CUDA / 合同 / 小型测试预检查 ====="
"$PY" -c 'import sys,torch; print("Python:",sys.executable,"Torch:",torch.__version__); assert torch.cuda.is_available(), "OccFM CUDA 不可用"; assert torch.cuda.is_bf16_supported(), "GPU 不支持 BF16"; print("GPU:",torch.cuda.get_device_name(0))'
"$PY" -m pytest -q tests/test_v18_source_interaction.py
mkdir -p "$RUN_DIR"
echo "===== 两组各 4086 windows × $EPOCHS epochs → TRAIN 独立场景校准 → 一次 dev64 ====="
echo "输出：$RUN_DIR；不改 Clean-E14，不构建新数据缓存，不运行生成模块。"
"$PY" -u tools/real_motion/train_p0_f9_v18_source_interaction.py \
  --config "$CONFIG" --train-cache "$TRAIN_V18" --dev-cache "$DEV_V18" \
  --population-manifest "$MANIFEST64" --base-checkpoint "$BASE_CKPT" \
  --dataroot "$DATAROOT" --train-info "$TRAIN_INFO" --dev-info "$DEV_INFO" --out-dir "$RUN_DIR/model" \
  --mode "$MODE" --device cuda --cpu-workers "$WORKERS" \
  --source-budget "$SOURCES" --window-budget "$WINDOWS" --epochs "$EPOCHS" \
  2>&1 | tee "$RUN_DIR/run.log"
echo "完成。请发回 $RUN_DIR/model/summary.txt；最后 checkpoint 不代表最佳，失败不自动重训。"
