#!/usr/bin/env bash
# Run as child bash, NOT source. Frozen V18 remains read-only.
set -euo pipefail
if [[ "${CONDA_DEFAULT_ENV:-}" != OccFM ]]; then
  echo "请先 conda activate OccFM，再用 bash 运行。" >&2; exit 2
fi
MODE="${1:-screen}"
case "$MODE" in smoke|screen) ;; *) echo "用法：bash $0 [smoke|screen]（不自动扩大）" >&2; exit 2 ;; esac
EMERGENCE_ROOT=/root/nas/occ/swfm
cd "$EMERGENCE_ROOT"
PY="$(command -v python)"
BASE_CKPT="$EMERGENCE_ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt"
TRAIN_V18="$EMERGENCE_ROOT/data/p0_f9_v18_se2_train_full.pt"
DEV_V18="$EMERGENCE_ROOT/data/p0_f9_v18_se2_val_all_4369.pt"
MANIFEST64="$EMERGENCE_ROOT/data/p0_f9_v21_dev64_manifest.json"
DATAROOT=/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes
TRAIN_INFO="$DATAROOT/nuscenes_infos_train_temporal_v3_scene.pkl"
DEV_INFO="$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl"
CONFIG="$EMERGENCE_ROOT/configs/real_motion_occfm.yaml"
WORKERS="${SPARSE_EMERGENCE_CPU_WORKERS:-8}"
BATCH="${SPARSE_EMERGENCE_BATCH_SIZE:-64}"
RUN_DIR="${SPARSE_EMERGENCE_OUT:-$EMERGENCE_ROOT/outputs/p0_f9_sparse_emergence/${MODE}_$(date +%Y%m%d_%H%M%S)_$(git rev-parse --short HEAD)}"
if [[ -e "$RUN_DIR" ]]; then echo "拒绝覆盖：$RUN_DIR" >&2; exit 2; fi
for input in "$BASE_CKPT" "$TRAIN_V18" "$DEV_V18" "$MANIFEST64" "$TRAIN_INFO" "$DEV_INFO" "$CONFIG"; do
  if [[ ! -f "$input" ]]; then echo "[MISSING] $input" >&2; exit 2; fi
done
if [[ ! -d "$DATAROOT" ]]; then echo "[MISSING] $DATAROOT" >&2; exit 2; fi
if [[ ! -f upstream_occfm/forecast/models/__init__.py ]]; then
  echo "upstream_occfm 子模块未初始化；请通过 Git 代理初始化仓库声明的子模块。" >&2; exit 2
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$EMERGENCE_ROOT:$EMERGENCE_ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
echo "===== 预检查（不训练 V18） ====="
"$PY" -c 'import sys,torch; print("Python:",sys.executable,"Torch:",torch.__version__); assert torch.cuda.is_available(), "OccFM CUDA 不可用"; print("GPU:",torch.cuda.get_device_name(0))'
"$PY" -m pytest -q tests/test_sparse_emergence.py
"$PY" tools/real_motion/train_p0_f9_sparse_emergence.py --help >/dev/null
mkdir -p "$RUN_DIR"
echo "===== 一次准备 → 两个小 decoder → TRAIN 校准 → 一次 dev64 ====="
echo "输出：$RUN_DIR；无需 Stage-1/prototype 重建；每个窗口/每32步打印进度。"
"$PY" -u tools/real_motion/train_p0_f9_sparse_emergence.py \
  --config "$CONFIG" --train-cache "$TRAIN_V18" --dev-cache "$DEV_V18" \
  --population-manifest "$MANIFEST64" --base-checkpoint "$BASE_CKPT" \
  --dataroot "$DATAROOT" --train-info "$TRAIN_INFO" --dev-info "$DEV_INFO" \
  --out-dir "$RUN_DIR/model" --mode "$MODE" --device cuda \
  --cpu-workers "$WORKERS" --batch-size "$BATCH" \
  2>&1 | tee "$RUN_DIR/run.log"
echo "完成。请发回 $RUN_DIR/model/summary.txt；失败也会生成摘要，不自动重训。"
