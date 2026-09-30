#!/usr/bin/env bash
# Child bash only. No oracle, generation or automatic full-training stage.
set -euo pipefail
if [[ "${CONDA_DEFAULT_ENV:-}" != OccFM ]]; then
  echo "请先 conda activate OccFM，再用 bash 运行。" >&2; exit 2
fi
MODE="${1:-screen}"
case "$MODE" in smoke|screen) ;; *) echo "用法：bash $0 [smoke|screen]" >&2; exit 2 ;; esac
XY_ROOT=/root/nas/occ/swfm
cd "$XY_ROOT"
PY="$(command -v python)"
BASE_CKPT="$XY_ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt"
TRAIN_V18="$XY_ROOT/data/p0_f9_v18_se2_train_full.pt"
DEV_V18="$XY_ROOT/data/p0_f9_v18_se2_val_all_4369.pt"
MANIFEST64="$XY_ROOT/data/p0_f9_v21_dev64_manifest.json"
DATAROOT=/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes
DEV_INFO="$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl"
CONFIG="$XY_ROOT/configs/real_motion_occfm.yaml"
WORKERS="${V18_XY_CPU_WORKERS:-8}"
BATCH="${V18_XY_BATCH_SIZE:-128}"
RUN_DIR="${V18_XY_OUT:-$XY_ROOT/outputs/p0_f9_v18_xy_trajectory/${MODE}_$(date +%Y%m%d_%H%M%S)_$(git rev-parse --short HEAD)}"
if [[ -e "$RUN_DIR" ]]; then echo "拒绝覆盖：$RUN_DIR" >&2; exit 2; fi
for file in "$BASE_CKPT" "$TRAIN_V18" "$DEV_V18" "$MANIFEST64" "$DEV_INFO" "$CONFIG"; do
  if [[ ! -f "$file" ]]; then echo "[MISSING] $file" >&2; exit 2; fi
done
if [[ ! -d "$DATAROOT" ]]; then echo "[MISSING] $DATAROOT" >&2; exit 2; fi
if [[ ! -f upstream_occfm/forecast/models/__init__.py ]]; then
  echo "upstream_occfm 子模块未初始化；请通过 Git 代理初始化仓库声明的子模块。" >&2; exit 2
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$XY_ROOT:$XY_ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
echo "===== 预检查（V18 冻结，不跑旧 oracle / 生成实验） ====="
"$PY" -c 'import sys,torch; print("Python:",sys.executable,"Torch:",torch.__version__); assert torch.cuda.is_available(), "OccFM CUDA 不可用"; print("GPU:",torch.cuda.get_device_name(0))'
"$PY" -m pytest -q tests/test_v18_xy_trajectory.py tests/test_p0_f9_v18_se2.py
"$PY" tools/real_motion/train_p0_f9_v18_xy_trajectory.py --help >/dev/null
mkdir -p "$RUN_DIR"
echo "===== 一次 latent 准备 → 三组真实 XY 训练 → TRAIN 校准 → 一次 dev64 ====="
echo "输出：$RUN_DIR；batch=$BATCH sources；不重建任何数据缓存或 prototype。"
"$PY" -u tools/real_motion/train_p0_f9_v18_xy_trajectory.py \
  --config "$CONFIG" --train-cache "$TRAIN_V18" --dev-cache "$DEV_V18" \
  --population-manifest "$MANIFEST64" --base-checkpoint "$BASE_CKPT" \
  --dataroot "$DATAROOT" --dev-info "$DEV_INFO" --out-dir "$RUN_DIR/model" \
  --mode "$MODE" --device cuda --cpu-workers "$WORKERS" --batch-size "$BATCH" \
  2>&1 | tee "$RUN_DIR/run.log"
echo "完成。请发回 $RUN_DIR/model/summary.txt；失败也生成摘要，不自动重训。"
