#!/usr/bin/env bash
# Execute as a child bash process, never source this into an interactive shell.
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "OccFM" ]]; then
  echo "请先 conda activate OccFM，再用 bash 运行。" >&2
  exit 2
fi
MODE="${1:-screen}"
case "$MODE" in smoke|screen|main) ;; *) echo "用法：bash $0 [smoke|screen|main]" >&2; exit 2 ;; esac
SELECTOR_ROOT=/root/nas/occ/swfm
cd "$SELECTOR_ROOT"
PY="$(command -v python)"
BASE_CKPT="$SELECTOR_ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt"
TRAIN_V18="$SELECTOR_ROOT/data/p0_f9_v18_se2_train_full.pt"
DEV_V18="$SELECTOR_ROOT/data/p0_f9_v18_se2_val_all_4369.pt"
DEV_STAGE1="$SELECTOR_ROOT/data/p0_f9_v20_stage1_v2_dev512_v19split_769edc52"
MANIFEST64="$SELECTOR_ROOT/data/p0_f9_v21_dev64_manifest.json"
DATAROOT=/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes
TRAIN_INFO="$DATAROOT/nuscenes_infos_train_temporal_v3_scene.pkl"
DEV_INFO="$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl"
CONFIG="$SELECTOR_ROOT/configs/real_motion_occfm.yaml"
CPU_WORKERS="${STATIC_SELECTOR_CPU_WORKERS:-8}"
RUN_DIR="${STATIC_SELECTOR_OUT:-$SELECTOR_ROOT/outputs/p0_f9_static_evidence_selector/${MODE}_$(date +%Y%m%d_%H%M%S)_$(git rev-parse --short HEAD)}"
if [[ -e "$RUN_DIR" ]]; then echo "拒绝覆盖现有目录：$RUN_DIR" >&2; exit 2; fi
for input in "$BASE_CKPT" "$TRAIN_V18" "$DEV_V18" "$TRAIN_INFO" "$DEV_INFO" "$CONFIG"; do
  if [[ ! -f "$input" ]]; then echo "[MISSING] $input" >&2; exit 2; fi
done
if [[ ! -d "$DATAROOT" ]]; then echo "[MISSING] $DATAROOT" >&2; exit 2; fi
if [[ ! -f upstream_occfm/forecast/models/__init__.py ]]; then
  echo "upstream_occfm 子模块未初始化。请通过 Git 代理初始化仓库声明的子模块后重试。" >&2
  exit 2
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$SELECTOR_ROOT:$SELECTOR_ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
echo "===== 环境与代码预检查 ====="
"$PY" -c 'import sys,torch; print("Python:",sys.executable,"Torch:",torch.__version__); assert torch.cuda.is_available(), "OccFM 中 CUDA 不可用"; print("GPU:",torch.cuda.get_device_name(0))'
"$PY" -m pytest -q tests/test_static_evidence_selector.py tests/test_source_evidence_audit.py
"$PY" tools/real_motion/train_p0_f9_static_evidence_selector.py --help >/dev/null
"$PY" tools/real_motion/eval_p0_f9_static_evidence_selector.py --help >/dev/null

MANIFEST="$MANIFEST64"
EXTRA=()
if [[ "$MODE" == main ]]; then
  if [[ -z "${STATIC_SELECTOR_SCREEN_DIR:-}" || ! -f "$STATIC_SELECTOR_SCREEN_DIR/model/summary.json" ]]; then
    echo "main 必须设置 STATIC_SELECTOR_SCREEN_DIR=已通过门槛的 screen 运行目录；不会自动扩大。" >&2
    exit 2
  fi
  EXTRA+=(--screen-summary "$STATIC_SELECTOR_SCREEN_DIR/model/summary.json")
  # This is a NEW manifest output, not an assumed pre-existing server input.
  MANIFEST="$SELECTOR_ROOT/data/p0_f9_static_selector_dev512_manifest.json"
  COUNT=0
else
  COUNT=64
fi
if [[ ! -e "$MANIFEST" ]]; then
  if [[ ! -e "$DEV_STAGE1" ]]; then echo "[MISSING] $DEV_STAGE1 (冻结 manifest 的唯一父 population)" >&2; exit 2; fi
  "$PY" -u tools/real_motion/build_p0_f9_v21_dev_manifest.py \
    --stage1-cache "$DEV_STAGE1" --count "$COUNT" --output "$MANIFEST"
fi
if [[ -n "${STATIC_SELECTOR_RESUME:-}" ]]; then
  if [[ ! -f "$STATIC_SELECTOR_RESUME" ]]; then echo "[MISSING] $STATIC_SELECTOR_RESUME" >&2; exit 2; fi
  EXTRA+=(--resume "$STATIC_SELECTOR_RESUME")
fi
mkdir -p "$RUN_DIR"
echo "===== 一次完成：历史特征准备 → selector 训练 → 固定 dev 评估 → best/last checkpoint ====="
echo "模式：$MODE；输出：$RUN_DIR"
echo "首次准备按窗口打印进度；准备完成后训练不再反复读取 nuScenes / 重跑 V18。"
"$PY" -u tools/real_motion/train_p0_f9_static_evidence_selector.py \
  --config "$CONFIG" --train-cache "$TRAIN_V18" --dev-cache "$DEV_V18" \
  --population-manifest "$MANIFEST" --base-checkpoint "$BASE_CKPT" \
  --dataroot "$DATAROOT" --train-info "$TRAIN_INFO" --dev-info "$DEV_INFO" \
  --out-dir "$RUN_DIR/model" --mode "$MODE" --device cuda \
  --cpu-workers "$CPU_WORKERS" --batch-size 512 "${EXTRA[@]}" \
  2>&1 | tee "$RUN_DIR/run.log"
"$PY" tools/real_motion/summarize_p0_f9_static_evidence_selector.py \
  "$RUN_DIR/model/summary.json" | tee "$RUN_DIR/summary.txt"
echo "完成。请发回 $RUN_DIR/summary.txt（门槛未通过也会正常生成，不自动重训）。"
