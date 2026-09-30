#!/usr/bin/env bash
# Run as a child bash process, NOT by sourcing into an interactive terminal.
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "OccFM" ]]; then
  echo "请先 conda activate OccFM，再用 bash 运行这个脚本。" >&2
  exit 2
fi
AUDIT_ROOT=/root/nas/occ/swfm
cd "$AUDIT_ROOT"
PY="$(command -v python)"
BASE_CKPT="$AUDIT_ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt"
DEV_V18="$AUDIT_ROOT/data/p0_f9_v18_se2_val_all_4369.pt"
DEV_STAGE1="$AUDIT_ROOT/data/p0_f9_v20_stage1_v2_dev512_v19split_769edc52"
MANIFEST="$AUDIT_ROOT/data/p0_f9_v21_dev64_manifest.json"
DATAROOT=/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes
DEV_INFO="$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl"
CONFIG="$AUDIT_ROOT/configs/real_motion_occfm.yaml"
CPU_WORKERS="${SOURCE_EVIDENCE_CPU_WORKERS:-8}"
RUN_DIR="${SOURCE_EVIDENCE_OUT:-$AUDIT_ROOT/outputs/p0_f9_source_evidence_audit/dev64_$(date +%Y%m%d_%H%M%S)_$(git rev-parse --short HEAD)}"

for input in "$BASE_CKPT" "$DEV_V18" "$DEV_INFO" "$CONFIG"; do
  if [[ ! -f "$input" ]]; then
    echo "[MISSING] $input" >&2
    exit 2
  fi
done
if [[ ! -d "$DATAROOT" ]]; then
  echo "[MISSING] $DATAROOT" >&2
  exit 2
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "输出目录已存在，拒绝覆盖：$RUN_DIR" >&2
  exit 2
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "===== 环境和代码预检查 ====="
"$PY" -c 'import sys,torch; print("Python:",sys.executable,"Torch:",torch.__version__); assert torch.cuda.is_available(), "OccFM 中 CUDA 不可用"; print("GPU:",torch.cuda.get_device_name(0))'
"$PY" -m pytest -q tests/test_source_evidence_audit.py
"$PY" tools/real_motion/eval_p0_f9_source_evidence_audit.py --help >/dev/null

if [[ ! -e "$MANIFEST" ]]; then
  if [[ ! -e "$DEV_STAGE1" ]]; then
    echo "缺少冻结 manifest，且找不到用于构建它的 $DEV_STAGE1" >&2
    exit 2
  fi
  echo "===== 仅在缺失时冻结 dev64 manifest ====="
  "$PY" -u tools/real_motion/build_p0_f9_v21_dev_manifest.py \
    --stage1-cache "$DEV_STAGE1" --count 64 --output "$MANIFEST"
fi
# Full fingerprint/order validation also runs in the evaluator. Never replace
# an existing population, silently take the first 64 records, or use full4369.
"$PY" -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x["selected_num_windows"] == 64, "需要冻结的 dev64 manifest"' "$MANIFEST"
CHECKPOINT_SHA="$(sha256sum "$BASE_CKPT")"
CHECKPOINT_SHA="${CHECKPOINT_SHA%% *}"
mkdir -p "$RUN_DIR"
echo "===== 一趟 dev64：运动 / 历史形状 / 因果配准 / 全类别静态记忆 ====="
echo "输入配置：$CONFIG"
echo "输出目录：$RUN_DIR"
"$PY" -u tools/real_motion/eval_p0_f9_source_evidence_audit.py \
  --config "$CONFIG" --val-cache "$DEV_V18" \
  --population-manifest "$MANIFEST" --checkpoint "$BASE_CKPT" \
  --expected-checkpoint-sha256 "$CHECKPOINT_SHA" \
  --dataroot "$DATAROOT" --info-pkl "$DEV_INFO" \
  --device cuda --cpu-workers "$CPU_WORKERS" --exactness-windows 1 \
  --static-patch-cells 4 --output "$RUN_DIR/audit.json" \
  2>&1 | tee "$RUN_DIR/run.log"
"$PY" tools/real_motion/summarize_p0_f9_source_evidence_audit.py \
  "$RUN_DIR/audit.json" | tee "$RUN_DIR/summary.txt"
echo "完成。请发回 $RUN_DIR/summary.txt"
