#!/usr/bin/env bash
# A child process: never source this into the interactive shell.
set -euo pipefail
if [[ "${CONDA_DEFAULT_ENV:-}" != OccFM ]]; then echo "请先 conda activate OccFM。" >&2; exit 2; fi
MOTION_ROOT=/root/nas/occ/swfm
cd "$MOTION_ROOT"
PY="$(command -v python)"
BASE_CKPT="$MOTION_ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt"
DEV_V18="$MOTION_ROOT/data/p0_f9_v18_se2_val_all_4369.pt"
MANIFEST="$MOTION_ROOT/data/p0_f9_v21_dev64_manifest.json"
DATAROOT=/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes
DEV_INFO="$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl"
CONFIG="$MOTION_ROOT/configs/real_motion_occfm.yaml"
# This actual path comes from the user-supplied prior audit.json provenance.
REFERENCE="${V18_MOTION_REFERENCE_AUDIT:-$MOTION_ROOT/outputs/p0_f9_source_evidence_audit/dev64_20260930_122107_575150d/audit.json}"
RUN_DIR="${V18_MOTION_OUT:-$MOTION_ROOT/outputs/p0_f9_v18_motion_gap/dev64_$(date +%Y%m%d_%H%M%S)_$(git rev-parse --short HEAD)}"
for file in "$BASE_CKPT" "$DEV_V18" "$MANIFEST" "$DEV_INFO" "$CONFIG"; do
  if [[ ! -f "$file" ]]; then echo "[MISSING] $file" >&2; exit 2; fi
done
if [[ ! -d "$DATAROOT" ]]; then echo "[MISSING] $DATAROOT" >&2; exit 2; fi
if [[ -e "$RUN_DIR" ]]; then echo "拒绝覆盖：$RUN_DIR" >&2; exit 2; fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$MOTION_ROOT:$MOTION_ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -c 'import sys,torch; print("Python:",sys.executable,"Torch:",torch.__version__); assert torch.cuda.is_available(), "CUDA 不可用"; print("GPU:",torch.cuda.get_device_name(0))'
"$PY" -m pytest -q tests/test_v18_motion_gap.py tests/test_source_evidence_audit.py tests/test_p0_f9_v18_se2.py
EXTRA=()
if [[ -f "$REFERENCE" ]]; then EXTRA+=(--reference-audit "$REFERENCE"); else
  if [[ -n "${V18_MOTION_REFERENCE_AUDIT:-}" ]]; then echo "[MISSING] $REFERENCE" >&2; exit 2; fi
  echo "旧审计文件已不存在：本次仍做完整运动分解，但 reference_check 会明确标为未执行。"
fi
mkdir -p "$RUN_DIR"
echo "===== 一趟 dev64：位置 / yaw / 联合 / 原训练监督覆盖 ====="
echo "输出：$RUN_DIR；不训练、不产生新 checkpoint、不重建 cache。"
"$PY" -u tools/real_motion/eval_p0_f9_v18_motion_gap.py \
  --config "$CONFIG" --val-cache "$DEV_V18" --population-manifest "$MANIFEST" \
  --checkpoint "$BASE_CKPT" --dataroot "$DATAROOT" --info-pkl "$DEV_INFO" \
  --out-dir "$RUN_DIR"/result --device cuda --cpu-workers "${V18_MOTION_CPU_WORKERS:-8}" \
  "${EXTRA[@]}" 2>&1 | tee "$RUN_DIR/run.log"
"$PY" tools/real_motion/summarize_p0_f9_v18_motion_gap.py "$RUN_DIR/result/audit.json" | tee "$RUN_DIR/summary.txt"
echo "请发回 $RUN_DIR/summary.txt。不会自动开始训练。"
