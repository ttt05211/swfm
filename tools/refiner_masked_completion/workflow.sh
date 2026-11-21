#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
DATA="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
CONTROL="${MASKED_REFINER_CONTROL:-$ROOT/outputs/p0_f9_shared_history/adaln_forward_single_free20_20261025_161644/C_train}"
OUT="${MASKED_REFINER_DIR:-$ROOT/outputs/p0_f9_shared_history/masked_refiner_20pct_1ep_$(date +%Y%m%d_%H%M%S)}"
TRAIN_GPUS="${PILOT_GPUS:-0,1}"
EVAL_GPU="${PILOT_EVAL_GPU:-${TRAIN_GPUS%%,*}}"
MODE="${1:-all}"
PY="$(command -v python)"

case "$MODE" in
  all|train|eval) ;;
  *) echo 'Usage: bash tools/refiner_masked_completion/workflow.sh [all|train|eval]' >&2; exit 2 ;;
esac
if [[ ( "$MODE" == eval || "${MASKED_REFINER_RESUME:-0}" == 1 ) && -z "${MASKED_REFINER_DIR:-}" ]]; then
  echo 'Set MASKED_REFINER_DIR to the existing run directory.' >&2
  exit 2
fi

echo "MASKED_REFINER_OUT=$OUT"
if [[ "$MODE" != eval ]]; then
  export CUDA_VISIBLE_DEVICES="$TRAIN_GPUS"
  "$PY" -c 'import torch; assert torch.cuda.device_count() == 2, "Two training GPUs required"'
  args=(--out-dir "$OUT" --control-dir "$CONTROL" --dataroot "$DATA"
    --train-info "$DATA/nuscenes_infos_train_temporal_v3_scene.pkl"
    --train-cache "${TRAIN_CACHE:-$ROOT/data/p0_f9_v18_se2_train_full.pt}")
  if [[ "${MASKED_REFINER_RESUME:-0}" == 1 ]]; then args+=(--resume); fi
  "$PY" -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
    --module tools.refiner_masked_completion.train "${args[@]}"
fi
if [[ "$MODE" != train ]]; then
  export CUDA_VISIBLE_DEVICES="$EVAL_GPU"
  "$PY" -u -m tools.refiner_masked_completion.evaluate --out-dir "$OUT" --dataroot "$DATA" \
    --dev-info "$DATA/nuscenes_infos_val_temporal_v3_scene.pkl" \
    --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}" \
    --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}"
  echo "SCORES=$OUT/summary.txt"
fi
