#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
RUNS="$ROOT/outputs/p0_f9_shared_history"
DATA="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
GPUS="${ABLATION_EVAL_GPUS:-0,1}"
OUT="${ABLATION_EVAL_DIR:-$RUNS/ablation_all_parallel_$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
PY="$(command -v python)"
exec "$PY" -u -m tools.paper_ablation_eval.workflow \
  --gpus "$GPUS" --out-dir "$OUT" --cpu-workers "${ABLATION_EVAL_CPU_WORKERS:-2}" \
  --train-dir "${ABLATION_TRAIN_DIR:-$RUNS/ablation_five_dual_20261107_203511}" \
  --epoch2 "${FORWARD_FULL_DIR:-$RUNS/adaln_forward_full_2ep_20261026_215351}/epoch_0002.pt" \
  --config "$ROOT/configs/real_motion_occfm.yaml" --dataroot "$DATA" \
  --dev-info "$DATA/nuscenes_infos_val_temporal_v3_scene.pkl" \
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}" \
  --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}" \
  --plan-cache "${PLAN_CACHE:-/root/nas/occ/new_code/cache/come_main_table/camera_pred}" "$@"
