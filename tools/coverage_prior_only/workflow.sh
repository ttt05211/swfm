#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${PILOT_GPUS:-0,1}"
EVAL_GPU="${PILOT_EVAL_GPU:-${CUDA_VISIBLE_DEVICES%%,*}}"
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
PY="$(command -v python)"
"$PY" -c 'import sys,torch; assert "OccFM" in sys.executable; assert torch.cuda.device_count()==2'
DATA="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
JOINT="${JOINT_COVERAGE_DIR:-$ROOT/outputs/p0_f9_shared_history/coverage_projection_dual_fix_20261120_190540}"
if [[ "${PRIOR_ONLY_RESUME:-0}" == 1 && -z "${PRIOR_ONLY_DIR:-}" ]]; then echo 'Set original PRIOR_ONLY_DIR to resume'; exit 1; fi
OUT="${PRIOR_ONLY_DIR:-$ROOT/outputs/p0_f9_shared_history/coverage_prior_only_20pct_1ep_$(date +%Y%m%d_%H%M%S)}"
"$PY" -u -m tools.coverage_prior_only.control --joint-dir "$JOINT"
echo "PRIOR_ONLY_OUT=$OUT; RANDOM TRAIN4086x1/1031; GLOBAL B4; TWO training GPUs; projection=OFF; no trained initialization"
args=(--joint-dir "$JOINT" --out-dir "$OUT" --dataroot "$DATA"
  --train-info "$DATA/nuscenes_infos_train_temporal_v3_scene.pkl"
  --train-cache "${TRAIN_CACHE:-$ROOT/data/p0_f9_v18_se2_train_full.pt}")
if [[ "${PRIOR_ONLY_RESUME:-0}" == 1 ]]; then args+=(--resume); fi
"$PY" -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
  --module tools.coverage_prior_only.run "${args[@]}"
export CUDA_VISIBLE_DEVICES="$EVAL_GPU"
"$PY" -u -m tools.coverage_prior_only.evaluate --out-dir "$OUT" --dataroot "$DATA" \
  --dev-info "$DATA/nuscenes_infos_val_temporal_v3_scene.pkl" \
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}" \
  --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}"
"$PY" -u -m tools.coverage_prior_only.visuals --root "$OUT/visuals"
echo "DONE: $OUT/summary.txt ; $OUT/visuals/index.html ; $OUT/visuals_and_metrics.zip"
