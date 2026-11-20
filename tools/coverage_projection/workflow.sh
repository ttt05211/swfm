#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${PILOT_GPU:-0}"
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
PY="$(command -v python)"
"$PY" -c 'import sys,torch; assert "OccFM" in sys.executable; assert torch.cuda.device_count()==1'
DATA="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
CONTROL="${COVERAGE_CONTROL_DIR:-$ROOT/outputs/p0_f9_shared_history/adaln_forward_single_free20_20261025_161644/C_train}"
if [[ "${COVERAGE_RESUME:-0}" == 1 && -z "${COVERAGE_DIR:-}" ]]; then echo 'Set original COVERAGE_DIR to resume'; exit 1; fi
OUT="${COVERAGE_DIR:-$ROOT/outputs/p0_f9_shared_history/coverage_projection_20pct_1ep_$(date +%Y%m%d_%H%M%S)}"
echo "COVERAGE_OUT=$OUT; RANDOM TRAIN4086x1; same current forward control; ONE GPU; no dense disk cache"
args=(--control-dir "$CONTROL" --out-dir "$OUT" --dataroot "$DATA"
  --train-info "$DATA/nuscenes_infos_train_temporal_v3_scene.pkl"
  --train-cache "${TRAIN_CACHE:-$ROOT/data/p0_f9_v18_se2_train_full.pt}")
if [[ "${COVERAGE_RESUME:-0}" == 1 ]]; then args+=(--resume); fi
"$PY" -u -m tools.coverage_projection.run "${args[@]}"
"$PY" -u -m tools.coverage_projection.evaluate --out-dir "$OUT" --dataroot "$DATA" \
  --dev-info "$DATA/nuscenes_infos_val_temporal_v3_scene.pkl" \
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}" \
  --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}"
"$PY" -u -m tools.coverage_projection.visuals --root "$OUT/visuals"
echo "DONE: $OUT/summary.txt ; $OUT/visuals/index.html ; $OUT/visuals_and_metrics.zip"
