#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
"$PY" -c 'import sys; assert "OccFM" in sys.executable, "conda activate OccFM first"'
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export SWFM_COLUMN_CPU_BACKEND=native
DATAROOT="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
EPOCHS="${EPOCHS:-2}"
if [[ -n "${RESUME:-}" && -z "${RUN_DIR:-}" ]]; then RUN_DIR="$(dirname "$RESUME")"; fi
RUN_DIR="${RUN_DIR:-$ROOT/outputs/p0_f9_shared_history/refiner_20pct_${EPOCHS}ep_$(date +%Y%m%d_%H%M%S)}"
args=(--config "$ROOT/configs/real_motion_occfm.yaml"
  --train-cache "${TRAIN_CACHE:-$ROOT/data/p0_f9_v18_se2_train_full.pt}"
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}"
  --population-manifest "${DEV_MANIFEST:-$ROOT/data/p0_f9_v21_dev64_manifest.json}"
  --base-checkpoint "${E14:-$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt}"
  --dataroot "$DATAROOT"
  --train-info "$DATAROOT/nuscenes_infos_train_temporal_v3_scene.pkl"
  --dev-info "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl"
  --ccr-history-cache "${CCR_HISTORY_CACHE:-$ROOT/cache/p0_f9_ccr_history_geometry_v1}"
  --ccr-val-history-cache "${CCR_VAL_CACHE:-$ROOT/cache/p0_f9_ccr_history_geometry_val_v1}"
  --out-dir "$RUN_DIR" --mode gt --epochs "$EPOCHS" --train-fraction .2 --eval-population dev512)
if [[ -n "${RESUME:-}" ]]; then args+=(--resume "$RESUME"); fi
echo "NEW semantic refiner; no old CCR/Adam import; TRAIN20% x$EPOCHS, GT, full6; no FPS claim."
echo "Output: $RUN_DIR; Ctrl-C saves complete update/window; final fixed DEV512."
MODULE="${REFINER_MODULE:-tools.shared_history_experiments.refine.run}"
exec "$PY" -u -m "$MODULE" "${args[@]}" "$@"
