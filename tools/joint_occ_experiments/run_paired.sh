#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
"$PY" -c 'import sys; assert "OccFM" in sys.executable, "conda activate OccFM first"'
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export SWFM_COLUMN_CPU_BACKEND=native
DATAROOT="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
MODE="${MODE:-paired}"
RUN_DIR="${RUN_DIR:-$ROOT/outputs/p0_f9_joint_occ_ego/${MODE}3_$(date +%Y%m%d_%H%M%S)}"
args=(
  --config "$ROOT/configs/real_motion_occfm.yaml"
  --train-cache "${TRAIN_CACHE:-$ROOT/data/p0_f9_v18_se2_train_full.pt}"
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}"
  --population-manifest "${DEV_MANIFEST:-$ROOT/data/p0_f9_v21_dev64_manifest.json}"
  --base-checkpoint "${E14:-$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt}"
  --dataroot "$DATAROOT"
  --train-info "$DATAROOT/nuscenes_infos_train_temporal_v3_scene.pkl"
  --dev-info "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl"
  --ccr-history-cache "${CCR_HISTORY_CACHE:-$ROOT/cache/p0_f9_ccr_history_geometry_v1}"
  --ccr-val-history-cache "${CCR_VAL_CACHE:-$ROOT/cache/p0_f9_ccr_history_geometry_val_v1}"
  --out-dir "$RUN_DIR" --mode "$MODE" --epochs 3 --train-fraction .2
  --window-batch-size 4 --source-budget 128 --eval-population "${EVAL_POPULATION:-dev512}"
)
if [[ -n "${RESUME:-}" ]]; then args+=(--resume "$RESUME"); fi
echo "Independent RANDOM mode=$MODE; full6 CE/Lovasz; reuse history caches READ ONLY; no new bank."
echo "Output: $RUN_DIR; Ctrl-C saves all active modes after a completed optimizer update."
exec "$PY" -u -m tools.joint_occ_experiments.run "${args[@]}" "$@"
