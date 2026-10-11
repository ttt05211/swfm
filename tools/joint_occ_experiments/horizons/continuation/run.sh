#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
"$PY" -c 'import sys; assert "OccFM" in sys.executable, "conda activate OccFM first"'
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export SWFM_COLUMN_CPU_BACKEND=native
: "${HORIZON_OUT:?Set HORIZON_OUT to the NEW continuation directory, or SAME directory when resuming}"
GT_RUN_DIR="${GT_RUN_DIR:-$ROOT/outputs/p0_f9_joint_occ_ego/gt3_fast_20261011_123726}"
SOURCE="${RESUME:-$GT_RUN_DIR/last.pt}"
test -f "$SOURCE" || { echo "Checkpoint missing: $SOURCE" >&2; exit 1; }
extra=(--accept-numerical-drift)
if [[ -n "${HORIZON_PROOF:-}" ]]; then extra+=(--horizon-proof "$HORIZON_PROOF"); fi
DATAROOT="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
echo "Continue GT to ORIGINAL three epochs; output=$HORIZON_OUT; source=$SOURCE"
echo 'Measured floating-point execution change, NOT lossless; no new probe or cache build.'
exec "$PY" -u -m tools.joint_occ_experiments.horizons.continuation.run \
  --config "$ROOT/configs/real_motion_occfm.yaml" \
  --train-cache "${TRAIN_CACHE:-$ROOT/data/p0_f9_v18_se2_train_full.pt}" \
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}" \
  --population-manifest "${DEV_MANIFEST:-$ROOT/data/p0_f9_v21_dev64_manifest.json}" \
  --base-checkpoint "${E14:-$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt}" \
  --dataroot "$DATAROOT" \
  --train-info "$DATAROOT/nuscenes_infos_train_temporal_v3_scene.pkl" \
  --dev-info "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl" \
  --ccr-history-cache "${CCR_HISTORY_CACHE:-$ROOT/cache/p0_f9_ccr_history_geometry_v1}" \
  --ccr-val-history-cache "${CCR_VAL_CACHE:-$ROOT/cache/p0_f9_ccr_history_geometry_val_v1}" \
  --out-dir "$HORIZON_OUT" --resume "$SOURCE" --mode gt --training-backend fused \
  --epochs 3 --train-fraction .2 --window-batch-size 4 --source-budget 128 \
  --eval-population dev512 --max-updates 0 "${extra[@]}" "$@"
