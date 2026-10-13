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
if command -v pgrep >/dev/null 2>&1; then
  if pgrep -af 'python.*tools\.(shared_history_experiments\.full_gt\.(run|workflow)|scene_ego_head\.experiment|joint_occ_experiments\.run)' >/dev/null; then
    echo 'Competing training found. Pause it with ONE Ctrl-C, wait for saved checkpoint, then rerun.'
    echo 'This probe never stops another process.'; exit 1
  fi
fi
DATAROOT="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
GT_CKPT="${GT_CKPT:-AUTO_FULL_GT}"
SPEED_RUN_DIR="${SPEED_RUN_DIR:-$ROOT/outputs/p0_f9_shared_history/refiner_b4_slice_speed_$(date +%Y%m%d_%H%M%S)}"
echo 'READ ONLY: original / slice / packed B4 / packed B4+slice. Full6 CE+Lovasz; approximate execution test only.'
echo "Source: $GT_CKPT; NEW output: $SPEED_RUN_DIR"
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
  --out-dir "$SPEED_RUN_DIR" --resume "$GT_CKPT" --mode gt --epochs 2 --train-fraction 1
  --eval-population dev512 --probe-steps 8 --passes 2)
exec "$PY" -u -m tools.shared_history_experiments.refiner_batch_readout.probe "${args[@]}" "$@"
