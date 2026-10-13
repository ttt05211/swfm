#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
"$PY" -c 'import sys; assert "OccFM" in sys.executable, "conda activate OccFM first"'
: "${GT_CKPT:?Set GT_CKPT to FINISHED full-GT train/epoch_0002.pt}"
: "${EGO_RUN_DIR:?Set EGO_RUN_DIR to a NEW stage directory}"
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
DATAROOT="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
args=(--config "$ROOT/configs/real_motion_occfm.yaml"
  --train-cache "${TRAIN_CACHE:-$ROOT/data/p0_f9_v18_se2_train_full.pt}"
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}"
  --population-manifest "${DEV_MANIFEST:-$ROOT/data/p0_f9_v21_dev64_manifest.json}"
  --base-checkpoint "${E14:-$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt}"
  --dataroot "$DATAROOT" --train-info "$DATAROOT/nuscenes_infos_train_temporal_v3_scene.pkl"
  --dev-info "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl"
  --ccr-history-cache "${CCR_HISTORY_CACHE:-$ROOT/cache/p0_f9_ccr_history_geometry_v1}"
  --ccr-val-history-cache "${CCR_VAL_CACHE:-$ROOT/cache/p0_f9_ccr_history_geometry_val_v1}"
  --out-dir "$EGO_RUN_DIR" --resume "$GT_CKPT" --mode gt --epochs 2 --train-fraction 1 --eval-population dev512
  --head-epochs "${HEAD_EPOCHS:-3}" --head-batch 64 --head-seed 21 --head-lr 0.0003
  --cpu-workers 4 --prefetch-workers 2 --frame-cache-mib 256 --geometry-ram-mib 128 --descriptor-ram-mib 128)
if [[ "${FROZEN_EGO_RESUME:-0}" == 1 ]]; then args+=(--resume-stage); fi
echo 'WAIT UNTIL GT epoch2 finishes. Freeze all body weights; TRAIN20430 head-only -> final INTERNAL DEV512.'
echo 'New compact HISTORY token bank belongs to this frozen body only; no old bank/cache overwrite.'
exec "$PY" -u -m tools.scene_ego_head.frozen_gt.run "${args[@]}" "$@"
