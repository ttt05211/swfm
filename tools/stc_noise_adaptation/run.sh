#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
"$PY" -c 'import sys; assert "OccFM" in sys.executable, "conda activate OccFM first"'
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
DATAROOT="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
GT_CKPT="${GT_CKPT:-$ROOT/outputs/p0_f9_shared_history/full_gt_ce_lovasz_2ep_20261013_123513/train/epoch_0002.pt}"
if [[ -z "${NOISE_MODEL:-}" ]]; then
  NOISE_MODEL="$("$PY" -c 'from pathlib import Path; from tools.stc_noise_adaptation.augmentation import load_model
paths=sorted(Path("outputs/p0_f9_shared_history").glob("stc_noise_fit_*/noise_model.json"))
assert paths, "Set NOISE_MODEL to the completed real-error profile noise_model.json"
p=paths[-1]; load_model(p); print(p.resolve())')"
fi
if [[ "${CALIB_RESUME:-0}" == 1 && -z "${CALIB_RUN_DIR:-}" ]]; then echo 'Set original CALIB_RUN_DIR to resume'; exit 1; fi
CALIB_RUN_DIR="${CALIB_RUN_DIR:-$ROOT/outputs/p0_f9_shared_history/stc_empirical_refiner500_$(date +%Y%m%d_%H%M%S)}"
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
  --gt-checkpoint "$GT_CKPT" --noise-model "$NOISE_MODEL"
  --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}" --out-dir "$CALIB_RUN_DIR"
  --cpu-workers "${CPU_WORKERS:-4}" --prefetch-workers 4
  --frame-cache-mib 1024 --geometry-ram-mib 512 --descriptor-ram-mib 256)
if [[ "${CALIB_RESUME:-0}" == 1 ]]; then args+=(--resume-workflow); fi
echo 'Two matched GTepoch2 Refiner-only 500-step arms: clean / empirical-noise. No precheck training run.'
echo 'Both TRAIN dense-valid; old caches read-only; corrupted histories rebuilt. Fixed remaining64, NOT old DEV64/512.'
echo "GT_CKPT=$GT_CKPT; NOISE_MODEL=$NOISE_MODEL; CALIB_RUN_DIR=$CALIB_RUN_DIR"
exec "$PY" -u -m tools.stc_noise_adaptation.workflow "${args[@]}" "$@"
