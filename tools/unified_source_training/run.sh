#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
"$PY" -c 'import sys; assert "OccFM" in sys.executable, "conda activate OccFM first"'
ORIGINAL="${ORIGINAL_SWFM:-/root/nas/occ/swfm}"
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm:$ORIGINAL/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
DATAROOT="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
if [[ -n "${RESUME:-}" ]]; then echo 'Use TRACK_TRAIN_RESUME=1 and original TRACK_TRAIN_DIR'; exit 1; fi
if [[ "${TRACK_TRAIN_RESUME:-0}" == 1 && -z "${TRACK_TRAIN_DIR:-}" ]]; then echo 'Set original TRACK_TRAIN_DIR'; exit 1; fi
TRACK_TRAIN_DIR="${TRACK_TRAIN_DIR:-$ORIGINAL/outputs/p0_f9_shared_history/track_matched_20pct_1ep_$(date +%Y%m%d_%H%M%S)}"
args=(--config "$ROOT/configs/real_motion_occfm.yaml"
  --train-cache "${TRAIN_CACHE:-$ORIGINAL/data/p0_f9_v18_se2_train_full.pt}"
  --dev-cache "${DEV_CACHE:-$ORIGINAL/data/p0_f9_v18_se2_val_all_4369.pt}"
  --population-manifest "${DEV_MANIFEST:-$ORIGINAL/data/p0_f9_v21_dev64_manifest.json}"
  --base-checkpoint "${E14:-$ORIGINAL/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt}"
  --dataroot "$DATAROOT" --train-info "$DATAROOT/nuscenes_infos_train_temporal_v3_scene.pkl"
  --dev-info "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl"
  --ccr-history-cache "${CCR_HISTORY_CACHE:-$ORIGINAL/cache/p0_f9_ccr_history_geometry_v1}"
  --ccr-val-history-cache "${CCR_VAL_CACHE:-$ORIGINAL/cache/p0_f9_ccr_history_geometry_val_v1}"
  --stc-root "${STC_ROOT:-$ORIGINAL/data/stc_camera/compact}" --out-dir "$TRACK_TRAIN_DIR"
  --epochs 1 --train-fraction .2 --mode gt --eval-population dev64
  --cpu-workers "${CPU_WORKERS:-4}" --prefetch-workers 4
  --frame-cache-mib 1024 --geometry-ram-mib 512 --descriptor-ram-mib 256)
if [[ "${TRACK_TRAIN_RESUME:-0}" == 1 ]]; then args+=(--resume-workflow); fi
echo "Matched RANDOM baseline/unified_fit TRAIN4086 x1 EACH; full6 CE+Lovasz, no ego; output=$TRACK_TRAIN_DIR"
exec "$PY" -u -m tools.unified_source_training.workflow "${args[@]}" "$@"
