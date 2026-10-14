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
GT_CKPT="${GT_CKPT:-/root/nas/occ/swfm/outputs/p0_f9_shared_history/full_gt_ce_lovasz_2ep_20261013_123513/train/epoch_0002.pt}"
if [[ "${TRACK_RESUME:-0}" == 1 && -z "${TRACK_DIR:-}" ]]; then echo 'Set original TRACK_DIR before resuming'; exit 1; fi
TRACK_DIR="${TRACK_DIR:-/root/nas/occ/swfm/outputs/p0_f9_shared_history/unified_tracks_dev64_$(date +%Y%m%d_%H%M%S)}"
DATAROOT="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
args=(--gt-checkpoint "$GT_CKPT" --out-dir "$TRACK_DIR" --dataroot "$DATAROOT"
  --dev-info "${DEV_INFO:-$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl}"
  --dev-cache "${DEV_CACHE:-/root/nas/occ/swfm/data/p0_f9_v18_se2_val_all_4369.pt}"
  --stc-root "${STC_ROOT:-/root/nas/occ/swfm/data/stc_camera/compact}"
  --config "$ROOT/configs/real_motion_occfm.yaml" --inputs "${TRACK_INPUTS:-both}"
  --cpu-workers "${CPU_WORKERS:-4}")
if [[ "${TRACK_RESUME:-0}" == 1 ]]; then args+=(--resume); fi
echo "Independent frozen track/KTA DEV64: $TRACK_DIR; old geometry caches READ ONLY; no training"
exec "$PY" -u -m tools.unified_source_tracks.run "${args[@]}" "$@"
