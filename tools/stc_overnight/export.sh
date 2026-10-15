#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
"$PY" -c 'import sys; assert "OccFM" in sys.executable, "conda activate OccFM"'
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
DATAROOT="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
STC_BUNDLE_DIR="${STC_BUNDLE_DIR:-$ROOT/outputs/p0_f9_shared_history/stc_overnight_bundle_$(date +%Y%m%d_%H%M%S)}"
args=(--out-dir "$STC_BUNDLE_DIR" --config "$ROOT/configs/real_motion_occfm.yaml" --dataroot "$DATAROOT"
  --dev-info "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl"
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}"
  --train-cache "${TRAIN_CACHE:-$ROOT/data/p0_f9_v18_se2_train_full.pt}"
  --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}")
if [[ -n "${DENSE_DDP_DIR:-}" ]]; then args+=(--dense-dir "$DENSE_DDP_DIR"); fi
if [[ -n "${CLEAN1000_CKPT:-}" ]]; then args+=(--clean-checkpoint "$CLEAN1000_CKPT"); fi
echo "SMALL CPU-only bundle: SAME64 VAL +32 clean TRAIN; compact lossless labels; dense+clean1000 only; no raw sensors."
exec "$PY" -u -m tools.stc_overnight.export "${args[@]}" "$@"
