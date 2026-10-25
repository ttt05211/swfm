#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${TRACK_GPU:-0}"
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
PY="$(command -v python)"
"$PY" -c 'import sys,torch; assert "OccFM" in sys.executable; assert torch.cuda.device_count()==1'
: "${TRACK_CKPT:?Set TRACK_CKPT to the explicit completed current FPN or AdaLN checkpoint}"
if [[ "${TRACK_RESUME:-0}" == 1 && -z "${TRACK_DIR:-}" ]]; then echo 'Set original TRACK_DIR'; exit 1; fi
export TRACK_DIR="${TRACK_DIR:-$ROOT/outputs/p0_f9_shared_history/forward_track64_$(date +%Y%m%d_%H%M%S)}"
DATA="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
args=(--checkpoint "$TRACK_CKPT" --out-dir "$TRACK_DIR" --dataroot "$DATA"
  --dev-info "$DATA/nuscenes_infos_val_temporal_v3_scene.pkl"
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}"
  --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}" --horizon-chunk 6)
if [[ "${TRACK_RESUME:-0}" == 1 ]]; then args+=(--resume); fi
echo "Frozen SAME weights / original64 / OCC+STC / paired source association; no optimizer. Output: $TRACK_DIR"
exec "$PY" -u -m tools.causal_forward_track.run "${args[@]}" "$@"
