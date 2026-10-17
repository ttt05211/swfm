#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY="$(command -v python)"
"$PY" -c 'import sys; assert "OccFM" in sys.executable, "conda activate OccFM"'
DATA="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
OUT="${PAIR_AUDIT_DIR:-$ROOT/outputs/p0_f9_shared_history/source_pair_audit128_$(date +%Y%m%d_%H%M%S)}"
args=(--dataroot "$DATA" --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}"
  --config "$ROOT/configs/real_motion_occfm.yaml" --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}" --out "$OUT")
if [[ "${PAIR_AUDIT_RESUME:-0}" == 1 ]]; then args+=(--resume); fi
echo "CPU-only last pair identity/velocity/registration audit. OUT=$OUT; no training/GPU/future occupancy."
exec "$PY" -u -m tools.source_scene_stage1.audit "${args[@]}" "$@"
