#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
"$PY" -c 'import sys; assert "OccFM" in sys.executable, "conda activate OccFM first"'
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
DATAROOT="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
KNOWN_CHECK_DIR="${KNOWN_CHECK_DIR:-$ROOT/outputs/p0_f9_shared_history/known_check6000_$(date +%Y%m%d_%H%M%S)}"
args=(--out-dir "$KNOWN_CHECK_DIR" --step "${KNOWN_CHECK_STEP:-6000}"
  --config "$ROOT/configs/real_motion_occfm.yaml" --dataroot "$DATAROOT"
  --dev-info "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl"
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}")
if [[ -n "${DENSE_DDP_DIR:-}" ]]; then args+=(--source-dir "$DENSE_DDP_DIR"); fi
exec "$PY" -u -m tools.dense_known_check.run "${args[@]}" "$@"
