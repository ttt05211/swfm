#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
"$PY" -c 'import sys; assert "OccFM" in sys.executable, "conda activate OccFM first"'
: "${GT_CKPT:?Set GT_CKPT to completed full GT epoch_0002.pt}"
PRIOR_DIAG_DIR="${PRIOR_DIAG_DIR:-$ROOT/outputs/p0_f9_shared_history/prior_strength_dev64_$(date +%Y%m%d_%H%M%S)}"
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
DATAROOT="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
args=(--config "$ROOT/configs/real_motion_occfm.yaml" --gt-checkpoint "$GT_CKPT" --out-dir "$PRIOR_DIAG_DIR"
  --dataroot "$DATAROOT" --dev-info "${DEV_INFO:-$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl}"
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}"
  --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}" --cpu-workers "${CPU_WORKERS:-3}")
if [[ -n "${STC_ZIP:-}" ]]; then args+=(--stc-archive "$STC_ZIP"); fi
if [[ "${PRIOR_DIAG_RESUME:-0}" == 1 ]]; then args+=(--resume); fi
echo "Frozen GTepoch2 / original DEV64 / c1 vs c0.5 / OCC+GT and STC+GT. NO training. Output: $PRIOR_DIAG_DIR"
exec "$PY" -u -m tools.prior_strength.run "${args[@]}" "$@"
