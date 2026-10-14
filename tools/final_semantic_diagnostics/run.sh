#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
"$PY" -c 'import sys; assert "OccFM" in sys.executable, "conda activate OccFM first"'
: "${GT_CKPT:?Set GT_CKPT to completed full GT epoch_0002.pt}"
: "${DIAG_DIR:?Set DIAG_DIR to a NEW diagnostic directory}"
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
DATAROOT="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
args=(--config "$ROOT/configs/real_motion_occfm.yaml" --gt-checkpoint "$GT_CKPT" --out-dir "$DIAG_DIR"
  --dataroot "$DATAROOT" --dev-info "${DEV_INFO:-$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl}"
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}"
  --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}" --cpu-workers "${CPU_WORKERS:-3}")
if [[ -n "${HEAD10_CKPT:-}" ]]; then args+=(--head10-checkpoint "$HEAD10_CKPT"); fi
if [[ -n "${STC_ZIP:-}" ]]; then
  [[ -f "$STC_ZIP" ]] || { echo 'Explicit STC_ZIP missing; not silently skipping archive audit' >&2; exit 2; }
  args+=(--stc-archive "$STC_ZIP")
elif [[ -f "$ROOT/data/stc_camera/stc-results.zip" ]]; then
  args+=(--stc-archive "$ROOT/data/stc_camera/stc-results.zip")
else
  echo 'Original STC ZIP not present: only extraction receipt/history bytes can be verified.'
fi
if [[ "${STC_DIAG_RESUME:-0}" == 1 ]]; then args+=(--resume); fi
echo 'Fixed64: six diagnostic routes; no training/epsilon changes/axis selection or automatic promotion.'
exec "$PY" -u -m tools.final_semantic_diagnostics.run "${args[@]}" "$@"
