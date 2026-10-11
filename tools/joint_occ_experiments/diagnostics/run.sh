#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
"$PY" -c 'import sys; assert "OccFM" in sys.executable, "conda activate OccFM first"'
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export SWFM_COLUMN_CPU_BACKEND=native
DIAG_OUT="${DIAG_OUT:-$ROOT/outputs/p0_f9_joint_occ_ego/ccr_loss_audit_$(date +%Y%m%d_%H%M%S)}"
DATAROOT="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
extra=()
if [[ -n "${DIAG_CHECKPOINT:-}" ]]; then extra+=(--resume "$DIAG_CHECKPOINT"); fi
if [[ "${DIAG_RESUME:-0}" == 1 ]]; then extra+=(--resume-audit); fi
echo 'Read-only TRAIN16 gradients + fixed DEV64 action/soft-hard diagnosis; no training or new cache.'
echo "Completed GT pilot auto-detected unless DIAG_CHECKPOINT is supplied. Output: $DIAG_OUT"
exec "$PY" -u -m tools.joint_occ_experiments.diagnostics.run \
  --config "$ROOT/configs/real_motion_occfm.yaml" \
  --train-cache "${TRAIN_CACHE:-$ROOT/data/p0_f9_v18_se2_train_full.pt}" \
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}" \
  --population-manifest "${DEV_MANIFEST:-$ROOT/data/p0_f9_v21_dev64_manifest.json}" \
  --base-checkpoint "${E14:-$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt}" \
  --dataroot "$DATAROOT" \
  --train-info "$DATAROOT/nuscenes_infos_train_temporal_v3_scene.pkl" \
  --dev-info "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl" \
  --ccr-history-cache "${CCR_HISTORY_CACHE:-$ROOT/cache/p0_f9_ccr_history_geometry_v1}" \
  --ccr-val-history-cache "${CCR_VAL_CACHE:-$ROOT/cache/p0_f9_ccr_history_geometry_val_v1}" \
  --out-dir "$DIAG_OUT" --checkpoint-root "$ROOT/outputs/p0_f9_joint_occ_ego" \
  --mode gt --eval-population dev64 --train-windows 16 "${extra[@]}" "$@"
