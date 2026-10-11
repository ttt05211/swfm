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
GT_RUN_DIR="${GT_RUN_DIR:-${RUN_DIR:-}}"
if [[ -z "$GT_RUN_DIR" || ! -f "$GT_RUN_DIR/last.pt" ]]; then
  echo 'Set GT_RUN_DIR to your EXISTING GT-only training directory (containing last.pt).' >&2
  exit 1
fi
GT_SPEED_OUT="${GT_SPEED_OUT:-$ROOT/outputs/p0_f9_joint_occ_ego/gt_speed_probe_$(date +%Y%m%d_%H%M%S)}"
DATAROOT="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
echo 'Read-only GT training speed A/B/C; SAME batch/source/loss/Adam/RNG. No published training updates.'
echo 'Pause competing GPU training with one Ctrl-C first. This script never stops another process.'
echo "Source: $GT_RUN_DIR/last.pt; NEW output: $GT_SPEED_OUT"
exec "$PY" -u -m tools.joint_occ_experiments.performance.probe \
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
  --out-dir "$GT_SPEED_OUT" --resume "$GT_RUN_DIR/last.pt" --mode gt \
  --probe-steps "${PROBE_STEPS:-8}" "$@"
