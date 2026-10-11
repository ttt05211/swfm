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
SOFT_CHECKPOINT="${SOFT_CHECKPOINT:-$ROOT/outputs/p0_f9_joint_occ_ego/gt3_horizon_float_resume_20261011_151144/last.pt}"
SOFT_OUT="${SOFT_OUT:-$ROOT/outputs/p0_f9_joint_occ_ego/soft_dev512_$(date +%Y%m%d_%H%M%S)}"
DATAROOT="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
extra=()
if [[ "${SOFT_RESUME:-0}" == 1 ]]; then extra+=(--resume-evaluation); fi
echo 'Frozen GT three-epoch pilot: exact DEV512 six readouts; then same 32 windows x2 soft/hard six-frame inference.'
echo 'No training/new bank; old history caches read only. Fresh Strong in timing; raw history/I/O excluded.'
echo "Source: $SOFT_CHECKPOINT; Output: $SOFT_OUT"
exec "$PY" -u -m tools.joint_occ_experiments.soft_inference.run \
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
  --out-dir "$SOFT_OUT" --resume "$SOFT_CHECKPOINT" \
  --mode gt --eval-population dev512 --speed-windows "${SOFT_SPEED_WINDOWS:-32}" \
  --speed-repeats "${SOFT_SPEED_REPEATS:-2}" "${extra[@]}" "$@"
