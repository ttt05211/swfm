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

if [[ -z "${SOURCE:-}" ]]; then
  SOURCE="$(find "$ROOT/outputs/p0_f9_shared_history" -maxdepth 2 -type f \
    -path '*/shared_20pct_*ep_*/last.pt' -printf '%T@ %p\n' \
    | sort -nr | sed -n '1s/^[^ ]* //p')"
fi

test -n "$SOURCE" && test -f "$SOURCE" || {
  echo "Shared-History source checkpoint not found. Set SOURCE=/path/to/last.pt" >&2
  exit 1
}

OUT="${SPEED_OUT:-$ROOT/outputs/p0_f9_shared_history_speed/full6_exec_$(date +%Y%m%d_%H%M%S)}"
DATAROOT="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"

echo "SOURCE=$SOURCE"
echo "SPEED_OUT=$OUT"
echo "REMOVE remains enabled in renderer candidates."
echo "Comparing current_h2, sequential_h3/h6, packed_h2/h3/h6 and scene_packed_h2/h3/h6."
echo "All candidates keep full six-horizon CE+Lovasz, motion/ego supervision and the generic REMOVE path."

exec "$PY" -u -m tools.shared_history_experiments.speed.probe \
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
  --out-dir "$OUT" --resume "$SOURCE" --mode gt --training-backend fused \
  --epochs 2 --train-fraction .2 --window-batch-size 4 --source-budget 128 \
  --eval-population dev512 --probe-steps "${PROBE_STEPS:-8}" --passes "${PASSES:-2}" "$@"
