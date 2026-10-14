#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
"$PY" -c 'import sys; assert "OccFM" in sys.executable, "conda activate OccFM first"'
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export SWFM_COLUMN_CPU_BACKEND=native
DATAROOT="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
if [[ -n "${RESUME:-}" ]]; then echo 'Use KNOWN_RESUME=1 and original RUN_DIR'; exit 1; fi
if [[ "${KNOWN_RESUME:-0}" == 1 && -z "${RUN_DIR:-}" ]]; then echo 'Set original RUN_DIR before resume'; exit 1; fi
RUN_DIR="${RUN_DIR:-$ROOT/outputs/p0_f9_shared_history/known_20pct_1ep_$(date +%Y%m%d_%H%M%S)}"
args=(--config "$ROOT/configs/real_motion_occfm.yaml"
  --train-cache "${TRAIN_CACHE:-$ROOT/data/p0_f9_v18_se2_train_full.pt}"
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}"
  --population-manifest "${DEV_MANIFEST:-$ROOT/data/p0_f9_v21_dev64_manifest.json}"
  --base-checkpoint "${E14:-$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt}"
  --dataroot "$DATAROOT"
  --train-info "$DATAROOT/nuscenes_infos_train_temporal_v3_scene.pkl"
  --dev-info "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl"
  --ccr-history-cache "${CCR_HISTORY_CACHE:-$ROOT/cache/p0_f9_ccr_history_geometry_v1}"
  --ccr-val-history-cache "${CCR_VAL_CACHE:-$ROOT/cache/p0_f9_ccr_history_geometry_val_v1}"
  --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}"
  --out-dir "$RUN_DIR" --epochs 1 --train-fraction .2 --mode gt --eval-population dev64
  --cpu-workers "${CPU_WORKERS:-4}" --prefetch-workers 4
  --frame-cache-mib 1024 --geometry-ram-mib 512 --descriptor-ram-mib 256)
if [[ "${KNOWN_RESUME:-0}" == 1 ]]; then args+=(--resume-workflow); fi
echo 'Matched RANDOM original-known/dense-valid, TRAIN20% x1 EACH. H6 CE+Lovasz ONLY; no ego head.'
echo 'No augmentation or learned prior gate; geometry caches read-only; final same DEV64 OCC+GT/STC+GT.'
echo "RUN_DIR=$RUN_DIR; Ctrl-C saves complete update/window."
exec "$PY" -u -m tools.history_validity.workflow "${args[@]}" "$@"
