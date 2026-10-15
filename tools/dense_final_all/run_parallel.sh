#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${DENSE_ALL_GPU:-0}"
PY="$(command -v python)"
"$PY" -c 'import sys,torch; assert "OccFM" in sys.executable; assert torch.cuda.device_count()==1, "ONE visible GPU required"'
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
DATAROOT="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
if [[ "${DENSE_ALL_RESUME:-0}" == 1 && -z "${DENSE_ALL_DIR:-}" ]]; then echo 'Set original DENSE_ALL_DIR'; exit 1; fi
DENSE_ALL_DIR="${DENSE_ALL_DIR:-$ROOT/outputs/p0_f9_shared_history/dense_epoch2_all_singleGPU2proc_$(date +%Y%m%d_%H%M%S)}"
args=(--out-dir "$DENSE_ALL_DIR" --config "$ROOT/configs/real_motion_occfm.yaml"
  --dataroot "$DATAROOT" --dev-info "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl"
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}"
  --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}" --cpu-workers "${DENSE_ALL_CPU_WORKERS:-2}")
if [[ -n "${DENSE_DDP_DIR:-}" ]]; then args+=(--source-dir "$DENSE_DDP_DIR"); fi
if [[ "${DENSE_ALL_RESUME:-0}" == 1 ]]; then args+=(--resume); fi
echo "ONE physical GPU / TWO inference processes; fixed epoch2/10292; disjoint4369 shards; raw-count merge."
echo "DENSE_ALL_DIR=$DENSE_ALL_DIR; no training; no promise of 2x speed."
exec "$PY" -u -m tools.dense_final_all.parallel "${args[@]}" "$@"
