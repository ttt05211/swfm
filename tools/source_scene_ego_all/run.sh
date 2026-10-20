#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${EGO_ALL_GPU:-0}"
PY="$(command -v python)"
"$PY" -c 'import sys,torch; assert "OccFM" in sys.executable; assert torch.cuda.device_count()==1, "ONE visible GPU required"'
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
DATAROOT="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
if [[ "${EGO_ALL_RESUME:-0}" == 1 && -z "${EGO_ALL_DIR:-}" ]]; then echo 'Set original EGO_ALL_DIR'; exit 1; fi
EGO_ALL_DIR="${EGO_ALL_DIR:-$ROOT/outputs/p0_f9_shared_history/scene_ego_query_epoch2_all_2proc_$(date +%Y%m%d_%H%M%S)}"
args=(--out-dir "$EGO_ALL_DIR" --config "$ROOT/configs/real_motion_occfm.yaml"
  --dataroot "$DATAROOT" --dev-info "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl"
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}"
  --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}" --cpu-workers "${EGO_ALL_CPU_WORKERS:-2}")
if [[ -n "${EGO_FULL_DIR:-}" ]]; then args+=(--source-dir "$EGO_FULL_DIR"); fi
if [[ "${EGO_ALL_RESUME:-0}" == 1 ]]; then args+=(--resume); fi
echo "ONE physical GPU / TWO inference processes; NEW ego-query/pure-memory/cross-first epoch2/10292; disjoint4369 shards; raw-count merge."
echo "EGO_ALL_DIR=$EGO_ALL_DIR; no training; no promise of 2x speed."
exec "$PY" -u -m tools.source_scene_ego_all.parallel "${args[@]}" "$@"
