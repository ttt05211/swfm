#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${STC_ONLY_GPU:-0}"
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
PY="$(command -v python)"
"$PY" -c 'import sys,torch; assert "OccFM" in sys.executable; assert torch.cuda.device_count()==1'
if [[ "${STC_ONLY_RESUME:-0}" == 1 && -z "${STC_ONLY_DIR:-}" ]]; then echo 'Set original STC_ONLY_DIR'; exit 1; fi
export STC_ONLY_DIR="${STC_ONLY_DIR:-$ROOT/outputs/p0_f9_shared_history/occ_raw_stc_adapt_$(date +%Y%m%d_%H%M%S)}"
DATA="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
SOURCE="${FORWARD_FULL_DIR:-$ROOT/outputs/p0_f9_shared_history/adaln_forward_full_2ep_20261026_215351}"
args=(--source-dir "$SOURCE" --out-dir "$STC_ONLY_DIR" --config "$ROOT/configs/real_motion_occfm.yaml"
  --dataroot "$DATA" --dev-info "$DATA/nuscenes_infos_val_temporal_v3_scene.pkl"
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}"
  --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}" --cpu-workers "${STC_ONLY_CPU_WORKERS:-2}")
if [[ "${STC_ONLY_RESUME:-0}" == 1 ]]; then args+=(--resume); fi
echo "ONE body / OCC original / STC-only adaptation / ALL4369 / ONE GPU TWO processes; NEW=$STC_ONLY_DIR"
exec "$PY" -u -m tools.causal_forward_stc_only.parallel "${args[@]}" "$@"
