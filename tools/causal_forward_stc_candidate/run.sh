#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${STC_CANDIDATE_GPU:-0}"
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
PY="$(command -v python)"
"$PY" -c 'import sys,torch; assert "OccFM" in sys.executable; assert torch.cuda.device_count()==1'
if [[ "${STC_CANDIDATE_RESUME:-0}" == 1 && -z "${STC_CANDIDATE_DIR:-}" ]]; then echo 'Set original STC_CANDIDATE_DIR'; exit 1; fi
export STC_CANDIDATE_DIR="${STC_CANDIDATE_DIR:-$ROOT/outputs/p0_f9_shared_history/stc_state_range_candidate_$(date +%Y%m%d_%H%M%S)}"
DATA="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
SOURCE="${FORWARD_FULL_DIR:-$ROOT/outputs/p0_f9_shared_history/adaln_forward_full_2ep_20261026_215351}"
args=(--source-dir "$SOURCE" --out-dir "$STC_CANDIDATE_DIR" --config "$ROOT/configs/real_motion_occfm.yaml"
  --dataroot "$DATA" --dev-info "$DATA/nuscenes_infos_val_temporal_v3_scene.pkl"
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}"
  --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}" --cpu-workers "${STC_CANDIDATE_CPU_WORKERS:-2}")
if [[ "${STC_CANDIDATE_RESUME:-0}" == 1 ]]; then args+=(--resume); fi
echo "Fixed latest body + causal CV state + two fitted range coefficients / ALL4369 GT+STC / ONE GPU TWO processes; NEW=$STC_CANDIDATE_DIR"
exec "$PY" -u -m tools.causal_forward_stc_candidate.parallel "${args[@]}" "$@"
