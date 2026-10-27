#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${FORWARD_MEAN_GPU:-0}"
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
PY="$(command -v python)"
"$PY" -c 'import sys,torch; assert "OccFM" in sys.executable; assert torch.cuda.device_count()==1'
if [[ "${FORWARD_MEAN_RESUME:-0}" == 1 && -z "${FORWARD_MEAN_DIR:-}" ]]; then echo 'Set original FORWARD_MEAN_DIR'; exit 1; fi
export FORWARD_MEAN_DIR="${FORWARD_MEAN_DIR:-$ROOT/outputs/p0_f9_shared_history/adaln_forward_mean3_$(date +%Y%m%d_%H%M%S)}"
DATA="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
args=(--out-dir "$FORWARD_MEAN_DIR" --config "$ROOT/configs/real_motion_occfm.yaml"
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}"
  --dataroot "$DATA" --dev-info "$DATA/nuscenes_infos_val_temporal_v3_scene.pkl"
  --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}" --cpu-workers 4 --device cuda)
if [[ -n "${FORWARD_FULL_DIR:-}" ]]; then args+=(--source-dir "$FORWARD_FULL_DIR"); fi
if [[ "${FORWARD_MEAN_RESUME:-0}" == 1 ]]; then args+=(--resume); fi
echo "Current forward AdaLN / mean(4000,10000,10292) / SAME64 / ONE GPU / FREE+age OFF; OUT=$FORWARD_MEAN_DIR"
exec "$PY" -u -m tools.causal_forward_average.run "${args[@]}" "$@"
