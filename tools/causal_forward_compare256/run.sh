#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${FORWARD_COMPARE_GPU:-0}"
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
PY="$(command -v python)"
"$PY" -c 'import sys,torch; assert "OccFM" in sys.executable; assert torch.cuda.device_count()==1'
if [[ "${FORWARD_COMPARE_RESUME:-0}" == 1 && -z "${FORWARD_COMPARE_DIR:-}" ]]; then echo 'Set original FORWARD_COMPARE_DIR'; exit 1; fi
export FORWARD_COMPARE_DIR="${FORWARD_COMPARE_DIR:-$ROOT/outputs/p0_f9_shared_history/adaln_epoch2_mean3_other256_$(date +%Y%m%d_%H%M%S)}"
DATA="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
args=(--out-dir "$FORWARD_COMPARE_DIR" --config "$ROOT/configs/real_motion_occfm.yaml"
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}" --dataroot "$DATA"
  --dev-info "$DATA/nuscenes_infos_val_temporal_v3_scene.pkl" --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}")
if [[ -n "${FORWARD_MEAN_DIR:-}" ]]; then args+=(--mean-dir "$FORWARD_MEAN_DIR"); fi
if [[ -n "${FORWARD_FULL_DIR:-}" ]]; then args+=(--full-dir "$FORWARD_FULL_DIR"); fi
if [[ "${FORWARD_COMPARE_RESUME:-0}" == 1 ]]; then args+=(--resume); fi
echo "Existing epoch2 vs mean3 / fixed256 OTHER scenes / ONE GPU / FREE+age OFF; NEW=$FORWARD_COMPARE_DIR"
exec "$PY" -u -m tools.causal_forward_compare256.run "${args[@]}" "$@"
