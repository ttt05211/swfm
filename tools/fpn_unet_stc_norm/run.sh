#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${STC_NORM_GPU:-0}"
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
PY="$(command -v python)"
"$PY" -c 'import sys,torch; assert "OccFM" in sys.executable; assert torch.cuda.device_count()==1'
if [[ "${STC_NORM_RESUME:-0}" == 1 && -z "${STC_NORM_DIR:-}" ]]; then echo 'Set original STC_NORM_DIR'; exit 1; fi
export STC_NORM_DIR="${STC_NORM_DIR:-$ROOT/outputs/p0_f9_shared_history/fpn_stc_norm320_$(date +%Y%m%d_%H%M%S)}"
DATA="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
args=(--out-dir "$STC_NORM_DIR" --config "$ROOT/configs/real_motion_occfm.yaml"
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}"
  --dev-info "$DATA/nuscenes_infos_val_temporal_v3_scene.pkl" --dataroot "$DATA"
  --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}")
if [[ -n "${FPN_UNET_FULL_DIR:-}" ]]; then args+=(--source-dir "$FPN_UNET_FULL_DIR"); fi
if [[ "${STC_NORM_RESUME:-0}" == 1 ]]; then args+=(--resume); fi
echo "STC-only GN320 unlabeled adaptation, fixed200 steps; scene-disjoint fit32/held32. Original GT/defaults unchanged. Output: $STC_NORM_DIR"
exec "$PY" -u -m tools.fpn_unet_stc_norm.run "${args[@]}" "$@"
