#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${FPN_UNET_ALL_GPU:-0}"
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
PY="$(command -v python)"
"$PY" -c 'import sys,torch; assert "OccFM" in sys.executable; assert torch.cuda.device_count()==1'
if [[ "${FPN_UNET_ALL_RESUME:-0}" == 1 && -z "${FPN_UNET_ALL_DIR:-}" ]]; then echo 'Set original FPN_UNET_ALL_DIR'; exit 1; fi
export FPN_UNET_ALL_DIR="${FPN_UNET_ALL_DIR:-$ROOT/outputs/p0_f9_shared_history/fpn_unet_all4369_$(date +%Y%m%d_%H%M%S)}"
DATA="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
args=(--out-dir "$FPN_UNET_ALL_DIR" --config "$ROOT/configs/real_motion_occfm.yaml"
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}"
  --dev-info "$DATA/nuscenes_infos_val_temporal_v3_scene.pkl" --dataroot "$DATA"
  --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}")
if [[ -n "${FPN_UNET_FULL_DIR:-}" ]]; then args+=(--source-dir "$FPN_UNET_FULL_DIR"); fi
if [[ "${FPN_UNET_ALL_RESUME:-0}" == 1 ]]; then args+=(--resume); fi
echo "Frozen FULL TRAINx2 on ALL4369; OCC original, STC fixed age0.5 memory; NO norm320/training/FPS. Output: $FPN_UNET_ALL_DIR"
exec "$PY" -u -m tools.fpn_unet_all.run "${args[@]}" "$@"
