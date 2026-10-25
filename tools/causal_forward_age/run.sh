#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
: "${FORWARD_DIR:?Set ORIGINAL forward-training workflow directory}"
export CUDA_VISIBLE_DEVICES="${FORWARD_EVAL_GPU:-0}"
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
PY="$(command -v python)"
"$PY" -c 'import sys,torch; assert "OccFM" in sys.executable; assert torch.cuda.device_count()==1'
if [[ "${FORWARD_AGE_RESUME:-0}" == 1 && -z "${FORWARD_AGE_DIR:-}" ]]; then echo 'Set original FORWARD_AGE_DIR'; exit 1; fi
export FORWARD_AGE_DIR="${FORWARD_AGE_DIR:-$FORWARD_DIR/age_off_on_$(date +%Y%m%d_%H%M%S)}"
DATA="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
args=(--checkpoint "$FORWARD_DIR/C_train/epoch_0001.pt" --out-dir "$FORWARD_AGE_DIR"
  --config "$ROOT/configs/real_motion_occfm.yaml" --dataroot "$DATA"
  --dev-info "$DATA/nuscenes_infos_val_temporal_v3_scene.pkl"
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}"
  --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}")
if [[ "${FORWARD_AGE_RESUME:-0}" == 1 ]]; then args+=(--resume); fi
echo "Frozen C / SAME64 OCC+STC / age off-on / FREE bias OFF / no training; NEW=$FORWARD_AGE_DIR"
exec "$PY" -u -m tools.causal_forward_age.run "${args[@]}" "$@"
