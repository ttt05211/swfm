#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${BASELINE_MOTION_GPU:-0}"
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
PY="$(command -v python)"
"$PY" -c 'import sys,torch; assert "OccFM" in sys.executable; assert torch.cuda.device_count()==1'
DATA="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
if [[ "${BASELINE_MOTION_RESUME:-0}" == 1 && -z "${BASELINE_MOTION_DIR:-}" ]]; then echo 'Set original BASELINE_MOTION_DIR'; exit 1; fi
export BASELINE_MOTION_DIR="${BASELINE_MOTION_DIR:-$ROOT/outputs/p0_f9_shared_history/baseline_motion_audit_$(date +%Y%m%d_%H%M%S)}"
args=(--out "$BASELINE_MOTION_DIR"
  --stage2-report "${STAGE2_AUDIT_DIR:-$ROOT/outputs/p0_f9_shared_history/stage2_motion_audit_20261018_144119}"
  --config "$ROOT/configs/real_motion_occfm.yaml" --dataroot "$DATA"
  --dev-info "$DATA/nuscenes_infos_val_temporal_v3_scene.pkl"
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}"
  --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}")
if [[ -n "${BASELINE_MOTION_CKPT:-}" ]]; then args+=(--checkpoint "$BASELINE_MOTION_CKPT");
elif [[ -n "${DENSE_DDP_DIR:-}" ]]; then args+=(--source-dir "$DENSE_DDP_DIR"); fi
if [[ "${BASELINE_MOTION_RESUME:-0}" == 1 ]]; then args+=(--resume); fi
echo "Original dense/shared-pinned epoch2 / SAME64 / ONE GPU / no optimizer. OUT=$BASELINE_MOTION_DIR"
exec "$PY" -u -m tools.source_motion_baseline_audit.run "${args[@]}" "$@"
