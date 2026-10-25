#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
: "${FORWARD_DIR:?Set FORWARD_DIR to the ORIGINAL failed workflow directory; do not create a new training run}"
export CUDA_VISIBLE_DEVICES="${FORWARD_EVAL_GPU:-0}"
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
PY="$(command -v python)"
"$PY" -c 'import sys,torch; assert "OccFM" in sys.executable; assert torch.cuda.device_count()==1'
if [[ "${RECOVERY_RESUME:-0}" == 1 && -z "${RECOVERY_DIR:-}" ]]; then echo 'Set original RECOVERY_DIR'; exit 1; fi
export RECOVERY_DIR="${RECOVERY_DIR:-$FORWARD_DIR/recovered_eval_$(date +%Y%m%d_%H%M%S)}"
DATA="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
args=(--workflow-dir "$FORWARD_DIR" --out-dir "$RECOVERY_DIR"
  --checkpoint "${FORWARD_RECOVERY_CKPT:-$FORWARD_DIR/C_train/epoch_0001.pt}"
  --config "$ROOT/configs/real_motion_occfm.yaml" --dataroot "$DATA"
  --dev-info "$DATA/nuscenes_infos_val_temporal_v3_scene.pkl"
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}"
  --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}")
if [[ "${RECOVERY_RESUME:-0}" == 1 ]]; then args+=(--resume); fi
echo "EVALUATION ONLY: original trained epoch1, no optimizer steps. NEW=$RECOVERY_DIR"
exec "$PY" -u -m tools.causal_forward_recovery.run "${args[@]}" "$@"
