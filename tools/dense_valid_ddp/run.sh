#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
"$PY" -c 'import sys, torch; assert "OccFM" in sys.executable, "conda activate OccFM first"; assert torch.cuda.device_count() == 2, "Expose exactly TWO GPUs with CUDA_VISIBLE_DEVICES=0,1"'
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
DATAROOT="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
REFERENCE_DIR="${REFERENCE_DIR:-$ROOT/outputs/p0_f9_shared_history/stc_empirical_refiner500_20261014_194954}"
if [[ "${DENSE_DDP_RESUME:-0}" == 1 && -z "${DENSE_DDP_DIR:-}" ]]; then echo 'Set original DENSE_DDP_DIR to resume'; exit 1; fi
DENSE_DDP_DIR="${DENSE_DDP_DIR:-$ROOT/outputs/p0_f9_shared_history/dense_valid_shared_full2_ddp_$(date +%Y%m%d_%H%M%S)}"
args=(--config "$ROOT/configs/real_motion_occfm.yaml"
  --train-cache "${TRAIN_CACHE:-$ROOT/data/p0_f9_v18_se2_train_full.pt}"
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}"
  --population-manifest "${DEV_MANIFEST:-$ROOT/data/p0_f9_v21_dev64_manifest.json}"
  --base-checkpoint "${E14:-$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt}"
  --dataroot "$DATAROOT"
  --train-info "$DATAROOT/nuscenes_infos_train_temporal_v3_scene.pkl"
  --dev-info "$DATAROOT/nuscenes_infos_val_temporal_v3_scene.pkl"
  --ccr-history-cache "${CCR_HISTORY_CACHE:-$ROOT/cache/p0_f9_ccr_history_geometry_v1}"
  --ccr-val-history-cache "${CCR_VAL_CACHE:-$ROOT/cache/p0_f9_ccr_history_geometry_val_v1}"
  --eval-reference "$REFERENCE_DIR/clean/results.json"
  --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}" --out-dir "$DENSE_DDP_DIR"
  --source-policy "${SOURCE_POLICY:-shared_pinned}"
  --track-cache "${TRACK_CACHE:-$ROOT/cache/shared_source_pinned_v1}")
if [[ "${DENSE_DDP_RESUME:-0}" == 1 ]]; then args+=(--resume "$DENSE_DDP_DIR/last.pt"); fi
echo "RANDOM full TRAIN x2; dense-valid; GLOBAL B4/H6; two GPUs; eval remaining64 OCC/STC every2000 + final."
echo "DENSE_DDP_DIR=$DENSE_DDP_DIR; old checkpoints/caches read only; no internal ego head."
echo "SOURCE_POLICY=${SOURCE_POLICY:-shared_pinned}; original pair/KTA pinned; new track cache builds on demand."
exec "$PY" -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
  --module tools.dense_valid_ddp.run "${args[@]}" "$@"
