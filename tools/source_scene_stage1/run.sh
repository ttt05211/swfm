#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
ARM="${1:?Usage: run.sh baseline|reader [extra fixed-protocol args]}"
shift
[[ "$ARM" == baseline || "$ARM" == reader ]] || { echo 'Unknown arm'; exit 1; }
export CUDA_VISIBLE_DEVICES="${SCENE_TRAIN_GPUS:-0}"
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
PY="$(command -v python)"
"$PY" -c 'import sys,torch; assert "OccFM" in sys.executable; assert torch.cuda.device_count() in (1,2), "Expose ONE or TWO training GPUs"'
DATA="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
REFERENCE="${REFERENCE_DIR:-$ROOT/outputs/p0_f9_shared_history/stc_empirical_refiner500_20261014_194954}"
OUT="${SCENE_ARM_DIR:?Set SCENE_ARM_DIR to a NEW isolated arm directory}"
args=(--arm "$ARM" --epochs "${SCENE_EPOCHS:-1}" --out-dir "$OUT" --config "$ROOT/configs/real_motion_occfm.yaml"
  --train-cache "${TRAIN_CACHE:-$ROOT/data/p0_f9_v18_se2_train_full.pt}"
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}"
  --population-manifest "${DEV_MANIFEST:-$ROOT/data/p0_f9_v21_dev64_manifest.json}"
  --base-checkpoint "${E14:-$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt}"
  --dataroot "$DATA" --train-info "$DATA/nuscenes_infos_train_temporal_v3_scene.pkl"
  --dev-info "$DATA/nuscenes_infos_val_temporal_v3_scene.pkl"
  --ccr-history-cache "${CCR_HISTORY_CACHE:-$ROOT/cache/p0_f9_ccr_history_geometry_v1}"
  --ccr-val-history-cache "${CCR_VAL_CACHE:-$ROOT/cache/p0_f9_ccr_history_geometry_val_v1}"
  --eval-reference "$REFERENCE/clean/results.json" --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}")
if [[ "${SCENE_RESUME:-0}" == 1 ]]; then args+=(--resume "$OUT/last.pt"); fi
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  exec "$PY" -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 --module tools.source_scene_stage1.run "${args[@]}" "$@"
else
  exec "$PY" -u -m tools.source_scene_stage1.run "${args[@]}" "$@"
fi
