#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${SCENE_TRAIN_GPUS:-0,1}"
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
PY="$(command -v python)"
"$PY" -c 'import sys,torch; assert "OccFM" in sys.executable; assert torch.cuda.device_count() in (1,2)'
PILOT="${SCENE_PILOT_DIR:-}"
if [[ -z "$PILOT" ]]; then
  candidates=()
  while IFS= read -r d; do
    [[ -f "$d/comparison.json" && -f "$d/reader/results.json" ]] && candidates+=("$d")
  done < <(find "$ROOT/outputs/p0_f9_shared_history" -maxdepth 1 -type d -name 'scene_reader_stage1_1ep_*' | sort)
  if [[ "${#candidates[@]}" != 1 ]]; then
    printf 'Set SCENE_PILOT_DIR to completed stage1 paired run; candidates:\n'
    printf '%s\n' "${candidates[@]}"; exit 1
  fi
  PILOT="${candidates[0]}"
fi
export SCENE_STAGE2_DIR="${SCENE_STAGE2_DIR:-$ROOT/outputs/p0_f9_shared_history/scene_stage2_1ep_$(date +%Y%m%d_%H%M%S)}"
DATA="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
REFERENCE="${REFERENCE_DIR:-$ROOT/outputs/p0_f9_shared_history/stc_empirical_refiner500_20261014_194954}"
args=(--out-dir "$SCENE_STAGE2_DIR" --reference-dir "$PILOT/reader" --epochs 1
  --config "$ROOT/configs/real_motion_occfm.yaml"
  --train-cache "${TRAIN_CACHE:-$ROOT/data/p0_f9_v18_se2_train_full.pt}"
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}"
  --population-manifest "${DEV_MANIFEST:-$ROOT/data/p0_f9_v21_dev64_manifest.json}"
  --base-checkpoint "${E14:-$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt}"
  --dataroot "$DATA" --train-info "$DATA/nuscenes_infos_train_temporal_v3_scene.pkl"
  --dev-info "$DATA/nuscenes_infos_val_temporal_v3_scene.pkl"
  --ccr-history-cache "${CCR_HISTORY_CACHE:-$ROOT/cache/p0_f9_ccr_history_geometry_v1}"
  --ccr-val-history-cache "${CCR_VAL_CACHE:-$ROOT/cache/p0_f9_ccr_history_geometry_val_v1}"
  --eval-reference "$REFERENCE/clean/results.json" --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}")
if [[ "${SCENE_STAGE2_RESUME:-0}" == 1 ]]; then args+=(--resume "$SCENE_STAGE2_DIR/last.pt"); fi
echo "Stage2 ONLY: RANDOM TRAIN20% x1; reference results ONLY=$PILOT/reader; NEW=$SCENE_STAGE2_DIR"
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  exec "$PY" -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 --module tools.source_scene_stage2.run "${args[@]}" "$@"
else
  exec "$PY" -u -m tools.source_scene_stage2.run "${args[@]}" "$@"
fi
