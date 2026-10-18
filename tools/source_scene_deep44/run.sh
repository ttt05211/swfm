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
CONTROL="$("$PY" -c 'import sys; from tools.source_scene_deep44.report import select_reference; print(select_reference(sys.argv[1],sys.argv[2] or None))' "$ROOT/outputs/p0_f9_shared_history" "${INTEGRATED_DIR:-}")"
if [[ "${DEEP44_RESUME:-0}" == 1 && -z "${DEEP44_DIR:-}" ]]; then echo 'Set original DEEP44_DIR'; exit 1; fi
export DEEP44_DIR="${DEEP44_DIR:-$ROOT/outputs/p0_f9_shared_history/scene_deep44_1ep_$(date +%Y%m%d_%H%M%S)}"
DATA="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
REFERENCE="${REFERENCE_DIR:-$ROOT/outputs/p0_f9_shared_history/stc_empirical_refiner500_20261014_194954}"
args=(--out-dir "$DEEP44_DIR" --reference-dir "$CONTROL" --epochs 1
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
if [[ "${DEEP44_RESUME:-0}" == 1 ]]; then args+=(--resume "$DEEP44_DIR/last.pt"); fi
echo "FOUR encoder + FOUR decoder / RANDOM TRAIN4086 x1 / completed TWO+TWO control reused; SAME AdaGN; no trained weights imported / control=$CONTROL / NEW=$DEEP44_DIR"
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  exec "$PY" -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 --module tools.source_scene_deep44.run "${args[@]}" "$@"
else
  exec "$PY" -u -m tools.source_scene_deep44.run "${args[@]}" "$@"
fi
