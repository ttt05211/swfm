#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
PY="$(command -v python)"
"$PY" -c 'import sys; assert "OccFM" in sys.executable'
RESUME="${FORWARD_RESUME:-0}"
if [[ "$RESUME" == 1 && -z "${FORWARD_DIR:-}" ]]; then echo 'Set original FORWARD_DIR'; exit 1; fi
export FORWARD_DIR="${FORWARD_DIR:-$ROOT/outputs/p0_f9_shared_history/adaln_forward_single_free20_$(date +%Y%m%d_%H%M%S)}"
if [[ "$RESUME" != 1 && -e "$FORWARD_DIR" ]]; then echo 'NEW FORWARD_DIR required; old results preserved'; exit 1; fi
if [[ "$RESUME" == 1 && ! -d "$FORWARD_DIR" ]]; then echo 'Original FORWARD_DIR does not exist'; exit 1; fi
CONTROL_ARGS=(--root "$ROOT/outputs/p0_f9_shared_history")
if [[ -n "${FORWARD_CONTROL_DIR:-}" ]]; then CONTROL_ARGS+=(--control-dir "$FORWARD_CONTROL_DIR"); fi
CONTROL="$("$PY" -m tools.causal_forward_train.control "${CONTROL_ARGS[@]}")"
mkdir -p "$FORWARD_DIR"
DATA="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
DEV_CACHE_PATH="${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}"
STC="${STC_ROOT:-$ROOT/data/stc_camera/compact}"
REFERENCE="${REFERENCE_DIR:-$ROOT/outputs/p0_f9_shared_history/stc_empirical_refiner500_20261014_194954}"
TRAIN_GPUS="${SCENE_TRAIN_GPUS:-0}"
EVAL_GPU="${FORWARD_EVAL_GPU:-${TRAIN_GPUS%%,*}}"
if [[ "$EVAL_GPU" == *,* || "$TRAIN_GPUS" == *,* ]]; then echo 'Training AND evaluation must each use ONE GPU; global batch remains4'; exit 1; fi
CHILD=''
trap 'if [[ -n "$CHILD" ]]; then kill -TERM "$CHILD" 2>/dev/null || true; wait "$CHILD" || true; fi; exit 130' INT TERM
run_child() { "$@" & CHILD=$!; wait "$CHILD"; CHILD=''; }
common=(--dataroot "$DATA" --dev-cache "$DEV_CACHE_PATH" --stc-root "$STC"
  --dev-info "$DATA/nuscenes_infos_val_temporal_v3_scene.pkl")
frozen=(--checkpoint "$CONTROL/epoch_0001.pt" --out-dir "$FORWARD_DIR/frozen_B" --horizon-chunk 6 --free-minus1)
audit=(--control-dir "$CONTROL" --out-dir "$FORWARD_DIR/pair_audit" --config "$ROOT/configs/real_motion_occfm.yaml")
if [[ "$RESUME" == 1 && -f "$FORWARD_DIR/frozen_B/results.json" ]]; then frozen+=(--resume); fi
if [[ "$RESUME" == 1 && -f "$FORWARD_DIR/pair_audit/results.json" ]]; then audit+=(--resume); fi
echo "AdaLN A/B/C / raw + fixed FREE-1 / SINGLE GPU; CONTROL=$CONTROL; NEW=$FORWARD_DIR; no full training."
echo 'Stage B: ONE GPU frozen original AdaLN / old vs forward histories / same64.'
run_child env CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PY" -u -m tools.causal_forward_track.run "${frozen[@]}" "${common[@]}"
echo 'Stage diagnostic: CPU / same64 / GT boxes AFTER causal matching.'
run_child env CUDA_VISIBLE_DEVICES='' "$PY" -u -m tools.causal_forward_train.audit "${audit[@]}" "${common[@]}"
args=(--out-dir "$FORWARD_DIR/C_train" --control-dir "$CONTROL" --epochs 1
  --config "$ROOT/configs/real_motion_occfm.yaml" --train-cache "${TRAIN_CACHE:-$ROOT/data/p0_f9_v18_se2_train_full.pt}"
  --dev-cache "$DEV_CACHE_PATH" --population-manifest "${DEV_MANIFEST:-$ROOT/data/p0_f9_v21_dev64_manifest.json}"
  --base-checkpoint "${E14:-$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt}"
  --dataroot "$DATA" --train-info "$DATA/nuscenes_infos_train_temporal_v3_scene.pkl"
  --dev-info "$DATA/nuscenes_infos_val_temporal_v3_scene.pkl"
  --ccr-history-cache "${CCR_HISTORY_CACHE:-$ROOT/cache/p0_f9_ccr_history_geometry_v1}"
  --ccr-val-history-cache "${CCR_VAL_CACHE:-$ROOT/cache/p0_f9_ccr_history_geometry_val_v1}"
  --eval-reference "$REFERENCE/clean/results.json" --stc-root "$STC")
if [[ "$RESUME" == 1 && -f "$FORWARD_DIR/C_train/last.pt" ]]; then args+=(--resume "$FORWARD_DIR/C_train/last.pt"); fi
export CUDA_VISIBLE_DEVICES="$TRAIN_GPUS"
"$PY" -c 'import torch; assert torch.cuda.device_count()==1, "Expose exactly ONE GPU"'
echo 'Stage C: RANDOM TRAIN20pct x1, same AdaLN initialization, forward TRAIN+eval.'
run_child "$PY" -u -m tools.causal_forward_train.run "${args[@]}"
run_child env CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PY" -u -m tools.causal_forward_train.report --control-dir "$CONTROL" --out-dir "$FORWARD_DIR"
