#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${FORWARD_FULL_GPUS:-0,1}"
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
PY="$(command -v python)"
"$PY" -c 'import sys,torch,os; assert "OccFM" in sys.executable; n=len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")); assert n in (1,2) and torch.cuda.device_count()==n'
if [[ "${FORWARD_FULL_RESUME:-0}" == 1 && -z "${FORWARD_FULL_DIR:-}" ]]; then echo 'Set original FORWARD_FULL_DIR'; exit 1; fi
export FORWARD_FULL_DIR="${FORWARD_FULL_DIR:-$ROOT/outputs/p0_f9_shared_history/adaln_forward_full_2ep_$(date +%Y%m%d_%H%M%S)}"
PILOT="${FORWARD_PILOT_CONTRACT:-$ROOT/outputs/p0_f9_shared_history/adaln_forward_single_free20_20261025_161644/C_train/contract.json}"
DATA="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
REFERENCE="${REFERENCE_DIR:-$ROOT/outputs/p0_f9_shared_history/stc_empirical_refiner500_20261014_194954}"
args=(--out-dir "$FORWARD_FULL_DIR" --pilot-contract "$PILOT" --epochs 2 --train-fraction 1 --checkpoint-every 2000
  --config "$ROOT/configs/real_motion_occfm.yaml" --train-cache "${TRAIN_CACHE:-$ROOT/data/p0_f9_v18_se2_train_full.pt}"
  --dev-cache "${DEV_CACHE:-$ROOT/data/p0_f9_v18_se2_val_all_4369.pt}"
  --population-manifest "${DEV_MANIFEST:-$ROOT/data/p0_f9_v21_dev64_manifest.json}"
  --base-checkpoint "${E14:-$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt}"
  --dataroot "$DATA" --train-info "$DATA/nuscenes_infos_train_temporal_v3_scene.pkl"
  --dev-info "$DATA/nuscenes_infos_val_temporal_v3_scene.pkl"
  --ccr-history-cache "${CCR_HISTORY_CACHE:-$ROOT/cache/p0_f9_ccr_history_geometry_v1}"
  --ccr-val-history-cache "${CCR_VAL_CACHE:-$ROOT/cache/p0_f9_ccr_history_geometry_val_v1}"
  --eval-reference "$REFERENCE/clean/results.json" --stc-root "${STC_ROOT:-$ROOT/data/stc_camera/compact}")
if [[ "${FORWARD_FULL_RESUME:-0}" == 1 ]]; then args+=(--resume "$FORWARD_FULL_DIR/last.pt"); fi
echo "RANDOM FULL TRAIN20430x2 / forward AdaLN / GLOBAL B4 / one-GPU eval / FREE+age OFF / every2000 save+eval; NEW=$FORWARD_FULL_DIR"
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  exec "$PY" -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 --module tools.causal_forward_full.run "${args[@]}" "$@"
else
  exec "$PY" -u -m tools.causal_forward_full.run "${args[@]}" "$@"
fi
