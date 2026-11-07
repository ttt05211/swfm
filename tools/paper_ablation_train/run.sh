#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
ARM="${1:?Usage: run.sh c_transport|d_kta_refiner|m0_no_memory|m1_ego_memory|s0_pool}"; shift
case "$ARM" in
  c_transport|d_kta_refiner)
    DEFAULT_GPUS=0,1
    REFERENCE="$ROOT/outputs/p0_f9_shared_history/adaln_forward_full_2ep_20261026_215351/contract.json" ;;
  m0_no_memory|m1_ego_memory|s0_pool)
    DEFAULT_GPUS=0
    REFERENCE="$ROOT/outputs/p0_f9_shared_history/adaln_forward_single_free20_20261025_161644/C_train/contract.json" ;;
  *) echo "Unknown arm: $ARM" >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES="${ABLATION_GPUS:-$DEFAULT_GPUS}"
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
PY="$(command -v python)"
"$PY" -c 'import sys,torch,os; assert "OccFM" in sys.executable; n=len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")); assert n in (1,2) and torch.cuda.device_count()==n'
if [[ "${ABLATION_RESUME:-0}" == 1 && -z "${ABLATION_DIR:-}" ]]; then echo 'Set original ABLATION_DIR to resume'; exit 2; fi
export ABLATION_DIR="${ABLATION_DIR:-$ROOT/outputs/p0_f9_shared_history/ablation_${ARM}_train1_$(date +%Y%m%d_%H%M%S)}"
DATA="${DATAROOT:-/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes}"
args=(--arm "$ARM" --out-dir "$ABLATION_DIR" --reference-contract "${ABLATION_REFERENCE:-$REFERENCE}"
  --config "$ROOT/configs/real_motion_occfm.yaml" --train-cache "${TRAIN_CACHE:-$ROOT/data/p0_f9_v18_se2_train_full.pt}"
  --base-checkpoint "${E14:-$ROOT/outputs/p0_f9_v18_se2_clean_tail15/epoch_0014.pt}"
  --dataroot "$DATA" --train-info "${TRAIN_INFO:-$DATA/nuscenes_infos_train_temporal_v3_scene.pkl}"
  --ccr-history-cache "${CCR_HISTORY_CACHE:-$ROOT/cache/p0_f9_ccr_history_geometry_v1}")
if [[ "${ABLATION_RESUME:-0}" == 1 ]]; then args+=(--resume "$ABLATION_DIR/last.pt"); fi
echo "ABLATION TRAIN ONLY: $ARM; ONE epoch; no VAL/STC/evaluation; OUT=$ABLATION_DIR"
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  exec "$PY" -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 --module tools.paper_ablation_train.run "${args[@]}" "$@"
else
  exec "$PY" -u -m tools.paper_ablation_train.run "${args[@]}" "$@"
fi
