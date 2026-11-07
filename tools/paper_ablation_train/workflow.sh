#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
export ABLATION_GPUS="${ABLATION_GPUS:-0,1}" CUDA_VISIBLE_DEVICES="${ABLATION_GPUS:-0,1}"
export ABLATION_SUITE_DIR="${ABLATION_SUITE_DIR:-$ROOT/outputs/p0_f9_shared_history/ablation_five_dual_$(date +%Y%m%d_%H%M%S)}"
PY="$(command -v python)"
"$PY" -c 'import sys,torch,os; assert "OccFM" in sys.executable; assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))==2 and torch.cuda.device_count()==2, "Exactly TWO visible GPUs required"'
args=(--out-dir "$ABLATION_SUITE_DIR" --gpus "$ABLATION_GPUS")
if [[ -n "${ABLATION_FULL_REFERENCE:-}" ]]; then args+=(--full-reference "$ABLATION_FULL_REFERENCE"); fi
if [[ -n "${ABLATION_SMALL_REFERENCE:-}" ]]; then args+=(--small-reference "$ABLATION_SMALL_REFERENCE"); fi
echo "FIVE arms / sequential TWO-GPU training / unchanged GLOBAL B4 / NO evaluation; SUITE=$ABLATION_SUITE_DIR"
exec "$PY" -u -m tools.paper_ablation_train.suite "${args[@]}" "$@"
