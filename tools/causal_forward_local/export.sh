#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=''
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY="$(command -v python)"
"$PY" -c 'import sys; assert "OccFM" in sys.executable'
export FORWARD_LOCAL_EXPORT_DIR="${FORWARD_LOCAL_EXPORT_DIR:-$ROOT/outputs/p0_f9_shared_history/forward_epoch2_local_pack_$(date +%Y%m%d_%H%M%S)}"
args=(--out-dir "$FORWARD_LOCAL_EXPORT_DIR" --config "$ROOT/configs/real_motion_occfm.yaml")
if [[ -n "${FORWARD_FULL_DIR:-}" ]]; then args+=(--source-dir "$FORWARD_FULL_DIR"); fi
echo "CPU-only latest epoch2 export; NO optimizer/RNG/raw data; existing local64 replay reused. OUT=$FORWARD_LOCAL_EXPORT_DIR"
exec "$PY" -u -m tools.causal_forward_local.export "${args[@]}" "$@"
