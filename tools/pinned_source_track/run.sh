#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
"$PY" -c 'import sys; assert "OccFM" in sys.executable, "conda activate OccFM first"'
ORIGINAL="${ORIGINAL_SWFM:-/root/nas/occ/swfm}"
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm:$ORIGINAL/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 SWFM_COLUMN_CPU_BACKEND=native
if [[ "${PAIR_RESUME:-0}" == 1 && -z "${PAIR_DIR:-}" ]]; then echo 'Set original PAIR_DIR before resume'; exit 1; fi
PAIR_DIR="${PAIR_DIR:-$ORIGINAL/outputs/p0_f9_shared_history/pinned_pair_20pct_1ep_$(date +%Y%m%d_%H%M%S)}"
args=(--out-dir "$PAIR_DIR" --search-root "$ORIGINAL/outputs/p0_f9_shared_history")
if [[ -n "${PINNED_REFERENCE_DIR:-}" ]]; then args+=(--reference-dir "$PINNED_REFERENCE_DIR"); fi
if [[ "${PAIR_RESUME:-0}" == 1 ]]; then args+=(--resume); fi
echo "Train ONLY unified_pair x1, reuse completed baseline; ORIGINAL KTA unchanged; output=$PAIR_DIR"
exec "$PY" -u -m tools.pinned_source_track.run "${args[@]}" "$@"
