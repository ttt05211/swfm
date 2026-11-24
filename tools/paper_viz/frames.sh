#!/usr/bin/env bash
# Paper-figure frames bundle (GT + ego poses, 4 history -> 12 future = 6 s). CPU only, read-only on inputs.
# Writes ONE NEW directory; refuses to run if it already exists. Nothing is deleted.
#   bash tools/paper_viz/frames.sh --dry-run          # selection + size estimate, writes nothing
#   bash tools/paper_viz/frames.sh                    # 400 windows (default)
#   WINDOWS=300 bash tools/paper_viz/frames.sh --moving
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
OUT="${PAPER_VIZ_FRAMES_DIR:-$ROOT/outputs/paper_viz/frames_$(date +%Y%m%d_%H%M%S)}"
exec python -u -m tools.paper_viz.frames --out-dir "$OUT" --windows "${WINDOWS:-400}" \
  --population "${POPULATION:-long6}" "$@"
