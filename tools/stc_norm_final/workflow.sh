#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
WORKFLOW="${NORM_WORKFLOW_DIR:?Set NORM_WORKFLOW_DIR to a NEW output directory}"
mkdir "$WORKFLOW"
export NORM_FINAL_DIR="$WORKFLOW/speed"
unset NORM_FINAL_RESUME
bash tools/stc_norm_final/run.sh probe
export NORM_SPEED_REPORT="$NORM_FINAL_DIR"
export NORM_FINAL_DIR="$WORKFLOW/all"
export NORM_FINAL_WORKERS=auto
bash tools/stc_norm_final/run.sh all
cat "$NORM_FINAL_DIR/summary.txt"
