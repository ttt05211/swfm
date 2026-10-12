#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
QUALITY_DIR="${REFINER_AUDIT_DIR:-$ROOT/outputs/p0_f9_shared_history/refiner_epoch2_audit_20261012_204921}"
test -f "$QUALITY_DIR/results.json" || { echo "Missing completed audit: $QUALITY_DIR"; exit 1; }
if [[ "${RETIME_RESUME:-0}" == 1 && -z "${RUN_DIR:-}" ]]; then echo 'Set original RUN_DIR'; exit 1; fi
export RUN_DIR="${RUN_DIR:-$ROOT/outputs/p0_f9_shared_history/refiner_epoch2_retime_$(date +%Y%m%d_%H%M%S)}"
export RESUME="$QUALITY_DIR/frozen/epoch2.pt" EPOCHS=2
export REFINER_MODULE=tools.shared_history_experiments.refiner_diagnostics.retime
args=(--reuse-audit "$QUALITY_DIR" --speed-repeats "${SPEED_REPEATS:-3}")
if [[ "${RETIME_RESUME:-0}" == 1 ]]; then args+=(--retime-resume); fi
echo 'Reuse existing DEV512/edit counts READ ONLY; remeasure ALL original32 windows; NO training.'
exec bash "$ROOT/tools/shared_history_experiments/refine/run.sh" "${args[@]}" "$@"
