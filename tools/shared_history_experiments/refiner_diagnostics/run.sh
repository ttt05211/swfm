#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export RESUME="${REFINER_CHECKPOINT:-$ROOT/outputs/p0_f9_shared_history/refiner_probe_2ep_20261012_184856/train/epoch_0002.pt}"
test -f "$RESUME" || { echo "Missing epoch2: $RESUME; set REFINER_CHECKPOINT"; exit 1; }
if [[ "${AUDIT_RESUME:-0}" == 1 && -z "${RUN_DIR:-}" ]]; then
  echo 'Set RUN_DIR to original diagnostic directory for resume'; exit 1
fi
export RUN_DIR="${RUN_DIR:-$ROOT/outputs/p0_f9_shared_history/refiner_epoch2_audit_$(date +%Y%m%d_%H%M%S)}"
export EPOCHS=2 REFINER_MODULE=tools.shared_history_experiments.refiner_diagnostics.run
args=(--speed-windows "${SPEED_WINDOWS:-32}" --speed-repeats "${SPEED_REPEATS:-3}")
if [[ "${AUDIT_RESUME:-0}" == 1 ]]; then args+=(--diagnostic-resume); fi
echo "Frozen epoch2 -> DEV512 edit audit + fresh Strong/KTA six-frame timing; NO training."
echo "Source: $RESUME; output: $RUN_DIR"
exec bash "$ROOT/tools/shared_history_experiments/refine/run.sh" "${args[@]}" "$@"
