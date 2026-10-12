#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
if [[ "${WORKFLOW_RESUME:-0}" == 1 && -z "${RUN_DIR:-}" ]]; then
  echo 'Please set RUN_DIR to the original workflow directory before resume'; exit 1
fi
export EPOCHS=2
export RUN_DIR="${RUN_DIR:-$ROOT/outputs/p0_f9_shared_history/refiner_probe_2ep_$(date +%Y%m%d_%H%M%S)}"
export REFINER_MODULE=tools.shared_history_experiments.refine.experiment.workflow
if [[ -n "${RESUME:-}" ]]; then
  echo 'Use WORKFLOW_RESUME=1 and original RUN_DIR (not RESUME=last.pt)'; exit 1
fi
extra=()
if [[ "${WORKFLOW_RESUME:-0}" == 1 ]]; then extra+=(--resume-workflow); fi
echo "Discarded probe -> fresh TRAIN4086 x2 -> DEV512 after epoch1 AND epoch2."
echo "WORKFLOW_DIR=$RUN_DIR; Ctrl-C safe boundary; resume SAME directory with WORKFLOW_RESUME=1."
exec bash "$ROOT/tools/shared_history_experiments/refine/run.sh" "${extra[@]}" "$@"
