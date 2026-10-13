#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
: "${GT_CKPT:?Set GT_CKPT to completed GT epoch_0002.pt}"
: "${RUN_DIR:?Set RUN_DIR to a NEW adaptation directory}"
export REFINER_MODULE=tools.scene_ego_head.pred_adapt.run EPOCHS=1
if [[ "${PRED_ADAPT_RESUME:-0}" == 1 ]]; then
  export RESUME="$RUN_DIR/train/last.pt"; test -f "$RESUME"
else
  export RESUME=""
fi
echo 'GT epoch2 body + RANDOM ego; freeze Encoder/STWM; Refiner1e-5 + ego1e-4; 500 updates -> DEV512.'
exec bash tools/shared_history_experiments/refine/run.sh --mode pred --gt-checkpoint "$GT_CKPT" "$@"
