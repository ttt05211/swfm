#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
: "${GT_CKPT:?Set GT_CKPT to the SAME finished epoch_0002.pt}"
: "${EGO_RUN_DIR:?Set EGO_RUN_DIR to a NEW hundred-epoch stage directory}"
PY="$(command -v python)"
"$PY" -c 'import sys; assert "OccFM" in sys.executable, "conda activate OccFM first"'
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "${EGO_REUSE_DIR:-}" ]]; then
  EGO_REUSE_DIR="$("$PY" -m tools.scene_ego_head.frozen_gt.reuse --find-stage "$ROOT/outputs/p0_f9_shared_history" "$GT_CKPT")"
fi
if [[ -z "${HEAD10_CKPT:-}" ]]; then
  HEAD10_CKPT="$("$PY" -m tools.scene_ego_head.frozen_gt.comparison --find-head10 "$ROOT/outputs/p0_f9_shared_history" "$GT_CKPT")"
fi
export EGO_REUSE_DIR HEAD10_CKPT HEAD_EPOCHS=100 EGO_FINAL_POPULATION=dev512
echo "History source=$EGO_REUSE_DIR; baseline head=$HEAD10_CKPT; both READ ONLY."
echo 'RANDOM new head x100; save50/80/100; fixed DEV512 head10/50/80/100, no model selection.'
exec bash tools/scene_ego_head/frozen_gt/run.sh --save-head-epochs 50 80 100 --comparison-head10 "$HEAD10_CKPT" "$@"
