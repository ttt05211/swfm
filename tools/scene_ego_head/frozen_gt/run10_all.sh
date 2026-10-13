#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
: "${GT_CKPT:?Set GT_CKPT to the SAME finished epoch_0002.pt}"
: "${EGO_RUN_DIR:?Set EGO_RUN_DIR to a NEW ten-epoch stage directory}"
PY="$(command -v python)"
"$PY" -c 'import sys; assert "OccFM" in sys.executable, "conda activate OccFM first"'
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "${EGO_REUSE_DIR:-}" ]]; then
  EGO_REUSE_DIR="$("$PY" -m tools.scene_ego_head.frozen_gt.reuse --find-stage "$ROOT/outputs/p0_f9_shared_history" "$GT_CKPT")"
fi
export EGO_REUSE_DIR HEAD_EPOCHS=10 EGO_FINAL_POPULATION=all
echo "History source=$EGO_REUSE_DIR; read only, no extraction or old head/Adam import."
echo 'RANDOM head x10 / full-cycle cosine / frozen epoch2 body -> ALL VAL4369, internal only.'
exec bash tools/scene_ego_head/frozen_gt/run.sh "$@"
