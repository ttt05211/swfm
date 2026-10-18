#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export SCENE_PILOT_DIR="${SCENE_PILOT_DIR:-$ROOT/outputs/p0_f9_shared_history/scene_reader_stage1_${SCENE_EPOCHS:-1}ep_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$SCENE_PILOT_DIR"
echo "Fixed TWO random runs, baseline THEN reader; TRAIN20% x${SCENE_EPOCHS:-1} each. OUT=$SCENE_PILOT_DIR"
echo 'Only history geometry caches reused. Configured final epoch(s) eval64 on ONE GPU, no best selection.'
for arm in baseline reader; do
  export SCENE_ARM_DIR="$SCENE_PILOT_DIR/$arm"
  if [[ -f "$SCENE_ARM_DIR/last.pt" ]]; then export SCENE_RESUME=1; else export SCENE_RESUME=0; fi
  bash tools/source_scene_stage1/run.sh "$arm" "$@"
  # A graceful early stop is not completion and must never silently start arm B.
  "$(command -v python)" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"]=="complete", "Arm stopped; resume same SCENE_PILOT_DIR before next arm"' "$SCENE_ARM_DIR/results.json"
done
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
"$(command -v python)" -u -m tools.source_scene_stage1.report --root "$SCENE_PILOT_DIR"
