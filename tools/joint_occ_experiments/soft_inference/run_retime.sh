#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
PY="$(command -v python)"
"$PY" -c 'import sys; assert "OccFM" in sys.executable, "conda activate OccFM first"'
export PYTHONPATH="$ROOT:$ROOT/upstream_occfm${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
if [[ -z "${SOFT_RETIME_FROM:-}" && "${SOFT_RESUME:-0}" != 1 ]]; then
  SOFT_RETIME_FROM="$("$PY" -c 'import sys; from tools.joint_occ_experiments.soft_inference.receipts import discover_completed; print(discover_completed(sys.argv[1]))' "$ROOT/outputs/p0_f9_joint_occ_ego")"
fi
export SOFT_OUT="${SOFT_OUT:-$ROOT/outputs/p0_f9_joint_occ_ego/soft_speed_aligned_$(date +%Y%m%d_%H%M%S)}"
args=()
if [[ -n "${SOFT_RETIME_FROM:-}" ]]; then args+=(--reuse-quality-from "$SOFT_RETIME_FROM"); fi
echo 'SPEED ONLY: verify/reuse completed DEV512 integer counts; discard old timings; NO training or 512-window re-scoring.'
echo 'Old FPS boundary: history extraction/matching/staging outside; fresh Strong/KTA + WM/CCR/projection/six dense CPU outputs inside.'
bash tools/joint_occ_experiments/soft_inference/run.sh "${args[@]}" "$@"
echo "RESULT: $SOFT_OUT/summary.txt"
