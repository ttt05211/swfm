#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export SOFT_OUT="${SOFT_OUT:-$ROOT/outputs/p0_f9_joint_occ_ego/soft_speed_optimized_$(date +%Y%m%d_%H%M%S)}"
echo 'Frozen speed-only: hard / old soft ADD / old soft ADD+REMOVE / fast soft ADD / fast soft ADD+REMOVE.'
echo 'Same 32 windows x2; verify all six output bytes; reuse DEV512 counts, NO training/re-scoring.'
bash tools/joint_occ_experiments/soft_inference/run_retime.sh --compare-optimized-speed "$@"
