#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${DENSE_MEAN_RESUME:-0}" != 1 && -z "${DENSE_MEAN_DIR:-}" ]]; then
  export DENSE_MEAN_DIR="$ROOT/outputs/p0_f9_shared_history/dense_mean4_wide_$(date +%Y%m%d_%H%M%S)"
fi
echo 'FIXED WIDE mean:4000/6000/8000/10292. New output; no old mean blended, no training, no auto full-VAL/model selection.'
exec bash "$ROOT/tools/dense_checkpoint_average/run.sh" --recipe wide "$@"
