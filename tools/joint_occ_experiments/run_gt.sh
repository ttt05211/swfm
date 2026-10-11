#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MODE=gt
exec bash "$ROOT/tools/joint_occ_experiments/run_paired.sh" "$@"
