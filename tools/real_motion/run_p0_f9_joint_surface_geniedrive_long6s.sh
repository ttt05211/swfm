#!/usr/bin/env bash
set -euo pipefail
[[ $# -le 2 ]] || { echo "用法: bash $0 [OUTPUT_DIR] [--resume]" >&2; exit 2; }
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-$ROOT/outputs/p0_f9_joint_surface_ccr/geniedrive_code_long6s_mean_$(date +%Y%m%d_%H%M%S)_$$}"
EXTRA=()
if [[ $# == 2 ]]; then
  [[ "$2" == --resume ]] || { echo '第二个参数只允许--resume' >&2; exit 2; }
  EXTRA+=(--resume)
fi
export SURFACE_LONG_ALIGNMENT=geniedrive_code10s
exec bash "$ROOT/tools/real_motion/run_p0_f9_joint_surface_long6s.sh" all "$OUT" "${EXTRA[@]}"
