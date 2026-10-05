#!/usr/bin/env bash
# Official PUBLIC CODE population; no GenieDrive framework/model installation.
set -euo pipefail
if [[ $# -gt 2 ]]; then echo "用法: bash $0 [OUTPUT_DIR] [--resume]" >&2; exit 2; fi
ROOT=/root/nas/occ/swfm
OUT="${1:-$ROOT/outputs/p0_f9_joint_causal_columns/geniedrive_code_long6s_epoch19_all_$(date +%Y%m%d_%H%M%S)_$$}"
EXTRA=()
if [[ $# == 2 ]]; then
  [[ "$2" == --resume ]] || { echo '第二个参数只允许--resume' >&2; exit 2; }
  EXTRA+=(--resume)
fi
[[ "${LONG_COMPARE_E14:-0}" == 0 ]] || { echo '此人口不允许六历史E14比较' >&2; exit 2; }
export LONG_POPULATION_ALIGNMENT=geniedrive_code10s
export LONG_HANDOFF_MODES="${LONG_HANDOFF_MODES:-redetect,reconciled}"
exec bash "$ROOT/tools/real_motion/run_p0_f9_joint_long_rollout.sh" all "$OUT" "${EXTRA[@]}"
