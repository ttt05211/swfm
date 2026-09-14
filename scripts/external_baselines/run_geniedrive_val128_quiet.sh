#!/usr/bin/env bash
set -euo pipefail

SWFM_ROOT="${SWFM_ROOT:-/root/nas/occ/swfm_geniedrive_eval}"
EXTERNAL_ROOT="${EXTERNAL_ROOT:-/root/nas/occ/external_baselines/geniedrive_val128}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EXTERNAL_ROOT}/results}"
LOG_PATH="${LOG_PATH:-${OUTPUT_ROOT}/run.log}"

mkdir -p "${OUTPUT_ROOT}"
echo "GenieDrive evaluation is running. Full log: ${LOG_PATH}"
if bash "${SWFM_ROOT}/scripts/external_baselines/run_geniedrive_val128.sh" \
  >"${LOG_PATH}" 2>&1; then
  bash "${SWFM_ROOT}/scripts/external_baselines/show_geniedrive_val128_result.sh"
else
  status=$?
  echo "GenieDrive evaluation failed (exit ${status}). Last 80 log lines:" >&2
  tail -n 80 "${LOG_PATH}" >&2
  exit "${status}"
fi
