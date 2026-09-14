#!/usr/bin/env bash
set -euo pipefail

SWFM_ROOT="${SWFM_ROOT:-/root/nas/occ/swfm_geniedrive_eval}"
EXTERNAL_ROOT="${EXTERNAL_ROOT:-/root/nas/occ/external_baselines/geniedrive_val128}"

python "${SWFM_ROOT}/tools/external_baselines/summarize_geniedrive_val128.py" \
  --results "${EXTERNAL_ROOT}/results"
