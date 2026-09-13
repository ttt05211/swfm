#!/usr/bin/env bash
set -euo pipefail

SWFM_ROOT="${SWFM_ROOT:-/root/nas/occ/swfm}"
EXTERNAL_ROOT="${EXTERNAL_ROOT:-/root/nas/occ/external_baselines/geniedrive_val128}"
GENIEDRIVE_ROOT="${GENIEDRIVE_ROOT:-${EXTERNAL_ROOT}/GenieDrive}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-${EXTERNAL_ROOT}/downloads}"
PREPARED_VAL="${PREPARED_VAL:-${SWFM_ROOT}/data/prepared_val}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EXTERNAL_ROOT}/results}"
GENIEDRIVE_ENV="${GENIEDRIVE_ENV:-geniedrive-occ}"
SWFM_ENV="${SWFM_ENV:-base}"
GPU_ID="${GPU_ID:-0}"

mkdir -p "${OUTPUT_ROOT}"

conda run --no-capture-output -n "${SWFM_ENV}" \
  python "${SWFM_ROOT}/tools/external_baselines/build_geniedrive_val128_manifest.py" \
  --prepared "${PREPARED_VAL}" \
  --output "${OUTPUT_ROOT}/val128_manifest.json" \
  --expected-count 128

CUDA_VISIBLE_DEVICES="${GPU_ID}" conda run --no-capture-output -n "${GENIEDRIVE_ENV}" \
  python "${SWFM_ROOT}/tools/external_baselines/export_geniedrive_val128.py" \
  --geniedrive-root "${GENIEDRIVE_ROOT}" \
  --manifest "${OUTPUT_ROOT}/val128_manifest.json" \
  --output-dir "${OUTPUT_ROOT}" \
  --checkpoint "${DOWNLOAD_ROOT}/genie_occ.pth" \
  --ann-file "${DOWNLOAD_ROOT}/world-nuscenes_infos_val.pkl" \
  --gpu-id 0

conda run --no-capture-output -n "${SWFM_ENV}" \
  python "${SWFM_ROOT}/tools/real_motion/evaluate_predictions.py" \
  --prepared "${PREPARED_VAL}" \
  --pred-dir "${OUTPUT_ROOT}/predictions" \
  --output "${OUTPUT_ROOT}/geniedrive_val128_moving_miou_v2.json"

echo "Result: ${OUTPUT_ROOT}/geniedrive_val128_moving_miou_v2.json"
