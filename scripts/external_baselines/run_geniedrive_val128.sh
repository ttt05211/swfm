#!/usr/bin/env bash
set -euo pipefail

SWFM_ROOT="${SWFM_ROOT:-/root/nas/occ/swfm}"
# Prevent a managed host's system Python 3.12 torch from leaking into the
# separate Python 3.8 GenieDrive environment.
source "${SWFM_ROOT}/scripts/external_baselines/sanitize_geniedrive_environment.sh"
EXTERNAL_ROOT="${EXTERNAL_ROOT:-/root/nas/occ/external_baselines/geniedrive_val128}"
GENIEDRIVE_ROOT="${GENIEDRIVE_ROOT:-${EXTERNAL_ROOT}/GenieDrive}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-${EXTERNAL_ROOT}/downloads}"
REFERENCE_CACHE="${REFERENCE_CACHE:-${PREPARED_VAL:-${SWFM_ROOT}/data/p0_f9_v2_wm_val_top2_128}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EXTERNAL_ROOT}/results}"
GENIEDRIVE_ENV="${GENIEDRIVE_ENV:-geniedrive-occ}"
GPU_ID="${GPU_ID:-0}"

mkdir -p "${OUTPUT_ROOT}"

GENIEDRIVE_PREFIX="$(conda run -n "${GENIEDRIVE_ENV}" python -c 'import sys; print(sys.prefix)')"
GENIEDRIVE_PYTHON="$(conda run -n "${GENIEDRIVE_ENV}" python -c 'import sys; print("python%d.%d" % sys.version_info[:2])')"
GENIEDRIVE_TORCH_LIB="${GENIEDRIVE_PREFIX}/lib/${GENIEDRIVE_PYTHON}/site-packages/torch/lib"
if [[ ! -d "${GENIEDRIVE_TORCH_LIB}" ]]; then
  echo "Missing GenieDrive torch library directory: ${GENIEDRIVE_TORCH_LIB}" >&2
  exit 5
fi

LD_LIBRARY_PATH="${GENIEDRIVE_TORCH_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
conda run --no-capture-output -n "${GENIEDRIVE_ENV}" \
  python "${SWFM_ROOT}/tools/external_baselines/build_geniedrive_val128_manifest.py" \
  --reference-cache "${REFERENCE_CACHE}" \
  --output "${OUTPUT_ROOT}/val128_manifest.json" \
  --expected-count 128

LD_LIBRARY_PATH="${GENIEDRIVE_TORCH_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
CUDA_VISIBLE_DEVICES="${GPU_ID}" conda run --no-capture-output -n "${GENIEDRIVE_ENV}" \
  python "${SWFM_ROOT}/tools/external_baselines/export_geniedrive_val128.py" \
  --geniedrive-root "${GENIEDRIVE_ROOT}" \
  --manifest "${OUTPUT_ROOT}/val128_manifest.json" \
  --output-dir "${OUTPUT_ROOT}" \
  --checkpoint "${DOWNLOAD_ROOT}/genie_occ.pth" \
  --ann-file "${DOWNLOAD_ROOT}/world-nuscenes_infos_val.pkl" \
  --gpu-id 0

LD_LIBRARY_PATH="${GENIEDRIVE_TORCH_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
conda run --no-capture-output -n "${GENIEDRIVE_ENV}" \
  python "${SWFM_ROOT}/tools/external_baselines/score_geniedrive_val128.py" \
  --reference-cache "${REFERENCE_CACHE}" \
  --manifest "${OUTPUT_ROOT}/val128_manifest.json" \
  --pred-dir "${OUTPUT_ROOT}/predictions" \
  --output "${OUTPUT_ROOT}/geniedrive_val128_moving_miou_v2.json"

echo "Result: ${OUTPUT_ROOT}/geniedrive_val128_moving_miou_v2.json"
