#!/usr/bin/env bash
set -euo pipefail

EXTERNAL_ROOT="${EXTERNAL_ROOT:-/root/nas/occ/external_baselines/geniedrive_val128}"
GENIEDRIVE_ROOT="${GENIEDRIVE_ROOT:-${EXTERNAL_ROOT}/GenieDrive}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-${EXTERNAL_ROOT}/downloads}"
NUSCENES_ROOT="${NUSCENES_ROOT:?Set NUSCENES_ROOT to the existing nuScenes/Occ3D root}"
OVERLAY_ROOT="${EXTERNAL_ROOT}/data/nuscenes"
GENIE_DATA_LINK="${GENIEDRIVE_ROOT}/occ_gen/data/nuscenes"

for required in gts v1.0-trainval; do
  if [[ ! -e "${NUSCENES_ROOT}/${required}" ]]; then
    echo "Missing ${NUSCENES_ROOT}/${required}" >&2
    exit 2
  fi
done
test -s "${DOWNLOAD_ROOT}/world-nuscenes_infos_val.pkl"

mkdir -p "${OVERLAY_ROOT}" "$(dirname "${GENIE_DATA_LINK}")"
for item in gts v1.0-trainval samples sweeps; do
  if [[ -e "${NUSCENES_ROOT}/${item}" && ! -e "${OVERLAY_ROOT}/${item}" ]]; then
    ln -s "${NUSCENES_ROOT}/${item}" "${OVERLAY_ROOT}/${item}"
  fi
done
if [[ ! -e "${OVERLAY_ROOT}/world-nuscenes_infos_val.pkl" ]]; then
  ln -s "${DOWNLOAD_ROOT}/world-nuscenes_infos_val.pkl" \
    "${OVERLAY_ROOT}/world-nuscenes_infos_val.pkl"
fi

if [[ -L "${GENIE_DATA_LINK}" ]]; then
  current="$(readlink -f "${GENIE_DATA_LINK}")"
  expected="$(readlink -f "${OVERLAY_ROOT}")"
  if [[ "${current}" != "${expected}" ]]; then
    echo "Refusing to replace existing link ${GENIE_DATA_LINK} -> ${current}" >&2
    exit 3
  fi
elif [[ -e "${GENIE_DATA_LINK}" ]]; then
  echo "Refusing to replace existing path ${GENIE_DATA_LINK}" >&2
  exit 3
else
  ln -s "${OVERLAY_ROOT}" "${GENIE_DATA_LINK}"
fi

echo "Read-only dataset overlay ready: ${OVERLAY_ROOT}"
