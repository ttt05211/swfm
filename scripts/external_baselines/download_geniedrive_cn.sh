#!/usr/bin/env bash
set -euo pipefail

# All external files live outside SWFM data/log/output trees.
EXTERNAL_ROOT="${EXTERNAL_ROOT:-/root/nas/occ/external_baselines/geniedrive_val128}"
GENIEDRIVE_ROOT="${GENIEDRIVE_ROOT:-${EXTERNAL_ROOT}/GenieDrive}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-${EXTERNAL_ROOT}/downloads}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
GENIEDRIVE_REV="${GENIEDRIVE_REV:-da48a529ffbe14136688e9b7a56f5d1061c366c5}"

mkdir -p "${EXTERNAL_ROOT}" "${DOWNLOAD_ROOT}"

fresh_clone=0
if [[ ! -d "${GENIEDRIVE_ROOT}/.git" ]]; then
  mirrors=(
    "${GITHUB_MIRROR_URL:-https://gh-proxy.com/https://github.com/Huster-YZY/GenieDrive.git}"
    "https://ghproxy.1888866.xyz/https://github.com/Huster-YZY/GenieDrive.git"
    "https://ghproxy.net/https://github.com/Huster-YZY/GenieDrive.git"
    "https://ghfast.top/https://github.com/Huster-YZY/GenieDrive.git"
  )
  cloned=0
  for url in "${mirrors[@]}"; do
    echo "Trying GitHub China mirror: ${url}"
    if git clone --depth 1 "${url}" "${GENIEDRIVE_ROOT}"; then
      cloned=1
      fresh_clone=1
      break
    fi
  done
  if [[ "${cloned}" -ne 1 ]]; then
    echo "All configured GitHub mirrors failed. Set GITHUB_MIRROR_URL to an available mirror." >&2
    exit 2
  fi
fi

if [[ "${fresh_clone}" -eq 1 ]]; then
  if ! git -C "${GENIEDRIVE_ROOT}" cat-file -e "${GENIEDRIVE_REV}^{commit}"; then
    git -C "${GENIEDRIVE_ROOT}" fetch --depth 1 origin "${GENIEDRIVE_REV}"
  fi
  git -C "${GENIEDRIVE_ROOT}" checkout --detach "${GENIEDRIVE_REV}"
fi
current_rev="$(git -C "${GENIEDRIVE_ROOT}" rev-parse HEAD)"
if [[ "${current_rev}" != "${GENIEDRIVE_REV}" ]]; then
  echo "GenieDrive checkout is ${current_rev}, expected tested revision ${GENIEDRIVE_REV}." >&2
  echo "Use an empty GENIEDRIVE_ROOT or set GENIEDRIVE_REV deliberately." >&2
  exit 4
fi

if command -v hf >/dev/null 2>&1; then
  HF_ENDPOINT="${HF_ENDPOINT}" hf download ANIYA673/GenieDrive \
    genie_occ.pth world-nuscenes_infos_val.pkl \
    --local-dir "${DOWNLOAD_ROOT}"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_ENDPOINT="${HF_ENDPOINT}" huggingface-cli download --resume-download \
    ANIYA673/GenieDrive \
    --include "genie_occ.pth" "world-nuscenes_infos_val.pkl" \
    --local-dir "${DOWNLOAD_ROOT}" --local-dir-use-symlinks False
else
  echo "Install huggingface_hub first: pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U huggingface_hub" >&2
  exit 3
fi

test -s "${DOWNLOAD_ROOT}/genie_occ.pth"
test -s "${DOWNLOAD_ROOT}/world-nuscenes_infos_val.pkl"
echo "GenieDrive root: ${GENIEDRIVE_ROOT}"
echo "GenieDrive rev:  ${current_rev}"
echo "Checkpoint:      ${DOWNLOAD_ROOT}/genie_occ.pth"
echo "Val annotation:  ${DOWNLOAD_ROOT}/world-nuscenes_infos_val.pkl"
