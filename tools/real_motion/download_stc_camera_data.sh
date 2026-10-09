#!/usr/bin/env bash
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == OccFM ]] || { echo '先 conda activate OccFM' >&2; exit 2; }
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$(command -v python)"
DEST="${STC_ZIP:-$ROOT/data/stc_camera/stc-results.zip}"
mkdir -p -- "$(dirname -- "$DEST")"
validate() {
  "$PY" - "$1" <<'PY'
import pathlib, sys, zipfile
p = pathlib.Path(sys.argv[1])
if p.stat().st_size != 67779055:
    raise SystemExit('STC ZIP大小不符，可能是HTML、未下完或官方更换了版本；保留文件，不自动删除。')
with zipfile.ZipFile(p) as z:
    members = [r for r in z.infolist() if not r.is_dir()]
    if not members or any(not r.filename.startswith('stc-results/') for r in members):
        raise SystemExit('不是预期的官方STC ZIP；保留文件。')
print('STC_ZIP_READY', p, p.stat().st_size)
PY
}
if [[ -e "$DEST" ]]; then
  validate "$DEST"
  echo '复用已存在ZIP，没有覆盖。'
  exit 0
fi
PART="$DEST.part"
[[ ! -e "$PART" ]] || { echo "已有未完成文件 $PART，保留不覆盖；请检查后换 STC_ZIP 路径再下载。" >&2; exit 2; }
"$PY" -c 'import gdown' || { echo '缺少gdown：请在OccFM执行 python -m pip install gdown' >&2; exit 2; }
"$PY" -m gdown 'https://drive.google.com/uc?id=1dXB9mtROLWChycBZlhYIf_JBLshXogBs' -O "$PART"
validate "$PART"
# Do not replace an artifact published by another process while downloading.
[[ ! -e "$DEST" ]] || { echo '下载期间目标文件出现；保留.part，不覆盖。' >&2; exit 2; }
mv -n -- "$PART" "$DEST"
echo "官方ZIP下载完成：$DEST；CRC/语义逐文件校验由解压入口完成。"
