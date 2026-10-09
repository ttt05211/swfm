#!/usr/bin/env python3
"""Download ONLY the pinned official validation metadata; no HF SDK needed."""
import sys
from pathlib import Path
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import http.client
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from tools.real_motion.geniedrive_eval_alignment import INFO_URL, INFO_BYTES, verify_info


class DownloadNetworkError(RuntimeError):
    """All configured transports failed; no official artifact was published."""


def download_urls(endpoint=None):
    """Only change the transport origin; keep the pinned repo/revision/file."""
    if endpoint is None:
        endpoint = os.environ.get('GENIEDRIVE_DOWNLOAD_ENDPOINT') or os.environ.get('HF_ENDPOINT')
    if not endpoint:
        return (INFO_URL, 'https://hf-mirror.com'+urllib.parse.urlsplit(INFO_URL).path)
    parsed = urllib.parse.urlsplit(endpoint)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.path not in ('', '/')):
        raise ValueError('download endpoint must be an HTTPS origin without credentials/path/query')
    return (endpoint.rstrip('/')+urllib.parse.urlsplit(INFO_URL).path,)


def _download_one(path, url, timeout):
    path = Path(path).resolve()
    if path.exists():
        verify_info(path)  # Never overwrite a wrong or user-owned artifact.
        print(f'OFFICIAL INFO already verified: {path}', flush=True)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive per-process temporary file; incomplete transfers cannot become
    # official metadata and an existing target is not overwritten.
    temporary = path.with_name(path.name+f'.{os.getpid()}.part')
    owned = False
    try:
        with temporary.open('xb') as target:
            owned = True
            request = urllib.request.Request(url, headers={'User-Agent': 'swfm-geniedrive-alignment/1'})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                size = 0
                while True:
                    chunk = response.read(2**20)
                    if not chunk: break
                    size += len(chunk)
                    if size > INFO_BYTES: raise RuntimeError('download exceeds pinned metadata size')
                    target.write(chunk)
                    if size % (16*2**20) == 0: print(f'download={size/2**20:.0f}/{INFO_BYTES/2**20:.1f} MiB', flush=True)
            target.flush(); os.fsync(target.fileno())
        verify_info(temporary)
        # link is atomic, same directory/filesystem, and fails if target exists.
        # Never os.replace: a concurrent download/user file must not be erased.
        try: os.link(temporary, path)
        except FileExistsError: verify_info(path)
        print(f'OFFICIAL INFO verified: {path}', flush=True)
        return path
    finally:
        if owned and temporary.is_file(): temporary.unlink()


def download(path, *, endpoint=None, timeout=30):
    path = Path(path).resolve()
    if path.exists():
        verify_info(path)  # Offline reuse; never download over existing files.
        print(f'OFFICIAL INFO already verified: {path}', flush=True)
        return path
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('download timeout must be finite and positive')
    failures = []
    for url in download_urls(endpoint):
        host = urllib.parse.urlsplit(url).netloc
        print(f'OFFICIAL INFO download: {host}; pinned revision, size and SHA256 enforced', flush=True)
        try:
            return _download_one(path, url, timeout)
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as exc:
            # Integrity errors and local filesystem failures are NOT retried or
            # disguised as network errors. Each owned .part is already removed.
            reason = getattr(exc, 'reason', exc)
            failures.append(f'{host}: {reason}')
            print(f'OFFICIAL INFO network failed: {host}: {reason}', flush=True)
    raise DownloadNetworkError(
        '无法下载官方 metadata（评估尚未启动）：'+'; '.join(failures)
        +'\n可设置 GENIEDRIVE_DOWNLOAD_ENDPOINT=https://可访问的镜像域名，'
        '或配置服务器 HTTPS_PROXY；完全离线时在可联网机器下载固定 revision 文件，'
        f'再上传至 {path}，脚本将先校验并直接复用。不要替换为其他 metadata。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--endpoint', help='HTTPS origin; overrides GENIEDRIVE_DOWNLOAD_ENDPOINT / HF_ENDPOINT')
    parser.add_argument('--timeout', type=float, default=30, help='per-connection timeout in seconds')
    args = parser.parse_args()
    try:
        download(args.out, endpoint=args.endpoint, timeout=args.timeout)
    except (DownloadNetworkError, ValueError) as exc:
        parser.exit(2, f'{exc}\n')


if __name__ == '__main__': main()
