#!/usr/bin/env python3
"""Download ONLY the pinned official validation metadata; no HF SDK needed."""
import sys
from pathlib import Path
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import os
import urllib.request
from tools.real_motion.geniedrive_eval_alignment import INFO_URL, INFO_BYTES, verify_info


def download(path):
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
            request = urllib.request.Request(INFO_URL, headers={'User-Agent': 'swfm-geniedrive-alignment/1'})
            with urllib.request.urlopen(request, timeout=60) as response:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    download(parser.parse_args().out)


if __name__ == '__main__': main()
