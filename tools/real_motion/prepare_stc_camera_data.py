#!/usr/bin/env python3
"""Safely unpack STCOcc-Res into lossless uint8 semantic-only NPZs.

No pickle, no model, no label remap/mask. Never overwrite an existing artifact
without explicit same-archive --resume. The original downloaded ZIP is kept.
"""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import hashlib
import io
import json
from pathlib import PurePosixPath
import re
import zipfile

import numpy as np

from real_motion.stc_camera_protocol import semantics
from real_motion.waymo_i2world import file_sha256, fingerprint
from tools.real_motion.waymo_zero_shot_common import write_json

OFFICIAL_ID = '1dXB9mtROLWChycBZlhYIf_JBLshXogBs'
EXPECTED_ZIP_BYTES = 67779055


def extract(archive, out, *, resume=False, shape=(200, 200, 16), expected_size=EXPECTED_ZIP_BYTES):
    archive, out = Path(archive).resolve(), Path(out).resolve()
    if out == archive or archive.is_relative_to(out):
        raise ValueError('archive must be outside extraction output')
    if expected_size and archive.stat().st_size != expected_size:
        raise ValueError(f'official ZIP size mismatch: expected {expected_size}; reject HTML/partial/replaced releases')
    digest = file_sha256(archive)
    contract = dict(protocol='stc_official_lossless_semantics_uint8_v1', archive_sha256=digest,
                    official_drive_id=OFFICIAL_ID, shape=list(shape), masks_applied=False,
                    archive_crc_checked=True, semantics_values_unchanged=True)
    state_path = out / 'extraction_state.json'
    if resume:
        state = json.loads(state_path.read_text())
        declared = state.pop('fingerprint', None)
        if declared != fingerprint(state) or state['contract'] != contract:
            raise RuntimeError('extraction resume archive/contract/state changed')
    else:
        if out.exists():
            raise FileExistsError('new output required; use --resume ONLY for this same archive')
        out.mkdir(parents=True)
        state = dict(contract=contract, files={}, status='extracting')
    def save():
        value = dict(state); value['fingerprint'] = fingerprint(value)
        write_json(state_path, value)
    save()
    try:
        with zipfile.ZipFile(archive) as zip_file:
            members = []; seen = set()
            for entry in zip_file.infolist():
                if entry.is_dir():
                    continue
                parts = PurePosixPath(entry.filename).parts
                if ('..' in parts or PurePosixPath(entry.filename).is_absolute()
                        or '\\' in entry.filename or (entry.external_attr >> 16) & 0o170000 == 0o120000):
                    raise ValueError('unsafe ZIP path/link')
                if len(parts) != 4 or parts[0] != 'stc-results' or parts[-1] != 'labels.npz':
                    raise ValueError('unexpected official STC ZIP member: ' + entry.filename)
                scene, token = parts[1:3]
                if not re.fullmatch(r'scene-[0-9]{4}', scene) or not re.fullmatch(r'[a-f0-9]{32}', token):
                    raise ValueError('invalid STC scene/token identity')
                key = scene + '/' + token
                if key in seen:
                    raise ValueError('duplicate STC frame in ZIP')
                seen.add(key); members.append((entry, key))
            if not members:
                raise ValueError('empty STC ZIP')
            if set(state['files']) - seen:
                raise RuntimeError('saved extraction has unexpected identities')
            for i, (entry, key) in enumerate(members, 1):
                # ZipFile.read verifies the outer member CRC; np.load checks
                # the inner NPZ. Validate BEFORE casting to avoid uint8 wrap.
                with np.load(io.BytesIO(zip_file.read(entry)), allow_pickle=False) as z:
                    sem = semantics(z['semantics'], shape)
                sha = hashlib.sha256(sem.tobytes()).hexdigest()
                destination = out / key / 'labels.npz'
                if destination.exists():
                    with np.load(destination, allow_pickle=False) as z:
                        actual = semantics(z['semantics'], shape)
                    if not np.array_equal(actual, sem) or (key in state['files'] and state['files'][key] != sha):
                        raise RuntimeError('existing compact STC semantics changed: ' + key)
                else:
                    if key in state['files']:
                        raise RuntimeError('previously extracted STC frame is missing: ' + key)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    tmp = destination.with_name('labels.tmp.npz')
                    np.savez_compressed(tmp, semantics=sem)
                    tmp.replace(destination)
                state['files'][key] = sha
                if i % 128 == 0 or i == len(members):
                    save(); print(f'STC_EXTRACT {i}/{len(members)} lossless_uint8', flush=True)
        if file_sha256(archive) != digest:
            raise RuntimeError('archive changed during extraction')
        state.update(status='complete', frames=len(members), scenes=len({k.split('/')[0] for k in seen}))
    finally:
        save()
    print(f'STC_READY frames={state["frames"]} scenes={state["scenes"]} root={out}', flush=True)
    return state


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', required=True); p.add_argument('--out-root', required=True)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args(argv)
    extract(a.archive, a.out_root, resume=a.resume)


if __name__ == '__main__':
    main()
