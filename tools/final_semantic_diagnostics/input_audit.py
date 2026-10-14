import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from real_motion.stc_camera_protocol import semantics
from real_motion.waymo_i2world import fingerprint, file_sha256
from .statistics import confusion


class ArtifactAudit:
    """Independent selected-history bytes check; never change input axes/labels."""
    def __init__(self, root, archive=None):
        self.root = Path(root); self.path = self.root/'extraction_state.json'
        self.sha = file_sha256(self.path); self.state = json.loads(self.path.read_text(encoding='utf-8'))
        value = dict(self.state); declared = value.pop('fingerprint', None)
        c = value['contract']
        if (declared != fingerprint(value) or value['status'] != 'complete'
                or c['protocol'] != 'stc_official_lossless_semantics_uint8_v1'
                or c['official_drive_id'] != '1dXB9mtROLWChycBZlhYIf_JBLshXogBs'
                or c['shape'] != [200, 200, 16] or c['masks_applied']
                or value['frames'] != len(value['files']) or value['frames'] != 6019 or value['scenes'] != 150):
            raise RuntimeError('STC extraction provenance/inventory differs')
        self.archive = Path(archive).resolve() if archive else None
        if self.archive and (self.archive.stat().st_size != 67779055 or file_sha256(self.archive) != c['archive_sha256']):
            raise RuntimeError('STC original ZIP SHA/size differs from extraction')
        self.zip = zipfile.ZipFile(self.archive) if self.archive else None
        self.checked = set()

    def check(self, scene, token, sem):
        key = scene+'/'+token
        if key in self.checked: return
        digest = hashlib.sha256(np.asarray(sem, np.uint8).tobytes()).hexdigest()
        if self.state['files'].get(key) != digest: raise RuntimeError('compact STC bytes differ from extraction: '+key)
        if self.zip:
            with np.load(io.BytesIO(self.zip.read('stc-results/'+key+'/labels.npz')), allow_pickle=False) as z:
                original = semantics(z['semantics'])
            if not np.array_equal(sem, original): raise RuntimeError('original ZIP/compact mismatch: '+key)
        self.checked.add(key)

    def verify(self):
        if file_sha256(self.path) != self.sha: raise RuntimeError('STC extraction receipt changed')
        if self.archive and file_sha256(self.archive) != self.state['contract']['archive_sha256']:
            raise RuntimeError('original ZIP changed')
        return dict(extraction_sha256=self.sha, selected_history_frames_checked=len(self.checked),
            archive_sha256=self.state['contract']['archive_sha256'], original_zip_independently_checked=bool(self.zip),
            caveat='byte/identity checks do NOT prove frontend quality or semantic-coordinate correctness')

    def close(self):
        if self.zip: self.zip.close()


def identity(source, w):
    nusc = source.source.nusc; tokens = (*w.history_tokens, *w.future_tokens)
    rows = [nusc.get('sample', t) for t in tokens]
    if (w.history_tokens[-1] != w.t0_token or len(set(tokens)) != len(tokens)
            or any(nusc.get('scene', r['scene_token'])['name'] != w.scene_name for r in rows)
            or any(a['next'] != b for a, b in zip(rows, tokens[1:]))):
        raise RuntimeError('STC/GT window metadata identity/chain differs')
    dt = np.diff([r['timestamp'] for r in rows])/1e6
    if np.any(dt < .25) or np.any(dt > .75): raise RuntimeError('history/future cadence corrupt')
    return dict(contiguous=True, interval_min_s=float(dt.min()), interval_max_s=float(dt.max()),
                history_tokens=list(w.history_tokens), t0=w.t0_token)


def input_confusions(camera, actual, previous, next_frame):
    if camera.shape[0] != camera.shape[1]: raise ValueError('XY orientation diagnostics require square lattice')
    result = {}
    for transpose in (False, True):
        base = camera.transpose(1, 0, 2) if transpose else camera
        for flipx, flipy in ((False, False), (True, False), (False, True), (True, True)):
            value = base[::-1] if flipx else base
            if flipy: value = value[:, ::-1]
            name = ('transposeXY' if transpose else 'identity')+('_flipX' if flipx else '')+('_flipY' if flipy else '')
            result[name] = confusion(value, actual).tolist()
    result['previous_GT_raw_not_pose_aligned'] = confusion(camera, previous).tolist()
    result['next_GT_raw_not_pose_aligned'] = confusion(camera, next_frame).tolist()
    return result
