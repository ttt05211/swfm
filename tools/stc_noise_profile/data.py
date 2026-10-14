"""Read selected same-frame STC/GT grids and historical metadata, without Torch."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from .statistics import SHAPE


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def quaternion_matrix(value):
    q = np.asarray(value, float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.dot(q, q) < 1e-16: raise ValueError('invalid quaternion')
    w, x, y, z = q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def pose(record):
    out = np.eye(4); out[:3, :3] = quaternion_matrix(record['rotation']); out[:3, 3] = record['translation']
    if not np.isfinite(out).all(): raise ValueError('invalid pose')
    return out


def receipt(root):
    path = Path(root)/'extraction_state.json'
    v = json.loads(path.read_text(encoding='utf-8')); declared = v.pop('fingerprint')
    c = v['contract']
    if (declared != fingerprint(v) or v['status'] != 'complete' or v['frames'] != len(v['files'])
            or v['frames'] != 6019 or v['scenes'] != 150 or c['shape'] != list(SHAPE)
            or c['protocol'] != 'stc_official_lossless_semantics_uint8_v1' or c['masks_applied']
            or c['official_drive_id'] != '1dXB9mtROLWChycBZlhYIf_JBLshXogBs'):
        raise ValueError('original compact STC extraction provenance required')
    scenes = {key.split('/')[0] for key in v['files']}
    if len(scenes) != 150: raise ValueError('STC scene inventory differs')
    return v, file_sha(path), sorted(scenes)


def split(prior, all_scenes):
    c = prior.get('contract', {}); keys = c.get('keys', [])
    if (prior.get('status') != 'complete' or prior.get('windows') != 64 or prior.get('source_epoch') != 2
            or prior.get('source_update') != 10292 or prior.get('scientific_updates') != 0
            or c.get('protocol') != 'frozen_final_prior_strength_DEV64_c1_c05_v1'
            or len(keys) != 64 or len({tuple(k) for k in keys}) != 64):
        raise ValueError('completed original final-prior DEV64 result required to freeze fitting scenes')
    fit = sorted({k[0] for k in keys}); available = set(all_scenes)
    if not set(fit) <= available or set(fit) == available: raise ValueError('fit/remaining scene split invalid')
    return dict(fit_scenes=fit, remaining_scenes=sorted(available-set(fit)), seed_DEV64_keys=keys,
        fitting_uses_all_released_frames_in_fit_scenes=True,
        remaining_scene_labels_read=False, remaining_is_pristine_test=False,
        disclosure='Same-frame VAL GT labels calibrate error statistics. Remaining scenes are excluded from THIS fit, '
                   'but earlier full VAL results were already inspected; NOT a new independent test.')


@dataclass(frozen=True)
class Frame:
    scene: str
    token: str
    timestamp: int
    prev: str
    next: str
    pose: np.ndarray


class Source:
    def __init__(self, dataroot, stc_root, development_split, extraction):
        self.root, self.stc = Path(dataroot).resolve(), Path(stc_root).resolve()
        self.fit = set(development_split['fit_scenes']); self.extraction = extraction
        self.metadata = []; base = self.root/'v1.0-trainval'
        def read(name):
            p = base/(name+'.json'); sha = file_sha(p)
            values = json.loads(p.read_text(encoding='utf-8'))
            if file_sha(p) != sha: raise RuntimeError('metadata changed during read')
            self.metadata.append(dict(path=str(p), sha256=sha)); return values
        scenes = {s['token']: s for s in read('scene') if s['name'] in self.fit}
        if {s['name'] for s in scenes.values()} != self.fit: raise ValueError('fit scene metadata missing')
        samples = {s['token']: s for s in read('sample') if s['scene_token'] in scenes}
        sensors = {s['token'] for s in read('sensor') if s['channel'] == 'LIDAR_TOP'}
        calibrated = {c['token'] for c in read('calibrated_sensor') if c['sensor_token'] in sensors}
        sample_poses = {}
        for sd in read('sample_data'):
            if sd['sample_token'] in samples and sd['is_key_frame'] and sd['calibrated_sensor_token'] in calibrated:
                if sd['sample_token'] in sample_poses: raise ValueError('duplicate keyframe pose')
                sample_poses[sd['sample_token']] = sd['ego_pose_token']
        needed = set(sample_poses.values()); poses = {r['token']: pose(r) for r in read('ego_pose') if r['token'] in needed}
        self.frames = {}
        for s in scenes.values():
            values, token, seen = [], s['first_sample_token'], set()
            while token:
                if token in seen or token not in samples: raise ValueError('cyclic/missing sample chain')
                seen.add(token); r = samples[token]
                if r['scene_token'] != s['token']: raise ValueError('cross-scene sample chain')
                values.append(Frame(s['name'], token, int(r['timestamp']), r['prev'], r['next'], poses[sample_poses[token]]))
                token = r['next']
            if len(values) < 4 or any(b.prev != a.token or not .25 <= (b.timestamp-a.timestamp)/1e6 <= .75
                                       for a, b in zip(values, values[1:])): raise ValueError('invalid fit-scene cadence/chain')
            self.frames[s['name']] = values
        categories = {c['token']: c['name'] for c in read('category')}
        instances = {i['token']: categories[i['category_token']] for i in read('instance')}
        mapping = (('vehicle.bicycle', 2), ('vehicle.bus', 3), ('vehicle.car', 4),
                   ('vehicle.construction', 5), ('vehicle.motorcycle', 6), ('human.pedestrian', 7),
                   ('vehicle.trailer', 9), ('vehicle.truck', 10))
        self.annotations = {t: [] for t in samples}
        for a in read('sample_annotation'):
            if a['sample_token'] not in samples: continue
            cid = next((cid for prefix, cid in mapping if instances[a['instance_token']].startswith(prefix)), None)
            if cid is None: continue
            w, length, height = a['size']
            if min(w, length, height) <= 0: raise ValueError('invalid annotation size')
            self.annotations[a['sample_token']].append(dict(instance_token=a['instance_token'], class_id=cid,
                translation=a['translation'], rotation_matrix=quaternion_matrix(a['rotation']), size_lwh=(length, w, height)))
        # Never stat/open any remaining-scene label file, even on preflight.
        self.inventory = []
        for scene in sorted(self.frames):
            expected = {k.split('/')[1] for k in extraction['files'] if k.split('/')[0] == scene}
            if expected != {f.token for f in self.frames[scene]}: raise ValueError('complete fit-scene STC frame coverage required')
            for frame in self.frames[scene]:
                for path in self.paths(frame):
                    st = path.stat()
                    if not path.is_file() or not st.st_size: raise ValueError('missing/empty selected label')
                    self.inventory.append([str(path), st.st_size, st.st_mtime_ns])
        self.by_path = {row[0]: tuple(row[1:]) for row in self.inventory}

    def paths(self, frame):
        if frame.scene not in self.fit: raise RuntimeError('remaining-scene labels MUST NOT be read')
        return (self.root/'gts'/frame.scene/frame.token/'labels.npz', self.stc/frame.scene/frame.token/'labels.npz')

    def check_inventory(self):
        for path, size, timestamp in self.inventory:
            s = Path(path).stat()
            if (s.st_size, s.st_mtime_ns) != (size, timestamp): raise RuntimeError('fit label files changed')

    def read(self, frame):
        paths = self.paths(frame)
        for path in paths:
            st = path.stat()
            if (st.st_size, st.st_mtime_ns) != self.by_path[str(path)]: raise RuntimeError('fit label changed before read')
        with np.load(paths[0], allow_pickle=False) as z:
            gt, known = np.asarray(z['semantics']), np.asarray(z['mask_lidar'], bool)
        with np.load(paths[1], allow_pickle=False) as z: pred = np.asarray(z['semantics'])
        if (gt.shape != SHAPE or pred.shape != SHAPE or known.shape != SHAPE
                or any(a.dtype.kind not in 'ui' or (a < 0).any() or (a > 17).any() for a in (gt, pred))): raise ValueError('unmapped XYZ grids 0..17 required')
        gt, pred = gt.astype(np.uint8), pred.astype(np.uint8)
        if hashlib.sha256(pred.tobytes()).hexdigest() != self.extraction['files'][frame.scene+'/'+frame.token]:
            raise RuntimeError('compact STC differs from extraction receipt')
        for path in paths:
            st = path.stat()
            if (st.st_size, st.st_mtime_ns) != self.by_path[str(path)]: raise RuntimeError('fit label changed during read')
        return dict(gt=gt, pred=pred, known=known, pose=frame.pose,
                    annotations=self.annotations[frame.token],
                    GT_semantics_sha256=hashlib.sha256(gt.tobytes()).hexdigest(),
                    GT_known_sha256=hashlib.sha256(known.tobytes()).hexdigest())
