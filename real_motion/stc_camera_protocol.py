"""Four-setting frozen STC evaluation: identity-addressed inputs, no GT masks.

Old COME records supply ONLY keys and predicted ego poses. Camera histories
are loaded independently from STCOcc-Res, never from those BEVStereo records.
"""
from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np

from real_motion.geometry import pose_matrix
from real_motion.waymo_i2world import SHAPE, file_sha256, fingerprint

PROTOCOL = 'p0_f9_surface_stc_four_settings_frozen_v1'
SETTINGS = ('occ_gt', 'occ_pred', 'stc_gt', 'stc_pred')
PLAN_TAG = 'come_main_table_v4_source_camera_masked_official_plan_yaw'
PLAN_SHA = '19c04eaf37f531148d5b5719e3cabf8caa35a7140bba9527df47c79aa6783afb'
UPSTREAM = 'https://github.com/lzzzzzm/II-World/tree/661d830f9b34ee03ce368db164a72753ab8764a3'


def validate_pose(value):
    a = np.array(value, dtype=np.float64, copy=True)
    if (a.shape != (4, 4) or not np.isfinite(a).all()
            or not np.allclose(a[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(a[:3, :3].T @ a[:3, :3], np.eye(3), atol=1e-5, rtol=0)
            or not np.isclose(np.linalg.det(a[:3, :3]), 1, atol=1e-5, rtol=0)):
        raise ValueError('invalid rigid ego-to-world pose')
    a.setflags(write=False)
    return a


def semantics(value, shape=SHAPE):
    a = np.asarray(value)
    if (a.shape != tuple(shape) or a.dtype.kind not in 'ui' or a.size == 0
            or a.min() < 0 or a.max() > 17):
        raise ValueError('STC/Occ3D requires XYZ integer semantics 0..17; no inferred remapping')
    a = np.array(a, dtype=np.uint8, copy=True, order='C')
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class Frame:
    token: str
    scene: str
    prev: str
    next: str
    timestamp: int
    pose: np.ndarray


@dataclass(frozen=True)
class Window:
    key: str
    scene: str
    t0: str
    history: tuple
    future: tuple


def load_catalog(dataroot, allowed_scenes):
    """Read standard nuScenes tables, NOT annotation tables or GT semantics.

    LIDAR_TOP's measured ego pose matches the existing Occ3D/source convention.
    No LiDAR occupancy/visibility is passed to the STC path.
    """
    base = Path(dataroot) / 'v1.0-trainval'
    provenance = []
    def read(name):
        p = base / (name + '.json'); sha = file_sha256(p)
        value = json.loads(p.read_text(encoding='utf-8'))
        if file_sha256(p) != sha:
            raise RuntimeError('nuScenes metadata changed during read')
        provenance.append(dict(path=str(p.resolve()), sha256=sha))
        return value
    scenes = {r['token']: r['name'] for r in read('scene') if r['name'] in allowed_scenes}
    samples = {r['token']: r for r in read('sample') if r['scene_token'] in scenes}
    sensors = {r['token'] for r in read('sensor') if r['channel'] == 'LIDAR_TOP'}
    calibrated = {r['token'] for r in read('calibrated_sensor') if r['sensor_token'] in sensors}
    sd = {}
    for r in read('sample_data'):
        if r['sample_token'] in samples and r['is_key_frame'] and r['calibrated_sensor_token'] in calibrated:
            if r['sample_token'] in sd:
                raise ValueError('duplicate LIDAR_TOP keyframe pose')
            sd[r['sample_token']] = r['ego_pose_token']
    needed = set(sd.values())
    poses = {r['token']: validate_pose(pose_matrix(r['translation'], r['rotation']))
             for r in read('ego_pose') if r['token'] in needed}
    catalog = {t: Frame(t, scenes[r['scene_token']], r['prev'], r['next'],
                         int(r['timestamp']), poses[sd[t]]) for t, r in samples.items()}
    return catalog, provenance


class STCFourSettingSource:
    def __init__(self, dataroot, stc_root, plan_cache, catalog, *, shape=SHAPE, cache_mib=512):
        self.root, self.stc_root, self.plan_cache = map(lambda p: Path(p).resolve(),
                                                     (dataroot, stc_root, plan_cache))
        self.shape, self.catalog = tuple(shape), catalog
        if not 0 <= cache_mib <= 4096:
            raise ValueError('invalid bounded frame RAM budget')
        manifest_path = self.plan_cache / 'manifest.json'
        self.manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        m = self.manifest
        if (m.get('format') != 'stochocc_transport_cache_v2'
                or m.get('come_protocol_setting') != 'camera_pred'
                or m.get('come_protocol_tag') != PLAN_TAG or m.get('future_ego') != 'predicted'
                or m.get('come_trajectory_sha256') != PLAN_SHA
                or m.get('predicted_yaw_semantics') != 'each_future_json_yaw_offset_relative_to_current_pose_non_cumulative'
                or m.get('hist_last') != 4):
            raise ValueError('trusted exact v4 camera_pred planner cache required; no GT fallback')
        keys = m.get('keys', [])
        if not keys or len(keys) != len(set(keys)) or len(keys) != m.get('num_windows'):
            raise ValueError('missing/duplicate planner population keys')
        self.windows = []
        for key in keys:
            if not re.fullmatch(r'[A-Za-z0-9_-]+__[A-Za-z0-9_-]+', key):
                raise ValueError('unsafe planner key')
            scene, t0 = key.split('__')
            if t0 not in catalog or catalog[t0].scene != scene:
                raise ValueError('planner identity missing from nuScenes: ' + key)
            hist = [t0]
            for _ in range(3):
                hist.insert(0, catalog[hist[0]].prev)
            future = []; t = t0
            for _ in range(6):
                t = catalog[t].next
                if not t:
                    raise ValueError('planner population has short future: ' + key)
                future.append(t)
            tokens = hist + future
            if (len(set(tokens)) != 10 or any(t not in catalog or catalog[t].scene != scene for t in tokens)):
                raise ValueError('incomplete/cross-scene window: ' + key)
            dt = np.diff([catalog[t].timestamp for t in tokens]) / 1e6
            if np.any(dt <= 0) or np.any(dt < .25) or np.any(dt > .75):
                raise ValueError('noncontiguous/wrong-cadence planner window: ' + key)
            self.windows.append(Window(key, scene, t0, tuple(hist), tuple(future)))
        self.by_key = {(w.scene, w.t0): w for w in self.windows}
        self.cache = OrderedDict(); self.cache_bytes = 0; self.cache_limit = int(cache_mib * 1024**2)
        self.inventory = {}; self.plans = {}; self.io_reads = self.cache_hits = 0
        self.metadata = dict(protocol=PROTOCOL, front_end='STCOcc-Res official release',
            planner='COME public BEV-Planner + yaw; externally predicted, not our ego head',
            planner_original_sha256=PLAN_SHA, manifest_sha256=file_sha256(manifest_path),
            declared_windows=len(self.windows), declared_scenes=m.get('num_scenes'),
            history_frames=4, future_frames=6, history_STC_visibility='dense prediction grid valid; NOT measured visibility',
            scoring_mask='none; use_image_mask=False/use_lidar_mask=False',
            population_note='same legacy planner-covered 6+6-start subset for ALL settings; not claimed identical to paper population',
            future_pose_convention='ego-to-global XYZ; cached independent t0-relative yaw; no GT z/tilt completion',
            future_target_frame='original GT ego frame; no post-hoc GT pose alignment',
            BEVStereo_history_used=False, learned_geometry_cache_used=False)

    @classmethod
    def from_files(cls, dataroot, stc_root, plan_cache, **kwargs):
        m = json.loads((Path(plan_cache) / 'manifest.json').read_text())
        scenes = {k.split('__')[0] for k in m.get('keys', [])}
        catalog, provenance = load_catalog(dataroot, scenes)
        source = cls(dataroot, stc_root, plan_cache, catalog, **kwargs)
        source.metadata['pose_source_files'] = provenance
        compact = Path(stc_root) / 'extraction_state.json'
        source.metadata['STC_release'] = (dict(path=str(compact.resolve()), sha256=file_sha256(compact))
                                         if compact.is_file() else dict(path=str(Path(stc_root).resolve()), supplied_directory=True))
        return source

    def select(self, population, parent_keys=None):
        if population == 'all':
            selected = list(self.windows); missing = []
        else:
            if population not in ('dev64', 'dev512') or not parent_keys:
                raise ValueError('dev population requires frozen parent_keys')
            keys = tuple(tuple(k) for k in parent_keys)
            if len(keys) != len(set(keys)):
                raise ValueError('duplicate frozen parent keys')
            missing = [list(k) for k in keys if k not in self.by_key]
            common = [k for k in keys if k in self.by_key]
            if population == 'dev64':
                from real_motion.v21_source_induction import select_scene_balanced_round_robin
                if len(common) < 64:
                    raise ValueError('fewer than 64 planner-covered parent windows')
                common = select_scene_balanced_round_robin(common, 64)
            selected = [self.by_key[k] for k in common]
        if not selected:
            raise ValueError('empty shared population')
        audit = dict(population=population, windows=len(selected), scenes=len({w.scene for w in selected}),
            missing_parent_keys=missing, selected_keys=[[w.scene, w.t0] for w in selected],
            selection='explicit planner-covered parent intersection; scene-balanced dev64, no quality selection')
        return selected, audit

    def _path(self, kind, scene, token):
        return (self.stc_root if kind == 'stc' else self.root / 'gts') / scene / token / 'labels.npz'

    def _stat(self, path):
        path = Path(path)
        st = path.stat()
        value = [st.st_size, st.st_mtime_ns]
        if not path.is_file() or not st.st_size:
            raise ValueError('empty/non-file input: ' + str(path))
        if str(path) in self.inventory and self.inventory[str(path)] != value:
            raise RuntimeError('data changed since preflight: ' + str(path))
        return value

    def preflight(self, selected):
        """Check all required files; missing STC is an ERROR, never GT substitution."""
        paths = set()
        for w in selected:
            paths.add(self.plan_cache / 'records' / (w.key + '.npz'))
            for t in w.history:
                for kind in ('stc', 'occ'):
                    paths.add(self._path(kind, w.scene, t))
            for t in w.future:
                paths.add(self._path('occ', w.scene, t))
        for p in sorted(paths):
            self.inventory[str(p)] = self._stat(p)
        for w in selected:
            p = self.plan_cache / 'records' / (w.key + '.npz')
            # These are the ONLY members read from legacy BEVStereo archives.
            with np.load(p, allow_pickle=False) as z:
                if (str(z['format'].item()) != 'stochocc_transport_cache_v2'
                        or str(z['protocol_tag'].item()) != PLAN_TAG or str(z['scene'].item()) != w.scene
                        or tuple(z['history_tokens'].tolist()) != w.history
                        or tuple(z['future_tokens'].tolist()) != w.future):
                    raise ValueError('planner record identity/order/tag mismatch: ' + w.key)
                poses = z['future_e2g']
                if poses.shape != (6, 4, 4):
                    raise ValueError('planner must provide six future poses')
                validated = tuple(validate_pose(pose) for pose in poses)
                current = self.catalog[w.t0].pose
                for pose in validated:
                    rel = current[:3, :3].T @ pose[:3, :3]
                    if (not np.isclose(pose[2, 3], current[2, 3], atol=1e-6, rtol=0)
                            or not np.allclose(rel[:, 2], [0, 0, 1], atol=1e-6, rtol=0)):
                        raise ValueError('cached planner z/tilt differs from t0-only SE2 construction: ' + w.key)
                self.plans[w.key] = validated
        return dict(files=len(paths), file_stat_fingerprint=fingerprint(self.inventory),
            selected_inputs_fingerprint=fingerprint([[w.key, list(w.history), list(w.future),
                [p.tolist() for p in self.plans[w.key]]] for w in selected]),
            integrity_scope='metadata SHA256 + exact pose/identity fingerprint; NPZ size/mtime on every read')

    def frame(self, kind, scene, token):
        key = (kind, scene, token); path = self._path(*key)
        self._stat(path)
        if key in self.cache:
            self.cache_hits += 1; self.cache.move_to_end(key)
            return self.cache[key]
        with np.load(path, allow_pickle=False) as z:
            sem = semantics(z['semantics'], self.shape)
            # The GT occupancy baseline retains its original historical lidar
            # evidence. STC never reads ANY GT visibility, even if in its NPZ.
            obs = np.ones(self.shape, bool) if kind == 'stc' else np.array(z['mask_lidar'], bool, copy=True)
            if obs.shape != self.shape:
                raise ValueError('wrong historical mask shape')
        obs.setflags(write=False); value = (sem, obs); self.io_reads += 1
        size = sem.nbytes + obs.nbytes
        if size <= self.cache_limit:
            while self.cache_bytes + size > self.cache_limit and self.cache:
                _, old = self.cache.popitem(last=False); self.cache_bytes -= sum(x.nbytes for x in old)
            self.cache[key] = value; self.cache_bytes += size
        return value

    def prediction_inputs(self, w, setting):
        if setting not in SETTINGS or w.key not in self.plans:
            raise ValueError('unknown setting or missing preflight')
        self._stat(self.plan_cache / 'records' / (w.key + '.npz'))
        kind = 'stc' if setting.startswith('stc') else 'occ'
        loaded = [self.frame(kind, w.scene, t) for t in w.history]
        future = (self.plans[w.key] if setting.endswith('pred') else
                  tuple(self.catalog[t].pose for t in w.future))
        raw = dict(history_occ=np.stack([a for a, _ in loaded]),
            history_observed=np.stack([b for _, b in loaded]),
            history_poses=[self.catalog[t].pose for t in w.history], future_poses=list(future), future_gt_occ=None)
        record = dict(scene_name=w.scene, t0_token=w.t0, history_tokens=w.history, future_tokens=w.future)
        return record, raw

    def metric_targets(self, w):
        result = []
        for h in (1, 3, 5):
            path = self._path('occ', w.scene, w.future[h]); self._stat(path)
            with np.load(path, allow_pickle=False) as z:
                result.append(semantics(z['semantics'], self.shape))
        return result
