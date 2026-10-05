"""Portable, bounded real-window replay. Future labels are never inference inputs.

Tensor-only .pt members use weights_only=True; checkpoints are copied separately
and are trusted repository artifacts, not arbitrary downloaded pickle objects.
No archive extraction, learned-feature cache, or resolution reduction is used.
"""
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
import torch

PROTOCOL = 'p0_f9_real_window_local_replay_v1'
MOTION_KEYS = ('features', 'local_semantic_tube', 'kta_displacement_xy_m',
               'frame_motion_features', 'target_source_mask_tube')
GEOMETRY_KEYS = ('anchors_xy_t0_m', 'source_class_id', 'source_centroid_xy_t0_m')
IDENTITY_KEYS = ('sample_id', 'scene_name', 't0_token', 'history_tokens', 'future_tokens')
LABEL_KEYS = ('supervised_source', 'se2_target_valid', 'target_source_residual_xy_m',
              'existence', 'target_yaw_rad', 'yaw_enabled', 'yaw_label_valid',
              'target_source_displacement_xy_m')
RAW_KEYS = ('history_occ', 'history_observed', 'history_poses', 'future_poses', 'trajectory')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''): h.update(block)
    return h.hexdigest()


def tensor(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    a = np.asarray(value)
    if a.dtype.hasobject: raise ValueError('object arrays cannot be exported')
    return torch.from_numpy(np.array(a, copy=True))


def pack_window(record, raw, moving):
    """Whitelist only. In particular target_source_mask_tube is t0 evidence,
    despite its legacy name; future target displacement/yaw live ONLY in labels.
    """
    inputs = {'identity': {k: record[k] for k in IDENTITY_KEYS},
              'record': {k: tensor(record[k]) for k in MOTION_KEYS+GEOMETRY_KEYS},
              'raw': {k: tensor(raw[k]) for k in RAW_KEYS}}
    # Serialize token metadata as ordinary Python strings/lists only.
    for k in IDENTITY_KEYS:
        v = inputs['identity'][k]
        inputs['identity'][k] = [str(x) for x in v] if k.endswith('_tokens') else str(v)
    labels = {'record': {k: tensor(record[k]) for k in LABEL_KEYS},
              'future_gt_occ': tensor(raw['future_gt_occ']), 'moving_support': tensor(moving)}
    return inputs, labels


def unpack_window(inputs, labels=None):
    """Without explicit labels, neither raw nor record contains future GT."""
    record = {**inputs['identity'], **inputs['record']}
    raw = {k: v.numpy() for k, v in inputs['raw'].items()}
    raw['future_gt_occ'] = None
    if labels is not None:
        record.update(labels['record'])
        raw['future_gt_occ'] = labels['future_gt_occ'].numpy()
    return record, raw


def validate_window(inputs, labels, grid_shape):
    if set(inputs) != {'identity', 'record', 'raw'} or set(labels) != {'record', 'future_gt_occ', 'moving_support'}:
        raise ValueError('replay input/label schema mismatch')
    if (set(inputs['identity']) != set(IDENTITY_KEYS) or set(inputs['record']) != set(MOTION_KEYS+GEOMETRY_KEYS)
            or set(inputs['raw']) != set(RAW_KEYS) or set(labels['record']) != set(LABEL_KEYS)):
        raise ValueError('unknown or missing replay fields; labels/features must not leak into inference')
    identity = inputs['identity']; r = inputs['record']; raw = inputs['raw']
    if any(not isinstance(identity[k], str) or not identity[k] for k in IDENTITY_KEYS[:3]):
        raise ValueError('invalid window identity')
    histories, futures = identity['history_tokens'], identity['future_tokens']
    if (len(histories) not in (4, 6) or len(futures) != 6 or histories[-1] != identity['t0_token']
            or any(not isinstance(x, str) or not x for x in (*histories, *futures))
            or len(set((*histories, *futures))) != len(histories)+len(futures)):
        raise ValueError('invalid history/future token order')
    for group in (r, raw, labels['record'], {k: labels[k] for k in ('future_gt_occ', 'moving_support')}):
        for k, value in group.items():
            if not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or value.requires_grad:
                raise ValueError('portable CPU tensors required: '+k)
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError('nonfinite tensor: '+k)
    shape = tuple(grid_shape)
    for k, group, count, dtype in (('history_occ', raw, 4, torch.uint8),
            ('history_observed', raw, 4, torch.bool), ('future_gt_occ', labels, 6, torch.uint8),
            ('moving_support', labels, 6, torch.bool)):
        if tuple(group[k].shape) != (count, *shape) or group[k].dtype != dtype:
            raise ValueError('full resolution four-history/six-future contract: '+k)
    if (torch.any(raw['history_occ'] > 17) or torch.any(labels['future_gt_occ'] > 17)):
        raise ValueError('invalid occupancy class')
    for k, count in (('history_poses', 4), ('future_poses', 6)):
        poses = raw[k]
        if poses.shape != (count, 4, 4) or not torch.allclose(poses[:, 3], poses.new_tensor([0, 0, 0, 1]).expand(count, -1)):
            raise ValueError('invalid SE3 pose: '+k)
        rot = poses[:, :3, :3]
        if (not torch.allclose(rot.transpose(1, 2)@rot, torch.eye(3, dtype=rot.dtype).expand(count, -1, -1), atol=1e-5, rtol=0)
                or not torch.allclose(torch.linalg.det(rot), rot.new_ones(count), atol=1e-5, rtol=0)):
            raise ValueError('nonrigid pose: '+k)
    if raw['trajectory'].shape != (12, 2) or torch.any(raw['trajectory'][:2] != 0):
        raise ValueError('official trajectory contract mismatch')
    n = len(r['features'])
    if (r['source_class_id'].shape != (n,) or r['source_centroid_xy_t0_m'].shape != (n, 2)
            or r['anchors_xy_t0_m'].shape != (n, 6, 2) or r['kta_displacement_xy_m'].shape != (n, 6, 2)
            or any(len(r[k]) != n for k in MOTION_KEYS) or any(len(labels['record'][k]) != n for k in LABEL_KEYS)):
        raise ValueError('source/input/target population mismatch')
    if not torch.allclose(r['anchors_xy_t0_m'].float(), r['source_centroid_xy_t0_m'].float()[:, None]+r['kta_displacement_xy_m'].float(), atol=2e-4, rtol=0):
        raise ValueError('KTA anchor identity mismatch')


def select_records(records, count=16, stress=4):
    """Causal identity-only representative subset + highest source-count tail.

    Stress windows are NOT representative throughput or quality samples. No GT
    positives, future labels, errors or validation scores enter selection.
    """
    if not 0 <= stress < count <= len(records): raise ValueError('invalid selection budgets')
    keyed = {(str(r['scene_name']), str(r['t0_token'])): r for r in records}
    if len(keyed) != len(records): raise ValueError('duplicate record identities')
    order = sorted(keyed, key=lambda k: (-len(keyed[k]['features']), k))
    heavy = order[:stress]
    groups = defaultdict(list)
    for key in sorted(keyed):
        if key not in heavy: groups[key[0]].append(key)
    # Uniform temporal positions within each scene; then scene round robin.
    for scene, keys in groups.items():
        ranks = np.linspace(0, len(keys)-1, min(len(keys), count-stress)).round().astype(int)
        groups[scene] = [keys[i] for i in ranks]
    normal = []
    for rank in range(count-stress):
        for scene in sorted(groups):
            if rank < len(groups[scene]): normal.append(groups[scene][rank])
            if len(normal) == count-stress: break
        if len(normal) == count-stress: break
    return [(keyed[k], 'representative') for k in normal]+[(keyed[k], 'high_source_stress') for k in heavy]


def safe_member(name):
    p = PurePosixPath(name)
    if '\\' in name or ':' in name or p.is_absolute() or '..' in p.parts or p.as_posix() != name:
        raise ValueError('unsafe archive member: '+name)
    return name


class BundleWriter:
    def __init__(self, path, max_bytes=4*2**30):
        self.path = Path(path); self.limit = max_bytes; self.members = {}; self.total = 0
        self.zip = zipfile.ZipFile(self.path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True)

    def add_stream(self, name, source):
        safe_member(name)
        if name in self.members or name == 'manifest.json': raise ValueError('duplicate/reserved member')
        h = hashlib.sha256(); size = 0
        with self.zip.open(name, 'w', force_zip64=True) as target:
            for block in iter(lambda: source.read(2**20), b''):
                size += len(block); self.total += len(block)
                if self.total > self.limit: raise ValueError('replay uncompressed size limit exceeded')
                h.update(block); target.write(block)
        self.members[name] = {'sha256': h.hexdigest(), 'bytes': size}

    def add_file(self, name, path):
        with Path(path).open('rb') as f: self.add_stream(name, f)

    def add_tensors(self, name, value):
        stream = io.BytesIO(); torch.save(value, stream); stream.seek(0); self.add_stream(name, stream)

    def finish(self, metadata):
        if set(metadata) & {'protocol', 'members', 'manifest_fingerprint'}:
            raise ValueError('reserved manifest metadata key')
        manifest = {**metadata, 'protocol': PROTOCOL, 'members': self.members}
        manifest['manifest_fingerprint'] = fingerprint(manifest)
        self.zip.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2))
        self.zip.close()
        return manifest

    def close(self): self.zip.close()


class ReplayBundle:
    def __init__(self, path, max_bytes=4*2**30):
        self.zip = zipfile.ZipFile(path); self.max_bytes = max_bytes
        try:
            infos = self.zip.infolist(); names = [safe_member(x.filename) for x in infos]
            if len(names) != len(set(names)) or sum(x.file_size for x in infos) > max_bytes:
                raise ValueError('duplicate members or oversized replay package')
            if 'manifest.json' not in names or self.zip.getinfo('manifest.json').file_size > 2**20:
                raise ValueError('missing/oversized replay manifest')
            self.manifest = json.loads(self.zip.read('manifest.json'))
            declared = self.manifest.copy(); digest = declared.pop('manifest_fingerprint', None)
            if self.manifest.get('protocol') != PROTOCOL or fingerprint(declared) != digest:
                raise ValueError('replay manifest fingerprint mismatch')
            if set(names) != {'manifest.json', *self.manifest['members']}:
                raise ValueError('undeclared/missing archive members')
            self.validate_members()
            rows = self.manifest['windows']; keys = [tuple(row['key']) for row in rows]
            if (not rows or len(keys) != len(set(keys)) or self.manifest.get('active_history_frames') != 4
                    or self.manifest.get('future_frames') != 6): raise ValueError('invalid replay population')
        except BaseException:
            self.zip.close(); raise

    def validate_members(self):
        for name, expected in self.manifest['members'].items():
            h = hashlib.sha256(); size = 0
            with self.zip.open(name) as f:
                for block in iter(lambda: f.read(2**20), b''): h.update(block); size += len(block)
            if size != expected['bytes'] or h.hexdigest() != expected['sha256']:
                raise ValueError('replay member fingerprint mismatch: '+name)

    def window(self, index, *, labels=False):
        row = self.manifest['windows'][index]
        inputs = torch.load(io.BytesIO(self.zip.read(row['inputs'])), map_location='cpu', weights_only=True)
        targets = torch.load(io.BytesIO(self.zip.read(row['labels'])), map_location='cpu', weights_only=True)
        validate_window(inputs, targets, self.manifest['grid_shape'])
        if tuple(row['key']) != (inputs['identity']['scene_name'], inputs['identity']['t0_token']):
            raise ValueError('manifest/window identity order mismatch')
        record, raw = unpack_window(inputs, targets if labels else None)
        return record, raw, targets if labels else None

    def copy_member(self, name, destination):
        """NEW local snapshots only; never write server paths from metadata."""
        if name not in self.manifest['members']: raise ValueError('undeclared member')
        with self.zip.open(name) as source, Path(destination).open('xb') as target:
            for block in iter(lambda: source.read(2**20), b''): target.write(block)

    def close(self): self.zip.close()
