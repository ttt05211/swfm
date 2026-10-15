"""Read portable replay data without nuScenes metadata or filesystem extraction."""
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import torch
from real_motion.prepared import PrepareConfig
from real_motion.geometry import OccupancyGrid
from real_motion.motion import PersistenceMotionConfig
from real_motion.kta import KTAConfig
from tools.dense_checkpoint_average.weights import model_from_spec
from .export import PROTOCOL


def prepare_config(mapping):
    values = dict(mapping)
    convert = lambda d: {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}
    values['grid'] = OccupancyGrid(**convert(values['grid']))
    values['motion'] = PersistenceMotionConfig(**convert(values['motion']))
    values['kta'] = KTAConfig(**convert(values['kta']))
    return PrepareConfig(**convert(values))


class Bundle:
    def __init__(self, path, cache_mib=256):
        self.path = Path(path).resolve(); self.zip = zipfile.ZipFile(self.path)
        if len(self.zip.namelist()) != len(set(self.zip.namelist())): raise ValueError('duplicate archive member')
        self.manifest = json.loads(self.zip.read('manifest.json'))
        if self.manifest['protocol'] != PROTOCOL: raise ValueError('unsupported portable bundle')
        self.pcfg = prepare_config(self.manifest['prepare_config'])
        self.inventory = {r['file']: r for r in self.manifest['inventory']}
        self.records = self.manifest['records']; self.cache = OrderedDict(); self.bytes = 0
        self.limit = int(cache_mib*2**20)
        keys = [(r['scene_name'], r['t0_token']) for r in self.records]
        if len(keys) != len(set(keys)): raise ValueError('duplicate replay window')
        self.by_key = dict(zip(keys, self.records))

    def close(self): self.zip.close(); self.cache.clear()

    def frame(self, kind, scene, token):
        name = f'frames/{kind}/{scene}/{token}/labels.npz'
        if name in self.cache: self.cache.move_to_end(name); return self.cache[name]
        payload = self.zip.read(name); info = self.inventory[name]
        if len(payload) != info['size'] or hashlib.sha256(payload).hexdigest() != info['sha256']:
            raise RuntimeError('portable frame digest differs: '+name)
        with np.load(io.BytesIO(payload), allow_pickle=False) as z:
            sem = np.array(z['semantics'], copy=True)
            if sem.dtype != np.uint8 or sem.shape != tuple(self.pcfg.grid.shape_hwd):
                raise ValueError('portable semantic grid/dtype differs: '+name)
            known = (sem < 18)
            lidar = None
            if 'mask_lidar_packed' in z:
                shape = tuple(map(int, z['mask_lidar_shape']))
                if shape != sem.shape: raise ValueError('portable mask grid differs')
                lidar = np.unpackbits(z['mask_lidar_packed'])[:sem.size].reshape(shape).astype(bool)
            elif 'mask_lidar' in z: lidar = np.array(z['mask_lidar'], bool, copy=True)
        value = dict(semantics=sem, dense_valid=known, lidar_known=lidar)
        size = sum(a.nbytes for a in value.values() if isinstance(a, np.ndarray))
        if size <= self.limit:
            while self.cache and self.bytes+size > self.limit:
                _, old = self.cache.popitem(last=False)
                self.bytes -= sum(a.nbytes for a in old.values() if isinstance(a, np.ndarray))
            self.cache[name] = value; self.bytes += size
        return value

    def history(self, row, kind='stc', policy='dense_valid'):
        if kind not in ('stc', 'gt') or policy not in ('dense_valid', 'lidar_known'): raise ValueError('history policy')
        if kind == 'stc' and policy != 'dense_valid': raise ValueError('STC cannot consume GT visibility')
        values = [self.frame(kind, row['scene_name'], t) for t in row['history_tokens']]
        masks = [v[policy] for v in values]
        if any(v is None for v in masks): raise ValueError('requested lidar history not exported')
        raw = dict(history_occ=np.stack([v['semantics'] for v in values]), history_observed=np.stack(masks),
                   history_poses=np.asarray(row['history_poses'], np.float64))
        rec = {k: row[k] for k in ('scene_name', 't0_token', 'history_tokens', 'future_tokens')}
        return rec, raw, np.asarray(row['history_times_s'], np.float64)

    def targets(self, row, indices=(1, 3, 5)):
        return np.stack([self.frame('gt', row['scene_name'], row['future_tokens'][h])['semantics'] for h in indices])

    def load_model(self, role, device='cpu'):
        matches = [v for v in self.manifest['weights'] if v['role'] == role]
        if len(matches) != 1: raise ValueError('missing or duplicate exported model role: '+role)
        value = matches[0]; payload = self.zip.read(value['file'])
        if hashlib.sha256(payload).hexdigest() != value['sha256']: raise RuntimeError('model archive SHA differs')
        saved = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
        if saved['protocol'] != PROTOCOL or saved.get('optimizer_included') is not False:
            raise ValueError('portable frozen body expected')
        body = model_from_spec(saved['model_configs']); body.load_state_dict(saved['state_dict'], strict=True)
        policy = saved['source_contract'].get('source_policy', 'baseline')
        if policy not in ('baseline', 'shared_pinned'): raise ValueError('unsupported source identity policy')
        return body.to(device).eval().requires_grad_(False), saved, policy
