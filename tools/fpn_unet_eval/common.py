"""Frozen source selection, non-score-based population, dense semantics only."""
import json
from pathlib import Path
import numpy as np
import torch
from real_motion.stc_camera_protocol import semantics
from tools.joint_occ_experiments import run as base
from tools.fpn_unet.model import checkpoint_model, VARIANT
from tools.fpn_unet.run import PROTOCOL as TRAIN_PROTOCOL
from tools.fpn_unet.adapter import forward_window
from tools.source_scene_stage2.adapter import DirectPredictor
from tools.final_semantic_evaluation.sources import NuscInputs, window
from tools.final_semantic_evaluation import metrics


def select_source(explicit, search):
    def valid(folder):
        p = folder/'results.json'
        value = json.loads(p.read_text(encoding='utf-8'))
        return (value.get('status') == 'complete' and value.get('contract', {}).get('protocol') == TRAIN_PROTOCOL
            and value.get('epoch') == 1 and value.get('updates') == value['contract']['total_updates']
            and (folder/'epoch_0001.pt').is_file())
    if explicit:
        folder = Path(explicit).resolve()
        if not folder.is_dir() or not valid(folder): raise RuntimeError('completed FPN+UNet TRAIN20%x1 directory required')
        return folder/'epoch_0001.pt'
    choices = []
    for path in Path(search).glob('fpn_unet_20pct_1ep_*/results.json'):
        try:
            if valid(path.parent): choices.append(path.parent.resolve())
        except (ValueError, KeyError): continue
    if len(choices) != 1: raise RuntimeError('set FPN_UNET_DIR; completed candidates='+str(choices))
    return choices[0]/'epoch_0001.pt'


def validate_source(saved):
    c = saved['contract']
    if (saved.get('protocol') != TRAIN_PROTOCOL or c.get('protocol') != TRAIN_PROTOCOL or
            saved.get('epoch') != 1 or saved.get('batch') != 0 or saved.get('updates') != c.get('total_updates') or
            c.get('epochs') != 1 or c.get('model', {}).get('variant') != VARIANT or
            c.get('known') != 'TRAIN_OCC_STC_dense_valid' or c.get('registration') != 'none' or
            c.get('source_policy') != 'shared_pinned' or
            c.get('objective') != dict(CE=1., Lovasz=1., motion=0., existence=0., ego=0.) or
            len(c.get('train_keys', [])) != 4086 or len(set(map(tuple, c['train_keys']))) != 4086 or
            len(c.get('eval_keys', [])) != 64 or len(set(map(tuple, c['eval_keys']))) != 64 or
            set(saved.get('models', {})) != {'gt'}):
        raise RuntimeError('fixed final FPN+UNet TRAIN4086x1 checkpoint required; no old/interim fallback')
    return c


def load_frozen(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    contract = validate_source(saved)
    return contract, checkpoint_model(saved, device).requires_grad_(False)


def select_jobs(records, original64, train_keys):
    keys = base.record_keys(records)
    if len(keys) != 4369 or len(set(keys)) != 4369 or len({k[0] for k in keys}) != 150:
        raise RuntimeError('complete unique VAL4369 / 150 scenes required')
    original64 = [tuple(map(str, k)) for k in original64]
    if len(original64) != 64 or len(set(original64)) != 64 or not set(original64) <= set(keys):
        raise RuntimeError('all original64 keys must exist, no replacements')
    if {k[0] for k in keys} & {k[0] for k in train_keys}: raise RuntimeError('TRAIN/VAL scene overlap')
    used = set(original64)
    additional = base.select_scene_balanced_round_robin(sorted(k for k in keys if k not in used), 448)
    ordered = original64+list(additional)
    if len(ordered) != 512 or len(set(ordered)) != 512: raise RuntimeError('exact unique 64+448 population required')
    selected = base.align_records(records, ordered)
    return [window(r) for r in selected], [list(k) for k in ordered]


class DenseInputs(NuscInputs):
    def frame(self, kind, scene, token):
        # Deliberately never load mask_lidar/mask_camera, even on the OCC side.
        path = self.path(kind, scene, token); self.stat(path); key = (kind, scene, token)
        if key in self.cache: self.cache.move_to_end(key); return self.cache[key]
        with np.load(path, allow_pickle=False) as z:
            labels = semantics(z['semantics'])
        value = (labels, (labels >= 0)&(labels < 18)); size = sum(a.nbytes for a in value)
        if size <= self.limit:
            while self.cache and self.bytes+size > self.limit:
                _, old = self.cache.popitem(last=False); self.bytes -= sum(a.nbytes for a in old)
            self.cache[key] = value; self.bytes += size
        return value


class ModalInputs:
    def __init__(self, source, kind): self.source, self.kind = source, kind
    def history(self, job, _='occ'): return self.source.history(job, self.kind)
    def poses(self, job): return self.source.poses(job)
    def targets(self, job, indices): return self.source.targets(job, indices)
    def moving(self, *args): return self.source.moving(*args)


class FPNPredictor(DirectPredictor):
    @torch.no_grad()
    def forward(self, row, *, poses):
        return forward_window(self.body, self.provider, row, poses=poses, fresh=True)


def merge_counts(parts, expected_keys, voxels):
    """Sum the non-overlapping 64/448 integer counts BEFORE computing metrics."""
    names = ('gt/transport', 'gt/semantic_refine')
    total = {name: metrics.empty(3) for name in names}; seen = []
    for part in parts:
        if set(part['counts']) != set(names): raise RuntimeError('routes differ')
        seen.extend(part['keys'])
        for name in names:
            c = part['counts'][name]
            if set(c) != {'confusion', 'moving'}: raise RuntimeError('count fields differ')
            for field, raw in c.items():
                a = np.asarray(raw)
                if a.shape != (3, 18, 18) or a.dtype.kind not in 'iu' or np.any(a < 0):
                    raise RuntimeError('invalid integer counts')
                if field == 'confusion' and np.any(a.sum((1, 2)) != len(part['keys'])*voxels):
                    raise RuntimeError('incomplete full-grid counts')
                total[name][field] += a
            if np.any(np.asarray(c['moving']) > np.asarray(c['confusion'])): raise RuntimeError('invalid moving support')
    if seen != expected_keys or len(set(map(tuple, seen))) != len(seen): raise RuntimeError('duplicated/missing/reordered population')
    return {name: metrics.report(value, (1., 2., 3.)) for name, value in total.items()}
