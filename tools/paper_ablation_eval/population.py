"""Prepare the shared VAL identities and poses once for the eight processes."""
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from real_motion.nuscenes_adapter import WindowTokens
from tools.causal_forward_all.common import validate_source
from tools.final_semantic_evaluation.sources import window
from tools.fpn_unet_eval.common import DenseInputs
from tools.joint_occ_experiments import run as base
from tools.scene_ego_head.experiment.planner import PlannerCache


def jobs_from_cache(path):
    _, records = base.load_cache(path)
    jobs = [window(record) for record in records]
    keys = [(job.scene_name, job.t0_token) for job in jobs]
    if len(jobs) != 4369 or len(set(keys)) != 4369 or len({key[0] for key in keys}) != 150:
        raise ValueError('Complete VAL4369 with 150 scenes is required')
    for job in jobs:
        if (len(job.history_tokens) != 4 or job.history_tokens[-1] != job.t0_token
                or len(job.future_tokens) != 6 or len(set(job.history_tokens + job.future_tokens)) != 10):
            raise ValueError(f'Invalid historical/future frame identities for {job.t0_token}')
    return jobs, keys


def prepare(args, epoch2):
    import torch

    saved = torch.load(epoch2, map_location='cpu', weights_only=False)
    reference = validate_source(saved)
    del saved
    files = {name: base.sha256(getattr(args, name)) for name in ('config', 'dev_info', 'dev_cache')}
    if any(reference['data'][name] != digest for name, digest in files.items()):
        raise RuntimeError('Config or VAL metadata differs from E')
    jobs, keys = jobs_from_cache(args.dev_cache)
    if {key[0] for key in keys} & {key[0] for key in reference['train_keys']}:
        raise RuntimeError('TRAIN and VAL scenes overlap')

    source = DenseInputs(args.dataroot, args.dev_info, stc_root=args.stc_root, cache_mib=0)
    for kind in ('occ', 'stc'):
        source.preflight(jobs, kind)
    tokens = sorted({token for job in jobs for token in job.history_tokens + job.future_tokens})
    poses = {token: np.asarray(source.source.pose(token)).tolist() for token in tokens}
    times = {token: source.source.nusc.get('sample', token)['timestamp'] / 1e6 for token in tokens}
    planner = PlannerCache(args.plan_cache)
    planner.prepare(keys)
    predicted = {}
    for job, key in zip(jobs, keys):
        if key in planner.rows:
            record = dict(scene_name=job.scene_name, t0_token=job.t0_token,
                          history_tokens=job.history_tokens, future_tokens=job.future_tokens)
            predicted[job.t0_token] = planner.poses(record, np.asarray(poses[job.t0_token])).tolist()

    return dict(dataroot=str(Path(args.dataroot).resolve()), stc_root=str(Path(args.stc_root).resolve()),
                files=files, overrides=reference.get('overrides'),
                jobs=[[job.scene_name, list(job.history_tokens), job.t0_token, list(job.future_tokens)] for job in jobs],
                poses=poses, times=times, predicted=predicted, planner=planner.metadata,
                inventory={path: list(stamp) for path, stamp in source.inventory.items()})


def select_jobs(population, predicted):
    jobs = [WindowTokens(scene, tuple(history), token, tuple(future))
            for scene, history, token, future in population['jobs']]
    if predicted:
        jobs = [job for job in jobs if job.t0_token in population['predicted']]
    return jobs


class PreparedInputs(DenseInputs):
    """Reuse dense frame/target I/O with the shared read-only pose table."""

    def __init__(self, population, cache_mib=128):
        self.source = SimpleNamespace(dataroot=population['dataroot'])
        self.stc = Path(population['stc_root'])
        self.cache = OrderedDict()
        self.bytes = 0
        self.limit = cache_mib * 2**20
        self.inventory = {path: tuple(stamp) for path, stamp in population['inventory'].items()}
        self.gt_poses = population['poses']
        self.predicted = population['predicted']
        self.times = population['times']

    def history(self, job, kind='occ'):
        record = dict(scene_name=job.scene_name, t0_token=job.t0_token,
                      history_tokens=job.history_tokens, future_tokens=job.future_tokens)
        frames = [self.frame(kind, job.scene_name, token) for token in job.history_tokens]
        raw = dict(history_occ=np.stack([frame[0] for frame in frames]),
                   history_observed=np.stack([frame[1] for frame in frames]),
                   history_poses=np.asarray([self.gt_poses[token] for token in job.history_tokens], dtype=np.float64))
        return record, raw, np.asarray([self.times[token] for token in job.history_tokens])

    def future_poses(self, job, predicted):
        poses = self.predicted[job.t0_token] if predicted else [self.gt_poses[token] for token in job.future_tokens]
        return np.asarray(poses, dtype=np.float64)
