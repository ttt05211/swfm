"""EXACT AdaLN tensors/initialization; association metadata is explicit."""
from pathlib import Path
import torch
from real_motion.local_st_world_model_v17 import config_from_mapping_v17
from tools.source_scene_stage1.model import ReaderConfig
from tools.fpn_adaln_zero.model import AdaLNBody, make_model as original_fresh
from tools.causal_forward_track.track import PROTOCOL as TRACK_PROTOCOL
from .control import sha

SOURCE_POLICY = 'causal_forward_CV_history'


class ForwardBody(AdaLNBody):
    def configs(self):
        return dict(super().configs(),source_policy=SOURCE_POLICY,association=TRACK_PROTOCOL)


def make_model(config, *, expected_control_spec=None, experiment=None, **kwargs):
    original = original_fresh(config,**kwargs)  # RANDOM, never trained control weights.
    if expected_control_spec is not None:
        expected = {k:v for k,v in expected_control_spec.items() if k != 'experiment'}
        actual = {k:v for k,v in original.configs().items() if k != 'experiment'}
        if actual != expected:raise RuntimeError('AdaLN control architecture differs')
    with torch.random.fork_rng(devices=[]):
        body = ForwardBody(config,experiment=experiment,**kwargs)
    body.load_state_dict(original.state_dict(),strict=True)
    return body


def checkpoint_model(saved,device='cpu'):
    from .run import PROTOCOL
    c = saved['contract']; spec = c['model']
    if (saved.get('protocol') != PROTOCOL or c.get('protocol') != PROTOCOL
            or c.get('source_policy') != SOURCE_POLICY or spec.get('source_policy') != SOURCE_POLICY):
        raise RuntimeError('only matched forward-association AdaLN checkpoint accepted')
    root = Path(__file__).resolve().parents[2]
    for name,digest in (spec.get('experiment') or {}).get('candidate_implementation',{}).items():
        path = (root/name).resolve()
        if root not in path.parents or sha(path) != digest:raise RuntimeError('forward implementation differs: '+name)
    body = ForwardBody(config_from_mapping_v17(spec['motion']),feature_width=spec['shared_history']['width'],
        z_bins=spec['shared_history']['z_bins'],memory_width=spec['repair']['memory_width'],
        reader_config=ReaderConfig(**spec['reader']),experiment=spec.get('experiment'))
    if body.configs() != spec:raise RuntimeError('forward checkpoint architecture differs')
    body.load_state_dict(saved['models']['gt'],strict=True)
    return body.to(device).eval().requires_grad_(False)
