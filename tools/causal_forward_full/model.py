"""Same fresh ForwardBody; a distinct full-training checkpoint protocol."""
from pathlib import Path
from real_motion.local_st_world_model_v17 import config_from_mapping_v17
from tools.source_scene_stage1.model import ReaderConfig
from tools.causal_forward_train import model as pilot
from tools.causal_forward_train.control import sha

SOURCE_POLICY = pilot.SOURCE_POLICY


def make_model(config,*,expected_pilot_spec=None,experiment=None,**kwargs):
    body = pilot.make_model(config,experiment=experiment,**kwargs)
    if expected_pilot_spec is not None:
        without_experiment = lambda value:{k:v for k,v in value.items() if k!='experiment'}
        if without_experiment(body.configs())!=without_experiment(expected_pilot_spec):
            raise RuntimeError('full model differs from validated forward AdaLN pilot')
    return body


def checkpoint_model(saved,device='cpu'):
    from .run import PROTOCOL
    c = saved['contract'];spec = c['model']
    if (saved.get('protocol')!=PROTOCOL or c.get('protocol')!=PROTOCOL
            or c.get('source_policy')!=SOURCE_POLICY or spec.get('source_policy')!=SOURCE_POLICY):
        raise RuntimeError('only full forward-associated AdaLN two-epoch checkpoint accepted')
    root = Path(__file__).resolve().parents[2]
    for name,digest in (spec.get('experiment') or {}).get('candidate_implementation',{}).items():
        path = (root/name).resolve()
        if root not in path.parents or sha(path)!=digest:raise RuntimeError('full forward implementation differs: '+name)
    body = pilot.ForwardBody(config_from_mapping_v17(spec['motion']),feature_width=spec['shared_history']['width'],
        z_bins=spec['shared_history']['z_bins'],memory_width=spec['repair']['memory_width'],
        reader_config=ReaderConfig(**spec['reader']),experiment=spec.get('experiment'))
    if body.configs()!=spec:raise RuntimeError('full forward architecture differs')
    body.load_state_dict(saved['models']['gt'],strict=True)
    return body.to(device).eval().requires_grad_(False)
