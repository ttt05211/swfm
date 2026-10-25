"""Full-training checkpoint loader; the completed one-epoch pilot is untouched."""
import hashlib
from pathlib import Path
from real_motion.local_st_world_model_v17 import config_from_mapping_v17
from tools.source_scene_stage1.model import ReaderConfig
from tools.fpn_adaln_zero.model import AdaLNBody, VARIANT, make_model


def checkpoint_model(saved, device='cpu'):
    from .run import PROTOCOL
    contract = saved['contract']; spec = contract['model']
    if (saved.get('protocol') != PROTOCOL or contract.get('protocol') != PROTOCOL
            or spec.get('variant') != VARIANT):
        raise RuntimeError('only full-TRAIN two-epoch AdaLN-Zero protocol accepted')
    root = Path(__file__).resolve().parents[2]
    for name, digest in (spec.get('experiment') or {}).get('candidate_implementation', {}).items():
        path = (root/name).resolve()
        if root not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError('AdaLN-Zero full source implementation differs: '+name)
    body = AdaLNBody(config_from_mapping_v17(spec['motion']),
        feature_width=spec['shared_history']['width'], z_bins=spec['shared_history']['z_bins'],
        memory_width=spec['repair']['memory_width'], reader_config=ReaderConfig(**spec['reader']),
        experiment=spec.get('experiment'))
    if body.configs() != spec: raise RuntimeError('AdaLN-Zero architecture differs')
    body.load_state_dict(saved['models']['gt'], strict=True)
    return body.to(device).eval().requires_grad_(False)
