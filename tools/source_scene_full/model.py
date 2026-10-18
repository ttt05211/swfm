"""Explicit inference loader for this full-training checkpoint protocol."""
from real_motion.local_st_world_model_v17 import config_from_mapping_v17
from tools.source_scene_stage1.model import ReaderConfig
from tools.source_scene_stage2.model import DirectBody


def checkpoint_model(saved, device='cpu'):
    from .run import PROTOCOL
    spec = saved['contract']['model']; shared, repair = spec['shared_history'], spec['repair']
    if saved.get('protocol') != PROTOCOL or spec.get('variant') != 'source_scene_stage2':
        raise RuntimeError('not this full-TRAIN stage2 checkpoint')
    model = DirectBody(config_from_mapping_v17(spec['motion']), feature_width=shared['width'],
        z_bins=shared['z_bins'], memory_width=repair['memory_width'], width=repair['width'],
        chunk_size=repair['chunk_size'], reader_config=ReaderConfig(**spec['reader']))
    model.columns.epsilon = repair['epsilon']; model.load_state_dict(saved['models']['gt'], strict=True)
    return model.to(device).eval().requires_grad_(False)
