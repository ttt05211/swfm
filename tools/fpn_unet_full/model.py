"""No architecture change; separate full-training loader rejects pilot resumes."""
from real_motion.local_st_world_model_v17 import config_from_mapping_v17
from tools.source_scene_stage1.model import ReaderConfig
from tools.fpn_unet.model import make_model, VARIANT


def checkpoint_model(saved, device='cpu'):
    from .run import PROTOCOL
    c = saved['contract']; spec = c['model']
    if saved.get('protocol') != PROTOCOL or c.get('protocol') != PROTOCOL or spec.get('variant') != VARIANT:
        raise RuntimeError('only full-TRAIN two-epoch FPN+UNet protocol accepted')
    shared, repair = spec['shared_history'], spec['repair']
    body = make_model(config_from_mapping_v17(spec['motion']), feature_width=shared['width'],
        z_bins=shared['z_bins'], memory_width=repair['memory_width'], reader_config=ReaderConfig(**spec['reader']))
    if body.configs() != spec: raise RuntimeError('FPN+UNet architecture changed')
    body.load_state_dict(saved['models']['gt'], strict=True)
    return body.to(device).eval().requires_grad_(False)
