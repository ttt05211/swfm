"""Same Integrated TWO+TWO model, explicit full-training protocol/loader."""
from real_motion.local_st_world_model_v17 import config_from_mapping_v17
from tools.source_scene_stage1.model import ReaderConfig
from tools.source_scene_integrated.model import IntegratedBody, make_model as fresh_integrated


def make_model(config, **kwargs):
    if config.decoder_blocks != 2: raise ValueError('latest TWO encoder / TWO decoder configuration required')
    body = fresh_integrated(config, **kwargs)
    if len(body.transport.encoder_blocks) != 2 or len(body.transport.decoder) != 2:
        raise RuntimeError('not Integrated TWO+TWO')
    return body


def checkpoint_model(saved, device='cpu'):
    from .run import PROTOCOL
    spec = saved['contract']['model']; shared, repair = spec['shared_history'], spec['repair']
    if (saved.get('protocol') != PROTOCOL or spec.get('variant') != 'source_scene_integrated_AdaGN' or
            spec['encoder']['blocks'] != 2 or spec['motion']['decoder_blocks'] != 2 or
            spec['reader']['layers'] != 2 or spec['reader']['temporal_layers'] != 2 or
            spec.get('kta_conditioning') is not True or spec.get('registration') != 'none' or
            repair.get('conditioning') != 'residual_branch_AdaGN_zero_projection' or repair.get('GN_affine') is not False):
        raise RuntimeError('not this full-TRAIN Integrated TWO+TWO checkpoint')
    body = IntegratedBody(config_from_mapping_v17(spec['motion']), feature_width=shared['width'],
        z_bins=shared['z_bins'], memory_width=repair['memory_width'], width=repair['width'],
        chunk_size=repair['chunk_size'], reader_config=ReaderConfig(**spec['reader']))
    body.columns.epsilon = repair['epsilon']; body.load_state_dict(saved['models']['gt'], strict=True)
    return body.to(device).eval().requires_grad_(False)
