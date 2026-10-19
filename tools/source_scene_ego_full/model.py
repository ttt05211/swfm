"""SAME three-change pilot architecture, separate full-training protocol and loader."""
from real_motion.local_st_world_model_v17 import config_from_mapping_v17
from tools.source_scene_stage1.model import ReaderConfig
from tools.source_scene_ego_conditioned.model import EgoBody, VARIANT, make_model as fresh_candidate

def make_model(config, **kwargs):
    if config.decoder_blocks != 2: raise ValueError('fixed TWO encoder / TWO decoder blocks required')
    body = fresh_candidate(config, **kwargs)
    if hasattr(body.columns, 'source_query'): raise RuntimeError('Q-to-memory module must be absent')
    return body


def checkpoint_model(saved, device='cpu'):
    from .run import PROTOCOL
    spec = saved['contract']['model']; sh, rep = spec['shared_history'], spec['repair']
    if (saved.get('protocol') != PROTOCOL or spec.get('variant') != VARIANT or
            spec['encoder']['blocks'] != 2 or spec['motion']['decoder_blocks'] != 2 or
            spec.get('registration') != 'none' or spec.get('kta_conditioning') is not True or
            spec.get('future_query', {}).get('decoder_order') != 'history_cross_future_self_FFN' or
            spec.get('memory', {}).get('future_query_content') is not False or
            spec.get('memory', {}).get('future_query_importance') is not False):
        raise RuntimeError('only SAME ego-query/history-only-memory/cross-first checkpoint allowed')
    body = EgoBody(config_from_mapping_v17(spec['motion']), feature_width=sh['width'], z_bins=sh['z_bins'],
        memory_width=rep['memory_width'], width=rep['width'], chunk_size=rep['chunk_size'], reader_config=ReaderConfig(**spec['reader']))
    body.columns.epsilon = rep['epsilon']
    if body.configs() != spec: raise RuntimeError('checkpoint model specification differs')
    body.load_state_dict(saved['models']['gt'], strict=True)
    return body.to(device).eval().requires_grad_(False)
