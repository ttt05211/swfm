"""ONLY append two Scene-Temporal and two future decoder blocks to Integrated."""
from dataclasses import replace
import torch
from tools.source_scene_stage1.model import ReaderConfig, MaskedFutureBlock
from tools.source_scene_integrated.attention import SceneTemporalBlock
from tools.source_scene_integrated.model import IntegratedTransport, IntegratedBody, make_model as fresh_control

DEEP_READER = ReaderConfig(layers=4, temporal_layers=4)


class DeepTransport(IntegratedTransport):
    def __init__(self, config, width=64, z_bins=16, reader_config=DEEP_READER):
        if config.decoder_blocks != 4 or reader_config.layers != 4 or reader_config.temporal_layers != 4:
            raise ValueError('explicit FOUR encoder and FOUR decoder blocks required')
        super().__init__(replace(config, decoder_blocks=2), width, z_bins,
                         replace(reader_config, layers=2, temporal_layers=2))
        self.config = self.v17_config = config
        self.reader_config = reader_config
        self.encoder_blocks.extend([SceneTemporalBlock(config.d_model, config.heads, reader_config.points) for _ in range(2)])
        self.decoder.extend([MaskedFutureBlock(config.d_model, config.heads) for _ in range(2)])
        # Inherit the EXACT old forward, including four history tokens, masks,
        # initial GAP only, KTA query/skip and t0-gauge learned history yaw.


class DeepBody(IntegratedBody):
    def __init__(self, config, *, feature_width=64, z_bins=16, memory_width=16, width=32,
                 chunk_size=65536, reader_config=DEEP_READER):
        super().__init__(replace(config, decoder_blocks=2), feature_width=feature_width, z_bins=z_bins,
            memory_width=memory_width, width=width, chunk_size=chunk_size,
            reader_config=replace(reader_config, layers=2, temporal_layers=2))
        self.transport = DeepTransport(config, feature_width, z_bins, reader_config)
        # SAME inherited AdaGN Refiner, no replacement/reinitialization here.

    def configs(self):
        c = super().configs()
        return {**c, 'variant': 'source_scene_integrated_deep44',
                'encoder': {**c['encoder'], 'blocks': 4}}


def make_model(config, *, reader_config=DEEP_READER, **kwargs):
    if config.decoder_blocks != 2: raise ValueError('matched TWO-decoder control architecture required')
    if reader_config != DEEP_READER: raise ValueError('fixed original sampling/PE configuration and FOUR layers required')
    control = fresh_control(config, **kwargs)  # Original factory defaults are the matched TWO+TWO reader.
    with torch.random.fork_rng(devices=[]):
        model = DeepBody(replace(config, decoder_blocks=4), reader_config=reader_config, **kwargs)
    previous = control.state_dict()
    model.load_state_dict({k: previous[k] if k in previous else v for k, v in model.state_dict().items()}, strict=True)
    return model


def checkpoint_model(saved, device='cpu'):
    from .run import PROTOCOL
    from real_motion.local_st_world_model_v17 import config_from_mapping_v17
    spec = saved['contract']['model']; sh, rep = spec['shared_history'], spec['repair']
    if (saved.get('protocol') != PROTOCOL or spec.get('variant') != 'source_scene_integrated_deep44' or
            spec['encoder']['blocks'] != 4 or spec['motion']['decoder_blocks'] != 4 or
            spec['reader']['layers'] != 4 or spec['reader']['temporal_layers'] != 4):
        raise RuntimeError('explicit integrated FOUR+FOUR checkpoint required')
    model = DeepBody(config_from_mapping_v17(spec['motion']), feature_width=sh['width'], z_bins=sh['z_bins'],
        memory_width=rep['memory_width'], width=rep['width'], chunk_size=rep['chunk_size'], reader_config=ReaderConfig(**spec['reader']))
    model.columns.epsilon = rep['epsilon']; model.load_state_dict(saved['models']['gt'], strict=True)
    return model.to(device).eval()
