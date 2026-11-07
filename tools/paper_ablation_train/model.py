"""Remove effective branches; copy only matching FRESH reference tensors."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from real_motion.causal_column_completion import ColumnConfig
from real_motion.local_st_world_model_v17 import config_from_mapping_v17
from tools.causal_forward_full.model import make_model as fresh_reference
from tools.causal_forward_train.model import ForwardBody
from tools.source_scene_stage1.model import ReaderConfig
from tools.fpn_unet.model import UNetRefiner

ARMS = {
    'c_transport': dict(table='C', fraction=1., motion=True, refiner=False, memory='none', pyramid='FPN'),
    'd_kta_refiner': dict(table='D', fraction=1., motion=False, refiner=True, memory='source', pyramid='FPN'),
    'm0_no_memory': dict(table='M0', fraction=.2, motion=True, refiner=True, memory='none', pyramid='FPN'),
    'm1_ego_memory': dict(table='M1', fraction=.2, motion=True, refiner=True, memory='ego_only', pyramid='FPN'),
    's0_pool': dict(table='S0', fraction=.2, motion=True, refiner=True, memory='source', pyramid='pooled_P2'),
}


class ZeroHistoryYaw(nn.Module):
    def forward(self, context): return context.new_zeros((*context.shape[:-1],1))


class PriorReadout(nn.Module):
    def __init__(self, columns):
        super().__init__()
        self.config = ColumnConfig(z_bins=columns.z_bins)
        self.z_bins, self.memory_width, self.epsilon = columns.z_bins, columns.memory_width, columns.epsilon

    def prior_logits(self, prior): return UNetRefiner.prior_logits(self, prior)


class KTATransport(nn.Module):
    """Only the shared spatial encoder remains; no dormant STWM parameters."""
    def __init__(self, original):
        super().__init__()
        self.config, self.reader_config = original.config, original.reader_config
        self.history_encoder = original.history_encoder

    def forward(self, fields, reads, times, classes, kta, ego_features=None):
        n = len(kta); field = fields[0]; dim = self.config.d_model
        if kta.shape != (n,6,2): raise ValueError('six KTA horizons required')
        return dict(residual_xy_m=field.new_zeros((n,6,2)), yaw_delta_rad=field.new_zeros((n,6)),
            existence_logits=field.new_zeros((n,6)), history_yaw_to_t0_rad=field.new_zeros((n,4)),
            history_source_context=field.new_zeros((n,dim)), future_transport_queries=field.new_zeros((n,6,dim)))


class PooledPyramid(nn.Module):
    """Native P2 stem/projection, then spatial pooling; no FPN/deep stages."""
    def __init__(self, original):
        super().__init__()
        self.z_bins, self.width = original.z_bins, original.width
        self.semantic, self.stem = original.semantic, original.stem
        self.lateral = nn.ModuleList([original.lateral[0]])
        self.smooth = nn.ModuleList([original.smooth[0]])

    def forward(self, semantics, observed):
        if (semantics.ndim!=4 or semantics.shape[0]!=4 or semantics.shape[-1]!=self.z_bins
                or semantics.shape!=observed.shape or semantics.shape[1]%8 or semantics.shape[2]%8):
            raise ValueError('FOUR histories and unchanged FPN grid contract required')
        known=observed.bool() & (semantics>=0) & (semantics<18)
        labels=torch.where(known,semantics.long(),18)
        x=torch.cat((self.semantic(labels),known[...,None].float()),-1)
        x=x.permute(0,3,4,1,2).flatten(1,2)
        p2=self.smooth[0](self.lateral[0](self.stem(x)))
        return p2,F.avg_pool2d(p2,2,2),F.avg_pool2d(p2,4,4)


class AblationBody(ForwardBody):
    def configure_ablation(self, arm, reference_spec, experiment):
        self.arm = arm; self.recipe = copy.deepcopy(ARMS[arm])
        self.reference_spec = copy.deepcopy(reference_spec); self.experiment = copy.deepcopy(experiment)
        if arm=='d_kta_refiner': self.transport=KTATransport(self.transport)
        if self.recipe['memory'] in ('none','ego_only'):
            self.transport.history_yaw = ZeroHistoryYaw()
        if not self.recipe['refiner']:
            self.columns=PriorReadout(self.columns)
        elif self.recipe['memory']=='none':
            # These cannot influence this arm, and must not enter Adam.
            del self.columns.lift, self.columns.importance, self.columns.empty
        if arm=='s0_pool': self.transport.history_encoder=PooledPyramid(self.transport.history_encoder)

    def configs(self):
        return dict(variant='CarryWorld_training_only_ablation_v1', arm=self.arm,
            recipe=copy.deepcopy(self.recipe), reference_model=copy.deepcopy(self.reference_spec),
            experiment=copy.deepcopy(self.experiment), source_policy='causal_forward_CV_history',
            disabled_history_yaw=self.recipe['memory'] in ('none','ego_only') or not self.recipe['motion'],
            weights_from_trained_model=False, network_parameters=sum(p.numel() for p in self.parameters()))


def make_model(config, arm, *, expected_reference_spec=None, experiment=None, **kwargs):
    if arm not in ARMS: raise ValueError('unknown ablation arm')
    reference=fresh_reference(config,expected_pilot_spec=expected_reference_spec,**kwargs)
    # The reference constructor fixes the same post-init RNG as existing E.
    with torch.random.fork_rng(devices=[]):
        body=AblationBody(config,**kwargs)
        body.configure_ablation(arm,reference.configs(),experiment)
    state=reference.state_dict()
    template=body.state_dict()
    if any(k not in state or state[k].shape!=v.shape or state[k].dtype!=v.dtype for k,v in template.items()):
        raise RuntimeError('ablation introduced an unmatched learned tensor')
    body.load_state_dict({k:state[k] for k in template},strict=True)
    return body


def from_spec(spec, device='cpu'):
    ref=spec['reference_model']
    body=make_model(config_from_mapping_v17(ref['motion']),spec['arm'],
        feature_width=ref['shared_history']['width'],z_bins=ref['shared_history']['z_bins'],
        memory_width=ref['repair']['memory_width'],reader_config=ReaderConfig(**ref['reader']),
        experiment=spec['experiment'])
    if body.configs()!=spec: raise RuntimeError('ablation model specification changed')
    return body.to(device)
