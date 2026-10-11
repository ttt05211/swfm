"""Isolated experimental models; no changes to legacy cache/inference signatures."""
import torch
from torch import nn
from torch.nn import functional as F
from real_motion.joint_surface_ccr import JointSurfaceCCR
from real_motion.surface_canonical_repair import SurfaceCanonicalRepairHead
from real_motion.ego_trajectory_head import command_indices
from tools.ego_experiments.ego_kinematic import history_state, integrate

PROTOCOL = 'p0_f9_joint_occ_ego_soft_train_hard_eval_v1'


class HistoryEgoHead(nn.Module):
    """The tested 8,710-parameter history control head, without frozen banks."""
    def __init__(self, width=64):
        super().__init__()
        self.history = nn.Sequential(nn.Linear(64, width), nn.SiLU(),
                                     nn.Linear(width, width), nn.SiLU())
        self.readout = nn.Linear(width, 6)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, history, commands):
        state = history_state(history)
        commands = command_indices(commands, batch=len(history), device=history.device)
        physical = torch.cat((state['velocity']/10., state['acceleration'][:, None]/4.,
            state['omega'][:, None], state['speed'][:, None]/10., state['jitter'][:, None]), -1)
        nav = F.one_hot(commands, 3).flatten(1).float()
        h = self.history(torch.cat((history.flatten(1), physical, nav), -1))
        controls = self.readout(h).reshape(-1, 3, 2).tanh()*h.new_tensor([4., .35])
        return dict(se2=integrate(state, controls), controls=controls)


class RepairBothActions(SurfaceCanonicalRepairHead):
    add_only_weighted = False

    def probabilities(self, logits, actors):
        # No fitted prior or action-specific BCE. Final semantic probabilities
        # are supervised instead. Old Surface head remains ADD-only unchanged.
        return logits.float().sigmoid()


class JointOccEgo(JointSurfaceCCR):
    def __init__(self, motion_config, *, width=64):
        super().__init__(motion_config, width=width)
        metadata = self.columns.config
        self.columns = RepairBothActions(motion_config.d_model, width)
        self.columns.config = metadata
        self.ego = HistoryEgoHead()

    def configs(self):
        return {**super().configs(), 'ego': dict(type='history_unicycle_control', width=64,
            parameters=sum(p.numel() for p in self.ego.parameters()), command_branches=3,
            history_frames=4, future_frames=6), 'repair_actions': ['ADD', 'REMOVE']}


def gradient_scale(value, scale):
    """Forward is ALWAYS predicted. Only occupancy->ego gradient is ramped."""
    return value.detach() + scale*(value-value.detach())


def torch_poses(current, se2):
    """Same world-Z/t0-tilt convention as ego_navigation.poses_from_se2."""
    current = torch.as_tensor(current, device=se2.device, dtype=torch.float64)
    a = se2.double()
    c, s = a[:, 2].cos(), a[:, 2].sin()
    z, o = torch.zeros_like(c), torch.ones_like(c)
    rotation = torch.stack((c, -s, z, s, c, z, z, z, o), -1).reshape(-1, 3, 3)
    r = current[:3, :3][None] @ rotation
    xy = a[:, :2] @ current[:2, :2].T + current[:2, 3]
    translation = torch.cat((xy, current[2, 3].expand(len(a), 1)), -1)
    top = torch.cat((r, translation[..., None]), -1)
    return torch.cat((top, current[3].expand(len(a), 1, 4)), 1)


def ego_geometry_loss(prediction, target, radius=10.):
    target = torch.as_tensor(target, device=prediction.device, dtype=prediction.dtype)
    points = prediction.new_tensor([[0, 0], [radius, 0], [-radius, 0], [0, radius], [0, -radius]])
    def transformed(a):
        c, s = a[..., 2, None].cos(), a[..., 2, None].sin()
        return torch.stack((c*points[:, 0]-s*points[:, 1]+a[..., 0, None],
                            s*points[:, 0]+c*points[:, 1]+a[..., 1, None]), -1)
    return F.smooth_l1_loss(transformed(prediction), transformed(target))


def source_auxiliary(output, record, state=None):
    """One geometry objective in metres + existence; no separate yaw/IoU loss.

    With live geometry, 32 evenly indexed observed source points are used.
    The small mask-only fallback is for standalone interface tests.
    Source-centre displacement targets keep V18's annotation offset correction.
    Invalid yaw labels fall back to translation-only, not fabricated angles.
    """
    device = output['residual_xy_m'].device
    get = lambda name: torch.as_tensor(record[name], device=device)
    pred = get('kta_displacement_xy_m').float()+output['residual_xy_m'].float()
    if not len(pred):
        zero = output['residual_xy_m'].sum()*0
        return zero, zero
    supervised = get('supervised_source').bool()
    valid = get('se2_target_valid').bool() & supervised[:, None]
    yaw_valid = valid & get('yaw_label_valid').bool() & get('yaw_enabled').bool()[:, None]
    target = get('target_source_displacement_xy_m').float()
    if state is not None:
        import numpy as np
        rows = []
        for cloud, current in zip(state['source_world_points'], state['current']):
            if not len(cloud): raise RuntimeError('empty reliable source point cloud')
            indices = np.linspace(0, len(cloud)-1, 32).astype(np.int64)
            rows.append(np.asarray(cloud)[indices, :2]-np.asarray(current['centroid_world'])[:2])
        points = torch.as_tensor(np.stack(rows), device=device, dtype=torch.float32)
        rotation = torch.as_tensor(np.asarray(state['current_pose'])[:2, :2], device=device, dtype=torch.float32)
        pred, target = pred @ rotation.T, target @ rotation.T
    else:
        points = pred.new_tensor([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])[None].expand(len(pred), -1, -1)
    def shape(d, yaw):
        c, s = yaw[..., None].cos(), yaw[..., None].sin()
        x, y = points[:, None, :, 0], points[:, None, :, 1]
        return torch.stack((c*x-s*y, s*x+c*y), -1)+d[:, :, None]
    py = torch.where(yaw_valid, output['yaw_delta_rad'].float(), torch.zeros_like(valid, dtype=torch.float32))
    ty = torch.where(yaw_valid, get('target_yaw_rad').float(), torch.zeros_like(py))
    geometry = (F.smooth_l1_loss(shape(pred, py)[valid], shape(target, ty)[valid])
                if bool(valid.any()) else pred.sum()*0)
    logits = output['existence_logits'].float()
    existence = (F.binary_cross_entropy_with_logits(logits[supervised], get('existence').float()[supervised])
                 if bool(supervised.any()) else logits.sum()*0)
    return geometry, existence
