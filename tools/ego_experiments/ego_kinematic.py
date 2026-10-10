"""Small causal control readout, isolated from legacy bank/code fingerprints.

This is a candidate ego predictor, NOT a change to the frozen WM/CCR. All
trajectories use the existing seven bank fields and the SAME navigation labels.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from real_motion.ego_trajectory_head import EgoHeadConfig, command_indices
from tools.real_motion.ego_trajectory_common import FEATURE_FIELDS

PROTOCOL = 'surface_ego_history_kinematic_control_v1'


def rotate(x, angle):
    c, s = angle.cos(), angle.sin()
    return torch.stack((c*x[..., 0]-s*x[..., 1], s*x[..., 0]+c*x[..., 1]), -1)


def linear_fit(t, value):
    """Batched straight-line fit evaluated at t=0; units stay physical."""
    tc = t-t.mean(-1, keepdim=True)
    slope = (tc[..., None]*value).sum(1)/tc.square().sum(-1).clamp_min(1e-6)[..., None]
    intercept = value.mean(1)-slope*t.mean(-1)[..., None]
    return intercept, slope


def history_state(history):
    """Recover t0 velocity from THREE historical secants, not the last average.

    Secants are unrotated using measured midpoint heading, then regressed at
    actual historical timestamps. This avoids assuming which LiDAR axis is
    forward while accounting for observed turning and acceleration.
    """
    e = history.float()
    if e.ndim != 3 or e.shape[1:] != (4, 10) or not torch.isfinite(e).all():
        raise ValueError('finite FOUR-frame ego history required')
    t = e[..., 9]*1.5
    dt = t[:, 1:]-t[:, :-1]
    if (dt <= 0).any() or t[:, -1].abs().max() > 1e-4:
        raise ValueError('increasing causal timestamps ending at t0 required')
    yaw = torch.atan2(e[..., 3], e[..., 4])
    dyaw = torch.atan2((yaw[:, 1:]-yaw[:, :-1]).sin(), (yaw[:, 1:]-yaw[:, :-1]).cos())
    # Reconstruct unwrapped heading anchored at CURRENT yaw=0.
    yaw = torch.cat((torch.zeros_like(yaw[:, :1]), dyaw.cumsum(1)), 1)
    yaw = yaw-yaw[:, -1:]
    mid = (t[:, 1:]+t[:, :-1])*.5
    mid_yaw = (yaw[:, 1:]+yaw[:, :-1])*.5
    velocity = (e[:, 1:, :2]-e[:, :-1, :2])*40./dt[..., None]
    body = rotate(velocity, -mid_yaw)
    # Exact chord correction for constant angular rate; stable at zero turn.
    body = body/torch.sinc(dyaw/(2*math.pi)).clamp_min(.5)[..., None]
    fit_v0, acceleration = linear_fit(mid, body)
    past_direction = body[:, -1]/torch.linalg.vector_norm(body[:, -1],dim=-1).clamp_min(.05)[:, None]
    intervals = mid[:, 1:]-mid[:, :-1]
    adjacent_a = ((body[:, 1:]-body[:, :-1])/intervals[..., None]*past_direction[:, None]).sum(-1)
    # Two consistent acceleration estimates are required. An unconstrained
    # polynomial fit amplified pose noise even in the synthetic constant-speed
    # negative control; reject opposite-sign/strongly inconsistent trends.
    magnitude = adjacent_a.abs()
    consistent = ((adjacent_a[:, 0]*adjacent_a[:, 1] > 0) &
        (magnitude.min(-1).values >= .25*magnitude.max(-1).values))
    v0 = torch.where(consistent[:, None],fit_v0,body[:, -1])
    speed = torch.linalg.vector_norm(v0, dim=-1)
    direction = v0/speed.clamp_min(.05)[..., None]
    # At a complete stop, recover the most recent measured moving direction.
    # If ALL four poses are stationary, +X is the vehicle forward direction in
    # the repository's ego-pose axes (NOT the LiDAR command-label axes). This
    # permits a learned launch while the zero-control prior remains stationary.
    body_speed = torch.linalg.vector_norm(body,dim=-1)
    last = torch.where(body_speed>.05,torch.arange(3,device=e.device)[None],-1).max(1).values
    recent = body[torch.arange(len(e),device=e.device),last.clamp_min(0)]
    recent = recent/torch.linalg.vector_norm(recent,dim=-1).clamp_min(.05)[:,None]
    fallback = torch.where((last>=0)[:,None],recent,e.new_tensor([1.,0.])[None])
    direction = torch.where((speed>.05)[:,None],direction,fallback)
    a = torch.where(consistent,(acceleration*direction).sum(-1),torch.zeros_like(speed)).clamp(-4., 4.)
    _, omega = linear_fit(t, yaw[..., None])
    omega = omega[:, 0].clamp(-.7, .7)
    residual = body-(fit_v0[:, None]+acceleration[:, None]*mid[..., None])
    jitter = residual.square().mean((1, 2)).sqrt()
    # No future heading, sensor mask or GT fallback is supplied.
    return dict(velocity=v0, speed=speed, direction=direction, acceleration=a,
                omega=omega, jitter=jitter)


def integrate(state, controls=None, *, substeps=8):
    """Midpoint unicycle integration with longitudinal acceleration + yaw rate.

    Signed past direction is retained (including reverse). Braking stops at
    zero speed instead of spuriously reversing. 48 substeps form one coherent
    trajectory; six poses are read at the original .5s slots.
    """
    speed0 = state['speed']; b = len(speed0)
    if controls is None:
        controls = speed0.new_zeros(b, 3, 2)
    if controls.shape != (b, 3, 2) or not torch.isfinite(controls).all():
        raise ValueError('three finite control knots required')
    n = 6*substeps; dt = .5/substeps
    knots = F.interpolate(controls.transpose(1, 2), size=n, mode='linear', align_corners=True).transpose(1, 2)
    acceleration = (state['acceleration'][:, None]+knots[..., 0]).clamp(-6., 6.)
    omega = (state['omega'][:, None]+knots[..., 1]).clamp(-.9, .9)
    # Vectorized reflection is EXACTLY the recurrence
    # v[k+1] = max(0, v[k] + a[k]*dt), including a subsequent restart. Clamping
    # only the final cumulative speed incorrectly swallows later acceleration.
    cumulative = speed0[:, None]+(acceleration*dt).cumsum(1)
    minimum = cumulative.cummin(1).values
    correction = torch.where(minimum < 0, minimum, torch.zeros_like(minimum))
    speed_end = cumulative-correction
    speed_start = torch.cat((speed0[:, None],speed_end[:, :-1]),1)
    speed_mid = (speed_start+speed_end)*.5
    angle = (omega*dt).cumsum(1); mid_angle = angle-omega*dt*.5
    motion = rotate(state['direction'][:, None]*speed_mid[..., None], mid_angle)*dt
    xy = motion.cumsum(1)
    ids = torch.arange(substeps-1, n, substeps, device=xy.device)
    return torch.cat((xy[:, ids], angle[:, ids, None]), -1)


class KinematicEgoHead(nn.Module):
    """Direct ego-dynamics path; optional attention to frozen scene evidence.

    The entire six-command sequence conditions ONE trajectory, not independently
    spliced per-horizon pose branches. This changes the head parameterization,
    not the available navigation information.
    """
    def __init__(self, config=EgoHeadConfig(), *, scene=True, width=64):
        super().__init__(); self.config = config; self.scene = bool(scene); self.width = width
        self.history = nn.Sequential(nn.Linear(46+18, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        if scene:
            self.objects = nn.Sequential(nn.Linear(config.object_dim+8, width), nn.LayerNorm(width), nn.SiLU())
            self.surfaces = nn.Sequential(nn.Linear(config.surface_dim+3, width), nn.LayerNorm(width), nn.SiLU())
            self.query = nn.Linear(width, width)
            self.fuse = nn.Sequential(nn.Linear(width*2, width), nn.SiLU())
        self.readout = nn.Linear(width, 6)
        nn.init.zeros_(self.readout.weight); nn.init.zeros_(self.readout.bias)

    def forward(self, bank, commands):
        if set(bank) != set(FEATURE_FIELDS):
            raise ValueError('head accepts ONLY seven historical feature fields')
        e = bank['ego_history'].float(); c = self.config; b = len(e)
        expected = dict(objects=(b,c.object_slots,c.object_dim), object_geometry=(b,c.object_slots,8),
            object_valid=(b,c.object_slots), surfaces=(b,c.surface_side**2,c.surface_dim),
            surface_geometry=(b,c.surface_side**2,3), surface_valid=(b,c.surface_side**2))
        if any(tuple(bank[k].shape) != shape for k, shape in expected.items()):
            raise ValueError('history bank shape mismatch')
        if any(not torch.isfinite(bank[k]).all() for k in FEATURE_FIELDS if not k.endswith('valid')):
            raise ValueError('nonfinite historical feature')
        commands = command_indices(commands, batch=b, device=e.device)
        state = history_state(e)
        physical = torch.cat((state['velocity']/10., state['acceleration'][:, None]/4.,
            state['omega'][:, None], state['speed'][:, None]/10., state['jitter'][:, None]), -1)
        nav = F.one_hot(commands, 3).flatten(1).float()
        h = self.history(torch.cat((e.flatten(1), physical, nav), -1))
        if self.scene:
            ob = self.objects(torch.cat((bank['objects'], bank['object_geometry']), -1).float())
            st = self.surfaces(torch.cat((bank['surfaces'], bank['surface_geometry']), -1).float())
            tokens = torch.cat((ob, st, torch.zeros_like(h[:, None])), 1)
            valid = torch.cat((bank['object_valid'].bool(), bank['surface_valid'].bool(),
                torch.ones((b, 1), device=e.device, dtype=torch.bool)), 1)
            scores = (self.query(h)[:, None]*tokens).sum(-1)/math.sqrt(self.width)
            weights = scores.masked_fill(~valid, -torch.inf).softmax(1)
            h = self.fuse(torch.cat((h, (tokens*weights[..., None]).sum(1)), -1))
        controls = self.readout(h).reshape(b, 3, 2).tanh()*h.new_tensor([4., .35])
        return dict(se2=integrate(state, controls), controls=controls)


class KinematicPrior(nn.Module):
    def __init__(self, config=EgoHeadConfig()):
        super().__init__(); self.config = config
        self.register_parameter('device_anchor', nn.Parameter(torch.zeros(()), requires_grad=False))

    def forward(self, bank, commands):
        command_indices(commands, batch=len(bank['ego_history']), device=bank['ego_history'].device)
        return dict(se2=integrate(history_state(bank['ego_history'])))
