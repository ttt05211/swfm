"""Learned XY-only controls on frozen V18 evidence and SE(2) geometry.

No future annotations, target validity, existence or class-dependent gate
enter inference. The base XY is retained and a zero-initialized correction is
added in FP32. Integration is invertible: it does NOT enforce constant speed.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .local_st_world_model_v18_se2 import soft_se2_transport_overlap_loss

PROTOCOL = "p0_f9_v18_xy_trajectory_screen_v1"
INPUT_PROTOCOL = "frozen_v18_per_window_bf16_latents_causal_kinematics_v1"
INPUT_KEYS = ("query", "context", "frame_motion", "kta", "base_xy")


@dataclass(frozen=True)
class TrajectoryConfig:
    dim: int = 128
    hidden: int = 128
    horizons: int = 6
    dt_s: float = .5

    def validate(self):
        if min(self.dim,self.hidden) < 1 or self.horizons != 6 or self.dt_s != .5:
            raise ValueError("invalid frozen six-horizon trajectory contract")


def check_inputs(query, context, frame_motion, kta, base_xy, config):
    n = len(query)
    shapes = ((n,6,config.dim),(n,config.dim),(n,6,5),(n,6,2),(n,6,2))
    values = (query,context,frame_motion,kta,base_xy)
    if any(tuple(v.shape) != s or not torch.isfinite(v).all() for v,s in zip(values,shapes)):
        raise ValueError("XY trajectory causal input shape/finite mismatch")
    if torch.any((frame_motion[...,4] < 0) | (frame_motion[...,4] > 1)):
        raise ValueError("history validity must be in [0,1]")


class LinearXYRefit(nn.Module):
    """Fine-tune a linear XY head as an additive parameter difference.

    Unlike re-evaluating the old BF16 linear head in FP32, delta=0 preserves
    the actual frozen output exactly at initialization.
    """
    kind = "linear_xy_refit"
    def __init__(self,config=TrajectoryConfig()):
        super().__init__(); config.validate(); self.config = config
        self.delta = nn.Linear(config.dim,2)
        nn.init.zeros_(self.delta.weight); nn.init.zeros_(self.delta.bias)

    def forward(self,query,context,frame_motion,kta,base_xy):
        check_inputs(query,context,frame_motion,kta,base_xy,self.config)
        return base_xy.float()+self.delta(query.float())


class IntegratedXYTrajectory(nn.Module):
    """Joint six-state kinematic-conditioned velocity residual -> position."""
    kind = "integrated_xy_trajectory"
    def __init__(self,config=TrajectoryConfig()):
        super().__init__(); config.validate(); self.config = config
        width = 7*config.dim+54
        self.encoder = nn.Sequential(nn.LayerNorm(width),nn.Linear(width,config.hidden),nn.GELU(),
                                     nn.Linear(config.hidden,config.hidden),nn.GELU())
        self.velocity = nn.Linear(config.hidden,12)
        nn.init.zeros_(self.velocity.weight); nn.init.zeros_(self.velocity.bias)

    def coefficients(self,query,context,frame_motion,kta,base_xy):
        check_inputs(query,context,frame_motion,kta,base_xy,self.config)
        fm = frame_motion.float()
        # Missing history must not inject invented offsets/velocities.
        fm = torch.cat((fm[...,:4]*(fm[...,4:5] > .5),fm[...,4:5]),-1)
        features = torch.cat((query.float().flatten(1),context.float(),fm.flatten(1),
                              kta.float().flatten(1)/20.,base_xy.float().flatten(1)/20.),1)
        return self.velocity(self.encoder(features)).reshape(-1,6,2)

    def forward(self,query,context,frame_motion,kta,base_xy):
        dv = self.coefficients(query,context,frame_motion,kta,base_xy)
        return base_xy.float()+self.config.dt_s*dv.cumsum(1)


class JointXYPosition(IntegratedXYTrajectory):
    """Same features/parameters as integration, but direct position residual."""
    kind = "joint_xy_position"
    def forward(self,query,context,frame_motion,kta,base_xy):
        return base_xy.float()+self.coefficients(query,context,frame_motion,kta,base_xy)


def make_xy_model(kind,config=TrajectoryConfig()):
    if kind == LinearXYRefit.kind: return LinearXYRefit(config)
    if kind == IntegratedXYTrajectory.kind: return IntegratedXYTrajectory(config)
    if kind == JointXYPosition.kind: return JointXYPosition(config)
    raise ValueError("unknown XY trajectory variant")


def xy_objective(pred_xy, inputs, labels, *, patch_resolution_m=.8):
    """Original active XY/shape groups, shared by all three arms.

    Frozen existence/yaw losses are constants and are not added as new knobs.
    Future GT validity masks supervise only; they never gate predictions.
    """
    valid = labels["valid"].bool() & labels["supervised"].bool()[:,None]
    target = labels["target_xy"].float()
    if pred_xy.shape != target.shape or valid.shape != pred_xy.shape[:2]:
        raise ValueError("XY target/valid shape mismatch")
    if not torch.isfinite(target[valid]).all(): raise ValueError("nonfinite valid XY labels")
    if not valid.any(): raise ValueError("batch has no valid supervised source-horizon")
    # Mask missing targets BEFORE geometry arithmetic: NaN padding must not
    # contaminate a grid_sample or its backward on an unusable horizon.
    safe_target = torch.where(valid[...,None],target,pred_xy.detach())
    trans = F.smooth_l1_loss(pred_xy[valid],safe_target[valid],beta=1.,reduction="mean")
    safe_yaw = torch.where(labels["yaw_valid"].bool() & valid,labels["target_yaw"].float(),labels["base_yaw"].float())
    shape,_ = soft_se2_transport_overlap_loss(inputs["kta"].float()+pred_xy,
        inputs["kta"].float()+safe_target,labels["base_yaw"].float(),safe_yaw,
        labels["footprint"].float(),valid,labels["yaw_enabled"],labels["yaw_valid"] & valid,
        patch_resolution_m=patch_resolution_m,materialize_stats=False)
    return trans+.25*shape,{"xy_smooth_l1":trans.detach(),"shape":shape.detach(),"valid_labels":valid.sum()}


def replace_xy_outputs(base_outputs, pred_xy):
    """ONLY XY changes; the yaw/existence tensors remain the same objects."""
    if pred_xy.shape != base_outputs["residual_xy_m"].shape or not torch.isfinite(pred_xy).all():
        raise ValueError("invalid learned XY output")
    return {**base_outputs,"residual_xy_m":pred_xy}


def xy_screen_gate(report):
    """A nonzero practical improvement, not a baseline/all-reject success."""
    import numpy as np
    d = report["delta_vs_v18_pp"]
    metrics = ("IoU","mIoU","MovingMacro","MovingMicro")
    horizons = d.get("per_horizon",{})
    rows = [d]+list(horizons.values())
    finite = (set(horizons) == {"1.0","2.0","3.0"}
              and all(row.get(k) is not None and np.isfinite(row[k]) for row in rows for k in metrics))
    nonnegative = finite and all(row[k] >= -1e-9 for row in rows for k in metrics)
    gain = finite and d["mIoU"] >= .30 and d["MovingMicro"] >= 1.
    return {"all_metrics_all_report_horizons_nonnegative":bool(nonnegative),
            "mIoU_ge_0_30pp":bool(finite and d["mIoU"] >= .30),
            "MovingMicro_ge_1_00pp":bool(finite and d["MovingMicro"] >= 1.),"pass":bool(nonnegative and gain)}
