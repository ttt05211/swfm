from dataclasses import dataclass
from typing import Sequence
import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class MotionTubeConfig:
    radii: Sequence[int] = (1, 2, 3, 4, 5, 6)
    latent_extra_radius: int = 1

def _dilate_2d(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.bool()
    if mask.ndim < 2:
        raise ValueError("mask must have spatial H,W dimensions")
    orig = mask.shape
    flat = mask.reshape(-1, 1, orig[-2], orig[-1]).float()
    k = 2 * radius + 1
    out = F.max_pool2d(flat, kernel_size=k, stride=1, padding=radius)
    return out.reshape(orig).bool()

def build_motion_tube(kta_support: torch.Tensor,
                      cfg: MotionTubeConfig = MotionTubeConfig()) -> torch.Tensor:
    if kta_support.ndim not in (3, 4):
        raise ValueError("kta_support must be [F,H,W] or [B,F,H,W]")
    has_batch = kta_support.ndim == 4
    x = kta_support if has_batch else kta_support.unsqueeze(0)
    f = x.shape[1]
    radii = list(cfg.radii)
    if not radii:
        raise ValueError("at least one radius is required")
    if len(radii) < f:
        radii += [radii[-1]] * (f - len(radii))
    out = torch.zeros_like(x, dtype=torch.bool)
    for h in range(f):
        out[:, h] = _dilate_2d(x[:, h], int(radii[h]))
    return out if has_batch else out[0]

def downsample_support(mask: torch.Tensor, out_hw, extra_radius: int = 1) -> torch.Tensor:
    if mask.ndim not in (3, 4):
        raise ValueError("mask must be [F,H,W] or [B,F,H,W]")
    had_batch = mask.ndim == 4
    x = mask if had_batch else mask.unsqueeze(0)
    b, f, h, w = x.shape
    pooled = F.adaptive_max_pool2d(x.reshape(b*f,1,h,w).float(), out_hw)
    pooled = pooled.reshape(b,f,*out_hw).bool()
    if extra_radius:
        pooled = _dilate_2d(pooled, extra_radius)
    return pooled if had_batch else pooled[0]

def coverage_and_active_ratio(gt_moving_support: torch.Tensor,
                              generation_support: torch.Tensor):
    gt = gt_moving_support.bool(); gen = generation_support.bool()
    if gt.shape != gen.shape:
        raise ValueError(f"shape mismatch: {gt.shape} vs {gen.shape}")
    inter = (gt & gen).sum(dtype=torch.float64); denom = gt.sum(dtype=torch.float64)
    coverage = (inter / denom).item() if denom.item() else 1.0
    active = gen.float().mean().item()
    return coverage, active
