from dataclasses import dataclass
from typing import List, Tuple
import torch

@dataclass(frozen=True)
class WindowPlan:
    origins: torch.Tensor      # [B,K,2], y/x top-left, -1 for padding
    valid: torch.Tensor        # [B,K]
    window_hw: Tuple[int, int]
    full_hw: Tuple[int, int]

class WindowPlanner:
    """Plan fixed windows that MUST cover required future support.

    ``context_support`` (typically historical moving + future KTA tube) is only
    a tie-break signal when multiple candidate windows cover the same amount of
    required future support. It never creates history-only windows by itself.
    """
    def __init__(self, window_hw=(20,20), max_windows=8):
        self.window_hw = tuple(window_hw)
        self.max_windows = int(max_windows)
        if min(self.window_hw) <= 0 or self.max_windows <= 0:
            raise ValueError("invalid window configuration")

    @staticmethod
    def _union_support(support: torch.Tensor) -> torch.Tensor:
        if support.ndim == 4:
            return support.bool().any(dim=1)
        if support.ndim == 3:
            return support.bool()
        raise ValueError("support must be [B,T,H,W] or [B,H,W]")

    def plan(self, required_support: torch.Tensor,
             context_support: torch.Tensor | None = None) -> WindowPlan:
        required = self._union_support(required_support)
        context = required if context_support is None else self._union_support(context_support)
        if required.shape != context.shape:
            raise ValueError("required_support and context_support must align in B,H,W")

        out_device = required.device
        required_cpu = required.detach().cpu()
        context_cpu = context.detach().cpu()
        b,h,w = required_cpu.shape
        wh,ww = self.window_hw
        if wh > h or ww > w:
            raise ValueError("window cannot be larger than latent map")

        origins = torch.full((b,self.max_windows,2), -1, dtype=torch.long)
        valid = torch.zeros((b,self.max_windows), dtype=torch.bool)

        # 50x50 planning is intentionally CPU-side: Python .item()/where() on a
        # CUDA tensor would synchronize repeatedly.
        for bi in range(b):
            remaining = required_cpu[bi].clone()
            context_map = context_cpu[bi]
            for ki in range(self.max_windows):
                if not remaining.any():
                    break

                ys,xs = torch.where(remaining)
                stride = max(1, len(ys)//64)
                best = None
                for y,x in zip(ys[::stride], xs[::stride]):
                    cy = max(0, min(int(y)-wh//2, h-wh))
                    cx = max(0, min(int(x)-ww//2, w-ww))
                    required_score = int(remaining[cy:cy+wh, cx:cx+ww].sum())
                    context_score = int(context_map[cy:cy+wh, cx:cx+ww].sum())
                    score = (required_score, context_score)
                    if best is None or score > best[0]:
                        best = (score, cy, cx)

                _, y0, x0 = best
                origins[bi,ki] = torch.tensor([y0,x0])
                valid[bi,ki] = True
                remaining[y0:y0+wh, x0:x0+ww] = False

        return WindowPlan(origins.to(out_device), valid.to(out_device),
                          self.window_hw, (h,w))

def crop_windows(x: torch.Tensor, plan: WindowPlan) -> torch.Tensor:
    """Crop [B,...,H,W] into [B,K,...,wh,ww], padding invalid slots with zero."""
    if x.shape[0] != plan.origins.shape[0] or tuple(x.shape[-2:]) != plan.full_hw:
        raise ValueError("input and WindowPlan shape mismatch")
    b = x.shape[0]; k = plan.origins.shape[1]; wh,ww = plan.window_hw
    out = x.new_zeros((b,k,*x.shape[1:-2],wh,ww))
    for bi in range(b):
        for ki in range(k):
            if not bool(plan.valid[bi,ki]):
                continue
            y,x0 = [int(v) for v in plan.origins[bi,ki]]
            out[bi,ki] = x[bi,...,y:y+wh,x0:x0+ww]
    return out

def scatter_windows(windows: torch.Tensor, plan: WindowPlan,
                    base: torch.Tensor | None = None,
                    weight: torch.Tensor | None = None) -> torch.Tensor:
    """Overlap-average [B,K,...,wh,ww] back to [B,...,H,W].

    ``base`` is used only where no valid window writes. For logits/latents this
    prevents overwrite-order dependence at window overlaps.
    """
    b,k = windows.shape[:2]; h,w = plan.full_hw; wh,ww = plan.window_hw
    if (b,k) != tuple(plan.valid.shape):
        raise ValueError("window batch/slot mismatch")
    out = windows.new_zeros((b,*windows.shape[2:-2],h,w))
    cnt = windows.new_zeros((b,*([1]*(windows.ndim-4)),h,w))
    if weight is not None and weight.shape[:2] != (b,k):
        raise ValueError("weight batch/slot mismatch")
    for bi in range(b):
        for ki in range(k):
            if not bool(plan.valid[bi,ki]): continue
            y,x0 = [int(v) for v in plan.origins[bi,ki]]
            wwgt = 1.0 if weight is None else weight[bi,ki]
            out[bi,...,y:y+wh,x0:x0+ww] += windows[bi,ki] * wwgt
            cnt[bi,...,y:y+wh,x0:x0+ww] += wwgt
    covered = cnt > 0
    out = out / cnt.clamp_min(1)
    if base is not None:
        if tuple(base.shape) != tuple(out.shape):
            raise ValueError("base shape mismatch")
        out = torch.where(covered.expand_as(out), out, base)
    return out


def window_coverage(support: torch.Tensor, plan: WindowPlan) -> torch.Tensor:
    """Per-sample fraction of union support covered by selected windows."""
    if support.ndim == 4:
        union=support.bool().any(dim=1)
    elif support.ndim == 3:
        union=support.bool()
    else:
        raise ValueError("support must be [B,T,H,W] or [B,H,W]")
    if tuple(union.shape[-2:]) != plan.full_hw or union.shape[0] != plan.valid.shape[0]:
        raise ValueError("support and plan shape mismatch")
    cover=torch.zeros_like(union)
    wh,ww=plan.window_hw
    for bi in range(union.shape[0]):
        for ki in range(plan.valid.shape[1]):
            if not bool(plan.valid[bi,ki]): continue
            y,x0=[int(v) for v in plan.origins[bi,ki]]
            cover[bi,y:y+wh,x0:x0+ww]=True
    numer=(cover & union).flatten(1).sum(1).float()
    denom=union.flatten(1).sum(1).float()
    return torch.where(denom>0,numer/denom,torch.ones_like(denom))
