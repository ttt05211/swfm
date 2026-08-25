from dataclasses import dataclass
from typing import Tuple
import torch

@dataclass(frozen=True)
class WindowPlan:
    origins: torch.Tensor
    valid: torch.Tensor
    window_hw: Tuple[int, int]
    full_hw: Tuple[int, int]

class WindowPlanner:
    """Greedy fixed-window planner over a latent support union."""
    def __init__(self, window_hw=(20,20), max_windows=8):
        self.window_hw = tuple(window_hw); self.max_windows = int(max_windows)
        if min(self.window_hw) <= 0 or self.max_windows <= 0:
            raise ValueError("invalid window configuration")

    def plan(self, support: torch.Tensor) -> WindowPlan:
        if support.ndim == 4:
            union = support.bool().any(dim=1)
        elif support.ndim == 3:
            union = support.bool()
        else:
            raise ValueError("support must be [B,T,H,W] or [B,H,W]")
        b,h,w = union.shape; wh,ww = self.window_hw
        if wh > h or ww > w:
            raise ValueError("window cannot be larger than latent map")
        origins = torch.full((b,self.max_windows,2), -1, dtype=torch.long, device=union.device)
        valid = torch.zeros((b,self.max_windows), dtype=torch.bool, device=union.device)
        for bi in range(b):
            remaining = union[bi].clone()
            for ki in range(self.max_windows):
                if not remaining.any(): break
                ys, xs = torch.where(remaining)
                y0 = max(0, min(int(ys[0])-wh//2, h-wh)); x0 = max(0, min(int(xs[0])-ww//2, w-ww))
                best=(int(remaining[y0:y0+wh,x0:x0+ww].sum()),y0,x0)
                stride=max(1,len(ys)//64)
                for y,x in zip(ys[::stride],xs[::stride]):
                    cy=max(0,min(int(y)-wh//2,h-wh)); cx=max(0,min(int(x)-ww//2,w-ww))
                    score=int(remaining[cy:cy+wh,cx:cx+ww].sum())
                    if score>best[0]: best=(score,cy,cx)
                _,y0,x0=best
                origins[bi,ki]=torch.tensor([y0,x0],device=origins.device); valid[bi,ki]=True
                remaining[y0:y0+wh,x0:x0+ww]=False
        return WindowPlan(origins,valid,self.window_hw,(h,w))

def crop_windows(x: torch.Tensor, plan: WindowPlan) -> torch.Tensor:
    if x.shape[0] != plan.origins.shape[0] or tuple(x.shape[-2:]) != plan.full_hw:
        raise ValueError("input and WindowPlan shape mismatch")
    b=x.shape[0]; k=plan.origins.shape[1]; wh,ww=plan.window_hw
    out=x.new_zeros((b,k,*x.shape[1:-2],wh,ww))
    for bi in range(b):
        for ki in range(k):
            if not bool(plan.valid[bi,ki]): continue
            y,x0=[int(v) for v in plan.origins[bi,ki]]
            out[bi,ki]=x[bi,...,y:y+wh,x0:x0+ww]
    return out

def scatter_windows(windows: torch.Tensor, plan: WindowPlan,
                    base: torch.Tensor | None = None,
                    weight: torch.Tensor | None = None) -> torch.Tensor:
    b,k=windows.shape[:2]; h,w=plan.full_hw; wh,ww=plan.window_hw
    if (b,k) != tuple(plan.valid.shape): raise ValueError("window batch/slot mismatch")
    out=windows.new_zeros((b,*windows.shape[2:-2],h,w))
    cnt=windows.new_zeros((b,*([1]*(windows.ndim-4)),h,w))
    if weight is not None and weight.shape[:2] != (b,k): raise ValueError("weight batch/slot mismatch")
    for bi in range(b):
        for ki in range(k):
            if not bool(plan.valid[bi,ki]): continue
            y,x0=[int(v) for v in plan.origins[bi,ki]]
            wwgt=1.0 if weight is None else weight[bi,ki]
            out[bi,...,y:y+wh,x0:x0+ww]+=windows[bi,ki]*wwgt
            cnt[bi,...,y:y+wh,x0:x0+ww]+=wwgt
    covered=cnt>0; out=out/cnt.clamp_min(1)
    if base is not None:
        if tuple(base.shape)!=tuple(out.shape): raise ValueError("base shape mismatch")
        out=torch.where(covered.expand_as(out),out,base)
    return out
