from dataclasses import dataclass
from typing import Tuple
import torch


@dataclass(frozen=True)
class WindowPlan:
    origins: torch.Tensor      # [B,K,2], y/x top-left, -1 for padding
    valid: torch.Tensor        # [B,K]
    window_hw: Tuple[int, int]
    full_hw: Tuple[int, int]


class WindowPlanner:
    """Fixed motion-window planner with future-target-first semantics.

    ``required_support`` is the only signal allowed to create a window.
    ``context_support`` breaks ties among windows that cover the same number of
    still-uncovered required cells. Candidate top-left positions are evaluated
    exhaustively on the tiny 50x50 latent map, so context can actually shift a
    window toward useful history instead of being restricted to target-centered
    crops.
    """
    def __init__(self, window_hw=(20, 20), max_windows=8):
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

    @staticmethod
    def _window_sums(mask_hw: torch.Tensor, wh: int, ww: int) -> torch.Tensor:
        # Exact integral-image window sums on CPU, output [H-wh+1,W-ww+1].
        x = mask_hw.to(torch.int32)
        integral = torch.zeros((x.shape[0] + 1, x.shape[1] + 1), dtype=torch.int32)
        integral[1:, 1:] = x.cumsum(0).cumsum(1)
        return (integral[wh:, ww:] - integral[:-wh, ww:]
                - integral[wh:, :-ww] + integral[:-wh, :-ww])

    def plan(self, required_support: torch.Tensor,
             context_support: torch.Tensor | None = None) -> WindowPlan:
        required = self._union_support(required_support)
        context = required if context_support is None else self._union_support(context_support)
        if required.shape != context.shape:
            raise ValueError("required_support/context_support shape mismatch")

        out_device = required.device
        req = required.detach().cpu()
        ctx = context.detach().cpu()
        B, H, W = req.shape
        wh, ww = self.window_hw
        if wh > H or ww > W:
            raise ValueError("window cannot be larger than latent map")

        origins = torch.full((B, self.max_windows, 2), -1, dtype=torch.long)
        valid = torch.zeros((B, self.max_windows), dtype=torch.bool)
        tie_base = wh * ww + 1

        for b in range(B):
            remaining = req[b].clone()
            context_map = ctx[b]
            for k in range(self.max_windows):
                if not bool(remaining.any()):
                    break
                req_score = self._window_sums(remaining, wh, ww)
                ctx_score = self._window_sums(context_map, wh, ww)
                # Lexicographic objective: required coverage dominates context.
                score = req_score * tie_base + ctx_score
                score = torch.where(req_score > 0, score, torch.full_like(score, -1))
                flat = int(score.reshape(-1).argmax())
                out_w = score.shape[1]
                y0, x0 = divmod(flat, out_w)
                origins[b, k] = torch.tensor([y0, x0])
                valid[b, k] = True
                remaining[y0:y0+wh, x0:x0+ww] = False

        return WindowPlan(origins.to(out_device), valid.to(out_device),
                          self.window_hw, (H, W))


def _linear_indices(plan: WindowPlan, device=None):
    device = device or plan.origins.device
    origins = plan.origins.to(device=device, dtype=torch.long)
    valid = plan.valid.to(device=device)
    B, K = valid.shape
    H, W = plan.full_hw
    wh, ww = plan.window_hw
    dy = torch.arange(wh, device=device).view(1, 1, wh, 1)
    dx = torch.arange(ww, device=device).view(1, 1, 1, ww)
    y = origins[..., 0].view(B, K, 1, 1) + dy
    x = origins[..., 1].view(B, K, 1, 1) + dx
    # Invalid padded windows have origin -1; clamp their indices and zero them
    # by ``valid`` in the caller.
    y = y.clamp(0, H - 1)
    x = x.clamp(0, W - 1)
    return (y * W + x).reshape(B, K, wh * ww), valid


def crop_windows(x: torch.Tensor, plan: WindowPlan) -> torch.Tensor:
    """Vectorized crop: [B,...,H,W] -> [B,K,...,wh,ww]."""
    if x.shape[0] != plan.origins.shape[0] or tuple(x.shape[-2:]) != plan.full_hw:
        raise ValueError("input and WindowPlan shape mismatch")
    B = x.shape[0]
    mid = x.shape[1:-2]
    H, W = plan.full_hw
    wh, ww = plan.window_hw
    C = 1
    for s in mid:
        C *= int(s)
    src = x.reshape(B, C, H * W)
    idx, valid = _linear_indices(plan, x.device)  # [B,K,P]
    K, P = idx.shape[1], idx.shape[2]
    gathered = torch.gather(
        src[:, None, :, :].expand(B, K, C, H * W),
        3,
        idx[:, :, None, :].expand(B, K, C, P),
    )
    gathered = gathered * valid[:, :, None, None].to(gathered.dtype)
    return gathered.reshape(B, K, *mid, wh, ww)


def scatter_windows(windows: torch.Tensor, plan: WindowPlan,
                    base: torch.Tensor | None = None,
                    weight: torch.Tensor | None = None) -> torch.Tensor:
    """Vectorized overlap-average scatter back to [B,...,H,W].

    ``weight`` may be [B,K] and is applied uniformly to each window. Invalid
    padded windows never contribute. ``base`` is used only where no window
    writes.
    """
    B, K = windows.shape[:2]
    if (B, K) != tuple(plan.valid.shape):
        raise ValueError("window batch/slot mismatch")
    mid = windows.shape[2:-2]
    wh, ww = plan.window_hw
    H, W = plan.full_hw
    if tuple(windows.shape[-2:]) != (wh, ww):
        raise ValueError("window spatial shape mismatch")

    C = 1
    for s in mid:
        C *= int(s)
    P = wh * ww
    vals = windows.reshape(B, K, C, P)
    idx, valid = _linear_indices(plan, windows.device)

    if weight is None:
        wk = torch.ones((B, K), device=windows.device, dtype=windows.dtype)
    else:
        if tuple(weight.shape) != (B, K):
            raise ValueError("weight must be [B,K]")
        wk = weight.to(device=windows.device, dtype=windows.dtype)
    wk = wk * valid.to(wk.dtype)
    vals = vals * wk[:, :, None, None]

    vals_flat = vals.permute(0, 2, 1, 3).reshape(B, C, K * P)
    idx_flat = idx.reshape(B, 1, K * P).expand(B, C, K * P)
    out = windows.new_zeros((B, C, H * W))
    out.scatter_add_(2, idx_flat, vals_flat)

    cnt_vals = wk[:, :, None].expand(B, K, P).reshape(B, 1, K * P)
    cnt_idx = idx.reshape(B, 1, K * P)
    cnt = windows.new_zeros((B, 1, H * W))
    cnt.scatter_add_(2, cnt_idx, cnt_vals)
    out = out / cnt.clamp_min(1)
    out = out.reshape(B, *mid, H, W)

    if base is not None:
        if tuple(base.shape) != tuple(out.shape):
            raise ValueError("base shape mismatch")
        covered = cnt.reshape(B, *([1] * len(mid)), H, W) > 0
        out = torch.where(covered.expand_as(out), out, base)
    return out


def window_coverage(support: torch.Tensor, plan: WindowPlan) -> torch.Tensor:
    """Per-sample fraction of union support covered by selected windows."""
    if support.ndim == 4:
        union = support.bool().any(dim=1)
    elif support.ndim == 3:
        union = support.bool()
    else:
        raise ValueError("support must be [B,T,H,W] or [B,H,W]")
    if union.shape[0] != plan.valid.shape[0] or tuple(union.shape[-2:]) != plan.full_hw:
        raise ValueError("support and plan shape mismatch")

    B, H, W = union.shape
    idx, valid = _linear_indices(plan, union.device)
    K, P = idx.shape[1:]
    cover = torch.zeros((B, 1, H * W), device=union.device, dtype=torch.float32)
    vals = valid.float()[:, :, None].expand(B, K, P).reshape(B, 1, K * P)
    cover.scatter_add_(2, idx.reshape(B, 1, K * P), vals)
    cover = cover.reshape(B, H, W) > 0
    numer = (cover & union).flatten(1).sum(1).float()
    denom = union.flatten(1).sum(1).float()
    return torch.where(denom > 0, numer / denom, torch.ones_like(denom))
