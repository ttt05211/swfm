"""Final Static-Protected Motion Composition."""
import torch


def static_protected_compose(static_occ: torch.Tensor,
                             wm_occ: torch.Tensor,
                             confident_static: torch.Tensor,
                             dynamic_classes,
                             write_support: torch.Tensor | None = None) -> torch.Tensor:
    """Compose dynamic WM output over deterministic static transport.

    Important distinction:
    - ``generation_support`` selects where the expensive WM is computed;
    - ``write_support`` (usually the occupancy-space causal motion tube) only
      gates *dynamic writes*; it never wholesale overwrites a region;
    - ``confident_static`` is the hard protection mask.

    This prevents decoder spillover outside the causal motion tube while still
    keeping ``support != overwrite mask``: within support, only predicted
    dynamic occupied voxels are written.
    """
    if static_occ.shape != wm_occ.shape:
        raise ValueError("static_occ and wm_occ must have identical shape")
    if confident_static.shape != static_occ.shape:
        try:
            confident_static = confident_static.expand_as(static_occ)
        except RuntimeError as e:
            raise ValueError("confident_static is not broadcastable") from e
    dyn = torch.zeros_like(wm_occ, dtype=torch.bool)
    for c in dynamic_classes:
        dyn |= wm_occ.eq(int(c))
    writable = (~confident_static.bool()) & dyn
    if write_support is not None:
        ws = write_support.bool()
        if ws.shape != static_occ.shape:
            # BEV [..,H,W] support may be expanded across occupancy height D.
            if ws.shape == static_occ.shape[:-1]:
                ws = ws.unsqueeze(-1).expand_as(static_occ)
            else:
                try:
                    ws = ws.expand_as(static_occ)
                except RuntimeError as e:
                    raise ValueError("write_support is not broadcastable") from e
        writable &= ws
    return torch.where(writable, wm_occ, static_occ)
