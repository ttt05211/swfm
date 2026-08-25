import torch

def static_protected_compose(static_occ: torch.Tensor,
                             wm_occ: torch.Tensor,
                             confident_static: torch.Tensor,
                             dynamic_classes) -> torch.Tensor:
    """Static-Protected Motion Composition.

    Generation support is intentionally absent: it controls computation, not
    overwrite permission. Confident-static voxels are never changed.
    """
    if static_occ.shape != wm_occ.shape:
        raise ValueError("static_occ and wm_occ must have identical shape")
    if confident_static.shape != static_occ.shape:
        try: confident_static = confident_static.expand_as(static_occ)
        except RuntimeError as e: raise ValueError("confident_static is not broadcastable") from e
    dyn=torch.zeros_like(wm_occ,dtype=torch.bool)
    for c in dynamic_classes: dyn |= wm_occ.eq(int(c))
    writable=(~confident_static.bool()) & dyn
    return torch.where(writable,wm_occ,static_occ)
