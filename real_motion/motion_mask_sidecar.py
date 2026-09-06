"""True-motion latent-mask sidecar and normalized FM weighting utilities.

The sidecar stores the same Moving-mIoU-v2 dual-box motion-associated support
used by the v8 diagnostic, after exact occupancy-BEV -> 50x50 latent any-pooling.
It is a training-label artifact only and is never a model input at inference.
"""
from __future__ import annotations

from pathlib import Path

import torch


MOTION_MASK_SIDECAR_VERSION = "p0_f9_v8_motion_mask_sidecar_v1"
MOTION_MASK_PROTOCOL = "moving_v2_dual_box_exact_4x4_any_pool_v1"
EXPECTED_MASK_SHAPE = (6, 50, 50)


class MotionMaskSidecar:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        obj = torch.load(self.path, map_location="cpu", weights_only=False)
        if obj.get("version") != MOTION_MASK_SIDECAR_VERSION:
            raise RuntimeError(
                f"unsupported motion-mask sidecar version {obj.get('version')!r}"
            )
        self.metadata = dict(obj.get("metadata") or {})
        if self.metadata.get("motion_mask_protocol") != MOTION_MASK_PROTOCOL:
            raise RuntimeError("motion-mask sidecar protocol mismatch")
        rows = list(obj.get("records") or [])
        if not rows:
            raise RuntimeError("empty motion-mask sidecar")
        self.records = {}
        for row in rows:
            sid = str(row.get("sample_id"))
            mask = row.get("motion_mask_latent")
            if sid in self.records:
                raise RuntimeError(f"duplicate motion-mask sample_id {sid}")
            if not torch.is_tensor(mask) or tuple(mask.shape) != EXPECTED_MASK_SHAPE:
                raise RuntimeError(
                    f"{sid}: motion mask must be {EXPECTED_MASK_SHAPE}, got "
                    f"{getattr(mask, 'shape', None)}"
                )
            self.records[sid] = mask.bool().contiguous()

    def __len__(self):
        return len(self.records)

    def validate_sample_ids(self, sample_ids) -> None:
        expected = {str(x) for x in sample_ids}
        actual = set(self.records)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise RuntimeError(
                "motion-mask sidecar sample set differs from train cache: "
                f"missing={missing[:3]} extra={extra[:3]}"
            )

    def get_batch(self, sample_ids, *, device=None) -> torch.Tensor:
        rows = []
        for sid in sample_ids:
            key = str(sid)
            if key not in self.records:
                raise KeyError(f"motion-mask sidecar missing sample {key}")
            rows.append(self.records[key])
        out = torch.stack(rows, dim=0)
        if device is not None:
            out = out.to(device=device, non_blocking=True)
        return out


def normalized_motion_weighted_mse(
    squared_error: torch.Tensor,
    motion_mask: torch.Tensor,
    motion_lambda: float,
) -> torch.Tensor:
    """Normalized spatially-weighted native FM MSE.

    Args:
        squared_error: [N,T,C,H,W], exactly ``(v_pred-v_target)^2``.
        motion_mask: [N,T,H,W] boolean motion-associated latent cells.
        motion_lambda: lambda in ``w = 1 + lambda * mask``.

    The denominator includes the channel broadcast, so lambda=0 is exactly the
    ordinary elementwise mean of ``squared_error`` (up to floating reduction
    roundoff of the same tensor).
    """
    if squared_error.ndim != 5:
        raise ValueError("squared_error must be [N,T,C,H,W]")
    expected = (
        int(squared_error.shape[0]),
        int(squared_error.shape[1]),
        int(squared_error.shape[-2]),
        int(squared_error.shape[-1]),
    )
    if motion_mask.ndim != 4 or tuple(motion_mask.shape) != expected:
        raise ValueError(
            f"motion_mask must be [N,T,H,W]={expected}, got {tuple(motion_mask.shape)}"
        )
    lam = float(motion_lambda)
    if lam < 0.0:
        raise ValueError("motion_lambda must be non-negative")
    weights = 1.0 + lam * motion_mask.to(dtype=squared_error.dtype)
    weights = weights[:, :, None].expand_as(squared_error)
    return (squared_error * weights).sum() / weights.sum().clamp_min(1.0)


def effective_motion_weight_mass(motion_fraction: float, motion_lambda: float) -> float:
    p = float(motion_fraction)
    lam = float(motion_lambda)
    if not 0.0 <= p <= 1.0 or lam < 0.0:
        raise ValueError("invalid motion fraction/lambda")
    return ((1.0 + lam) * p) / max(1.0 + lam * p, 1e-12)
