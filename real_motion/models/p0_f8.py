"""P0-F8 Strong-W2Det anchor World Model with an anchor-relative edit head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from real_motion.edit_repair import NUM_ACTIONS, NUM_RESULT_CLASSES
from real_motion.semantic_repair import collapse_dynamic_logits

from .anchor_cfm import AnchorWindowCFM
from .p0_f4 import make_p0_f4_model


class AnchorRelativeEditHead(nn.Module):
    """Predict KEEP/CLEAR/WRITE from WM semantic evidence and anchor context."""

    def __init__(
        self,
        *,
        semantic_classes: int = NUM_RESULT_CLASSES,
        anchor_embed_dim: int = 16,
        horizon_embed_dim: int = 8,
        hidden_dim: int = 64,
        future_frames: int = 6,
        keep_bias: float = 2.0,
    ):
        super().__init__()
        self.semantic_classes = int(semantic_classes)
        self.future_frames = int(future_frames)
        self.anchor_embed = nn.Embedding(self.semantic_classes, int(anchor_embed_dim))
        self.horizon_embed = nn.Embedding(self.future_frames, int(horizon_embed_dim))
        in_dim = self.semantic_classes + int(anchor_embed_dim) + int(horizon_embed_dim)
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, int(hidden_dim))
        self.fc2 = nn.Linear(int(hidden_dim), NUM_ACTIONS)

        nn.init.normal_(self.anchor_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.horizon_embed.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        # KEEP bias makes the initial argmax anchor-preserving.  The final-layer
        # weights must nevertheless be large enough that edit supervision has a
        # useful gradient path back through frozen decoder evidence into the WM;
        # an almost-zero head would make shared-gradient lambda calibration
        # artificially explode.
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc2.bias)
        with torch.no_grad():
            self.fc2.bias[0] = float(keep_bias)

    def forward(
        self,
        semantic_logits: torch.Tensor,
        anchor_slots: torch.Tensor,
        horizons: torch.Tensor,
        *,
        already_collapsed: bool = False,
    ) -> torch.Tensor:
        if semantic_logits.ndim != 2:
            raise ValueError("semantic logits must be [N,C]")
        if already_collapsed:
            collapsed = semantic_logits.float()
            if collapsed.shape[-1] != self.semantic_classes:
                raise ValueError("collapsed semantic class count mismatch")
        else:
            collapsed = collapse_dynamic_logits(semantic_logits).float()
        n = int(collapsed.shape[0])
        anchor = anchor_slots.to(device=collapsed.device, dtype=torch.long).reshape(-1)
        horizon = horizons.to(device=collapsed.device, dtype=torch.long).reshape(-1)
        if anchor.numel() != n or horizon.numel() != n:
            raise ValueError("anchor/horizon length mismatch")
        if n:
            if int(anchor.min()) < 0 or int(anchor.max()) >= self.semantic_classes:
                raise ValueError("anchor slot out of range")
            if int(horizon.min()) < 0 or int(horizon.max()) >= self.future_frames:
                raise ValueError("horizon index out of range")
        # Log-probabilities provide a stable semantic evidence scale independent
        # of the frozen decoder logit magnitude.
        sem = F.log_softmax(collapsed, dim=-1)
        x = torch.cat([
            sem,
            self.anchor_embed(anchor),
            self.horizon_embed(horizon),
        ], dim=-1)
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)


class AnchorRelativeEditWM(AnchorWindowCFM):
    def __init__(self, transition, *, edit_head: AnchorRelativeEditHead, **kwargs):
        super().__init__(transition, **kwargs)
        self.edit_head = edit_head


def make_p0_f8_model(
    window=20,
    *,
    sample_steps=10,
    source_noise_std=0.0,
    keep_bias=2.0,
):
    """Reuse the P0-F6 uniform-FM transition and add only the edit head."""
    base = make_p0_f4_model(
        window,
        sample_steps=sample_steps,
        source_noise_std=source_noise_std,
    )
    return AnchorRelativeEditWM(
        base.transition,
        edit_head=AnchorRelativeEditHead(keep_bias=float(keep_bias)),
        rescale_factor=base.rescale_factor,
        sample_steps=base.sample_steps,
        alpha_shift=base.alpha_shift,
        source_noise_std=base.source_noise_std,
    )
