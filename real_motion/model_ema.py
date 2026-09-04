"""Small FP32 EMA utility used by P0-F9 training."""
from __future__ import annotations

import copy
import math

import torch


class ModelEMA:
    """FP32 exponential moving average with GenieDrive/BEVDepth-style ramp."""

    def __init__(self, model, *, decay: float = 0.999, updates: int = 0):
        if not 0.0 < float(decay) < 1.0:
            raise ValueError("EMA decay must be in (0,1)")
        self.decay = float(decay)
        self.updates = int(updates)
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)
        # Keep the smoothing accumulator in FP32 even under BF16 AMP.
        for p in self.model.parameters():
            if p.is_floating_point():
                p.data = p.data.float()
        for name, b in self.model.named_buffers():
            if b.is_floating_point():
                b.data = b.data.float()

    def current_decay(self) -> float:
        return self.decay * (1.0 - math.exp(-float(self.updates) / 2000.0))

    @torch.no_grad()
    def update(self, model) -> float:
        self.updates += 1
        d = self.current_decay()
        source = model.state_dict()
        target = self.model.state_dict()
        if source.keys() != target.keys():
            raise RuntimeError("EMA/model state_dict keys differ")
        for key, value in target.items():
            src = source[key].detach().to(device=value.device)
            if value.is_floating_point():
                value.mul_(d).add_(src.float(), alpha=1.0 - d)
            else:
                value.copy_(src)
        return d

    def state_dict(self):
        return {
            "updates": int(self.updates),
            "decay": float(self.decay),
            "state_dict": self.model.state_dict(),
        }

    def load_state_dict(self, obj):
        if not isinstance(obj, dict) or "state_dict" not in obj:
            raise ValueError("invalid EMA state")
        if float(obj.get("decay", self.decay)) != float(self.decay):
            raise RuntimeError("EMA decay differs from checkpoint")
        self.model.load_state_dict(obj["state_dict"], strict=True)
        self.updates = int(obj.get("updates", 0))
