"""One small spatiotemporal Transformer and one calibrated utility objective."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .static_evidence_selector import HISTORY_DIM, CONTEXT_DIM, FEATURE_PROTOCOL, PROTOCOL


@dataclass(frozen=True)
class SelectorConfig:
    width: int = 64
    heads: int = 4
    layers: int = 2
    keep_prior: float = .05


class StaticEvidenceSelector(nn.Module):
    """30 causal history tokens (6 times x cross5) + one future-query token.

    Emits only a whole-patch correctness probability. No class/shape decoder,
    future occupancy input, new-source generator or V18 parameter update.
    """
    def __init__(self, config=SelectorConfig()):
        super().__init__()
        if config.width < 4 or config.heads < 1 or config.width % config.heads or config.layers < 1 or not 0 < config.keep_prior < .5:
            raise ValueError("invalid selector architecture")
        self.config = config
        self.history_encoder = nn.Linear(HISTORY_DIM, config.width)
        self.query_encoder = nn.Linear(CONTEXT_DIM, config.width)
        self.time_embedding = nn.Parameter(torch.randn(1, 6, 1, config.width) * .02)
        self.space_embedding = nn.Parameter(torch.randn(1, 1, 5, config.width) * .02)
        self.query_embedding = nn.Parameter(torch.randn(1, 1, config.width) * .02)
        block = nn.TransformerEncoderLayer(config.width, config.heads, dim_feedforward=config.width * 2,
                                          dropout=0., activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(block, config.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.width)
        self.keep_head = nn.Linear(config.width, 1)
        nn.init.zeros_(self.keep_head.weight)
        nn.init.constant_(self.keep_head.bias, math.log(config.keep_prior / (1 - config.keep_prior)))

    def forward(self, history, context):
        if history.ndim != 4 or tuple(history.shape[1:]) != (6, 5, HISTORY_DIM) or tuple(context.shape) != (len(history), CONTEXT_DIM):
            raise ValueError("selector tensor contract mismatch")
        if not len(history):
            return context.new_empty((0,))
        x = self.history_encoder(history.float()) + self.time_embedding + self.space_embedding
        q = self.query_encoder(context.float()).unsqueeze(1) + self.query_embedding
        tokens = torch.cat((x.flatten(1, 2), q), dim=1)
        return self.keep_head(self.norm(self.transformer(tokens)[:, -1])).squeeze(-1)

    def contract(self):
        return {"protocol": PROTOCOL, "feature_protocol": FEATURE_PROTOCOL, "model_config": asdict(self.config)}


def utility_loss(logits, correct, wrong, sampling_weight):
    """ONE BCE: correct*softplus(-logit) + wrong*softplus(logit).

    Its calibrated optimum is P(correct semantic voxel | whole patch evidence).
    At the frozen p>=.5 gate, expected correct-minus-wrong utility is positive.
    Importance weights undo sign-balanced sampling; no focal or pos_weight.
    """
    if any(x.shape != logits.shape for x in (correct, wrong, sampling_weight)):
        raise ValueError("utility objective shapes differ")
    if any(not torch.isfinite(x).all() for x in (logits, correct, wrong, sampling_weight)) or torch.any(correct < 0) or torch.any(wrong < 0) or torch.any(sampling_weight <= 0):
        raise ValueError("nonfinite or invalid utility objective")
    correct, wrong, w = correct.float(), wrong.float(), sampling_weight.float()
    denominator = ((correct + wrong) * w).sum()
    if denominator <= 0:
        raise ValueError("empty utility objective")
    return ((correct * F.softplus(-logits.float()) + wrong * F.softplus(logits.float())) * w).sum() / denominator
