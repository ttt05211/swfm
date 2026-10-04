"""Shared full-Z local encoder + cross-attention; distinct geometry/action heads."""
from __future__ import annotations
from dataclasses import asdict
import torch
from torch import nn
from torch.nn import functional as F
from .causal_column_completion import ColumnConfig, GENERATE, KEEP, ADD, REMOVE, CONTEXT_DIM


class CrossBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.qnorm, self.knorm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2*width), nn.GELU(), nn.Linear(2*width, width))

    def forward(self, q, memory, invalid):
        k = self.knorm(memory)
        q = q + self.attention(self.qnorm(q), k, k, key_padding_mask=invalid, need_weights=False)[0]
        return q+self.ff(q)


class CausalColumnModel(nn.Module):
    def __init__(self, config=ColumnConfig(), *, history_frames=6):
        super().__init__(); config.validate(); self.config = config
        if history_frames not in (4, 6): raise ValueError('four or six history observations required')
        self.history_frames = history_frames
        d, z, e = config.width, config.z_bins, config.semantic_dim
        self.semantic = nn.Embedding(19, e)
        # Ordered Z flattening preserves bin identity; no height mean/top-only collapse.
        self.column = nn.Linear(z*(e+2), d)
        self.spatial = nn.Sequential(nn.Conv2d(d, d, 3, padding=1), nn.GELU(), nn.Conv2d(d, d, 3, padding=1))
        self.position = nn.Linear(3, d)
        self.query = nn.Linear(CONTEXT_DIM+2*z*e, d)
        self.kind = nn.Embedding(2, d); self.classes = nn.Embedding(17, d)
        self.decoder = nn.ModuleList([CrossBlock(d, config.heads) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(d)
        self.generation = nn.Linear(d, z)
        self.refinement = nn.Linear(d, z*3)
        nn.init.zeros_(self.generation.weight); nn.init.constant_(self.generation.bias, -4.)
        nn.init.zeros_(self.refinement.weight); nn.init.zeros_(self.refinement.bias)
        with torch.no_grad(): self.refinement.bias.reshape(z, 3)[:, KEEP] = 4.
        # Finite TRAIN-derived corrections persisted with checkpoint.
        self.register_buffer("generation_pos_weight", torch.ones(()))
        self.register_buffer("refine_class_weights", torch.ones(3))
        coords = torch.stack(torch.meshgrid(torch.linspace(-1, 0, history_frames), torch.linspace(-1, 1, config.patch),
            torch.linspace(-1, 1, config.patch), indexing="ij"), dim=-1)
        self.register_buffer("memory_coordinates", coords.reshape(1, history_frames*config.patch**2, 3), persistent=False)

    def encode_local(self, history, flags, base, fallback, context, kind, classes, *, query_extra=None):
        n, z, p, d = len(kind), self.config.z_bins, self.config.patch, self.config.width
        t = self.history_frames
        if (history.shape != (n, t, p, p, z) or flags.shape != history.shape
                or base.shape != (n, z) or fallback.shape != (n, z) or context.shape != (n, CONTEXT_DIM)
                or classes.shape != (n,)):
            raise ValueError("column feature shapes differ from checkpoint contract")
        if n == 0: return context.new_empty((0, d))
        # UNKNOWN semantics are zeroed BEFORE spatial convolution; they may not
        # masquerade as free or become learned content through an embedding.
        valid = history != 18
        emb = self.semantic(history.long())*valid[..., None]
        bits = torch.stack(((flags & 1) != 0, (flags & 2) != 0), dim=-1).to(emb.dtype)*valid[..., None]
        x = self.column(torch.cat((emb, bits), dim=-1).flatten(-2))*valid.any(-1)[..., None]
        x = x.reshape(n*t, p, p, d).permute(0, 3, 1, 2)
        x = self.spatial(x).permute(0, 2, 3, 1).reshape(n, t*p*p, d)
        x = x+self.position(self.memory_coordinates.to(x.dtype))
        invalid = ~valid.any(-1).flatten(1)
        # An all-unknown query gets a ZERO dummy token, never all-masked NaNs.
        empty = invalid.all(1)
        if empty.any():
            x = x.clone(); invalid = invalid.clone(); x[empty, 0] = 0; invalid[empty, 0] = False
        q = (self.query(torch.cat((context.float(), self.semantic(base.long()).flatten(1),
                                  self.semantic(fallback.long()).flatten(1)), dim=1))
             +self.kind(kind.long())+self.classes(classes.long())).unsqueeze(1)
        if query_extra is not None:
            if query_extra.shape != (n, d): raise ValueError('continuous source query shape mismatch')
            q = q + query_extra[:, None].to(q.dtype)
        for block in self.decoder: q = block(q, x, invalid)
        return self.norm(q[:, 0])

    def forward(self, history, flags, base, fallback, context, kind, classes, *, query_extra=None):
        q = self.encode_local(history, flags, base, fallback, context, kind, classes, query_extra=query_extra)
        return self.generation(q), self.refinement(q).reshape(len(kind), self.config.z_bins, 3)

    def calibrated_probabilities(self, generation, refinement, kind, legal):
        if (not torch.isfinite(generation).all() or not torch.isfinite(refinement).all()
                or not torch.isfinite(self.generation_pos_weight).all()
                or not torch.isfinite(self.refine_class_weights).all()
                or self.generation_pos_weight <= 0 or torch.any(self.refine_class_weights <= 0)):
            raise RuntimeError("nonfinite prediction or invalid TRAIN calibration weights")
        g = generation.float()-self.generation_pos_weight.log()
        r = refinement.float()-self.refine_class_weights.log()
        r = r.masked_fill(~legal.bool(), -torch.inf)
        p = r.softmax(-1)
        gp = torch.stack((1-g.sigmoid(), g.sigmoid(), torch.zeros_like(g)), dim=-1)
        gp = gp*legal
        gp[..., KEEP] = 1-gp[..., ADD]
        return torch.where((kind == GENERATE)[:, None, None], gp, p)

    def contract(self): return asdict(self.config)


def column_loss(model, generation, refinement, kind, legal, target, weight, *, materialize_stats=True, distributed=None):
    """Two task losses, averaged by type; importance restores query sampling.

    Refine target is action utility with KEEP-on-tie, NOT indiscriminate deletion
    whenever GT is a different class. Illegal actions do not enter softmax.
    """
    if target.shape != generation.shape or legal.shape != refinement.shape or weight.shape != kind.shape:
        raise ValueError("loss shape mismatch")
    if not torch.isfinite(weight).all() or torch.any(weight <= 0): raise ValueError("invalid sampling weights")
    if not torch.gather(legal.bool(), -1, target.long()[..., None]).all(): raise ValueError("illegal target action")
    terms, stats = [], {}
    by_task = {}; denominators = generation.new_zeros(2, dtype=torch.float32)
    for task in (0, 1):
        take = kind == task
        if not take.any(): continue
        w = weight[take, None].float()
        if task == GENERATE:
            mask = legal[take, :, ADD].float()
            loss = F.binary_cross_entropy_with_logits(generation[take].float(), (target[take] == ADD).float(),
                        pos_weight=model.generation_pos_weight, reduction="none")
        else:
            mask = (legal[take, :, ADD] | legal[take, :, REMOVE]).float()
            logits = refinement[take].float().masked_fill(~legal[take].bool(), -1e9)
            loss = F.cross_entropy(logits.flatten(0, 1), target[take].long().flatten(),
                                   weight=model.refine_class_weights, reduction="none").reshape_as(mask)
        denom = (w*mask).sum()
        if denom <= 0: continue
        value = (loss*w*mask).sum()/denom
        by_task[task] = value; denominators[task] = denom
        terms.append(value); stats["generation_bce" if task == 0 else "refine_action_ce"] = float(value.detach()) if materialize_stats else value.detach()
    if distributed is not None:
        total, values = distributed.columns(by_task, denominators, generation.new_zeros(()))
        return total, {k: float(v.detach()) if materialize_stats else v.detach() for k, v in values.items()}
    if not terms: raise RuntimeError("batch has no legal supervised edits")
    total = torch.stack(terms).mean()
    if not torch.isfinite(total): raise RuntimeError("nonfinite completion loss")
    return total, stats
