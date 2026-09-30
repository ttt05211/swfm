"""End-to-end V18 with causal, window-local source interaction.

No new sources, future labels, motion safety routing, or renderer changes.
The zero-initialized attention output makes initialization exactly V18.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn

from .local_st_world_model_v17 import LocalSTWMV17Config
from .local_st_world_model_v18_se2 import LocalSpatialTemporalWorldModelV18SE2

PROTOCOL = "p0_f9_v18_source_interaction_paired20_v1"
ARMS = ("v18_continuation", "v18_source_interaction")
INPUT_KEYS = ("features", "local_semantic_tube", "kta_displacement_xy_m",
              "frame_motion_features", "target_source_mask_tube")


@dataclass(frozen=True)
class InteractionConfig:
    neighbors: int = 16
    radius_m: float = 30.0

    def __post_init__(self):
        if self.neighbors < 1 or not 0 < self.radius_m < float("inf"):
            raise ValueError("invalid causal neighbor budget")


def neighbor_graph(centers, classes, frame_motion, window_ids, config):
    """CPU preparation, bounded N x K memory on GPU; NEVER prune by GT validity.

    Input source order is retained. A singleton/no-neighbor source has a masked
    dummy self index; its attention correction is explicitly zero.
    """
    centers = torch.as_tensor(centers).detach().cpu().float()
    classes = torch.as_tensor(classes).detach().cpu().long()
    motion = torch.as_tensor(frame_motion).detach().cpu().float()
    windows = torch.as_tensor(window_ids).detach().cpu().long()
    n, k = len(centers), config.neighbors
    if centers.shape != (n, 2) or classes.shape != (n,) or windows.shape != (n,) or motion.shape != (n, 6, 5):
        raise ValueError("causal graph input shape mismatch")
    if not torch.isfinite(centers).all() or not torch.isfinite(motion).all():
        raise ValueError("nonfinite causal graph inputs")
    indices = torch.arange(n)[:, None].expand(n, k).clone()
    valid = torch.zeros(n, k, dtype=torch.bool)
    edge = torch.zeros(n, k, 5)
    for w in windows.unique(sorted=True):
        ids = torch.nonzero(windows == w, as_tuple=True)[0]
        xy = centers[ids]
        distance = torch.cdist(xy, xy)
        distance.fill_diagonal_(float("inf"))
        count = min(k, len(ids))
        # Stable ties are deterministic relative to the frozen source order.
        local = distance.argsort(dim=1, stable=True)[:, :count]
        neigh = ids[local]
        legal = distance.gather(1, local) <= config.radius_m
        indices[ids, :count] = neigh
        valid[ids, :count] = legal
        delta = (centers[neigh] - xy[:, None]) / config.radius_m
        velocity = motion[:, -1, 2:4] * motion[:, -1, 4:5]
        relative_velocity = velocity[neigh] - velocity[ids, None]
        same_class = (classes[neigh] == classes[ids, None]).float()[..., None]
        edge[ids, :count] = torch.cat((delta, relative_velocity, same_class), dim=-1)
    return {"neighbor_indices": indices, "neighbor_valid": valid, "neighbor_edge": edge}


class CausalSourceAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.edge = nn.Sequential(nn.Linear(5, dim), nn.GELU(), nn.Linear(dim, dim))
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.)
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)

    def forward(self, queries, context, neighbor_indices, neighbor_valid, neighbor_edge):
        n = len(queries)
        if n == 0:
            return queries
        k = neighbor_indices.shape[1]
        if (neighbor_indices.shape != (n, k) or neighbor_valid.shape != (n, k)
                or neighbor_edge.shape != (n, k, 5) or k < 1):
            raise ValueError("neighbor graph shape mismatch")
        mask = neighbor_valid.bool()
        has_neighbor = mask.any(dim=1)
        safe_mask = mask.clone()
        safe_mask[:, 0] |= ~has_neighbor  # prevent all-masked softmax NaNs
        memory = self.memory_norm(context)[neighbor_indices] + self.edge(neighbor_edge.to(context.dtype))
        correction, _ = self.attention(self.query_norm(queries), memory, memory,
                                      key_padding_mask=~safe_mask, need_weights=False)
        return queries + correction * has_neighbor[:, None, None].to(correction.dtype)


class SourceInteractionV18(LocalSpatialTemporalWorldModelV18SE2):
    """Original shared encoder/decoder/XY/yaw are ALL trainable."""
    def __init__(self, config=LocalSTWMV17Config()):
        super().__init__(config)
        self.source_interaction = CausalSourceAttention(config.d_model, config.heads)

    def forward(self, *args, neighbor_indices, neighbor_valid, neighbor_edge,
                return_latents=False, decode_outputs=True, **kwargs):
        latent = super().forward(*args, return_latents=True, decode_outputs=False, **kwargs)
        q = self.source_interaction(latent["future_transport_queries"], latent["history_source_context"],
                                    neighbor_indices, neighbor_valid, neighbor_edge)
        out = self.decode_transport_queries(q) if decode_outputs else {}
        if return_latents:
            out.update(history_source_context=latent["history_source_context"], future_transport_queries=q)
        return out


def causal_forward(model, batch):
    args = [batch[key] for key in INPUT_KEYS]
    if isinstance(model, SourceInteractionV18):
        return model(*args, **{k: batch[k] for k in ("neighbor_indices", "neighbor_valid", "neighbor_edge")})
    return model(*args)
