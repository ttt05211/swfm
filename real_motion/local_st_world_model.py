"""Local spatial-temporal world model for structured rigid motion transport.

This is the first final-model replacement for the tiny v13 MLP probe.  It keeps
what the probe proved useful -- causal Strong sources, a KTA displacement prior,
the v13 displacement-residual target, and rigid source-shape transport -- while
replacing the flattened two-layer MLP with an object-centric local semantic
occupancy world model.

The learned module predicts six KTA displacement residuals.  It never predicts
absolute annotation centers and never generates the source object's voxel shape.
A zero-initialized residual head makes a fresh model exactly equal to KTA.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from .geometry import OccupancyGrid, relative_transform, warp_semantic_grid
from .metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from .motion_transport import FEATURE_DIM, FEATURE_NAMES, FUTURE_FRAMES, HISTORY_FRAMES

LOCAL_STWM_CACHE_VERSION = "p0_f9_local_stwm_v1"
LOCAL_TUBE_CONTRACT = "causal_source_centered_top_surface_semantic_bev_16m_0p8m_v1"
MODEL_PROTOCOL = "p0_f9_v16_local_spatial_temporal_world_model_v1"
DEFAULT_PATCH_SIZE_M = 16.0
DEFAULT_PATCH_RESOLUTION_M = 0.8
SEMANTIC_CLASSES = 18


def _check_semantic_grid(semantics: np.ndarray, grid: OccupancyGrid) -> np.ndarray:
    sem = np.asarray(semantics)
    if tuple(sem.shape) != tuple(grid.shape_hwd):
        raise ValueError(f"semantic grid {sem.shape} != {grid.shape_hwd}")
    return sem


def top_surface_semantic(
    semantics: np.ndarray,
    *,
    grid: OccupancyGrid = OccupancyGrid(),
    free_label: int = 17,
) -> np.ndarray:
    """Collapse [X,Y,Z] occupancy to a semantic top-surface BEV.

    The highest occupied voxel in every column is retained.  This keeps road and
    sidewalk where no object is above them, while a vehicle/pedestrian remains
    visible instead of being hidden by the ground voxel below it.
    """
    sem = _check_semantic_grid(semantics, grid)
    occupied = sem != int(free_label)
    has = occupied.any(axis=2)
    rev = occupied[:, :, ::-1]
    top_from_back = np.argmax(rev, axis=2)
    z = sem.shape[2] - 1 - top_from_back
    out = np.full(sem.shape[:2], int(free_label), dtype=np.uint8)
    if bool(has.any()):
        ix, iy = np.nonzero(has)
        out[ix, iy] = sem[ix, iy, z[ix, iy]].astype(np.uint8)
    return out


def _priority_lut(free_label: int) -> np.ndarray:
    """Rank labels for deterministic 2x2 pooling.

    Any dynamic class outranks static semantics; any occupied static class
    outranks free.  The exact ordering within the dynamic/static groups is only a
    deterministic collision rule and is not a learned semantic prior.
    """
    lut = np.full(256, 240, dtype=np.int16)
    dyn = [int(c) for c in DYNAMIC_CLASS_IDS]
    dyn_set = set(dyn)
    for rank, c in enumerate(dyn):
        lut[c] = rank
    base = len(dyn) + 8
    for c in range(SEMANTIC_CLASSES):
        if c == int(free_label) or c in dyn_set:
            continue
        lut[c] = base + c
    lut[int(free_label)] = 255
    return lut


def priority_pool2x2(labels: np.ndarray, *, free_label: int = 17) -> np.ndarray:
    """Downsample a semantic BEV by 2 while preserving small dynamic objects."""
    x = np.asarray(labels, dtype=np.uint8)
    if x.ndim != 2 or x.shape[0] % 2 or x.shape[1] % 2:
        raise ValueError("labels must be an even [X,Y] array")
    X, Y = x.shape
    cells = x.reshape(X // 2, 2, Y // 2, 2).transpose(0, 2, 1, 3).reshape(X // 2, Y // 2, 4)
    lut = _priority_lut(int(free_label))
    ranks = lut[cells]
    pick = np.argmin(ranks, axis=2)
    return np.take_along_axis(cells, pick[..., None], axis=2)[..., 0].astype(np.uint8)


def extract_bev_patch(
    bev: np.ndarray,
    center_xy_m: np.ndarray,
    *,
    grid: OccupancyGrid = OccupancyGrid(),
    patch_voxels: int = 40,
    free_label: int = 17,
) -> np.ndarray:
    """Extract a fixed metric patch around one center in the t0 ego frame."""
    if patch_voxels <= 0 or patch_voxels % 2:
        raise ValueError("patch_voxels must be a positive even integer")
    arr = np.asarray(bev, dtype=np.uint8)
    if tuple(arr.shape) != tuple(grid.shape_hwd[:2]):
        raise ValueError("BEV/grid shape mismatch")
    xy = np.asarray(center_xy_m, dtype=np.float64)
    if xy.shape != (2,):
        raise ValueError("center_xy_m must be [2]")
    vx, vy = float(grid.voxel_size[0]), float(grid.voxel_size[1])
    cx = int(np.floor((xy[0] - float(grid.x_min)) / vx))
    cy = int(np.floor((xy[1] - float(grid.y_min)) / vy))
    half = patch_voxels // 2
    x0, y0 = cx - half, cy - half
    x1, y1 = x0 + patch_voxels, y0 + patch_voxels
    out = np.full((patch_voxels, patch_voxels), int(free_label), dtype=np.uint8)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x1, arr.shape[0]), min(y1, arr.shape[1])
    if sx0 < sx1 and sy0 < sy1:
        out[sx0 - x0:sx1 - x0, sy0 - y0:sy1 - y0] = arr[sx0:sx1, sy0:sy1]
    return out


def history_offsets_from_features(features: torch.Tensor | np.ndarray) -> np.ndarray:
    """Recover the v13 six-frame t0-relative center offsets in meters."""
    x = features.detach().cpu().numpy() if isinstance(features, torch.Tensor) else np.asarray(features)
    if x.ndim != 2 or x.shape[1] != FEATURE_DIM:
        raise ValueError(f"features must be [N,{FEATURE_DIM}]")
    out = np.zeros((x.shape[0], HISTORY_FRAMES, 2), dtype=np.float32)
    name_to_i = {name: i for i, name in enumerate(FEATURE_NAMES)}
    for t in range(HISTORY_FRAMES):
        out[:, t, 0] = x[:, name_to_i[f"hist_offset_{t}_x"]] * 20.0
        out[:, t, 1] = x[:, name_to_i[f"hist_offset_{t}_y"]] * 20.0
    return out


def build_local_semantic_tubes(
    history_semantics: Sequence[np.ndarray],
    history_poses: Sequence[np.ndarray],
    source_xy_t0_m: np.ndarray,
    history_offsets_xy_t0_m: np.ndarray,
    track_valid: np.ndarray,
    *,
    grid: OccupancyGrid = OccupancyGrid(),
    free_label: int = 17,
    patch_size_m: float = DEFAULT_PATCH_SIZE_M,
    patch_resolution_m: float = DEFAULT_PATCH_RESOLUTION_M,
) -> np.ndarray:
    """Build causal source-centered semantic tubes [N,T,H,W] uint8.

    Every history occupancy is ego-compensated into the current t0 frame.  Each
    frame is then cropped around that source's causally backtracked center.  Thus
    spatial coordinates are object-relative while the scalar feature stream
    retains the absolute displacement/velocity history proven useful by v13.
    No future occupancy or future annotation is read here.
    """
    if len(history_semantics) != HISTORY_FRAMES or len(history_poses) != HISTORY_FRAMES:
        raise ValueError("expected six history semantics and poses")
    src = np.asarray(source_xy_t0_m, dtype=np.float32)
    offs = np.asarray(history_offsets_xy_t0_m, dtype=np.float32)
    valid = np.asarray(track_valid, dtype=bool)
    if src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("source_xy_t0_m must be [N,2]")
    if offs.shape != (src.shape[0], HISTORY_FRAMES, 2) or valid.shape != (src.shape[0], HISTORY_FRAMES):
        raise ValueError("history offset/valid shape mismatch")

    native = float(grid.voxel_size[0])
    if abs(float(grid.voxel_size[1]) - native) > 1e-9:
        raise ValueError("local tube requires square XY voxels")
    raw_vox = int(round(float(patch_size_m) / native))
    pool = int(round(float(patch_resolution_m) / native))
    if raw_vox <= 0 or raw_vox % 2 or pool != 2 or raw_vox % pool:
        raise ValueError("v1 local tube requires an even patch and exactly 2x XY pooling")
    out_hw = raw_vox // pool

    t0_pose = np.asarray(history_poses[-1], dtype=np.float64)
    bevs = []
    for t, (sem, pose) in enumerate(zip(history_semantics, history_poses)):
        sem = _check_semantic_grid(sem, grid)
        if t == HISTORY_FRAMES - 1:
            aligned = sem
        else:
            aligned = warp_semantic_grid(
                sem,
                relative_transform(np.asarray(pose, dtype=np.float64), t0_pose),
                grid=grid,
                free_label=int(free_label),
            )
        bevs.append(top_surface_semantic(aligned, grid=grid, free_label=int(free_label)))

    tubes = np.full(
        (src.shape[0], HISTORY_FRAMES, out_hw, out_hw),
        int(free_label),
        dtype=np.uint8,
    )
    for i in range(src.shape[0]):
        for t in range(HISTORY_FRAMES):
            center = src[i] + offs[i, t] if valid[i, t] else src[i]
            patch = extract_bev_patch(
                bevs[t], center, grid=grid, patch_voxels=raw_vox, free_label=int(free_label)
            )
            tubes[i, t] = priority_pool2x2(patch, free_label=int(free_label))
    return tubes


class SpatialTemporalBlock(nn.Module):
    """Efficient factorized spatial-temporal block.

    Spatial mixing is a ConvNeXt-style depthwise convolution on each local BEV
    frame.  Temporal mixing is self-attention over the six history frames at each
    source-relative spatial location.  This avoids the cost and ordering issues
    of full 4D attention while preserving explicit temporal reasoning.
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4, kernel_size: int = 5):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        pad = kernel_size // 2
        self.spatial_norm = nn.GroupNorm(1, dim)
        self.spatial_dw = nn.Conv2d(dim, dim, kernel_size, padding=pad, groups=dim)
        self.spatial_pw1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.spatial_pw2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.spatial_act = nn.GELU()
        self.temporal_norm = nn.LayerNorm(dim)
        self.temporal_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.temporal_ffn_norm = nn.LayerNorm(dim)
        self.temporal_ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("ST block expects [B,T,C,H,W]")
        B, T, C, H, W = x.shape
        s = x.reshape(B * T, C, H, W)
        y = self.spatial_pw2(self.spatial_act(self.spatial_pw1(self.spatial_dw(self.spatial_norm(s)))))
        s = s + y
        x = s.reshape(B, T, C, H, W)

        seq = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)
        q = self.temporal_norm(seq)
        y, _ = self.temporal_attn(q, q, q, need_weights=False)
        seq = seq + y
        seq = seq + self.temporal_ffn(self.temporal_ffn_norm(seq))
        return seq.reshape(B, H, W, T, C).permute(0, 3, 4, 1, 2).contiguous()


class FutureQueryBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_q_norm = nn.LayerNorm(dim)
        self.cross_ctx_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, q: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        z = self.self_norm(q)
        y, _ = self.self_attn(z, z, z, need_weights=False)
        q = q + y
        y, _ = self.cross_attn(
            self.cross_q_norm(q), self.cross_ctx_norm(context), self.cross_ctx_norm(context),
            need_weights=False,
        )
        q = q + y
        return q + self.ffn(self.ffn_norm(q))


@dataclass(frozen=True)
class LocalSTWMConfig:
    d_model: int = 128
    semantic_dim: int = 32
    heads: int = 4
    blocks: int = 4
    decoder_blocks: int = 2
    tube_hw: int = 20
    history_frames: int = HISTORY_FRAMES
    future_frames: int = FUTURE_FRAMES


class LocalSpatialTemporalWorldModel(nn.Module):
    """Object-centric local WM: history semantics + kinematics -> KTA residual."""

    def __init__(self, config: LocalSTWMConfig = LocalSTWMConfig()):
        super().__init__()
        cfg = config
        if cfg.d_model % cfg.heads:
            raise ValueError("d_model must be divisible by heads")
        if cfg.history_frames != HISTORY_FRAMES or cfg.future_frames != FUTURE_FRAMES:
            raise ValueError("v16 requires the frozen 6-history + 6-future contract")
        if cfg.tube_hw % 2:
            raise ValueError("tube_hw must be even for the stride-2 spatial stem")
        self.config = cfg
        self.semantic_embedding = nn.Embedding(SEMANTIC_CLASSES, cfg.semantic_dim)
        self.spatial_stem = nn.Sequential(
            nn.Conv2d(cfg.semantic_dim, cfg.d_model, 3, stride=2, padding=1),
            nn.GroupNorm(1, cfg.d_model), nn.GELU(),
            nn.Conv2d(cfg.d_model, cfg.d_model, 3, padding=1),
            nn.GroupNorm(1, cfg.d_model), nn.GELU(),
        )
        self.kinematic_proj = nn.Sequential(
            nn.Linear(FEATURE_DIM, cfg.d_model), nn.GELU(), nn.LayerNorm(cfg.d_model)
        )
        self.time_embedding = nn.Parameter(torch.zeros(1, HISTORY_FRAMES, cfg.d_model, 1, 1))
        stem_hw = cfg.tube_hw // 2
        self.spatial_embedding = nn.Parameter(torch.zeros(1, 1, cfg.d_model, stem_hw, stem_hw))
        self.blocks = nn.ModuleList([
            SpatialTemporalBlock(cfg.d_model, cfg.heads) for _ in range(cfg.blocks)
        ])
        self.future_query = nn.Parameter(torch.zeros(1, FUTURE_FRAMES, cfg.d_model))
        self.future_time_embedding = nn.Parameter(torch.zeros(1, FUTURE_FRAMES, cfg.d_model))
        self.kta_future_proj = nn.Sequential(
            nn.Linear(2, cfg.d_model), nn.GELU(), nn.LayerNorm(cfg.d_model)
        )
        self.decoder = nn.ModuleList([
            FutureQueryBlock(cfg.d_model, cfg.heads) for _ in range(cfg.decoder_blocks)
        ])
        self.residual_head = nn.Linear(cfg.d_model, 2)
        self.existence_head = nn.Linear(cfg.d_model, 1)

        nn.init.trunc_normal_(self.time_embedding, std=0.02)
        nn.init.trunc_normal_(self.spatial_embedding, std=0.02)
        nn.init.trunc_normal_(self.future_query, std=0.02)
        nn.init.trunc_normal_(self.future_time_embedding, std=0.02)
        # Safety contract: a fresh model is exactly the KTA displacement predictor.
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.existence_head.weight)
        nn.init.zeros_(self.existence_head.bias)

    def forward(
        self,
        features: torch.Tensor,
        local_semantic_tube: torch.Tensor,
        kta_displacement_xy_m: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cfg = self.config
        if features.ndim != 2 or features.shape[-1] != FEATURE_DIM:
            raise ValueError(f"features must be [B,{FEATURE_DIM}]")
        if local_semantic_tube.ndim != 4 or tuple(local_semantic_tube.shape[1:]) != (
            HISTORY_FRAMES, cfg.tube_hw, cfg.tube_hw
        ):
            raise ValueError("local_semantic_tube shape mismatch")
        if kta_displacement_xy_m.shape != (features.shape[0], FUTURE_FRAMES, 2):
            raise ValueError("kta_displacement_xy_m must be [B,6,2]")
        labels = local_semantic_tube.long()
        if bool((labels < 0).any()) or bool((labels >= SEMANTIC_CLASSES).any()):
            raise ValueError("semantic tube contains labels outside [0,17]")

        B = features.shape[0]
        # Validation can legitimately contain windows with no Strong dynamic
        # source. PyTorch SDPA rejects the resulting zero-sized attention batch,
        # so preserve the semantic contract explicitly: no sources means no
        # residual/existence predictions and the caller keeps the KTA anchor.
        if B == 0:
            return {
                "residual_xy_m": features.new_empty((0, FUTURE_FRAMES, 2)),
                "existence_logits": features.new_empty((0, FUTURE_FRAMES)),
            }

        emb = self.semantic_embedding(labels)  # [B,T,H,W,E]
        x = emb.permute(0, 1, 4, 2, 3).reshape(
            B * HISTORY_FRAMES, cfg.semantic_dim, cfg.tube_hw, cfg.tube_hw
        )
        x = self.spatial_stem(x)
        Hs, Ws = x.shape[-2:]
        x = x.reshape(B, HISTORY_FRAMES, cfg.d_model, Hs, Ws)
        obj = self.kinematic_proj(features).view(B, 1, cfg.d_model, 1, 1)
        x = x + obj + self.time_embedding + self.spatial_embedding
        for block in self.blocks:
            x = block(x)

        context = x.permute(0, 1, 3, 4, 2).reshape(B, HISTORY_FRAMES * Hs * Ws, cfg.d_model)
        q = self.future_query.expand(B, -1, -1) + self.future_time_embedding
        q = q + self.kinematic_proj(features).unsqueeze(1)
        q = q + self.kta_future_proj(kta_displacement_xy_m.to(q.dtype) / 20.0)
        for block in self.decoder:
            q = block(q, context)
        residual = self.residual_head(q)
        existence = self.existence_head(q)[..., 0]
        return {"residual_xy_m": residual, "existence_logits": existence}


def config_from_mapping(raw: Mapping | None) -> LocalSTWMConfig:
    if not raw:
        return LocalSTWMConfig()
    fields = LocalSTWMConfig.__dataclass_fields__
    return LocalSTWMConfig(**{k: int(v) for k, v in raw.items() if k in fields})