"""Fast V17 + KTA-backtrace 3D residual-branch probe.

This module is intentionally a capacity probe, not the final MT-V1 model.  It
keeps the complete V17-RL predictor and injects one extra future-query residual
computed from causal KTA-backtraced 3D occupancy crops.

The treatment has three important contracts:
  * no future occupancy/annotation is read by the branch;
  * gamma==0 bypasses the branch and calls V17 exactly;
  * the branch preserves all 16 Occ3D height bins instead of top-surface BEV.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import OccupancyGrid
from .local_st_world_model_v17 import (
    LocalSpatialTemporalWorldModelV17,
    LocalSTWMV17Config,
)
from .motion_transport import FEATURE_NAMES, HISTORY_FRAMES, FUTURE_FRAMES

BACKTRACE3D_PROTOCOL = "p0_f9_v17_backtrace3d_query_residual_probe_v1"
BACKTRACE3D_CROP_CONTRACT = "causal_kta_backtrace_full3d_25p6m_0p4m_v1"
BACKTRACE3D_FUSION_CONTRACT = "v17_future_query_plus_gamma_backtrace3d_residual_v1"
BACKTRACE3D_SHAPE = (64, 64, 16)
BACKTRACE3D_XY_RESOLUTION_M = 0.4
BACKTRACE3D_OOB_LABEL = 18
BACKTRACE3D_CLASS_TOKENS = 19

_NAME_TO_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


@lru_cache(maxsize=4)
def _local_grid(shape_xyz=BACKTRACE3D_SHAPE, xy_resolution_m=BACKTRACE3D_XY_RESOLUTION_M,
                z_min=-1.0, z_step=0.4):
    cx, cy, cz = map(int, shape_xyz)
    gx = (np.arange(cx, dtype=np.float64) - (cx - 1) / 2.0) * float(xy_resolution_m)
    gy = (np.arange(cy, dtype=np.float64) - (cy - 1) / 2.0) * float(xy_resolution_m)
    gz = float(z_min) + (np.arange(cz, dtype=np.float64) + 0.5) * float(z_step)
    xx, yy, zz = np.meshgrid(gx, gy, gz, indexing="ij")
    return np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)


def _metric_to_index(points_xyz: np.ndarray, grid: OccupancyGrid):
    p = np.asarray(points_xyz, dtype=np.float64)
    origin = np.asarray([grid.x_min, grid.y_min, grid.z_min], dtype=np.float64)
    step = np.asarray(grid.voxel_size, dtype=np.float64)
    idx = np.floor((p - origin[None]) / step[None]).astype(np.int64)
    shape = np.asarray(grid.shape_hwd, dtype=np.int64)
    ok = ((idx >= 0) & (idx < shape[None])).all(axis=1)
    return idx, ok


def _native_source_mask_64(record: Mapping, source_indices: Sequence[int]) -> np.ndarray:
    native = record.get("native_source_footprint_mask")
    if native is None:
        raise RuntimeError(
            "backtrace3D probe requires native_source_footprint_mask; use the nativefp V17 cache"
        )
    arr = native.detach().cpu().numpy() if torch.is_tensor(native) else np.asarray(native)
    arr = arr[np.asarray(source_indices, dtype=np.int64)]
    if arr.ndim != 3:
        raise RuntimeError(f"native source footprint must be [N,H,W], got {arr.shape}")
    if arr.shape[-2:] != (40, 40):
        raise RuntimeError(
            f"expected native 16m@0.4m source mask [40,40], got {arr.shape[-2:]}"
        )
    out = np.zeros((len(source_indices), 64, 64), dtype=bool)
    lo = (64 - 40) // 2
    out[:, lo:lo + 40, lo:lo + 40] = arr.astype(bool)
    return out


def velocity_xy_t0_from_features(features: torch.Tensor | np.ndarray) -> np.ndarray:
    x = features.detach().cpu().numpy() if torch.is_tensor(features) else np.asarray(features)
    vx = x[:, _NAME_TO_INDEX["current_vx_norm"]] * 20.0
    vy = x[:, _NAME_TO_INDEX["current_vy_norm"]] * 20.0
    return np.stack((vx, vy), axis=1).astype(np.float64)


def build_kta_backtrace_3d_crops(
    source,
    record: Mapping,
    source_indices: Sequence[int],
    *,
    grid: OccupancyGrid = OccupancyGrid(),
    frame_dt_s: float = 0.5,
    device: torch.device | str | None = None,
):
    """Build causal full-height crops for selected V17 sources in one window.

    Crop centers follow constant-velocity KTA *backward* from t0.  Therefore any
    deviation of the observed object from constant velocity remains visible as a
    spatial offset inside the crop instead of being removed by per-frame source
    recentering.
    """
    ids = np.asarray(source_indices, dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError("source_indices must be one-dimensional")
    if len(ids) == 0:
        return {
            "semantics": torch.empty((0, HISTORY_FRAMES, *BACKTRACE3D_SHAPE), dtype=torch.uint8),
            "valid": torch.empty((0, HISTORY_FRAMES, *BACKTRACE3D_SHAPE), dtype=torch.bool),
            "source_mask": torch.empty((0, 64, 64), dtype=torch.bool),
            "relative_times": torch.empty((0, HISTORY_FRAMES), dtype=torch.float32),
        }
    tokens = tuple(str(x) for x in record["history_tokens"])
    if len(tokens) != HISTORY_FRAMES:
        raise RuntimeError("backtrace3D requires six history frames")
    scene = str(record["scene_name"])
    occ = []
    obs = []
    poses = []
    for tok in tokens:
        sem, valid = source.load_occ3d(scene, tok, require_lidar_mask=True)
        occ.append(np.asarray(sem))
        obs.append(np.asarray(valid, dtype=bool))
        poses.append(np.asarray(source.pose(tok), dtype=np.float64))

    if tuple(occ[-1].shape) != tuple(grid.shape_hwd):
        raise RuntimeError(f"Occ3D shape {occ[-1].shape} != grid {grid.shape_hwd}")
    if int(grid.shape_hwd[2]) != BACKTRACE3D_SHAPE[2]:
        raise RuntimeError("backtrace3D probe expects all 16 height bins")

    src_xy = record["source_centroid_xy_t0_m"]
    src_xy = src_xy.detach().cpu().numpy() if torch.is_tensor(src_xy) else np.asarray(src_xy)
    src_xy = np.asarray(src_xy[ids], dtype=np.float64)
    vel = velocity_xy_t0_from_features(record["features"])[ids]
    rel_t = (np.arange(HISTORY_FRAMES, dtype=np.float64) - (HISTORY_FRAMES - 1)) * float(frame_dt_s)
    native_mask = _native_source_mask_64(record, ids)

    m = len(ids)
    cx, cy, cz = BACKTRACE3D_SHAPE

    target_device = torch.device(device) if device is not None else torch.device("cpu")
    if target_device.type != "cpu":
        # Fast path: for each history frame, sample all sources from the same
        # Occ3D volume in one 5D nearest-neighbour grid_sample call.  Input
        # layout is [N,C,Z,Y,X]; output is converted back to [N,X,Y,Z].
        src_t = torch.as_tensor(src_xy, device=target_device, dtype=torch.float32)
        vel_t = torch.as_tensor(vel, device=target_device, dtype=torch.float32)
        rel_t_t = torch.as_tensor(rel_t, device=target_device, dtype=torch.float32)
        gx = (torch.arange(cx, device=target_device, dtype=torch.float32) - (cx - 1) / 2.0) * float(BACKTRACE3D_XY_RESOLUTION_M)
        gy = (torch.arange(cy, device=target_device, dtype=torch.float32) - (cy - 1) / 2.0) * float(BACKTRACE3D_XY_RESOLUTION_M)
        gz = float(grid.z_min) + (torch.arange(cz, device=target_device, dtype=torch.float32) + 0.5) * float(grid.voxel_size[2])
        # grid_sample 5D output order is [D,H,W].  Use [Z,X,Y] so the sampled
        # tensor can be permuted directly to the repository's [X,Y,Z].
        zz, xx, yy = torch.meshgrid(gz, gx, gy, indexing="ij")
        base = torch.stack((xx, yy, zz), dim=-1)[None]  # [1,Z,X,Y,3]
        origin = torch.tensor(
            [grid.x_min, grid.y_min, grid.z_min],
            device=target_device, dtype=torch.float32,
        )
        extent = torch.tensor(
            [
                grid.voxel_size[0] * grid.shape_hwd[0],
                grid.voxel_size[1] * grid.shape_hwd[1],
                grid.voxel_size[2] * grid.shape_hwd[2],
            ],
            device=target_device, dtype=torch.float32,
        )
        sem_frames = []
        valid_frames = []
        t0_pose_t = torch.as_tensor(t0_pose, device=target_device, dtype=torch.float32)
        for ti in range(HISTORY_FRAMES):
            center = src_t + vel_t * rel_t_t[ti]
            p0 = base.expand(m, -1, -1, -1, -1).clone()
            p0[..., 0] += center[:, None, None, None, 0]
            p0[..., 1] += center[:, None, None, None, 1]

            hist_pose_t = torch.as_tensor(poses[ti], device=target_device, dtype=torch.float32)
            hist_from_t0 = torch.linalg.inv(hist_pose_t) @ t0_pose_t
            ph = torch.einsum("...j,ij->...i", p0, hist_from_t0[:3, :3]) + hist_from_t0[:3, 3]
            norm = 2.0 * (ph - origin) / extent - 1.0
            # Use the repository's metric half-open grid contract rather than
            # normalized [-1,1], because align_corners=False places x=1 at the
            # outer half-voxel boundary.
            in_bounds = ((ph >= origin) & (ph < (origin + extent))).all(dim=-1)

            sem_vol = torch.as_tensor(
                occ[ti], device=target_device, dtype=torch.float32
            ).permute(2, 1, 0)[None, None].expand(m, -1, -1, -1, -1)
            obs_vol = torch.as_tensor(
                obs[ti], device=target_device, dtype=torch.float32
            ).permute(2, 1, 0)[None, None].expand(m, -1, -1, -1, -1)
            sampled_sem = F.grid_sample(
                sem_vol, norm, mode="nearest", padding_mode="zeros", align_corners=False
            )[:, 0]
            sampled_obs = F.grid_sample(
                obs_vol, norm, mode="nearest", padding_mode="zeros", align_corners=False
            )[:, 0] > 0.5
            sampled_sem = sampled_sem.to(torch.uint8)
            sampled_sem = torch.where(
                in_bounds,
                sampled_sem,
                torch.full_like(sampled_sem, BACKTRACE3D_OOB_LABEL),
            )
            sampled_obs = sampled_obs & in_bounds
            sem_frames.append(sampled_sem.permute(0, 2, 3, 1).contiguous())
            valid_frames.append(sampled_obs.permute(0, 2, 3, 1).contiguous())

        return {
            "semantics": torch.stack(sem_frames, dim=1),
            "valid": torch.stack(valid_frames, dim=1),
            "source_mask": torch.as_tensor(native_mask, device=target_device, dtype=torch.bool),
            "relative_times": rel_t_t[None].expand(m, -1).contiguous(),
        }

    sem_out = np.full(
        (m, HISTORY_FRAMES, cx, cy, cz), BACKTRACE3D_OOB_LABEL, dtype=np.uint8
    )
    valid_out = np.zeros((m, HISTORY_FRAMES, cx, cy, cz), dtype=bool)

    base = _local_grid(
        BACKTRACE3D_SHAPE,
        BACKTRACE3D_XY_RESOLUTION_M,
        float(grid.z_min),
        float(grid.voxel_size[2]),
    )
    t0_pose = poses[-1]
    for ti in range(HISTORY_FRAMES):
        hist_from_t0 = np.linalg.inv(poses[ti]) @ t0_pose
        R = hist_from_t0[:3, :3]
        t = hist_from_t0[:3, 3]
        for mi in range(m):
            center = src_xy[mi] + vel[mi] * float(rel_t[ti])
            p0 = base.copy()
            p0[:, 0] += center[0]
            p0[:, 1] += center[1]
            ph = p0 @ R.T + t[None]
            idx, ok = _metric_to_index(ph, grid)
            flat_sem = np.full((len(idx),), BACKTRACE3D_OOB_LABEL, dtype=np.uint8)
            flat_valid = np.zeros((len(idx),), dtype=bool)
            q = idx[ok]
            flat_sem[ok] = occ[ti][q[:, 0], q[:, 1], q[:, 2]].astype(np.uint8)
            flat_valid[ok] = obs[ti][q[:, 0], q[:, 1], q[:, 2]]
            sem_out[mi, ti] = flat_sem.reshape(cx, cy, cz)
            valid_out[mi, ti] = flat_valid.reshape(cx, cy, cz)

    return {
        "semantics": torch.from_numpy(sem_out),
        "valid": torch.from_numpy(valid_out),
        "source_mask": torch.from_numpy(native_mask),
        "relative_times": torch.from_numpy(
            np.broadcast_to(rel_t.astype(np.float32), (m, HISTORY_FRAMES)).copy()
        ),
    }


class _ConvGN(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        groups = 8
        while cout % groups:
            groups -= 1
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.norm = nn.GroupNorm(groups, cout)

    def forward(self, x):
        return F.gelu(self.norm(self.conv(x)))


@dataclass(frozen=True)
class Backtrace3DProbeConfig:
    embedding_dim: int = 4
    branch_dim: int = 128
    temporal_layers: int = 2
    heads: int = 4
    source_microbatch: int = 12


class Backtrace3DFutureQueryBranch(nn.Module):
    """Height-preserving causal branch producing six V17 query residuals."""

    def __init__(self, cfg: Backtrace3DProbeConfig = Backtrace3DProbeConfig()):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(BACKTRACE3D_CLASS_TOKENS, int(cfg.embedding_dim))
        # Z*embedding=64, per-height validity=16, source mask=1, XY=2, time=1.
        in_channels = 16 * int(cfg.embedding_dim) + 16 + 1 + 2 + 1
        if in_channels != 84:
            raise AssertionError(in_channels)
        self.stem = nn.Sequential(
            _ConvGN(84, 32),
            _ConvGN(32, 64, stride=2),
            _ConvGN(64, 128, stride=2),
            _ConvGN(128, 128, stride=2),
        )
        self.frame_proj = nn.Sequential(
            nn.Linear(256, int(cfg.branch_dim)),
            nn.GELU(),
            nn.LayerNorm(int(cfg.branch_dim)),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=int(cfg.branch_dim),
            nhead=int(cfg.heads),
            dim_feedforward=2 * int(cfg.branch_dim),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=int(cfg.temporal_layers))
        self.future_query = nn.Parameter(torch.zeros(1, FUTURE_FRAMES, int(cfg.branch_dim)))
        self.future_time = nn.Parameter(torch.zeros(1, FUTURE_FRAMES, int(cfg.branch_dim)))
        nn.init.normal_(self.future_query, mean=0.0, std=0.02)
        nn.init.normal_(self.future_time, mean=0.0, std=0.02)
        self.cross_norm_q = nn.LayerNorm(int(cfg.branch_dim))
        self.cross_norm_kv = nn.LayerNorm(int(cfg.branch_dim))
        self.cross = nn.MultiheadAttention(
            int(cfg.branch_dim), int(cfg.heads), dropout=0.0, batch_first=True
        )
        self.out = nn.Sequential(
            nn.LayerNorm(int(cfg.branch_dim)),
            nn.Linear(int(cfg.branch_dim), int(cfg.branch_dim)),
        )

    def _one(self, sem, valid, source_mask, relative_times):
        device = self.embedding.weight.device
        sem = sem.to(device=device, non_blocking=True).long()
        valid = valid.to(device=device, non_blocking=True).bool()
        source_mask = source_mask.to(device=device, non_blocking=True).bool()
        relative_times = relative_times.to(device=device, non_blocking=True).float()
        b, t, x, y, z = sem.shape
        if (t, x, y, z) != (HISTORY_FRAMES, *BACKTRACE3D_SHAPE):
            raise ValueError(f"unexpected backtrace crop shape: {tuple(sem.shape)}")

        emb = self.embedding(sem.clamp(0, BACKTRACE3D_CLASS_TOKENS - 1))
        emb = emb.permute(0, 1, 4, 5, 2, 3).contiguous().reshape(
            b, t, z * self.cfg.embedding_dim, x, y
        )
        vh = valid.permute(0, 1, 4, 2, 3).float()
        sm = source_mask[:, None, None].expand(b, t, 1, x, y).float()
        gx = (torch.arange(x, device=device, dtype=torch.float32) + 0.5) / x * 2.0 - 1.0
        gy = (torch.arange(y, device=device, dtype=torch.float32) + 0.5) / y * 2.0 - 1.0
        xx, yy = torch.meshgrid(gx, gy, indexing="ij")
        xy = torch.stack((xx, yy), dim=0)[None, None].expand(b, t, 2, x, y)
        tau = (relative_times / 2.5)[:, :, None, None, None].expand(b, t, 1, x, y)
        feat = torch.cat((emb, vh, sm, xy, tau), dim=2)
        h = self.stem(feat.reshape(b * t, 84, x, y))
        h = h.reshape(b, t, h.shape[1], h.shape[2], h.shape[3])
        mean = h.mean(dim=(-2, -1))
        mx = h.amax(dim=(-2, -1))
        frame = self.frame_proj(torch.cat((mean, mx), dim=-1))
        hist = self.temporal(frame)
        q = (self.future_query + self.future_time).expand(b, -1, -1)
        attn, _ = self.cross(
            self.cross_norm_q(q),
            self.cross_norm_kv(hist),
            self.cross_norm_kv(hist),
            need_weights=False,
        )
        return self.out(q + attn)

    def forward(self, semantics, valid, source_mask, relative_times):
        n = int(semantics.shape[0])
        if n == 0:
            return self.future_query.new_empty((0, FUTURE_FRAMES, self.cfg.branch_dim))
        step = max(1, int(self.cfg.source_microbatch))
        rows = []
        for lo in range(0, n, step):
            hi = min(n, lo + step)
            rows.append(
                self._one(
                    semantics[lo:hi],
                    valid[lo:hi],
                    source_mask[lo:hi],
                    relative_times[lo:hi],
                )
            )
        return torch.cat(rows, dim=0)


class LocalSpatialTemporalWorldModelV17Backtrace3D(LocalSpatialTemporalWorldModelV17):
    """V17-RL plus a causal backtrace-3D future-query residual branch."""

    def __init__(
        self,
        v17_config: LocalSTWMV17Config = LocalSTWMV17Config(),
        branch_config: Backtrace3DProbeConfig = Backtrace3DProbeConfig(),
    ):
        super().__init__(v17_config)
        if int(branch_config.branch_dim) != int(v17_config.d_model):
            raise ValueError("branch_dim must match V17 d_model for additive query fusion")
        self.backtrace3d_config = branch_config
        self.backtrace3d_branch = Backtrace3DFutureQueryBranch(branch_config)

    def base_named_parameters(self):
        for name, p in self.named_parameters():
            if not name.startswith("backtrace3d_branch."):
                yield name, p

    def branch_named_parameters(self):
        for name, p in self.named_parameters():
            if name.startswith("backtrace3d_branch."):
                yield name, p

    def forward(
        self,
        features,
        local_semantic_tube,
        kta_displacement_xy_m,
        frame_motion_features=None,
        target_source_mask_tube=None,
        *,
        backtrace_semantics=None,
        backtrace_valid=None,
        backtrace_source_mask=None,
        backtrace_relative_times=None,
        branch_gamma: float = 1.0,
    ):
        gamma = float(branch_gamma)
        if gamma == 0.0:
            # Important: exact historical V17 code path for step-0 identity.
            return super().forward(
                features,
                local_semantic_tube,
                kta_displacement_xy_m,
                frame_motion_features,
                target_source_mask_tube,
            )
        if any(
            x is None
            for x in (
                backtrace_semantics,
                backtrace_valid,
                backtrace_source_mask,
                backtrace_relative_times,
            )
        ):
            raise ValueError("backtrace3D tensors are required when branch_gamma != 0")
        residual = self.backtrace3d_branch(
            backtrace_semantics,
            backtrace_valid,
            backtrace_source_mask,
            backtrace_relative_times,
        )
        return super().forward(
            features,
            local_semantic_tube,
            kta_displacement_xy_m,
            frame_motion_features,
            target_source_mask_tube,
            future_query_residual=gamma * residual,
        )
