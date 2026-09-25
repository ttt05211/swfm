"""V20 lightweight 3D scene encoder and task heads.

Design goals:
* six historical frames are encoded once in canonical coordinates;
* Z remains a spatial dimension throughout the scene path;
* the global trunk is low-resolution 3D;
* high-resolution semantics are decoded tile-wise;
* V18 source tokens interact locally with the scene volume;
* Dormant runs only on memory-only sources;
* Birth uses a small fixed query set, one query = one object across six horizons.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .local_st_world_model import SEMANTIC_CLASSES
from .motion_transport import FUTURE_FRAMES, HISTORY_FRAMES
from .v20_history_world import DYNAMIC_IDS, FREE_LABEL

V20_MODEL_PROTOCOL = "p0_f9_v20_3d_history_model_v1"


@dataclass(frozen=True)
class V20SceneConfig:
    semantic_dim: int = 8
    base_dim: int = 32
    source_dim: int = 96
    tile_dim: int = 48
    birth_queries: int = 8
    vertical_bins: int = 16
    dynamic_classes: int = len(DYNAMIC_IDS)


class HistoricalEvidence3DEncoder(nn.Module):
    """Encode canonical history evidence [B,T,X,Y,Z] once.

    Unknown, observed-free and observed-occupied are distinct channels.
    Time is fused by a tiny 3D conv over concatenated per-frame embeddings.
    The caller already supplies the configured low-resolution Ωmax lattice, so
    this encoder preserves its spatial resolution rather than downsampling it
    a second time.
    """

    def __init__(self, cfg: V20SceneConfig = V20SceneConfig()):
        super().__init__()
        self.cfg = cfg
        d = int(cfg.semantic_dim)
        b = int(cfg.base_dim)
        self.semantic_embedding = nn.Embedding(SEMANTIC_CLASSES, d)
        # Per frame: semantic embedding + observed + observed_free.
        in_ch = HISTORY_FRAMES * (d + 2)
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, b, 3, stride=1, padding=1),
            nn.GroupNorm(1, b),
            nn.GELU(),
            nn.Conv3d(b, 2 * b, 3, stride=1, padding=1),
            nn.GroupNorm(1, 2 * b),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv3d(2 * b, 2 * b, 3, padding=1),
            nn.GroupNorm(1, 2 * b),
            nn.GELU(),
            nn.Conv3d(2 * b, 2 * b, 3, padding=1),
            nn.GroupNorm(1, 2 * b),
            nn.GELU(),
        )

    @property
    def output_dim(self) -> int:
        return 2 * int(self.cfg.base_dim)

    def forward(
        self,
        semantic: torch.Tensor,
        observed: torch.Tensor,
        observed_free: torch.Tensor,
    ) -> torch.Tensor:
        if semantic.ndim != 5:
            raise ValueError("semantic must be [B,T,X,Y,Z]")
        B, T, X, Y, Z = semantic.shape
        if T != HISTORY_FRAMES:
            raise ValueError("V20 requires six history frames")
        if observed.shape != semantic.shape or observed_free.shape != semantic.shape:
            raise ValueError("observed masks must match semantic")
        if bool((observed_free.bool() & ~observed.bool()).any()):
            raise ValueError("observed_free cannot occur in unknown voxels")
        emb = self.semantic_embedding(semantic.long())  # B,T,X,Y,Z,D
        obs = observed.to(emb.dtype).unsqueeze(-1)
        free = observed_free.to(emb.dtype).unsqueeze(-1)
        x = torch.cat((emb, obs, free), dim=-1)
        # B,T,X,Y,Z,C -> B,T*C,X,Y,Z
        x = x.permute(0, 1, 5, 2, 3, 4).reshape(B, -1, X, Y, Z)
        x = self.stem(x)
        return x + self.refine(x)


class SourceSceneFusion(nn.Module):
    """Local source-to-volume interaction without a global token transformer.

    The caller supplies a sampled local scene vector for each source token.
    This module intentionally does not attend over the full voxel grid.
    """

    def __init__(self, scene_dim: int, source_dim: int, out_dim: int):
        super().__init__()
        self.scene_proj = nn.Linear(int(scene_dim), int(out_dim))
        self.source_proj = nn.Linear(int(source_dim), int(out_dim))
        self.mix = nn.Sequential(
            nn.Linear(2 * int(out_dim), int(out_dim)),
            nn.GELU(),
            nn.Linear(int(out_dim), int(out_dim)),
        )

    def forward(self, source_token: torch.Tensor, sampled_scene: torch.Tensor) -> torch.Tensor:
        if source_token.ndim != 2 or sampled_scene.ndim != 2:
            raise ValueError("source token/scene sample must be [N,C]")
        if source_token.shape[0] != sampled_scene.shape[0]:
            raise ValueError("source/scene batch mismatch")
        a = self.source_proj(source_token)
        b = self.scene_proj(sampled_scene)
        return self.mix(torch.cat((a, b), dim=-1))


class StaticWorldHead(nn.Module):
    """Predict one canonical 3D static semantic field.

    Global logits are produced at coarse resolution.  Native-resolution logits
    are produced only for supplied candidate tiles with trilinear features.
    """

    def __init__(self, scene_dim: int, cfg: V20SceneConfig = V20SceneConfig()):
        super().__init__()
        self.cfg = cfg
        td = int(cfg.tile_dim)
        self.coarse_head = nn.Conv3d(int(scene_dim), SEMANTIC_CLASSES, 1)
        self.tile_refine = nn.Sequential(
            nn.Conv3d(int(scene_dim) + 3, td, 3, padding=1),
            nn.GroupNorm(1, td),
            nn.GELU(),
            nn.Conv3d(td, td, 3, padding=1),
            nn.GroupNorm(1, td),
            nn.GELU(),
            nn.Conv3d(td, SEMANTIC_CLASSES, 1),
        )

    def forward_coarse(self, scene: torch.Tensor) -> torch.Tensor:
        return self.coarse_head(scene)

    def refine_tiles(
        self,
        scene_features: torch.Tensor,
        *,
        sample_grid: torch.Tensor,
        query_mask: torch.Tensor,
        seen_mask: torch.Tensor,
        t0_missing_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Refine one packed native-resolution tile batch.

        sample_grid is the normalized grid_sample grid [B,D,H,W,3].
        Context channels distinguish query support, history-seen, and t0-missing.
        """
        if sample_grid.ndim != 5 or sample_grid.shape[-1] != 3:
            raise ValueError("sample_grid must be [B,D,H,W,3]")
        x = F.grid_sample(
            scene_features,
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        masks = torch.stack(
            (
                query_mask.to(x.dtype),
                seen_mask.to(x.dtype),
                t0_missing_mask.to(x.dtype),
            ),
            dim=1,
        )
        if masks.shape[2:] != x.shape[2:]:
            raise ValueError("tile context mask shape mismatch")
        return self.tile_refine(torch.cat((x, masks), dim=1))


class DormantSourceHead(nn.Module):
    """Residual adapter only for history-observed, t0-missing sources."""

    def __init__(self, source_dim: int, scene_dim: int, hidden_dim: int = 96):
        super().__init__()
        h = int(hidden_dim)
        self.fusion = SourceSceneFusion(scene_dim, source_dim, h)
        self.time = nn.Parameter(torch.zeros(1, FUTURE_FRAMES, h))
        self.decoder = nn.Sequential(
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.GELU(),
        )
        self.xy = nn.Linear(h, 2)
        self.yaw = nn.Linear(h, 1)
        self.exist = nn.Linear(h, 1)
        nn.init.zeros_(self.xy.weight); nn.init.zeros_(self.xy.bias)
        nn.init.zeros_(self.yaw.weight); nn.init.zeros_(self.yaw.bias)

    def forward(self, source_token: torch.Tensor, local_scene: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.fusion(source_token, local_scene)
        q = h[:, None, :] + self.time
        q = self.decoder(q)
        return {
            "residual_xy_m": self.xy(q),
            "yaw_delta_rad": self.yaw(q)[..., 0],
            "existence_logits": self.exist(q)[..., 0],
        }


class BirthQueryHead(nn.Module):
    """Set prediction for ancestor-free dynamic objects.

    One learned query predicts one identity-consistent object over all six
    horizons.  First-appearance is derived from existence logits rather than a
    separate head.
    """

    def __init__(
        self,
        scene_dim: int,
        cfg: V20SceneConfig = V20SceneConfig(),
        *,
        shape_size_xyz: tuple[int, int, int] = (12, 12, 8),
    ):
        super().__init__()
        self.cfg = cfg
        self.Q = int(cfg.birth_queries)
        if int(cfg.dynamic_classes) != len(DYNAMIC_IDS):
            raise ValueError(
                "birth dynamic_classes must match frozen DYNAMIC_IDS "
                f"({len(DYNAMIC_IDS)})"
            )
        self.shape_size_xyz = tuple(int(x) for x in shape_size_xyz)
        h = 2 * int(cfg.base_dim)
        self.query = nn.Parameter(torch.zeros(1, self.Q, h))
        self.global_proj = nn.Linear(int(scene_dim), h)
        enc = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=4,
            dim_feedforward=4 * h,
            batch_first=True,
            norm_first=True,
        )
        # This transformer only mixes Q object queries; it never sees all scene voxels.
        self.query_mixer = nn.TransformerEncoder(enc, num_layers=1)
        self.class_head = nn.Linear(h, int(cfg.dynamic_classes) + 1)  # + no-object
        self.exist_head = nn.Linear(h, FUTURE_FRAMES)
        self.traj_head = nn.Linear(h, FUTURE_FRAMES * 3)  # x,y,yaw
        sx, sy, sz = self.shape_size_xyz
        self.shape_head = nn.Linear(h, sx * sy * sz)

    def forward(self, scene: torch.Tensor) -> dict[str, torch.Tensor]:
        if scene.ndim != 5:
            raise ValueError("scene must be [B,C,X,Y,Z]")
        pooled = scene.mean(dim=(2, 3, 4))
        q = self.query.expand(scene.shape[0], -1, -1)
        q = q + self.global_proj(pooled).unsqueeze(1)
        q = self.query_mixer(q)
        traj = self.traj_head(q).reshape(
            scene.shape[0], self.Q, FUTURE_FRAMES, 3
        )
        sx, sy, sz = self.shape_size_xyz
        shape = self.shape_head(q).reshape(scene.shape[0], self.Q, sx, sy, sz)
        return {
            "class_logits": self.class_head(q),
            "existence_logits": self.exist_head(q),
            "trajectory_xy_yaw": traj,
            "shape_logits": shape,
        }

    @staticmethod
    def first_appearance(existence_logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        active = torch.sigmoid(existence_logits) >= float(threshold)
        B, Q, Fh = active.shape
        ids = torch.arange(Fh, device=active.device).view(1, 1, Fh).expand(B, Q, Fh)
        sentinel = torch.full_like(ids, Fh)
        return torch.where(active, ids, sentinel).amin(dim=-1)


class V20HistoryWorldModel(nn.Module):
    """Shared V20 scene path plus Static/Dormant/Birth heads."""

    def __init__(self, cfg: V20SceneConfig = V20SceneConfig()):
        super().__init__()
        self.cfg = cfg
        self.encoder = HistoricalEvidence3DEncoder(cfg)
        d = self.encoder.output_dim
        self.static = StaticWorldHead(d, cfg)
        self.dormant = DormantSourceHead(cfg.source_dim, d)
        self.birth = BirthQueryHead(d, cfg)

    def encode_history(
        self,
        semantic: torch.Tensor,
        observed: torch.Tensor,
        observed_free: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(semantic, observed, observed_free)
