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
    source_dim: int = 128
    tile_dim: int = 48
    birth_queries: int = 8
    birth_shape_size_xyz: tuple[int, int, int] = (36, 18, 12)
    birth_shape_voxel_size_m: float = 0.4
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
        self.prefer_channels_last_3d = False
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

    def set_channels_last_3d(self, enabled: bool = True) -> None:
        """Select 3D channels-last for the convolutional history encoder."""
        self.prefer_channels_last_3d = bool(enabled)
        if self.prefer_channels_last_3d:
            self.stem.to(memory_format=torch.channels_last_3d)
            self.refine.to(memory_format=torch.channels_last_3d)

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
        invalid_free = (observed_free.bool() & ~observed.bool()).any()
        if invalid_free.device.type == "cuda" and hasattr(torch, "_assert_async"):
            torch._assert_async(
                ~invalid_free,
                "observed_free cannot occur in unknown voxels",
            )
        elif bool(invalid_free.item()):
            raise ValueError("observed_free cannot occur in unknown voxels")
        emb = self.semantic_embedding(semantic.long())  # B,T,X,Y,Z,D
        obs = observed.to(emb.dtype).unsqueeze(-1)
        free = observed_free.to(emb.dtype).unsqueeze(-1)
        x = torch.cat((emb, obs, free), dim=-1)
        # B,T,X,Y,Z,C -> B,T*C,X,Y,Z
        x = x.permute(0, 1, 5, 2, 3, 4).reshape(B, -1, X, Y, Z)
        if self.prefer_channels_last_3d:
            x = x.contiguous(memory_format=torch.channels_last_3d)
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
        self.prefer_channels_last_3d = False
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
        # Zero-contribution initialization: before training, Static decodes to
        # free everywhere. This is stronger than merely disabling the branch.
        nn.init.zeros_(self.coarse_head.weight)
        nn.init.zeros_(self.coarse_head.bias)
        with torch.no_grad():
            self.coarse_head.bias[FREE_LABEL] = 8.0
        last = self.tile_refine[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        with torch.no_grad():
            last.bias[FREE_LABEL] = 8.0

    def set_tile_channels_last_3d(self, enabled: bool = True) -> None:
        """Select channels-last-3d for the expensive Static tile Conv3D path.

        This changes only tensor memory format, never tensor values or the
        state-dict contract.  It is enabled by the Static Repair trainer on
        CUDA because cuDNN otherwise inserts repeated NCDHW<->NDHWC converts.
        """
        self.prefer_channels_last_3d = bool(enabled)
        if self.prefer_channels_last_3d:
            self.tile_refine.to(memory_format=torch.channels_last_3d)

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
        output_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Refine one packed native-resolution tile batch.

        sample_grid is the normalized grid_sample grid [B,D,H,W,3].
        Context channels distinguish query support, history-seen, and t0-missing.
        """
        if sample_grid.ndim != 5 or sample_grid.shape[-1] != 3:
            raise ValueError("sample_grid must be [B,D,H,W,3]")
        Bgrid, D, H, W, _ = sample_grid.shape
        if scene_features.shape[0] == 1 and Bgrid > 1:
            # All tiles come from the same canonical scene. Pack their output
            # grids along D and sample the scene once instead of logically
            # expanding the full coarse 3D input B times.
            packed_grid = sample_grid.reshape(1, Bgrid * D, H, W, 3)
            packed = F.grid_sample(
                scene_features,
                packed_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            C = int(packed.shape[1])
            x = packed[0].reshape(C, Bgrid, D, H, W).permute(
                1, 0, 2, 3, 4
            ).contiguous()
        else:
            if scene_features.shape[0] not in {1, Bgrid}:
                raise ValueError(
                    "scene/grid batch mismatch for static tile refinement"
                )
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
        tile_input = torch.cat((x, masks), dim=1)
        if self.prefer_channels_last_3d:
            tile_input = tile_input.contiguous(
                memory_format=torch.channels_last_3d
            )

        if output_ids is None:
            return self.tile_refine(tile_input)

        # Exact subset path: the final layer is a 1x1x1 Conv3d, so selecting
        # its output channels before convolution is algebraically identical to
        # full 18-class convolution followed by index_select. It avoids
        # computing unused dynamic-class logits during Static Repair.
        h = self.tile_refine[:-1](tile_input)
        last = self.tile_refine[-1]
        ids = output_ids.to(device=last.weight.device, dtype=torch.long)
        weight = last.weight.index_select(0, ids)
        bias = (
            None
            if last.bias is None
            else last.bias.index_select(0, ids)
        )
        return F.conv3d(
            h,
            weight,
            bias,
            stride=last.stride,
            padding=last.padding,
            dilation=last.dilation,
            groups=last.groups,
        )


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
        nn.init.zeros_(self.exist.weight); nn.init.constant_(self.exist.bias, -8.0)

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
    """Persistent Birth queries over spatial scene evidence and V18 source state.

    Each query attends to:
      1) a small spatial token grid pooled from the shared 3D history volume;
      2) frozen V18 current-source context tokens;
      3) frozen V18 six-horizon transport-query tokens at predicted positions.

    This keeps Q small while making Birth explicitly aware of where historical
    evidence exists and which dynamic sources are already owned by V18.
    """

    def __init__(
        self,
        scene_dim: int,
        cfg: V20SceneConfig = V20SceneConfig(),
        *,
        shape_size_xyz: tuple[int, int, int] | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.Q = int(cfg.birth_queries)
        if int(cfg.dynamic_classes) != len(DYNAMIC_IDS):
            raise ValueError(
                "birth dynamic_classes must match frozen DYNAMIC_IDS "
                f"({len(DYNAMIC_IDS)})"
            )
        self.shape_size_xyz = tuple(
            int(x) for x in (
                cfg.birth_shape_size_xyz if shape_size_xyz is None else shape_size_xyz
            )
        )
        if len(self.shape_size_xyz) != 3 or min(self.shape_size_xyz) <= 0:
            raise ValueError("invalid Birth shape lattice")
        if float(cfg.birth_shape_voxel_size_m) <= 0:
            raise ValueError("birth_shape_voxel_size_m must be positive")

        h = 2 * int(cfg.base_dim)
        self.hidden_dim = h
        self.scene_token_grid = (4, 4, 2)
        self.query = nn.Parameter(torch.zeros(1, self.Q, h))
        self.global_proj = nn.Linear(int(scene_dim), h)
        self.scene_token_proj = nn.Linear(int(scene_dim), h)
        self.scene_pos_proj = nn.Sequential(
            nn.Linear(3, h),
            nn.GELU(),
            nn.Linear(h, h),
        )
        self.source_token_proj = nn.Linear(int(cfg.source_dim), h)
        self.source_pos_proj = nn.Sequential(
            nn.Linear(3, h),
            nn.GELU(),
            nn.Linear(h, h),
        )
        self.current_source_type = nn.Parameter(torch.zeros(1, 1, h))
        self.future_source_type = nn.Parameter(torch.zeros(1, 1, h))
        self.future_time = nn.Parameter(torch.zeros(1, FUTURE_FRAMES, h))
        self.scene_attn = nn.MultiheadAttention(h, 4, batch_first=True)
        self.source_attn = nn.MultiheadAttention(h, 4, batch_first=True)
        self.scene_norm = nn.LayerNorm(h)
        self.source_norm = nn.LayerNorm(h)

        enc = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=4,
            dim_feedforward=4 * h,
            batch_first=True,
            norm_first=True,
        )
        self.query_mixer = nn.TransformerEncoder(enc, num_layers=1)
        self.class_head = nn.Linear(h, int(cfg.dynamic_classes) + 1)
        self.exist_head = nn.Linear(h, FUTURE_FRAMES)
        self.traj_head = nn.Linear(h, FUTURE_FRAMES * 4)
        sx, sy, sz = self.shape_size_xyz
        self.shape_head = nn.Linear(h, sx * sy * sz)

        nn.init.trunc_normal_(self.query, std=0.02)
        nn.init.trunc_normal_(self.future_time, std=0.02)
        nn.init.zeros_(self.current_source_type)
        nn.init.zeros_(self.future_source_type)
        # Zero-contribution initialization: no object/no existence/no shape.
        nn.init.zeros_(self.class_head.weight)
        nn.init.zeros_(self.class_head.bias)
        nn.init.zeros_(self.exist_head.weight)
        nn.init.constant_(self.exist_head.bias, -8.0)
        nn.init.zeros_(self.shape_head.weight)
        nn.init.constant_(self.shape_head.bias, -8.0)
        with torch.no_grad():
            self.class_head.bias[int(cfg.dynamic_classes)] = 8.0

    def _scene_tokens(self, scene: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool3d(scene, self.scene_token_grid)
        B, C, X, Y, Z = pooled.shape
        tok = pooled.flatten(2).transpose(1, 2)
        xs = torch.linspace(-1.0, 1.0, X, device=scene.device, dtype=scene.dtype)
        ys = torch.linspace(-1.0, 1.0, Y, device=scene.device, dtype=scene.dtype)
        zs = torch.linspace(-1.0, 1.0, Z, device=scene.device, dtype=scene.dtype)
        xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
        pos = torch.stack((xx, yy, zz), dim=-1).reshape(1, X * Y * Z, 3)
        return self.scene_token_proj(tok) + self.scene_pos_proj(pos).expand(B, -1, -1)

    def _source_tokens(
        self,
        *,
        current_source_tokens: torch.Tensor | None,
        current_source_xyz_norm: torch.Tensor | None,
        future_source_tokens: torch.Tensor | None,
        future_source_xyz_norm: torch.Tensor | None,
    ) -> torch.Tensor | None:
        rows = []
        if current_source_tokens is not None:
            if current_source_xyz_norm is None:
                raise ValueError("current source positions are required with source tokens")
            if current_source_tokens.ndim != 3 or current_source_xyz_norm.shape != (
                current_source_tokens.shape[0], current_source_tokens.shape[1], 3
            ):
                raise ValueError("current source token/position shape mismatch")
            rows.append(
                self.source_token_proj(current_source_tokens)
                + self.source_pos_proj(current_source_xyz_norm.to(current_source_tokens.dtype))
                + self.current_source_type
            )
        if future_source_tokens is not None:
            if future_source_xyz_norm is None:
                raise ValueError("future source positions are required with future tokens")
            if future_source_tokens.ndim != 4 or future_source_tokens.shape[2] != FUTURE_FRAMES:
                raise ValueError("future_source_tokens must be [B,N,6,C]")
            if future_source_xyz_norm.shape != future_source_tokens.shape[:3] + (3,):
                raise ValueError("future source token/position shape mismatch")
            B, N, Fh, C = future_source_tokens.shape
            ft = self.source_token_proj(
                future_source_tokens.reshape(B, N * Fh, C)
            )
            fp = self.source_pos_proj(
                future_source_xyz_norm.to(future_source_tokens.dtype).reshape(B, N * Fh, 3)
            )
            tm = self.future_time[:, None].expand(B, N, Fh, -1).reshape(B, N * Fh, -1)
            rows.append(ft + fp + tm + self.future_source_type)
        if not rows:
            return None
        return torch.cat(rows, dim=1)

    def forward(
        self,
        scene: torch.Tensor,
        *,
        current_source_tokens: torch.Tensor | None = None,
        current_source_xyz_norm: torch.Tensor | None = None,
        future_source_tokens: torch.Tensor | None = None,
        future_source_xyz_norm: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if scene.ndim != 5:
            raise ValueError("scene must be [B,C,X,Y,Z]")
        B = scene.shape[0]
        pooled = scene.mean(dim=(2, 3, 4))
        q = self.query.expand(B, -1, -1)
        q = q + self.global_proj(pooled).unsqueeze(1)

        scene_kv = self._scene_tokens(scene)
        scene_delta, _ = self.scene_attn(q, scene_kv, scene_kv, need_weights=False)
        q = self.scene_norm(q + scene_delta)

        source_kv = self._source_tokens(
            current_source_tokens=current_source_tokens,
            current_source_xyz_norm=current_source_xyz_norm,
            future_source_tokens=future_source_tokens,
            future_source_xyz_norm=future_source_xyz_norm,
        )
        if source_kv is not None and source_kv.shape[1] > 0:
            source_delta, _ = self.source_attn(q, source_kv, source_kv, need_weights=False)
            q = self.source_norm(q + source_delta)

        q = self.query_mixer(q)
        traj = self.traj_head(q).reshape(B, self.Q, FUTURE_FRAMES, 4)
        sx, sy, sz = self.shape_size_xyz
        shape = self.shape_head(q).reshape(B, self.Q, sx, sy, sz)
        return {
            "class_logits": self.class_head(q),
            "existence_logits": self.exist_head(q),
            "trajectory_xyz_yaw": traj,
            "shape_logits": shape,
        }

    @staticmethod
    def first_appearance(
        existence_logits: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        active = torch.sigmoid(existence_logits) >= float(threshold)
        B, Q, Fh = active.shape
        ids = torch.arange(
            Fh, device=active.device
        ).view(1, 1, Fh).expand(B, Q, Fh)
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
