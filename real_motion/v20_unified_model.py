"""V20 unified transport-completion model.

The model has one V18 source path and one 18-class completion path.  Static,
Dormant and Birth are evaluation groups only; there is no routing head here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .local_st_world_model import SEMANTIC_CLASSES
from .local_st_world_model_v18_se2 import LocalSpatialTemporalWorldModelV18SE2
from .motion_transport import FUTURE_FRAMES
from .v20_history_world import FREE_LABEL, CanonicalLattice
from .v20_scene_model import HistoricalEvidence3DEncoder, V20SceneConfig
from .v20_unified_data import (
    CompletionTileQuery,
    UnifiedHistoryInput,
    UnifiedSourceInput,
    audit_inference_input_names,
)
from .v20_unified_loss import compute_training_loss as _compute_training_loss
from .v20_unified_runtime import prepare_runtime_queries as _prepare_runtime_queries

V20_UNIFIED_MODEL_PROTOCOL = "p0_f9_v20_unified_transport_completion_v1"


@dataclass(frozen=True)
class V20UnifiedConfig:
    scene_dim: int = 64
    future_dim: int = 64
    tile_dim: int = 48
    completion_classes: int = SEMANTIC_CLASSES
    free_logit_bias: float = 2.0
    completion_last_std: float = 1.0e-3
    tile_decode_batch_size: int = 8
    runtime_query_chunk: int = 32
    checkpoint_completion_tiles: bool = True


@dataclass(frozen=True)
class HistoryEncoding:
    features: torch.Tensor
    global_features: torch.Tensor
    observation_summary: torch.Tensor
    future_ego_to_t0: torch.Tensor


@dataclass(frozen=True)
class SourceFusionOutput:
    shared_queries: torch.Tensor
    adapter_delta: torch.Tensor
    source_positions_xyz_t0_m: torch.Tensor


@dataclass(frozen=True)
class SourceScatterReport:
    requested_points: int
    in_bounds_points: int
    out_of_bounds_points: int


def _group_count(channels: int) -> int:
    for value in (8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


class V20UnifiedTransportCompletion(nn.Module):
    def __init__(
        self,
        v18: LocalSpatialTemporalWorldModelV18SE2,
        *,
        coarse_lattice: CanonicalLattice,
        config: V20UnifiedConfig = V20UnifiedConfig(),
    ):
        super().__init__()
        if int(config.completion_classes) != SEMANTIC_CLASSES:
            raise ValueError("V20 completion must have exactly 18 classes")
        self.v18 = v18
        self.coarse_lattice = coarse_lattice
        self.config = config
        d_model = int(v18.config.d_model)
        history_cfg = V20SceneConfig(base_dim=int(config.scene_dim) // 2)
        self.history_encoder = HistoricalEvidence3DEncoder(history_cfg)
        if self.history_encoder.output_dim != int(config.scene_dim):
            raise ValueError("scene_dim must be twice the history encoder base_dim")

        # q0, source history, current/future scene, global scene, positions/time.
        fusion_in = 2 * d_model + 3 * int(config.scene_dim) + 8
        self.source_adapter = nn.Sequential(
            nn.Linear(fusion_in, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        nn.init.zeros_(self.source_adapter[-1].weight)
        nn.init.zeros_(self.source_adapter[-1].bias)

        c = int(config.future_dim)
        self.scatter_projection = nn.Linear(2 * d_model, c)
        self.transport_condition_projection = nn.Conv3d(SEMANTIC_CLASSES + 1, c, 1)
        self.future_input = nn.Conv3d(int(config.scene_dim) + 2 * c + 1, c, 1)
        self.future_blocks = nn.Sequential(
            nn.Conv3d(c, c, 3, padding=1),
            nn.GroupNorm(_group_count(c), c),
            nn.GELU(),
            nn.Conv3d(c, c, 3, padding=1),
            nn.GroupNorm(_group_count(c), c),
            nn.GELU(),
        )
        self.future_film = nn.Sequential(
            nn.Linear(int(config.scene_dim) + 12 + 2, 2 * c),
            nn.GELU(),
            nn.Linear(2 * c, 2 * c),
        )
        nn.init.zeros_(self.future_film[-1].weight)
        nn.init.zeros_(self.future_film[-1].bias)

        td = int(config.tile_dim)
        # F_t + history observed/free + normalized xyz + query-valid.
        self.completion_trunk = nn.Sequential(
            nn.Conv3d(c + 2 + 3 + 1, td, 3, padding=1),
            nn.GroupNorm(_group_count(td), td),
            nn.GELU(),
            nn.Conv3d(td, td, 3, padding=1),
            nn.GroupNorm(_group_count(td), td),
            nn.GELU(),
        )
        self.completion_head = nn.Conv3d(td, SEMANTIC_CLASSES, 1)
        nn.init.normal_(
            self.completion_head.weight,
            mean=0.0,
            std=float(config.completion_last_std),
        )
        nn.init.zeros_(self.completion_head.bias)
        with torch.no_grad():
            self.completion_head.bias[FREE_LABEL] = float(config.free_logit_bias)

    @property
    def d_model(self) -> int:
        return int(self.v18.config.d_model)

    def encode_history(self, history: UnifiedHistoryInput) -> HistoryEncoding:
        audit_inference_input_names(history)
        history.validate()
        features = self.history_encoder(
            history.semantic, history.observed, history.observed_free
        )
        summary = torch.stack(
            (
                history.observed.float().mean(dim=1),
                history.observed_free.float().mean(dim=1),
            ),
            dim=1,
        ).to(features.dtype)
        return HistoryEncoding(
            features=features,
            global_features=features.mean(dim=(2, 3, 4)),
            observation_summary=summary,
            future_ego_to_t0=history.future_ego_to_t0,
        )

    def encode_sources(
        self,
        *,
        features: torch.Tensor,
        local_semantic_tube: torch.Tensor,
        kta_displacement_xy_m: torch.Tensor,
        frame_motion_features: torch.Tensor,
        target_source_mask_tube: torch.Tensor,
        source_anchor_xyz_t0_m: torch.Tensor,
        window_index: torch.Tensor,
    ) -> tuple[UnifiedSourceInput, dict[str, torch.Tensor]]:
        out = self.v18(
            features,
            local_semantic_tube,
            kta_displacement_xy_m,
            frame_motion_features,
            target_source_mask_tube,
            return_latents=True,
            decode_outputs=False,
        )
        source = UnifiedSourceInput(
            history_source_context=out["history_source_context"],
            future_transport_queries=out["future_transport_queries"],
            source_anchor_xyz_t0_m=source_anchor_xyz_t0_m,
            kta_displacement_xy_m=kta_displacement_xy_m,
            window_index=window_index.long(),
        )
        source.validate(self.d_model)
        return source, out

    def _normalized_xyz(self, xyz: torch.Tensor) -> torch.Tensor:
        origin = torch.as_tensor(
            self.coarse_lattice.origin_xyz_m, device=xyz.device, dtype=xyz.dtype
        )
        step = torch.as_tensor(
            self.coarse_lattice.voxel_size_xyz_m, device=xyz.device, dtype=xyz.dtype
        )
        shape = torch.as_tensor(
            self.coarse_lattice.shape_xyz, device=xyz.device, dtype=xyz.dtype
        )
        fractional_index = (xyz - origin) / step - 0.5
        return 2.0 * fractional_index / (shape - 1.0).clamp_min(1.0) - 1.0

    def _sample_volume(
        self,
        volume: torch.Tensor,
        points_xyz: torch.Tensor,
        window_index: torch.Tensor,
    ) -> torch.Tensor:
        """Sample [B,C,X,Y,Z] using explicit grid_sample XYZ ordering."""
        if volume.ndim != 5 or tuple(volume.shape[2:]) != self.coarse_lattice.shape_xyz:
            raise ValueError("volume shape does not match coarse lattice")
        if points_xyz.shape[-1] != 3 or tuple(window_index.shape) != points_xyz.shape[:-1]:
            raise ValueError("window_index must match point leading dimensions")
        leading = points_xyz.shape[:-1]
        flat_points = points_xyz.reshape(-1, 3)
        flat_windows = window_index.reshape(-1).long()
        result = volume.new_zeros((flat_points.shape[0], volume.shape[1]))
        norm = self._normalized_xyz(flat_points)
        for b in range(int(volume.shape[0])):
            select = flat_windows == b
            if not bool(select.any()):
                continue
            # Input [D,H,W]=[X,Y,Z], grid tuple=(W,H,D)=(Z,Y,X).
            grid = norm[select][:, (2, 1, 0)].view(1, -1, 1, 1, 3)
            sampled = F.grid_sample(
                volume[b : b + 1],
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            result[select] = sampled[0, :, :, 0, 0].transpose(0, 1)
        return result.reshape(*leading, volume.shape[1])

    def fuse_source_scene(
        self,
        history: HistoryEncoding,
        sources: UnifiedSourceInput,
        *,
        adapter_enabled: bool,
    ) -> SourceFusionOutput:
        sources.validate(self.d_model)
        q0 = sources.future_transport_queries
        n = int(q0.shape[0])
        pos = sources.source_anchor_xyz_t0_m[:, None, :].expand(-1, FUTURE_FRAMES, -1).clone()
        pos[..., :2] = pos[..., :2] + sources.kta_displacement_xy_m.to(pos.dtype)
        if n == 0:
            delta = torch.zeros_like(q0)
            return SourceFusionOutput(q0, delta, pos)

        windows = sources.window_index[:, None].expand(-1, FUTURE_FRAMES)
        current_scene = self._sample_volume(
            history.features,
            sources.source_anchor_xyz_t0_m,
            sources.window_index,
        )[:, None, :].expand(-1, FUTURE_FRAMES, -1)
        future_scene = self._sample_volume(history.features, pos, windows)
        global_scene = history.global_features.index_select(0, sources.window_index.long())
        global_scene = global_scene[:, None, :].expand(-1, FUTURE_FRAMES, -1)
        source_history = sources.history_source_context[:, None, :].expand(
            -1, FUTURE_FRAMES, -1
        )
        current_norm = self._normalized_xyz(sources.source_anchor_xyz_t0_m)
        current_norm = current_norm[:, None, :].expand(-1, FUTURE_FRAMES, -1)
        future_norm = self._normalized_xyz(pos)
        phase = torch.arange(FUTURE_FRAMES, device=q0.device, dtype=q0.dtype)
        phase = (phase + 1.0) / float(FUTURE_FRAMES)
        time = torch.stack((torch.sin(torch.pi * phase), torch.cos(torch.pi * phase)), dim=-1)
        time = time[None].expand(n, -1, -1)
        adapter_input = torch.cat(
            (
                q0,
                source_history,
                current_scene,
                future_scene,
                global_scene,
                current_norm,
                future_norm,
                time,
            ),
            dim=-1,
        )
        delta = self.source_adapter(adapter_input)
        shared = q0 + delta if adapter_enabled else q0
        return SourceFusionOutput(shared, delta, pos)

    def decode_transport(self, shared_queries: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.v18.decode_transport_queries(shared_queries)

    def source_positions_from_transport(
        self,
        sources: UnifiedSourceInput,
        transport_outputs: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return detached current predicted future source centers for scatter."""
        residual = transport_outputs["residual_xy_m"]
        expected = (
            sources.future_transport_queries.shape[0],
            FUTURE_FRAMES,
            2,
        )
        if tuple(residual.shape) != expected:
            raise ValueError("transport residual shape mismatch")
        pos = sources.source_anchor_xyz_t0_m[:, None, :].expand(
            -1, FUTURE_FRAMES, -1
        ).clone()
        pos[..., :2] = (
            pos[..., :2]
            + sources.kta_displacement_xy_m.to(pos.dtype)
            + residual.to(pos.dtype)
        )
        return pos.detach()

    def _scatter_source_tokens(
        self,
        history: HistoryEncoding,
        sources: UnifiedSourceInput,
        fusion: SourceFusionOutput,
    ) -> tuple[torch.Tensor, torch.Tensor, SourceScatterReport]:
        B = int(history.features.shape[0])
        T = FUTURE_FRAMES
        C = int(self.config.future_dim)
        shape = self.coarse_lattice.shape_xyz
        flat_voxels = shape[0] * shape[1] * shape[2]
        field = history.features.new_zeros((B, T, flat_voxels, C))
        weight = history.features.new_zeros((B, T, flat_voxels, 1))
        token = self.scatter_projection(
            torch.cat(
                (
                    sources.history_source_context[:, None, :].expand(-1, T, -1),
                    fusion.shared_queries,
                ),
                dim=-1,
            )
        )
        pos = fusion.source_positions_xyz_t0_m.detach()
        origin = torch.as_tensor(
            self.coarse_lattice.origin_xyz_m, device=pos.device, dtype=pos.dtype
        )
        step = torch.as_tensor(
            self.coarse_lattice.voxel_size_xyz_m, device=pos.device, dtype=pos.dtype
        )
        fidx = (pos - origin) / step - 0.5
        base = torch.floor(fidx).long()
        frac = fidx - base.to(fidx.dtype)
        requested = int(pos.shape[0] * pos.shape[1])
        maximum = origin + step * torch.as_tensor(
            shape, device=pos.device, dtype=pos.dtype
        )
        in_bounds_point = ((pos >= origin) & (pos < maximum)).all(dim=-1)

        # Vectorize source/window/horizon; only eight trilinear corners remain.
        field_flat = field.view(B * T * flat_voxels, C)
        weight_flat = weight.view(B * T * flat_voxels, 1)
        batch_index = sources.window_index.long()[:, None].expand(-1, T)
        horizon_index = torch.arange(T, device=pos.device)[None].expand(
            pos.shape[0], -1
        )
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    offset = torch.tensor((dx, dy, dz), device=base.device)
                    idx = base + offset
                    valid = (
                        in_bounds_point
                        & (idx[..., 0] >= 0)
                        & (idx[..., 0] < shape[0])
                        & (idx[..., 1] >= 0)
                        & (idx[..., 1] < shape[1])
                        & (idx[..., 2] >= 0)
                        & (idx[..., 2] < shape[2])
                    )
                    if not bool(valid.any()):
                        continue
                    linear = (
                        idx[..., 0] * (shape[1] * shape[2])
                        + idx[..., 1] * shape[2]
                        + idx[..., 2]
                    )
                    global_linear = (
                        (batch_index * T + horizon_index) * flat_voxels + linear
                    )[valid]
                    corner_weight = (
                        (frac[..., 0] if dx else 1.0 - frac[..., 0])
                        * (frac[..., 1] if dy else 1.0 - frac[..., 1])
                        * (frac[..., 2] if dz else 1.0 - frac[..., 2])
                    )
                    w = corner_weight[valid].to(field.dtype).unsqueeze(-1)
                    field_flat.index_add_(0, global_linear, token[valid] * w)
                    weight_flat.index_add_(0, global_linear, w)

        field = field / weight.clamp_min(1.0e-6)
        field = field.view(B, T, *shape, C).permute(0, 1, 5, 2, 3, 4)
        density = weight.view(B, T, *shape, 1).permute(0, 1, 5, 2, 3, 4)
        in_count = int(in_bounds_point.sum().item())
        return field, density, SourceScatterReport(
            requested, in_count, requested - in_count
        )

    def build_future_features(
        self,
        history: HistoryEncoding,
        sources: UnifiedSourceInput,
        fusion: SourceFusionOutput,
        current_transport_condition: torch.Tensor,
    ) -> tuple[torch.Tensor, SourceScatterReport]:
        expected_condition = (
            history.features.shape[0],
            FUTURE_FRAMES,
            SEMANTIC_CLASSES + 1,
            *self.coarse_lattice.shape_xyz,
        )
        if tuple(current_transport_condition.shape) != expected_condition:
            raise ValueError(
                "transport condition must be [B,6,19,Xc,Yc,Zc]"
            )
        source_field, density, report = self._scatter_source_tokens(history, sources, fusion)
        B, T = int(history.features.shape[0]), FUTURE_FRAMES
        scene = history.features[:, None].expand(-1, T, -1, -1, -1, -1)
        cond_input = current_transport_condition.to(scene.dtype).reshape(
            B * T,
            SEMANTIC_CLASSES + 1,
            *self.coarse_lattice.shape_xyz,
        )
        cond = self.transport_condition_projection(cond_input).view(
            B, T, -1, *self.coarse_lattice.shape_xyz
        )
        x = torch.cat((scene, source_field, density, cond), dim=2)
        x = x.reshape(B * T, x.shape[2], *x.shape[3:])
        x = self.future_input(x)
        x = x + self.future_blocks(x)
        pose = history.future_ego_to_t0[:, :, :3, :4].reshape(B, T, 12).to(x.dtype)
        global_scene = history.global_features[:, None].expand(-1, T, -1)
        phase = (torch.arange(T, device=x.device, dtype=x.dtype) + 1.0) / float(T)
        time = torch.stack((torch.sin(torch.pi * phase), torch.cos(torch.pi * phase)), dim=-1)
        time = time[None].expand(B, -1, -1)
        scale, shift = self.future_film(
            torch.cat((global_scene, pose, time), dim=-1)
        ).chunk(2, dim=-1)
        x = x.view(B, T, x.shape[1], *x.shape[2:])
        x = x * (1.0 + scale[..., None, None, None]) + shift[..., None, None, None]
        return x, report

    def decode_completion_from_features(
        self,
        history: HistoryEncoding,
        future_features: torch.Tensor,
        queries: Sequence[CompletionTileQuery],
    ) -> list[torch.Tensor]:
        """Decode shape-homogeneous tiles in bounded micro-batches."""
        if future_features.ndim != 6 or future_features.shape[1] != FUTURE_FRAMES:
            raise ValueError("future_features must be [B,6,C,Xc,Yc,Zc]")
        if not queries:
            return []
        outputs: list[torch.Tensor | None] = [None] * len(queries)
        groups: dict[
            tuple[int, int, int], list[tuple[int, CompletionTileQuery]]
        ] = {}
        for qi, query in enumerate(queries):
            query.validate()
            groups.setdefault(
                tuple(query.tile.halo_shape_xyz), []
            ).append((qi, query))

        B, T = int(future_features.shape[0]), FUTURE_FRAMES
        flat_future = future_features.reshape(
            B * T, future_features.shape[2], *future_features.shape[3:]
        )
        micro = max(int(self.config.tile_decode_batch_size), 1)
        for rows in groups.values():
            for start in range(0, len(rows), micro):
                part = rows[start : start + micro]
                points = torch.stack(
                    [q.points_xyz_t0_m for _, q in part], dim=0
                )
                future_windows = torch.stack(
                    [
                        torch.full(
                            q.points_xyz_t0_m.shape[:-1],
                            q.tile.window_index * T + q.tile.horizon,
                            device=points.device,
                            dtype=torch.long,
                        )
                        for _, q in part
                    ],
                    dim=0,
                )
                history_windows = torch.stack(
                    [
                        torch.full(
                            q.points_xyz_t0_m.shape[:-1],
                            q.tile.window_index,
                            device=points.device,
                            dtype=torch.long,
                        )
                        for _, q in part
                    ],
                    dim=0,
                )
                local_future = self._sample_volume(
                    flat_future, points, future_windows
                )
                local_history = self._sample_volume(
                    history.observation_summary, points, history_windows
                )
                xyz = self._normalized_xyz(points)
                valid = torch.stack(
                    [q.geometry_valid for _, q in part], dim=0
                ).to(local_future.dtype).unsqueeze(-1)
                tile_input = torch.cat(
                    (local_future, local_history, xyz, valid), dim=-1
                ).permute(0, 4, 1, 2, 3)
                if (
                    self.training
                    and torch.is_grad_enabled()
                    and bool(self.config.checkpoint_completion_tiles)
                ):
                    decoded = activation_checkpoint(
                        self._decode_completion_batch,
                        tile_input,
                        use_reentrant=False,
                    )
                else:
                    decoded = self._decode_completion_batch(tile_input)
                batched_logits = decoded.permute(0, 2, 3, 4, 1)
                for bi, (qi, _) in enumerate(part):
                    outputs[qi] = batched_logits[bi]
        if any(x is None for x in outputs):
            raise RuntimeError("completion tile batching lost an output")
        return [x for x in outputs if x is not None]

    def _decode_completion_batch(self, tile_input: torch.Tensor) -> torch.Tensor:
        """Checkpointable tile trunk+head; returns [M,18,X,Y,Z]."""
        return self.completion_head(self.completion_trunk(tile_input))

    def decode_completion(
        self,
        history: HistoryEncoding,
        sources: UnifiedSourceInput,
        fusion: SourceFusionOutput,
        current_transport_condition: torch.Tensor,
        queries: Sequence[CompletionTileQuery],
    ) -> tuple[list[torch.Tensor], SourceScatterReport]:
        future, report = self.build_future_features(
            history, sources, fusion, current_transport_condition
        )
        return self.decode_completion_from_features(history, future, queries), report

    def prepare_runtime_queries(self, *args, **kwargs):
        return _prepare_runtime_queries(*args, coarse_lattice=self.coarse_lattice, **kwargs)

    def compose(self, *args, **kwargs):
        from .v20_unified_runtime import compose

        return compose(*args, **kwargs)

    def compute_training_loss(self, *args, **kwargs):
        return _compute_training_loss(*args, **kwargs)

    def forward(
        self,
        history: UnifiedHistoryInput,
        *,
        features: torch.Tensor,
        local_semantic_tube: torch.Tensor,
        kta_displacement_xy_m: torch.Tensor,
        frame_motion_features: torch.Tensor,
        target_source_mask_tube: torch.Tensor,
        source_anchor_xyz_t0_m: torch.Tensor,
        window_index: torch.Tensor,
        adapter_enabled: bool,
    ) -> dict[str, object]:
        """Observation-only first stage; a hard renderer supplies transport next."""
        encoded_history = self.encode_history(history)
        sources, original = self.encode_sources(
            features=features,
            local_semantic_tube=local_semantic_tube,
            kta_displacement_xy_m=kta_displacement_xy_m,
            frame_motion_features=frame_motion_features,
            target_source_mask_tube=target_source_mask_tube,
            source_anchor_xyz_t0_m=source_anchor_xyz_t0_m,
            window_index=window_index,
        )
        fusion = self.fuse_source_scene(
            encoded_history, sources, adapter_enabled=adapter_enabled
        )
        transport = self.decode_transport(fusion.shared_queries)
        fusion = SourceFusionOutput(
            shared_queries=fusion.shared_queries,
            adapter_delta=fusion.adapter_delta,
            source_positions_xyz_t0_m=self.source_positions_from_transport(
                sources, transport
            ),
        )
        return {
            "history": encoded_history,
            "sources": sources,
            "fusion": fusion,
            "transport": transport,
            "v18_original": original,
        }
