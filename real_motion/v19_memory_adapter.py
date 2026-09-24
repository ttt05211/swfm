"""V19 memory-only adapter for the frozen Clean-E14 V18 source forecaster.

The adapter is structurally gated. A source detected in the current block
state has gate=0, so its path is mathematically identical to Clean-E14 even
after the memory adapter has been trained. Real-observation age is tracked
separately; only memory-only sources receive the new status embedding and
zero-initialized residual heads.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .local_st_world_model import SEMANTIC_CLASSES
from .local_st_world_model_v17 import FRAME_MOTION_DIM, LocalSTWMV17Config
from .local_st_world_model_v18_se2 import LocalSpatialTemporalWorldModelV18SE2
from .motion_transport import FEATURE_DIM, FUTURE_FRAMES, HISTORY_FRAMES
from .v19_scene_memory import SOURCE_STATUS_DIM


class MemoryAdaptedV18SE2(LocalSpatialTemporalWorldModelV18SE2):
    """Frozen V18 core plus a gated, non-interfering source-memory adapter."""

    def __init__(
        self,
        config: LocalSTWMV17Config = LocalSTWMV17Config(),
        status_dim: int = SOURCE_STATUS_DIM,
    ):
        super().__init__(config)
        d = int(config.d_model)
        self.status_dim = int(status_dim)
        self.status_proj = nn.Sequential(
            nn.Linear(self.status_dim, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.memory_xy_delta = nn.Linear(d, 2)
        self.memory_yaw_delta = nn.Linear(d, 1)
        self.survival_head = nn.Sequential(
            nn.Linear(d + self.status_dim, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 1),
        )

        # Exact initialization contract: loading Clean-E14 plus these zero
        # adapters leaves every current-source output unchanged.
        nn.init.zeros_(self.status_proj[-1].weight)
        nn.init.zeros_(self.status_proj[-1].bias)
        nn.init.zeros_(self.memory_xy_delta.weight)
        nn.init.zeros_(self.memory_xy_delta.bias)
        nn.init.zeros_(self.memory_yaw_delta.weight)
        nn.init.zeros_(self.memory_yaw_delta.bias)
        nn.init.zeros_(self.survival_head[-1].weight)
        nn.init.zeros_(self.survival_head[-1].bias)

    def freeze_clean_core(self) -> None:
        """Freeze every inherited Clean-E14 parameter; train only memory modules."""
        trainable_prefixes = (
            "status_proj.",
            "memory_xy_delta.",
            "memory_yaw_delta.",
            "survival_head.",
        )
        for name, p in self.named_parameters():
            p.requires_grad = name.startswith(trainable_prefixes)

    def memory_parameter_names(self) -> list[str]:
        return [
            name
            for name, p in self.named_parameters()
            if p.requires_grad
        ]

    def forward(
        self,
        features: torch.Tensor,
        local_semantic_tube: torch.Tensor,
        kta_displacement_xy_m: torch.Tensor,
        frame_motion_features: torch.Tensor | None = None,
        target_source_mask_tube: torch.Tensor | None = None,
        history_status: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        cfg = self.config
        if features.ndim != 2 or features.shape[-1] != FEATURE_DIM:
            raise ValueError(f"features must be [B,{FEATURE_DIM}]")
        B = int(features.shape[0])
        if local_semantic_tube.ndim != 4 or tuple(
            local_semantic_tube.shape[1:]
        ) != (HISTORY_FRAMES, cfg.tube_hw, cfg.tube_hw):
            raise ValueError("local_semantic_tube shape mismatch")
        if kta_displacement_xy_m.shape != (B, FUTURE_FRAMES, 2):
            raise ValueError("kta_displacement_xy_m must be [B,6,2]")
        if frame_motion_features is None or frame_motion_features.shape != (
            B, HISTORY_FRAMES, FRAME_MOTION_DIM
        ):
            raise ValueError("frame_motion_features must be [B,6,5]")
        if (
            target_source_mask_tube is None
            or target_source_mask_tube.shape != local_semantic_tube.shape
        ):
            raise ValueError("target_source_mask_tube must match local_semantic_tube")
        if history_status is None or history_status.shape != (
            B, self.status_dim
        ):
            raise ValueError(
                f"history_status must be [B,{self.status_dim}]"
            )

        labels = local_semantic_tube.long()
        if bool((labels < 0).any()) or bool(
            (labels >= SEMANTIC_CLASSES).any()
        ):
            raise ValueError("semantic tube contains labels outside [0,17]")
        mask_labels = target_source_mask_tube.long()
        if bool((mask_labels < 0).any()) or bool((mask_labels > 1).any()):
            raise ValueError("target source mask must be binary")

        if B == 0:
            return {
                "residual_xy_m": features.new_empty((0, FUTURE_FRAMES, 2)),
                "existence_logits": features.new_empty((0, FUTURE_FRAMES)),
                "yaw_delta_rad": features.new_empty((0, FUTURE_FRAMES)),
                "survival_logits": features.new_empty((0, FUTURE_FRAMES)),
                "memory_gate": features.new_empty((0,)),
            }

        status = history_status.to(features.dtype)
        # status[:,0] = 1 when the source is detected in the current block
        # state. This may be a real observation in block 1 or a re-detected
        # source from generated occupancy in open-loop rollout. Such sources
        # are protected from all new modules.
        gate = (1.0 - status[:, 0]).clamp(0.0, 1.0)
        status_emb = self.status_proj(status) * gate[:, None]

        emb = self.semantic_embedding(labels) + self.source_mask_embedding(
            mask_labels
        )
        x = emb.permute(0, 1, 4, 2, 3).reshape(
            B * HISTORY_FRAMES,
            cfg.semantic_dim,
            cfg.tube_hw,
            cfg.tube_hw,
        )
        x = self.spatial_stem(x)
        Hs, Ws = x.shape[-2:]
        x = x.reshape(B, HISTORY_FRAMES, cfg.d_model, Hs, Ws)
        obj = self.kinematic_proj(features).view(
            B, 1, cfg.d_model, 1, 1
        )
        fm = self.frame_motion_proj(
            frame_motion_features.to(x.dtype)
        ).view(B, HISTORY_FRAMES, cfg.d_model, 1, 1)
        mem = status_emb.to(x.dtype).view(B, 1, cfg.d_model, 1, 1)
        x = (
            x
            + obj
            + fm
            + mem
            + self.time_embedding
            + self.spatial_embedding
        )
        for block in self.blocks:
            x = block(x)

        context = x.permute(0, 1, 3, 4, 2).reshape(
            B, HISTORY_FRAMES * Hs * Ws, cfg.d_model
        )
        q = (
            self.future_query.expand(B, -1, -1)
            + self.future_time_embedding
        )
        q = q + self.kinematic_proj(features).unsqueeze(1)
        q = q + self.kta_future_proj(
            kta_displacement_xy_m.to(q.dtype) / 20.0
        )
        q = q + status_emb.to(q.dtype).unsqueeze(1)
        for block in self.decoder:
            q = block(q, context)

        base_xy = self.residual_head(q)
        base_yaw = self.yaw_head(q)[..., 0]
        base_exist = self.existence_head(q)[..., 0]
        g = gate[:, None, None].to(q.dtype)
        gy = gate[:, None].to(q.dtype)

        # Additional capacity is also gate-isolated.  Current-source outputs are
        # exactly the inherited V18 heads regardless of learned adapter weights.
        xy = base_xy + g * self.memory_xy_delta(q)
        yaw = base_yaw + gy * self.memory_yaw_delta(q)[..., 0]
        status_rep = status[:, None, :].expand(
            B, FUTURE_FRAMES, self.status_dim
        ).to(q.dtype)
        survival = self.survival_head(
            torch.cat((q, status_rep), dim=-1)
        )[..., 0]

        return {
            "residual_xy_m": xy,
            "existence_logits": base_exist,
            "yaw_delta_rad": yaw,
            "survival_logits": survival,
            "memory_gate": gate,
        }


def load_clean_e14_into_memory_model(
    model: MemoryAdaptedV18SE2,
    clean_state_dict: dict[str, torch.Tensor],
) -> tuple[list[str], list[str]]:
    """Load a frozen V18 state dict while requiring only V19 keys to be missing."""
    result = model.load_state_dict(clean_state_dict, strict=False)
    expected_missing = {
        name
        for name in model.state_dict()
        if name.startswith(
            (
                "status_proj.",
                "memory_xy_delta.",
                "memory_yaw_delta.",
                "survival_head.",
            )
        )
    }
    got_missing = set(result.missing_keys)
    if got_missing != expected_missing:
        raise RuntimeError(
            "Clean-E14 load missing-key contract changed: "
            f"expected={sorted(expected_missing)}, got={sorted(got_missing)}"
        )
    if result.unexpected_keys:
        raise RuntimeError(
            f"unexpected Clean-E14 keys: {result.unexpected_keys}"
        )
    return list(result.missing_keys), list(result.unexpected_keys)
