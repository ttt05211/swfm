"""Lightweight causal Motion Support Proposal (MSP) feasibility utilities.

This module is deliberately isolated from the formal SWFM world-model training
path.  It answers a narrower pre-training question: can a tiny object/component
proposal head learn *where* future motion support should be placed under a
fixed sparse budget?

Causality contract
-----------------
Model inputs come only from the current occupancy decomposition and historical
component motion estimated from ego-aligned occupancy.  nuScenes instance
annotations are used only to build supervision/diagnostics and are never copied
into the model feature vector.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import OccupancyGrid, relative_transform
from .kta import KTAConfig, MotionComponent, estimate_components
from .metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS, SPEED_THRESHOLD_MPS
from .nuscenes_adapter import category_to_dynamic_class

MSP_CACHE_VERSION = "real_motion_msp_probe_v1"
STATE_OBSERVED_MOVING = 0
STATE_DORMANT = 1
STATE_NAMES = {
    STATE_OBSERVED_MOVING: "observed_moving",
    STATE_DORMANT: "dormant",
}
SOURCE_A = "A_observed_moving"
SOURCE_B = "B_dormant"
SOURCE_C = "C_no_causal_source"
CLASS_TO_SLOT = {int(c): i for i, c in enumerate(DYNAMIC_CLASS_IDS)}
FEATURE_NAMES = (
    "x_norm", "y_norm", "vx_norm", "vy_norm", "speed_norm",
    "log_voxel_count", "extent_x_norm", "extent_y_norm", "kta_matched",
    "state_observed_moving", "state_dormant",
    *tuple(f"class_{int(c)}" for c in DYNAMIC_CLASS_IDS),
)
FEATURE_DIM = len(FEATURE_NAMES)


@dataclass(frozen=True)
class MSPCandidate:
    class_id: int
    state: int
    centroid_xy_m: np.ndarray
    velocity_xy_mps: np.ndarray
    extent_xy_m: np.ndarray
    voxel_count: int
    kta_matched: bool

    def __post_init__(self):
        if int(self.class_id) not in CLASS_TO_SLOT:
            raise ValueError(f"MSP candidate class {self.class_id} is not motion-eligible")
        if int(self.state) not in STATE_NAMES:
            raise ValueError(f"invalid MSP state {self.state}")
        if np.asarray(self.centroid_xy_m).shape != (2,):
            raise ValueError("centroid_xy_m must be [2]")
        if np.asarray(self.velocity_xy_mps).shape != (2,):
            raise ValueError("velocity_xy_mps must be [2]")
        if np.asarray(self.extent_xy_m).shape != (2,):
            raise ValueError("extent_xy_m must be [2]")
        if int(self.voxel_count) <= 0:
            raise ValueError("voxel_count must be positive")


def _component_extent_xy(comp: MotionComponent, grid: OccupancyGrid) -> np.ndarray:
    cells = np.asarray(comp.bev_cells, dtype=np.int64)
    if cells.ndim != 2 or cells.shape[1] != 2 or len(cells) == 0:
        raise ValueError("MotionComponent.bev_cells must be non-empty [N,2]")
    vx, vy, _ = grid.voxel_size
    sx = (int(cells[:, 0].max()) - int(cells[:, 0].min()) + 1) * float(vx)
    sy = (int(cells[:, 1].max()) - int(cells[:, 1].min()) + 1) * float(vy)
    return np.asarray([sx, sy], dtype=np.float32)


def extract_msp_candidates(
    aligned_history: np.ndarray,
    moving_mask: np.ndarray,
    dormant_mask: np.ndarray,
    grid: OccupancyGrid = OccupancyGrid(),
    kta_cfg: KTAConfig = KTAConfig(),
) -> list[MSPCandidate]:
    """Extract causal MOVING/DORMANT occupancy components for the MSP probe.

    `moving_mask` and `dormant_mask` must be the mutually-exclusive masks from
    the current real-motion decomposition.  The only history-derived motion
    feature is the same last-step constant-velocity estimate already available
    to KTA; no future annotation enters this function.
    """
    hist = np.asarray(aligned_history)
    moving = np.asarray(moving_mask, dtype=bool)
    dormant = np.asarray(dormant_mask, dtype=bool)
    if hist.ndim != 4:
        raise ValueError("aligned_history must be [T,X,Y,Z]")
    if moving.shape != hist.shape[1:] or dormant.shape != hist.shape[1:]:
        raise ValueError("moving/dormant masks must align with occupancy grid")
    if np.any(moving & dormant):
        raise ValueError("moving and dormant masks must be disjoint")

    out: list[MSPCandidate] = []
    for state, mask in (
        (STATE_OBSERVED_MOVING, moving),
        (STATE_DORMANT, dormant),
    ):
        comps = estimate_components(hist, mask, grid=grid, cfg=kta_cfg)
        for comp in comps:
            if int(comp.class_id) not in CLASS_TO_SLOT:
                continue
            out.append(
                MSPCandidate(
                    class_id=int(comp.class_id),
                    state=int(state),
                    centroid_xy_m=np.asarray(comp.centroid_xy_m, dtype=np.float32),
                    velocity_xy_mps=np.asarray(comp.velocity_xy_mps, dtype=np.float32),
                    extent_xy_m=_component_extent_xy(comp, grid),
                    voxel_count=int(len(comp.voxel_indices)),
                    kta_matched=bool(comp.matched),
                )
            )
    out.sort(
        key=lambda c: (
            int(c.state), int(c.class_id),
            float(c.centroid_xy_m[0]), float(c.centroid_xy_m[1]),
        )
    )
    return out


def candidate_feature(candidate: MSPCandidate) -> np.ndarray:
    """Return the frozen, annotation-free feature vector used by the probe."""
    c = candidate
    x, y = [float(v) for v in c.centroid_xy_m]
    vx, vy = [float(v) for v in c.velocity_xy_mps]
    speed = math.hypot(vx, vy)
    ex, ey = [float(v) for v in c.extent_xy_m]
    feat = [
        x / 40.0,
        y / 40.0,
        vx / 20.0,
        vy / 20.0,
        speed / 20.0,
        math.log1p(float(c.voxel_count)) / 8.0,
        ex / 10.0,
        ey / 10.0,
        1.0 if c.kta_matched else 0.0,
        1.0 if c.state == STATE_OBSERVED_MOVING else 0.0,
        1.0 if c.state == STATE_DORMANT else 0.0,
    ]
    one_hot = [0.0] * len(DYNAMIC_CLASS_IDS)
    one_hot[CLASS_TO_SLOT[int(c.class_id)]] = 1.0
    feat.extend(one_hot)
    arr = np.asarray(feat, dtype=np.float32)
    if arr.shape != (FEATURE_DIM,):
        raise AssertionError(f"unexpected MSP feature shape {arr.shape}")
    return arr


def candidates_to_feature_tensor(candidates: Sequence[MSPCandidate]) -> torch.Tensor:
    if not candidates:
        return torch.zeros((0, FEATURE_DIM), dtype=torch.float32)
    return torch.from_numpy(np.stack([candidate_feature(c) for c in candidates], axis=0))


def dynamic_t0_instances(nusc, t0_token: str, t0_ego_to_world: np.ndarray) -> list[dict]:
    """Training-label helper: dynamic GT instances at t0 in the t0 ego frame."""
    sample = nusc.get("sample", t0_token)
    world_to_t0 = np.linalg.inv(np.asarray(t0_ego_to_world, dtype=np.float64))
    rows = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        cid = category_to_dynamic_class(ann["category_name"])
        if cid is None:
            continue
        center_world = np.asarray(ann["translation"], dtype=np.float64)
        center_t0 = (world_to_t0 @ np.r_[center_world, 1.0])[:2]
        rows.append({
            "instance_token": str(ann["instance_token"]),
            "class_id": int(cid),
            "center_xy_t0_m": center_t0.astype(np.float32),
            "center_world": center_world.astype(np.float64),
        })
    rows.sort(key=lambda r: (r["class_id"], r["instance_token"]))
    return rows


def match_candidates_to_instances(
    candidates: Sequence[MSPCandidate],
    instances: Sequence[Mapping],
    max_distance_m: float = 4.0,
) -> tuple[list[str | None], dict[str, int]]:
    """Greedy one-to-one same-class matching used only to attach GT labels.

    Matching identity is *not* a model input.  The deterministic sort makes the
    result reproducible when multiple pairs have the same distance.
    """
    if max_distance_m <= 0:
        raise ValueError("max_distance_m must be positive")
    pairs = []
    for ci, cand in enumerate(candidates):
        cc = np.asarray(cand.centroid_xy_m, dtype=np.float64)
        for gi, inst in enumerate(instances):
            if int(inst["class_id"]) != int(cand.class_id):
                continue
            gc = np.asarray(inst["center_xy_t0_m"], dtype=np.float64)
            d = float(np.linalg.norm(cc - gc))
            if d <= float(max_distance_m):
                pairs.append((d, ci, gi, str(inst["instance_token"])))
    pairs.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    used_c, used_g = set(), set()
    candidate_tokens: list[str | None] = [None] * len(candidates)
    token_to_candidate: dict[str, int] = {}
    for _, ci, gi, token in pairs:
        if ci in used_c or gi in used_g:
            continue
        used_c.add(ci)
        used_g.add(gi)
        candidate_tokens[ci] = token
        token_to_candidate[token] = ci
    return candidate_tokens, token_to_candidate


def source_type_for_token(
    token: str,
    token_to_candidate: Mapping[str, int],
    candidates: Sequence[MSPCandidate],
) -> str:
    ci = token_to_candidate.get(str(token))
    if ci is None:
        return SOURCE_C
    state = int(candidates[int(ci)].state)
    if state == STATE_OBSERVED_MOVING:
        return SOURCE_A
    if state == STATE_DORMANT:
        return SOURCE_B
    raise ValueError(f"unexpected candidate state {state}")


def _sample_ann_map(nusc, sample_token: str) -> dict[str, Mapping]:
    sample = nusc.get("sample", sample_token)
    out = {}
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        out[str(ann["instance_token"])] = ann
    return out


def build_probe_record(
    source,
    window,
    pcfg,
    *,
    match_max_distance_m: float = 4.0,
) -> dict:
    """Build one GT-supervised MSP probe record from causal occupancy features.

    Future annotations populate only `activation` and `target_xy_t0_m`.  The
    feature tensor and KTA/zero anchors are fully causal.
    """
    from .geometry import ego_compensate_sequence
    from .motion import decompose_masks
    from .prepared import _align_observation_sequence, load_nuscenes_window_raw

    raw = load_nuscenes_window_raw(source, window, pcfg, include_gt=True)
    aligned_hist = ego_compensate_sequence(
        raw["history_occ"], raw["history_poses"], -1, pcfg.grid, pcfg.free_label
    )
    aligned_obs = _align_observation_sequence(
        raw["history_observed"], raw["history_poses"], -1, pcfg.grid
    )
    t0_masks = decompose_masks(aligned_hist, pcfg.motion, history_observed=aligned_obs)
    candidates = extract_msp_candidates(
        aligned_hist,
        t0_masks.moving,
        t0_masks.uncertain,
        grid=pcfg.grid,
        kta_cfg=pcfg.kta,
    )

    t0_pose = np.asarray(raw["history_poses"][-1], dtype=np.float64)
    instances = dynamic_t0_instances(source.nusc, window.t0_token, t0_pose)
    candidate_tokens, token_to_candidate = match_candidates_to_instances(
        candidates, instances, max_distance_m=match_max_distance_m
    )
    inst_by_token = {str(r["instance_token"]): r for r in instances}

    horizons = np.asarray(
        [(i + 1) * float(pcfg.frame_dt_s) for i in range(pcfg.future_frames)],
        dtype=np.float32,
    )
    N, H = len(candidates), len(horizons)
    features = candidates_to_feature_tensor(candidates)
    anchors = torch.zeros((N, H, 2), dtype=torch.float32)
    activation = torch.zeros((N, H), dtype=torch.float32)
    activation_valid = torch.zeros((N, H), dtype=torch.bool)
    target_xy = torch.zeros((N, H, 2), dtype=torch.float32)
    target_valid = torch.zeros((N, H), dtype=torch.bool)

    world_to_t0 = np.linalg.inv(t0_pose)
    future_maps = [_sample_ann_map(source.nusc, tok) for tok in window.future_tokens]
    for ci, cand in enumerate(candidates):
        center = torch.as_tensor(cand.centroid_xy_m, dtype=torch.float32)
        vel = torch.as_tensor(cand.velocity_xy_mps, dtype=torch.float32)
        for hi, h in enumerate(horizons.tolist()):
            anchors[ci, hi] = center + (vel * float(h) if cand.state == STATE_OBSERVED_MOVING else 0.0)

        token = candidate_tokens[ci]
        if token is None:
            continue
        ann0 = inst_by_token[token]
        c0w = np.asarray(ann0["center_world"], dtype=np.float64)
        for hi, (h, fmap) in enumerate(zip(horizons.tolist(), future_maps)):
            activation_valid[ci, hi] = True
            annh = fmap.get(token)
            if annh is None:
                activation[ci, hi] = 0.0
                continue
            chw = np.asarray(annh["translation"], dtype=np.float64)
            speed = float(np.linalg.norm(chw[:2] - c0w[:2]) / float(h))
            is_active = speed >= float(SPEED_THRESHOLD_MPS)
            activation[ci, hi] = 1.0 if is_active else 0.0
            ch_t0 = (world_to_t0 @ np.r_[chw, 1.0])[:2]
            target_xy[ci, hi] = torch.as_tensor(ch_t0, dtype=torch.float32)
            target_valid[ci, hi] = bool(is_active)

    rel = np.stack([
        relative_transform(t0_pose, np.asarray(p, dtype=np.float64))
        for p in raw["future_poses"]
    ], axis=0).astype(np.float32)

    return {
        "sample_id": f"{window.scene_name}:{window.t0_token}",
        "scene_name": str(window.scene_name),
        "history_tokens": tuple(window.history_tokens),
        "t0_token": str(window.t0_token),
        "future_tokens": tuple(window.future_tokens),
        "features": features,
        "anchors_xy_t0_m": anchors,
        "activation": activation,
        "activation_valid": activation_valid,
        "target_xy_t0_m": target_xy,
        "target_valid": target_valid,
        "future_rel_t0_to_ego": torch.from_numpy(rel),
        "candidate_class_id": torch.tensor([c.class_id for c in candidates], dtype=torch.long),
        "candidate_state": torch.tensor([c.state for c in candidates], dtype=torch.long),
        "candidate_extent_xy_m": torch.from_numpy(
            np.stack([c.extent_xy_m for c in candidates], axis=0).astype(np.float32)
        ) if candidates else torch.zeros((0, 2), dtype=torch.float32),
        "candidate_centroid_xy_m": torch.from_numpy(
            np.stack([c.centroid_xy_m for c in candidates], axis=0).astype(np.float32)
        ) if candidates else torch.zeros((0, 2), dtype=torch.float32),
        "candidate_velocity_xy_mps": torch.from_numpy(
            np.stack([c.velocity_xy_mps for c in candidates], axis=0).astype(np.float32)
        ) if candidates else torch.zeros((0, 2), dtype=torch.float32),
        "candidate_instance_token": tuple(candidate_tokens),
        "num_candidates": int(N),
        "num_matched_candidates": int(sum(t is not None for t in candidate_tokens)),
    }


def validate_probe_record(record: Mapping, future_frames: int = 6) -> None:
    required = (
        "features", "anchors_xy_t0_m", "activation", "activation_valid",
        "target_xy_t0_m", "target_valid", "future_rel_t0_to_ego",
        "candidate_state", "candidate_extent_xy_m",
    )
    missing = [k for k in required if k not in record]
    if missing:
        raise KeyError(f"MSP record missing keys {missing}")
    f = record["features"]
    if not torch.is_tensor(f) or f.ndim != 2 or f.shape[1] != FEATURE_DIM:
        raise ValueError(f"features must be [N,{FEATURE_DIM}]")
    n = int(f.shape[0])
    if tuple(record["anchors_xy_t0_m"].shape) != (n, future_frames, 2):
        raise ValueError("anchors shape mismatch")
    if tuple(record["activation"].shape) != (n, future_frames):
        raise ValueError("activation shape mismatch")
    if tuple(record["target_xy_t0_m"].shape) != (n, future_frames, 2):
        raise ValueError("target shape mismatch")
    if tuple(record["future_rel_t0_to_ego"].shape) != (future_frames, 4, 4):
        raise ValueError("future relative transform shape mismatch")
    if tuple(record["candidate_state"].shape) != (n,):
        raise ValueError("candidate_state shape mismatch")


def collate_probe_records(records: Sequence[Mapping]) -> dict:
    if not records:
        raise ValueError("cannot collate an empty record list")
    for r in records:
        validate_probe_record(r)
    B = len(records)
    H = int(records[0]["activation"].shape[1])
    max_n = max(1, max(int(r["features"].shape[0]) for r in records))

    def zeros(*shape, dtype=torch.float32):
        return torch.zeros(shape, dtype=dtype)

    out = {
        "features": zeros(B, max_n, FEATURE_DIM),
        "anchors_xy_t0_m": zeros(B, max_n, H, 2),
        "activation": zeros(B, max_n, H),
        "activation_valid": zeros(B, max_n, H, dtype=torch.bool),
        "target_xy_t0_m": zeros(B, max_n, H, 2),
        "target_valid": zeros(B, max_n, H, dtype=torch.bool),
        "candidate_mask": zeros(B, max_n, dtype=torch.bool),
        "candidate_extent_xy_m": zeros(B, max_n, 2),
        "future_rel_t0_to_ego": torch.stack([r["future_rel_t0_to_ego"] for r in records]),
        "sample_id": [str(r.get("sample_id", "")) for r in records],
        "scene_name": [str(r.get("scene_name", "")) for r in records],
    }
    for bi, r in enumerate(records):
        n = int(r["features"].shape[0])
        if n == 0:
            continue
        out["candidate_mask"][bi, :n] = True
        for key in (
            "features", "anchors_xy_t0_m", "activation", "activation_valid",
            "target_xy_t0_m", "target_valid", "candidate_extent_xy_m",
        ):
            out[key][bi, :n] = r[key]
    return out


class MSPProbeHead(nn.Module):
    """One-layer object-attention MSP used only for the feasibility gate."""
    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        hidden_dim: int = 96,
        num_heads: int = 4,
        num_modes: int = 4,
        future_frames: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim <= 0 or hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if num_modes <= 0 or future_frames <= 0:
            raise ValueError("num_modes/future_frames must be positive")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_modes = int(num_modes)
        self.future_frames = int(future_frames)
        self.input_proj = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=self.hidden_dim * 2,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.object_encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.activation_head = nn.Linear(self.hidden_dim, self.future_frames)
        # Per mode: residual dx, residual dy, raw sigma, mode logit.
        self.proposal_head = nn.Linear(
            self.hidden_dim, self.future_frames * self.num_modes * 4
        )

    def forward(self, features: torch.Tensor, candidate_mask: torch.Tensor) -> dict:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError("features must be [B,N,D]")
        if candidate_mask.shape != features.shape[:2]:
            raise ValueError("candidate_mask shape mismatch")
        mask = candidate_mask.bool()
        safe_mask = mask.clone()
        empty = ~safe_mask.any(dim=1)
        if bool(empty.any()):
            safe_mask[empty, 0] = True
        x = self.input_proj(features)
        x = self.object_encoder(x, src_key_padding_mask=~safe_mask)
        x = x * mask.unsqueeze(-1).to(x.dtype)
        act = self.activation_head(x)
        raw = self.proposal_head(x).view(
            features.shape[0], features.shape[1],
            self.future_frames, self.num_modes, 4,
        )
        return {
            "activation_logits": act,
            "mu_residual_xy_m": raw[..., :2],
            "raw_sigma": raw[..., 2],
            "mode_logits": raw[..., 3],
        }


def msp_probe_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    positive_weight: float = 2.0,
    sigma_floor_m: float = 0.35,
    sigma_max_m: float = 8.0,
) -> tuple[torch.Tensor, dict]:
    """Activation BCE + multimodal Gaussian NLL for active future centers."""
    act_logits = outputs["activation_logits"]
    valid = batch["activation_valid"].bool() & batch["candidate_mask"].unsqueeze(-1)
    labels = batch["activation"].to(act_logits.dtype)
    if bool(valid.any()):
        pos_weight = torch.as_tensor(float(positive_weight), device=act_logits.device)
        bce_raw = F.binary_cross_entropy_with_logits(
            act_logits, labels, reduction="none", pos_weight=pos_weight
        )
        bce = bce_raw[valid].mean()
    else:
        bce = act_logits.sum() * 0.0

    mu = outputs["mu_residual_xy_m"]
    sigma = F.softplus(outputs["raw_sigma"]) + float(sigma_floor_m)
    sigma = sigma.clamp(max=float(sigma_max_m))
    mode_logp = F.log_softmax(outputs["mode_logits"], dim=-1)
    target_residual = (
        batch["target_xy_t0_m"] - batch["anchors_xy_t0_m"]
    ).unsqueeze(-2)
    diff = (target_residual - mu) / sigma.unsqueeze(-1)
    log_gauss = -0.5 * diff.square().sum(dim=-1)
    log_gauss = log_gauss - 2.0 * torch.log(sigma) - math.log(2.0 * math.pi)
    mix_logp = torch.logsumexp(mode_logp + log_gauss, dim=-1)
    loc_valid = batch["target_valid"].bool() & batch["candidate_mask"].unsqueeze(-1)
    if bool(loc_valid.any()):
        nll = -mix_logp[loc_valid].mean()
    else:
        nll = mix_logp.sum() * 0.0
    total = bce + nll
    return total, {
        "loss": float(total.detach().cpu()),
        "activation_bce": float(bce.detach().cpu()),
        "location_nll": float(nll.detach().cpu()),
        "num_activation_labels": int(valid.sum().item()),
        "num_location_labels": int(loc_valid.sum().item()),
    }


def _transform_xy_t0_to_future(
    xy: torch.Tensor,
    rel_t0_to_future: torch.Tensor,
) -> torch.Tensor:
    """Apply planar part of [4,4] t0->future ego rigid transforms."""
    if xy.shape[-1] != 2:
        raise ValueError("xy must end in dimension 2")
    r = rel_t0_to_future
    x = xy[..., 0]
    y = xy[..., 1]
    out_x = r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 3]
    out_y = r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 3]
    return torch.stack([out_x, out_y], dim=-1)


def rasterize_msp_scores(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    latent_hw: tuple[int, int] = (50, 50),
    grid: OccupancyGrid = OccupancyGrid(),
    sigma_floor_m: float = 0.35,
    sigma_max_m: float = 8.0,
) -> torch.Tensor:
    """Rasterize MSP mixture proposals to [B,H,LX,LY] ranking score maps.

    The head predicts positions in the t0 ego frame.  Mapping to each future ego
    frame uses only the experiment's already-allowed future ego transform; this
    transform is not an MSP network input.
    """
    act = torch.sigmoid(outputs["activation_logits"])
    mode_p = F.softmax(outputs["mode_logits"], dim=-1)
    sigma = (F.softplus(outputs["raw_sigma"]) + float(sigma_floor_m)).clamp(
        max=float(sigma_max_m)
    )
    centers_t0 = batch["anchors_xy_t0_m"].unsqueeze(-2) + outputs["mu_residual_xy_m"]
    B, N, H, K, _ = centers_t0.shape
    if tuple(batch["future_rel_t0_to_ego"].shape) != (B, H, 4, 4):
        raise ValueError("future_rel_t0_to_ego shape mismatch")
    if tuple(batch["candidate_mask"].shape) != (B, N):
        raise ValueError("candidate_mask shape mismatch")

    rel = batch["future_rel_t0_to_ego"][:, None, :, None, :, :].expand(B, N, H, K, 4, 4)
    centers = _transform_xy_t0_to_future(centers_t0, rel)
    LX, LY = map(int, latent_hw)
    if LX <= 0 or LY <= 0:
        raise ValueError("latent_hw must be positive")
    x_step = (grid.x_max - grid.x_min) / float(LX)
    y_step = (grid.y_max - grid.y_min) / float(LY)
    xs = torch.linspace(
        grid.x_min + 0.5 * x_step,
        grid.x_max - 0.5 * x_step,
        LX,
        device=centers.device,
        dtype=centers.dtype,
    )
    ys = torch.linspace(
        grid.y_min + 0.5 * y_step,
        grid.y_max - 0.5 * y_step,
        LY,
        device=centers.device,
        dtype=centers.dtype,
    )
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    scores = torch.zeros((B, H, LX, LY), device=centers.device, dtype=centers.dtype)
    candidate_mask = batch["candidate_mask"].bool()
    ext = batch["candidate_extent_xy_m"].to(centers.dtype)

    # Small B/H loops keep memory bounded while vectorizing all object modes.
    for bi in range(B):
        valid_idx = torch.nonzero(candidate_mask[bi], as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            continue
        for hi in range(H):
            c = centers[bi, valid_idx, hi].reshape(-1, 2)
            s = sigma[bi, valid_idx, hi].reshape(-1)
            # Current footprint is a causal lower bound on useful spatial width.
            footprint = 0.25 * ext[bi, valid_idx].sum(dim=-1)
            footprint = footprint[:, None].expand(-1, K).reshape(-1)
            s_eff = torch.sqrt(s.square() + footprint.square().clamp_min(0.05))
            w = (
                act[bi, valid_idx, hi, None] * mode_p[bi, valid_idx, hi]
            ).reshape(-1)
            dx = gx.unsqueeze(0) - c[:, 0, None, None]
            dy = gy.unsqueeze(0) - c[:, 1, None, None]
            g = torch.exp(-0.5 * (dx.square() + dy.square()) / s_eff[:, None, None].square())
            score = (w[:, None, None] * g).amax(dim=0)
            scores[bi, hi] = score
    return scores


def top_budget_support(score_maps: torch.Tensor, budget_ratio: float) -> torch.Tensor:
    """Select at most a fixed fraction of latent cells independently per horizon."""
    if score_maps.ndim != 4:
        raise ValueError("score_maps must be [B,H,X,Y]")
    ratio = float(budget_ratio)
    if not 0.0 < ratio <= 1.0:
        raise ValueError("budget_ratio must be in (0,1]")
    B, H, X, Y = score_maps.shape
    total = X * Y
    k = max(1, min(total, int(math.ceil(ratio * total))))
    flat = score_maps.reshape(B, H, total)
    out = torch.zeros_like(flat, dtype=torch.bool)
    for bi in range(B):
        for hi in range(H):
            row = flat[bi, hi]
            if not bool(torch.isfinite(row).all()):
                raise ValueError("non-finite MSP support score")
            if float(row.max().item()) <= 0.0:
                continue
            vals, idx = torch.topk(row, k=k, largest=True, sorted=False)
            keep = vals > 0
            out[bi, hi, idx[keep]] = True
    return out.reshape(B, H, X, Y)


def latent_support_to_bev(
    latent_support: torch.Tensor,
    bev_hw: tuple[int, int] = (200, 200),
) -> torch.Tensor:
    """Nearest block expansion for the aligned 50x50 -> 200x200 SWFM grid."""
    if latent_support.ndim < 2:
        raise ValueError("latent_support must include X,Y dimensions")
    X, Y = map(int, latent_support.shape[-2:])
    bx, by = map(int, bev_hw)
    if bx % X or by % Y:
        raise ValueError("BEV dimensions must be integer multiples of latent dimensions")
    return latent_support.repeat_interleave(bx // X, dim=-2).repeat_interleave(by // Y, dim=-1)
