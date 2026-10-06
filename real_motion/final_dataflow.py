"""Final causal history -> six-frame dense forecast dataflow.

Implementation counterpart of
docs/FINAL_METHOD_ARCHITECTURE_AND_FPS_PROTOCOL_20261006_CN.md.

The split is deliberate:
  prepare_history(): history-only deterministic representation, outside FPS.
  forecast_six(): every forecast-dependent operation, inside FPS.

No future GT, learned activation, Strong/KTA future prior, ownership/fallback,
or horizon repair score may be stored in CausalHistoryState.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import time
from typing import Callable

import numpy as np
import torch

from real_motion.canonical_causal_repair import (
    CanonicalEvidence,
    CanonicalRepairHead,
    build_canonical_evidence,
    compose_canonical,
    map_canonical_evidence,
)
from real_motion.motion_transport import world_points_to_t0, world_vec_to_t0
from real_motion.rigid_transport import rigid_source_points_world
from real_motion.runtime_fastpath import baseline_clear_flat_indices, extract_instances_cropped_exact
from real_motion.v18_execution_trial import reuse_v18_projections
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
from tools.real_motion.causal_column_common import (
    PreparedColumns,
    FrozenColumns,
    causal_source_history,
    render_column_layers,
)
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record


HISTORY_INPUT_KEYS = (
    "features",
    "local_semantic_tube",
    "frame_motion_features",
    "target_source_mask_tube",
)
IDENTITY_KEYS = (
    "sample_id",
    "scene_name",
    "t0_token",
    "source_class_id",
    "source_centroid_xy_t0_m",
)
FUTURE_FRAMES = 6


@dataclass
class CausalHistoryState:
    """Immutable model-ready history representation.

    All fields are history-only except future_poses, which is an explicit
    externally supplied future-ego condition. It is not a learned prediction
    and never carries future occupancy labels.

    gpu_history contains only history representation tensors. KTA is built live
    in forecast_six().
    """
    record: dict
    window: object
    raw_history: dict
    current_sem: np.ndarray
    previous_sem: np.ndarray
    current_pose: np.ndarray
    previous_pose: np.ndarray
    future_poses: tuple
    current: list
    previous: list
    velocities: dict
    source_world_points: list
    source_rel_xy: list
    source_z_t0: np.ndarray
    registrations: list
    source_audit: dict
    canonical_evidence: CanonicalEvidence
    gpu_history: dict
    preparation_seconds: dict

    @property
    def source_count(self):
        return len(self.current)


def _slim_history_record(record):
    """Keep only identity + deterministic history representation fields."""
    out = {}
    for key in (*IDENTITY_KEYS, *HISTORY_INPUT_KEYS):
        if key in record:
            out[key] = record[key]
    missing = [k for k in (
        "features", "local_semantic_tube", "frame_motion_features",
        "target_source_mask_tube", "source_class_id", "source_centroid_xy_t0_m"
    ) if k not in out]
    if missing:
        raise KeyError("history record missing: " + ",".join(missing))
    return out


def _stage_history_inputs(record, device):
    """Stage history representation only; no KTA/future prediction."""
    return {
        "features": torch.as_tensor(record["features"], device=device).float(),
        "tube": torch.as_tensor(record["local_semantic_tube"], device=device),
        "frame_motion": torch.as_tensor(record["frame_motion_features"], device=device).float(),
        "source_mask": torch.as_tensor(record["target_source_mask_tube"], device=device),
    }


def _history_source_geometry(raw, record, pcfg, strong, workers):
    """History-only source extraction, matching, registration and rigid geometry."""
    hist = np.asarray(raw["history_occ"], dtype=np.uint8)
    poses = np.asarray(raw["history_poses"], dtype=np.float64)
    if hist.shape[0] < 2 or poses.shape[0] < 2:
        raise ValueError("at least two causal history observations are required")
    current_sem, previous_sem = hist[-1], hist[-2]
    current_pose, previous_pose = poses[-1], poses[-2]
    current = extract_instances_cropped_exact(current_sem, current_pose, grid=pcfg.grid, cfg=strong)
    previous = extract_instances_cropped_exact(previous_sem, previous_pose, grid=pcfg.grid, cfg=strong)
    velocities = runtime.match_instances(
        previous, current, float(pcfg.frame_dt_s), max_speed_mps=strong.max_match_speed_mps
    )
    if len(current) != int(record["features"].shape[0]):
        raise RuntimeError(f"{record.get('sample_id','?')}: Strong/source count mismatch")
    got = [int(c["class_id"]) for c in current]
    exp = [int(x) for x in torch.as_tensor(record["source_class_id"]).tolist()]
    if got != exp:
        raise RuntimeError(f"{record.get('sample_id','?')}: Strong/source order mismatch")

    source_world_points = [
        rigid_source_points_world(c["voxel_indices"], current_pose, grid=pcfg.grid)
        for c in current
    ]
    source_rel_xy = [
        np.asarray(points, dtype=np.float64)[:, :2]
        - np.asarray(comp["centroid_world"], dtype=np.float64)[None, :2]
        for points, comp in zip(source_world_points, current)
    ]
    source_z_t0 = np.asarray([
        world_points_to_t0(
            np.asarray(comp["centroid_world"], dtype=np.float64)[None], current_pose
        )[0, 2]
        for comp in current
    ], dtype=np.float64)
    state = dict(
        current=current,
        previous=previous,
        velocities=velocities,
        source_world_points=source_world_points,
        current_pose=current_pose,
    )
    registrations, _, _, audit = causal_source_history(
        hist, poses, state, pcfg.grid, strong, workers, previous_instances=previous
    )
    return (
        current_sem, previous_sem, current_pose, previous_pose, current, previous,
        velocities, source_world_points, source_rel_xy, source_z_t0, registrations, audit,
    )


def _history_prepared_view(window, raw, current_pose, current, registrations, audit):
    """Minimal PreparedColumns view used only to build history-only CCR evidence."""
    state = {"current_pose": current_pose, "current": current}
    return PreparedColumns(
        window=window,
        raw=raw,
        state=state,
        baseline=[],
        owners=[],
        fallbacks=[],
        components=[],
        targets=[],
        yaws=[],
        registrations=registrations,
        footprints=None,
        memory=None,
        source_audit=audit,
        outputs=None,
    )


@torch.no_grad()
def prepare_history(provider, source, record, *, kernels=None, executor=None):
    """Build the one reusable causal history representation.

    This bypasses provider overrides that also build forecast-dependent Strong
    or future static-memory state. Only the base raw-history loader is used.
    """
    started = time.perf_counter()
    raw = FrozenColumns.load_raw_columns(provider, source, record, include_gt=False)
    if raw.get("future_gt_occ") is not None:
        raise RuntimeError("future GT must not enter CausalHistoryState")
    raw_at = time.perf_counter()
    window = window_from_record(record)
    (
        current_sem, previous_sem, current_pose, previous_pose, current, previous,
        velocities, source_world_points, source_rel_xy, source_z_t0, registrations, audit,
    ) = _history_source_geometry(raw, record, provider.pcfg, provider.strong, provider.workers)
    geometry_at = time.perf_counter()

    prep = _history_prepared_view(window, raw, current_pose, current, registrations, audit)
    evidence = build_canonical_evidence(prep, provider.pcfg.grid, kernels=kernels, executor=executor)
    evidence_at = time.perf_counter()
    gpu_history = _stage_history_inputs(record, provider.device)
    if provider.device.type == "cuda":
        torch.cuda.synchronize(provider.device)
    staged_at = time.perf_counter()

    centers = world_points_to_t0(
        np.asarray([c["centroid_world"] for c in current], dtype=np.float64).reshape(-1, 3),
        current_pose,
    )[:, :2]
    expected_centers = torch.as_tensor(record["source_centroid_xy_t0_m"]).cpu().numpy()
    if not np.allclose(centers, expected_centers, rtol=0, atol=2e-4):
        raise RuntimeError("history source-centre identity mismatch")

    slim = _slim_history_record(record)
    future_poses = tuple(np.asarray(p, dtype=np.float64) for p in raw["future_poses"])
    if len(future_poses) != FUTURE_FRAMES:
        raise RuntimeError("six future ego poses required")

    return CausalHistoryState(
        record=slim,
        window=window,
        raw_history=raw,
        current_sem=current_sem,
        previous_sem=previous_sem,
        current_pose=current_pose,
        previous_pose=previous_pose,
        future_poses=future_poses,
        current=current,
        previous=previous,
        velocities=velocities,
        source_world_points=source_world_points,
        source_rel_xy=source_rel_xy,
        source_z_t0=source_z_t0,
        registrations=registrations,
        source_audit=audit,
        canonical_evidence=evidence,
        gpu_history=gpu_history,
        preparation_seconds={
            "raw_io": raw_at-started,
            "source_extract_match_register": geometry_at-raw_at,
            "canonical_history_evidence": evidence_at-geometry_at,
            "history_h2d": staged_at-evidence_at,
            "total": staged_at-started,
        },
    )


def build_causal_motion_prior(history, frame_dt_s):
    """Recompute exact six-horizon KTA displacement/anchor from causal velocity."""
    n = history.source_count
    source_xy = world_points_to_t0(
        np.asarray([c["centroid_world"] for c in history.current], dtype=np.float64).reshape(-1, 3),
        history.current_pose,
    )[:, :2]
    kta = np.zeros((n, FUTURE_FRAMES, 2), dtype=np.float32)
    anchors = np.zeros_like(kta)
    for i in range(n):
        velocity_world = np.asarray(history.velocities.get(i, np.zeros(3)), dtype=np.float64)
        velocity_t0 = world_vec_to_t0(velocity_world, history.current_pose)[:2]
        for h in range(FUTURE_FRAMES):
            displacement = velocity_t0 * ((h+1)*float(frame_dt_s))
            kta[i, h] = displacement.astype(np.float32)
            anchors[i, h] = (source_xy[i] + displacement).astype(np.float32)
    return kta, anchors


def _runtime_record(history, kta, anchors):
    rec = dict(history.record)
    rec["kta_displacement_xy_m"] = torch.from_numpy(np.ascontiguousarray(kta))
    rec["anchors_xy_t0_m"] = torch.from_numpy(np.ascontiguousarray(anchors))
    return rec


def _gpu_inputs(history, kta, device):
    result = dict(history.gpu_history)
    result["kta"] = torch.as_tensor(kta, device=device).float()
    return result


def _forecast_prepared(history, state, record, output, grid):
    baseline, owners, fallbacks, components, targets, yaws = render_column_layers(
        state, record, output, grid
    )
    return PreparedColumns(
        window=history.window,
        raw=history.raw_history,
        state=state,
        baseline=baseline,
        owners=owners,
        fallbacks=fallbacks,
        components=components,
        targets=targets,
        yaws=yaws,
        registrations=history.registrations,
        footprints=None,
        memory=None,
        source_audit=history.source_audit,
        outputs=output,
    )


@torch.no_grad()
def forecast_six(
    history,
    provider,
    model,
    head,
    probability_fn: Callable,
    *,
    kernels=None,
    executor=None,
    majority_backend="native",
    reuse_motion_projections=True,
):
    """Official Dense Forecast computation from history state to dense x6.

    Callers own outer CUDA synchronize + wall-clock timing. This function
    contains every forecast-dependent operation required by the formal FPS
    protocol.
    """
    if history.raw_history.get("future_gt_occ") is not None:
        raise RuntimeError("Dense Forecast FPS cannot see future GT")
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("frozen read-only motion model required")
    if head.training or any(p.requires_grad for p in head.parameters()):
        raise RuntimeError("frozen read-only CCR head required")
    if majority_backend not in ("native", "dense_cuda", "sparse_cuda"):
        raise ValueError("unsupported Strong majority backend")

    stages = {}
    def call(name, fn):
        tick = time.perf_counter()
        value = fn()
        stages[name] = time.perf_counter()-tick
        return value

    kta, anchors_xy = call(
        "causal_motion_prior",
        lambda: build_causal_motion_prior(history, provider.pcfg.frame_dt_s),
    )

    prior_profile = {}
    anchors, baseline_by_hi = call(
        "strong_prior",
        lambda: runtime._strong_all_horizons(
            history.current_sem,
            history.current_pose,
            history.future_poses,
            history.current,
            history.velocities,
            history.source_world_points,
            frame_dt_s=provider.pcfg.frame_dt_s,
            grid=provider.pcfg.grid,
            cfg=provider.strong,
            runtime_device=provider.device,
            majority_backend=majority_backend,
            profile=prior_profile,
        ),
    )
    clear = call(
        "strong_clear_index",
        lambda: [baseline_clear_flat_indices(rows, grid=provider.pcfg.grid)
                 for rows in baseline_by_hi],
    )
    state = dict(
        rec=None,
        window=history.window,
        current_sem=history.current_sem,
        previous_sem=history.previous_sem,
        current_pose=history.current_pose,
        previous_pose=history.previous_pose,
        future_poses=list(history.future_poses),
        current=history.current,
        previous=history.previous,
        velocities=history.velocities,
        source_world_points=history.source_world_points,
        source_rel_xy=history.source_rel_xy,
        source_z_t0=history.source_z_t0,
        anchors=anchors,
        baseline_by_hi=baseline_by_hi,
        baseline_clear_flat_by_hi=clear,
        world_to_future=[np.linalg.inv(p) for p in history.future_poses],
        gpu=None,
    )
    record = _runtime_record(history, kta, anchors_xy)
    state["rec"] = record
    gi = call("stage_live_kta", lambda: _gpu_inputs(history, kta, provider.device))

    cm = reuse_v18_projections(model) if reuse_motion_projections else nullcontext()
    with cm:
        output = call(
            "motion_forward",
            lambda: runtime._model_forward(model, gi, provider.device, return_latents=True),
        )

    prep = call(
        "source_transport_and_layers",
        lambda: _forecast_prepared(history, state, record, output, provider.pcfg.grid),
    )
    evidence = history.canonical_evidence
    plan = call(
        "six_projection_ownership_legality",
        lambda: map_canonical_evidence(
            evidence, prep, provider.pcfg.grid, kernels=kernels, executor=executor
        ),
    )
    probability = call(
        "shared_encode_six_readouts",
        lambda: probability_fn(head, evidence, plan, output, provider.device),
    )
    dense = call(
        "constrained_dense_composition",
        lambda: compose_canonical(
            prep.baseline,
            evidence,
            plan,
            probability[..., 0],
            probability[..., 1],
            thresholds=(.5, .95),
        ),
    )
    if (
        len(dense) != FUTURE_FRAMES
        or any(x.shape != tuple(provider.pcfg.grid.shape_hwd) for x in dense)
        or not np.isfinite(probability).all()
    ):
        raise RuntimeError("six complete dense outputs required")
    return {
        "dense": dense,
        "probability": probability,
        "motion": output,
        "stages_seconds": stages,
        "strong_profile_ms": prior_profile,
        "six_complete_dense": True,
    }


def batch_frozen_motion(teacher, rows, device):
    """One frozen V18 forward for a multi-window batch, split by source count."""
    if not rows:
        raise ValueError("non-empty training rows required")
    keys = (
        "features",
        "local_semantic_tube",
        "kta_displacement_xy_m",
        "frame_motion_features",
        "target_source_mask_tube",
    )
    sizes = [int(r["features"].shape[0]) for r, _ in rows]
    merged = {key: torch.cat([torch.as_tensor(r[key]) for r, _ in rows], dim=0) for key in keys}
    output = teacher.motion(merged, device)
    split = []
    for values in zip(*(v.split(sizes) for v in output.values())):
        split.append({k: v for k, v in zip(output, values)})
    if len(split) != len(rows):
        raise RuntimeError("batched frozen motion split mismatch")
    return split
