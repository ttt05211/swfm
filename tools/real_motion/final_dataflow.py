"""Final causal-history -> dense-forecast dataflow.

This module is the executable counterpart of
docs/FINAL_METHOD_ARCHITECTURE_AND_FPS_PROTOCOL_20261006_CN.md.

The contract is deliberately narrow:

* prepare_history may use ONLY observed-history information plus immutable
  cached model inputs derived from that history. It performs source extraction,
  causal association/registration, canonical evidence construction and optional
  staging of history-only tensors.
* forecast_six receives future ego poses explicitly and performs EVERY
  forecast-dependent operation: fresh KTA, Strong prior, V18 motion, rigid
  transport/layering, future projection/ownership, Point CCR and dense
  composition.

No future occupancy/GT, learned activation or cached Strong result is permitted
inside CausalHistoryState.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
import hashlib
import math
import time

import numpy as np
import torch

from real_motion.canonical_causal_repair import (
    CanonicalEvidence,
    build_canonical_evidence,
    map_canonical_evidence,
    compose_canonical,
)
from real_motion.canonical_repair_context import fixed_history_digest
from real_motion.motion_transport import world_points_to_t0, world_vec_to_t0
from real_motion.rigid_transport import rigid_source_points_world
from real_motion.runtime_fastpath import (
    baseline_clear_flat_indices,
    compose_component_replacements_fast_exact,
)
from real_motion.v18_execution_trial import reuse_v18_projections
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
from tools.real_motion.causal_column_common import (
    FrozenColumns,
    PreparedColumns,
    causal_source_history,
    render_column_layers,
)
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record

FUTURE_FRAMES = 6


@dataclass(frozen=True)
class FutureEgoCondition:
    poses: tuple[np.ndarray, ...]

    def __post_init__(self):
        if len(self.poses) != FUTURE_FRAMES:
            raise ValueError("FutureEgoCondition requires exactly six poses")
        for pose in self.poses:
            if np.asarray(pose).shape != (4, 4):
                raise ValueError("future ego pose must be 4x4")


@dataclass
class CausalHistoryState:
    """History representation used by the final forecast path.

    raw_history contains only history occupancy/visibility/poses. Future ego
    poses live in FutureEgoCondition; future semantic GT is never stored.
    gpu_history contains only history-derived model inputs. KTA is excluded
    because the official Dense Forecast FPS rebuilds it inside the timer.
    """

    sample_id: str
    scene_name: str
    window: object
    raw_history: dict
    current_sem: np.ndarray
    current_pose: np.ndarray
    previous_sem: np.ndarray
    previous_pose: np.ndarray
    current: list
    previous: list
    velocities: dict
    source_world_points: list
    source_rel_xy: list
    source_z_t0: np.ndarray
    source_centers_xy_t0: np.ndarray
    registrations: list
    source_audit: dict
    canonical_evidence: CanonicalEvidence
    gpu_history: dict
    record_static: dict
    content_sha256: str
    prepare_seconds: dict


def _history_only_record(record):
    """Keep only inference inputs that are functions of observed history."""
    keep = (
        "sample_id", "scene_name", "history_tokens", "t0_token", "future_tokens",
        "features", "local_semantic_tube", "frame_motion_features",
        "target_source_mask_tube", "source_class_id", "source_voxel_count",
    )
    out = {k: record[k] for k in keep if k in record}
    required = ("features", "local_semantic_tube", "frame_motion_features",
                "target_source_mask_tube", "source_class_id")
    missing = [k for k in required if k not in out]
    if missing:
        raise KeyError("missing history/model input fields: " + ",".join(missing))
    return out


def _stage_history_inputs(record, device):
    """Match frozen V18 input dtypes without staging future KTA."""
    return {
        "features": record["features"].float().to(device),
        "tube": record["local_semantic_tube"].to(device),
        "frame_motion": record["frame_motion_features"].float().to(device),
        "source_mask": record["target_source_mask_tube"].to(device),
    }


def _source_core(raw_history, record, provider):
    hist = np.asarray(raw_history["history_occ"], dtype=np.uint8)
    poses = np.asarray(raw_history["history_poses"], dtype=np.float64)
    if hist.shape[0] != 4 or poses.shape[0] != 4:
        raise ValueError("final Point CCR history contract requires exactly four frames")
    current_sem, previous_sem = hist[-1], hist[-2]
    current_pose, previous_pose = poses[-1], poses[-2]
    current = runtime.extract_instances_cropped_exact(
        current_sem, current_pose, grid=provider.pcfg.grid, cfg=provider.strong)
    previous = runtime.extract_instances_cropped_exact(
        previous_sem, previous_pose, grid=provider.pcfg.grid, cfg=provider.strong)
    velocities = runtime.match_instances(
        previous, current, float(provider.pcfg.frame_dt_s),
        max_speed_mps=provider.strong.max_match_speed_mps)
    if len(current) != int(record["features"].shape[0]):
        raise RuntimeError(f"{record['sample_id']}: source count mismatch")
    classes = [int(x["class_id"]) for x in current]
    expected = [int(x) for x in record["source_class_id"].tolist()]
    if classes != expected:
        raise RuntimeError(f"{record['sample_id']}: source order/class mismatch")
    source_world_points = [
        rigid_source_points_world(c["voxel_indices"], current_pose, grid=provider.pcfg.grid)
        for c in current
    ]
    source_rel_xy = [
        np.asarray(points, np.float64)[:, :2]
        - np.asarray(comp["centroid_world"], np.float64)[None, :2]
        for points, comp in zip(source_world_points, current)
    ]
    # Preserve the frozen runtime's per-source homogeneous transform order.
    # Vectorizing this matmul can change the last floating bit on some BLASes,
    # which is unnecessary risk for the byte-exact parity gate.
    centers_t0_full = [
        world_points_to_t0(
            np.asarray(comp["centroid_world"], dtype=np.float64)[None],
            current_pose,
        )[0]
        for comp in current
    ]
    centers_t0 = (
        np.asarray([p[:2] for p in centers_t0_full], dtype=np.float64)
        if centers_t0_full else np.empty((0, 2), np.float64)
    )
    source_z_t0 = (
        np.asarray([p[2] for p in centers_t0_full], dtype=np.float64)
        if centers_t0_full else np.empty((0,), np.float64)
    )
    state = dict(
        current=current,
        previous=previous,
        velocities=velocities,
        source_world_points=source_world_points,
        current_pose=current_pose,
    )
    registrations, _, _, audit = causal_source_history(
        hist, poses, state, provider.pcfg.grid, provider.strong, provider.workers,
        previous_instances=previous,
    )
    return (
        current_sem, previous_sem, current_pose, previous_pose,
        current, previous, velocities, source_world_points, source_rel_xy,
        source_z_t0, centers_t0, registrations, audit,
    )


def _derive_kta(current, velocities, current_pose, frame_dt_s):
    """Rebuild the causal constant-velocity displacement contract."""
    n = len(current)
    centers = np.zeros((n, 2), np.float64)
    kta = np.zeros((n, FUTURE_FRAMES, 2), np.float32)
    anchors = np.zeros((n, FUTURE_FRAMES, 2), np.float32)
    for i, comp in enumerate(current):
        cur_world = np.asarray(comp["centroid_world"], dtype=np.float64)
        cur_t0 = world_points_to_t0(cur_world[None], current_pose)[0, :2]
        centers[i] = cur_t0
        v_world = np.asarray(velocities.get(i, np.zeros(3)), dtype=np.float64)
        v_t0 = world_vec_to_t0(v_world, current_pose)[:2]
        for h in range(FUTURE_FRAMES):
            kd = v_t0 * ((h + 1) * float(frame_dt_s))
            kta[i, h] = kd.astype(np.float32)
            anchors[i, h] = (cur_t0 + kd).astype(np.float32)
    return centers, kta, anchors


def _verify_cached_motion_prior(record, centers, kta, anchors):
    """Audit only; timed forecast still rebuilds KTA rather than reusing cache."""
    report = {}
    fields = (
        ("source_centroid_xy_t0_m", centers.astype(np.float32)),
        ("kta_displacement_xy_m", kta),
        ("anchors_xy_t0_m", anchors),
    )
    for name, actual in fields:
        if name not in record:
            report[name] = "missing_not_required"
            continue
        expected = (
            record[name].detach().cpu().numpy()
            if isinstance(record[name], torch.Tensor) else np.asarray(record[name])
        )
        exact = np.array_equal(expected, actual)
        close = np.allclose(expected, actual, rtol=0, atol=2e-4)
        if not close:
            raise RuntimeError(f"{record['sample_id']}: derived {name} differs from frozen cache")
        report[name] = "byte_exact" if exact else "within_2e-4"
    return report


def prepare_history(provider, source, record, *, device, kernels=None, executor=None):
    """Build the single history-only state used by inference/FPS.

    This deliberately bypasses PilotProvider.load_raw_columns because that
    legacy helper also precomputes six-horizon Strong geometry. Future poses are
    returned separately and no forecast-dependent result enters the state.
    """
    total_started = time.perf_counter()
    stages = {}

    tick = time.perf_counter()
    raw = FrozenColumns.load_raw_columns(provider, source, record, include_gt=False)
    if raw.get("future_gt_occ") is not None:
        raise RuntimeError("history preparation must not load future occupancy")
    history_raw = {
        "history_occ": np.asarray(raw["history_occ"]),
        "history_observed": np.asarray(raw["history_observed"], dtype=bool),
        "history_poses": np.asarray(raw["history_poses"], dtype=np.float64),
    }
    future = FutureEgoCondition(
        tuple(np.asarray(p, np.float64) for p in raw["future_poses"]))
    stages["raw_history_io"] = time.perf_counter() - tick

    tick = time.perf_counter()
    core = _source_core(history_raw, record, provider)
    (current_sem, previous_sem, current_pose, previous_pose, current, previous,
     velocities, source_world_points, source_rel_xy, source_z_t0, centers_t0,
     registrations, source_audit) = core
    stages["source_extract_associate_register"] = time.perf_counter() - tick

    tick = time.perf_counter()
    minimal = SimpleNamespace(
        raw=history_raw,
        state={"current_pose": current_pose, "current": current},
        registrations=registrations,
    )
    evidence = build_canonical_evidence(
        minimal, provider.pcfg.grid, kernels=kernels, executor=executor)
    digest = fixed_history_digest(minimal, provider.pcfg.grid)
    evidence.fixed_history_sha256 = digest
    stages["canonical_history_evidence"] = time.perf_counter() - tick

    tick = time.perf_counter()
    gpu_history = _stage_history_inputs(record, device)
    stages["history_input_h2d"] = time.perf_counter() - tick

    centers_check, kta_check, anchors_check = _derive_kta(
        current, velocities, current_pose, provider.pcfg.frame_dt_s)
    if not np.allclose(centers_check, centers_t0, rtol=0, atol=1e-12):
        raise RuntimeError("internal source-centre derivation mismatch")
    prior_audit = _verify_cached_motion_prior(
        record, centers_check, kta_check, anchors_check)
    source_audit = {**dict(source_audit), "cached_motion_prior": prior_audit}

    stages["total"] = time.perf_counter() - total_started
    state = CausalHistoryState(
        sample_id=str(record["sample_id"]),
        scene_name=str(record["scene_name"]),
        window=window_from_record(record),
        raw_history=history_raw,
        current_sem=current_sem,
        current_pose=current_pose,
        previous_sem=previous_sem,
        previous_pose=previous_pose,
        current=current,
        previous=previous,
        velocities=velocities,
        source_world_points=source_world_points,
        source_rel_xy=source_rel_xy,
        source_z_t0=source_z_t0,
        source_centers_xy_t0=centers_t0,
        registrations=registrations,
        source_audit=source_audit,
        canonical_evidence=evidence,
        gpu_history=gpu_history,
        record_static=_history_only_record(record),
        content_sha256=digest,
        prepare_seconds=stages,
    )
    return state, future


def _forecast_record(history, kta, anchors):
    record = dict(history.record_static)
    record["kta_displacement_xy_m"] = torch.from_numpy(np.ascontiguousarray(kta))
    record["anchors_xy_t0_m"] = torch.from_numpy(np.ascontiguousarray(anchors))
    return record


def _forecast_state(history, future, provider, *, native_majority=True, profile=None):
    """Build every future-dependent deterministic prior inside FPS boundary."""
    future_poses = [np.asarray(p, np.float64) for p in future.poses]
    strong_anchors, strong_components = runtime._strong_all_horizons(
        history.current_sem,
        history.current_pose,
        future_poses,
        history.current,
        history.velocities,
        history.source_world_points,
        frame_dt_s=provider.pcfg.frame_dt_s,
        grid=provider.pcfg.grid,
        cfg=provider.strong,
        runtime_device=provider.device,
        majority_backend="native" if native_majority else "dense_cuda",
        profile=profile,
    )
    clear = [
        baseline_clear_flat_indices(rows, grid=provider.pcfg.grid)
        for rows in strong_components
    ]
    backgrounds = [
        compose_component_replacements_fast_exact(
            anchor, rows, [],
            dynamic_class_ids=runtime.DYNAMIC_CLASS_IDS,
            free_label=17,
            grid=provider.pcfg.grid,
            precomputed_clear_flat_indices=flat,
        )
        for anchor, rows, flat in zip(strong_anchors, strong_components, clear)
    ]
    return {
        "current_sem": history.current_sem,
        "previous_sem": history.previous_sem,
        "current_pose": history.current_pose,
        "previous_pose": history.previous_pose,
        "future_poses": future_poses,
        "current": history.current,
        "previous": history.previous,
        "velocities": history.velocities,
        "source_world_points": history.source_world_points,
        "source_rel_xy": history.source_rel_xy,
        "source_z_t0": history.source_z_t0,
        "anchors": strong_anchors,
        "baseline_by_hi": strong_components,
        "baseline_clear_flat_by_hi": clear,
        "column_backgrounds": backgrounds,
        "world_to_future": [np.linalg.inv(p) for p in future_poses],
        "gpu": None,
    }


def _result_signature(dense, probability, output):
    def fp(value):
        arr = np.ascontiguousarray(value)
        return hashlib.sha256(arr.view(np.uint8)).hexdigest()
    motion = {}
    for key, value in output.items():
        if not isinstance(value, torch.Tensor):
            continue
        raw = value.detach().contiguous().view(torch.uint8).cpu().numpy()
        motion[key] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": fp(raw),
        }
    return {
        "dense": [fp(x) for x in dense],
        "probability": fp(probability),
        "motion": motion,
    }


@torch.no_grad()
def forecast_six(history, future, provider, model, head, *, kernels=None,
                 executor=None, optimize_motion=True, native_majority=True,
                 return_signature=False):
    """Official Dense Forecast FPS execution path.

    Timer begins before fresh KTA/Strong construction and stops only after all
    six dense semantic occupancy grids are complete and CUDA is synchronized.
    """
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("formal FPS requires frozen eval motion model")
    if head.training or any(p.requires_grad for p in head.parameters()):
        raise RuntimeError("formal FPS requires frozen eval repair head")
    if history.canonical_evidence is None:
        raise RuntimeError("history state missing canonical evidence")
    device = provider.device
    if torch.device(device).type != "cuda":
        raise RuntimeError("formal Dense Forecast FPS requires CUDA")

    def sync():
        torch.cuda.synchronize(device)

    stages = {}
    prior_profile = {}

    def call(name, fn):
        tick = time.perf_counter()
        value = fn()
        stages[name] = stages.get(name, 0.0) + time.perf_counter() - tick
        return value

    sync()
    started = time.perf_counter()

    centers, kta, source_anchors = call(
        "causal_motion_prior",
        lambda: _derive_kta(
            history.current, history.velocities, history.current_pose,
            provider.pcfg.frame_dt_s,
        ),
    )
    if not np.allclose(
            centers, history.source_centers_xy_t0, rtol=0, atol=1e-12):
        raise RuntimeError("history/source centre changed after preparation")
    record = _forecast_record(history, kta, source_anchors)

    state = call(
        "fresh_strong_kta_prior",
        lambda: _forecast_state(
            history, future, provider, native_majority=native_majority,
            profile=prior_profile,
        ),
    )

    gpu = dict(history.gpu_history)
    gpu["kta"] = call(
        "kta_h2d",
        lambda: torch.as_tensor(
            np.ascontiguousarray(kta), device=device).float(),
    )
    context = reuse_v18_projections(model) if optimize_motion else nullcontext()
    with context:
        output = call(
            "motion_forward",
            lambda: runtime._model_forward(
                model, gpu, device, return_latents=True),
        )

    raw = {
        **history.raw_history,
        "future_poses": np.stack(future.poses),
        "future_gt_occ": None,
    }
    state["rec"] = record
    baseline, owners, fallbacks, components, targets, yaws = call(
        "source_transport_and_layering",
        lambda: render_column_layers(
            state, record, output, provider.pcfg.grid),
    )
    prep = PreparedColumns(
        history.window, raw, state, baseline, owners, fallbacks, components,
        targets, yaws, history.registrations, None, None, history.source_audit,
        output, None, None,
    )

    plan = call(
        "future_projection_ownership",
        lambda: map_canonical_evidence(
            history.canonical_evidence, prep, provider.pcfg.grid,
            kernels=kernels, executor=executor,
        ),
    )
    from tools.real_motion.pilot_p0_f9_canonical_causal_repair import probabilities
    probability = call(
        "ccr_shared_encode_six_readouts",
        lambda: probabilities(
            head, history.canonical_evidence, plan, output, device),
    )
    dense = call(
        "constrained_dense_composition",
        lambda: compose_canonical(
            prep.baseline, history.canonical_evidence, plan,
            probability[..., 0], probability[..., 1],
            thresholds=(.5, .95),
        ),
    )

    sync()
    elapsed = time.perf_counter() - started
    if (
        len(dense) != FUTURE_FRAMES
        or any(x.shape != tuple(provider.pcfg.grid.shape_hwd) for x in dense)
        or not math.isfinite(elapsed)
        or elapsed <= 0
    ):
        raise RuntimeError("six finished dense semantic outputs required")

    result = {
        "seconds": elapsed,
        "stages_seconds": stages,
        "prior_profile_ms": prior_profile,
        "six_complete_dense": True,
        "dense": dense,
        "probability": probability,
        "motion_output": output,
    }
    if return_signature:
        result["signature"] = _result_signature(dense, probability, output)
    return result


def batched_frozen_motion(teacher, records, device):
    """One frozen V18 forward for several windows, then split by source count.

    This is training-side dataflow cleanup only: no target/support/loss changes.
    """
    if not records:
        return []
    keys = (
        "features", "local_semantic_tube", "kta_displacement_xy_m",
        "frame_motion_features", "target_source_mask_tube",
    )
    sizes = [int(r["features"].shape[0]) for r in records]
    merged = {
        key: torch.cat([torch.as_tensor(r[key]) for r in records], dim=0)
        for key in keys
    }
    with torch.no_grad():
        output = teacher.motion(merged, device)
    if any(
        not isinstance(v, torch.Tensor) or v.shape[0] != sum(sizes)
        for v in output.values()
    ):
        raise RuntimeError("batched frozen motion output/source population mismatch")
    split = {k: v.split(sizes) for k, v in output.items()}
    return [
        {k: split[k][i] for k in split}
        for i in range(len(records))
    ]
