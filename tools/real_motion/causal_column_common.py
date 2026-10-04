"""Causal preparation, exact layered edits and one-pass four-way evaluation."""
from __future__ import annotations
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time
import numpy as np
import torch
from scipy.ndimage import binary_dilation

from real_motion.causal_column_completion import (ColumnConfig, ColumnPlan, GENERATE, REFINE, KEEP, ADD, REMOVE,
    FREE, UNKNOWN, CONTEXT_DIM, action_targets, compose_sparse, sparse_layout, actions_from_probabilities, sparse_counts, acceptance_gate)
from real_motion.source_evidence_audit import (associate_backwards, register_history_shape, transform_points, planar_move,
                                              raster_flat, edit_quality)
from real_motion.rigid_transport import rigid_source_points_world
from real_motion.runtime_fastpath import extract_instances_cropped_exact, compose_component_replacements_fast_exact
from real_motion.v19_static_novelty import history_grid_footprint_bev_sequence
from real_motion.v22_causal_emergence import build_future_static_memory_only, build_surface_frontier
from real_motion.nuscenes_adapter import gt_moving_support_sequence
from real_motion.v18_motion_gap import numpy
from tools.real_motion.v18_xy_trajectory_common import FrozenXYV18
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import Metrics, DYN, delta
from real_motion.causal_column_sampling import ColumnFeatureSampler
from real_motion.column_runtime_pipeline import prefetch_raw_columns
from real_motion.prepared import load_nuscenes_window_raw
from real_motion.motion_transport import world_points_to_t0
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record

REPORT = (1, 3, 5)
FEATURE_KEYS = ("history", "flags", "base", "fallback", "context", "kind", "classes")
VARIANTS = ("generation", "refine", "joint")


def component_layers(background, components):
    """Frozen last-source-wins order, with explicit visible owner/fallback."""
    shape = background.shape
    base = np.asarray(background).copy(); fallback = base.copy()
    owner = np.full(shape, -1, np.int32)
    for i, comp in enumerate(components):
        idx = np.asarray(comp.voxel_indices, np.int64)
        if not len(idx): continue
        at = tuple(idx.T)
        fallback[at] = base[at]; base[at] = comp.class_id; owner[at] = i
    return base, owner, fallback


def pose_motion(center, target, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    out = np.eye(4); out[:2, :2] = ((c, -s), (s, c))
    out[:2, 3] = np.asarray(target)[:2]-out[:2, :2]@np.asarray(center)[:2]
    # Match frozen world-Z preservation (no vertical centre translation).
    return out


def causal_source_history(raw_history, poses, state, grid, strong, workers):
    """Causal same-class association/ICP only. No annotation or future GT."""
    t = len(raw_history)
    if t not in (4, 6) or len(poses) != t: raise ValueError('four or six aligned historical observations required')
    with ThreadPoolExecutor(max_workers=min(workers, t-1)) as pool:
        frames = list(pool.map(lambda f: extract_instances_cropped_exact(raw_history[f], poses[f], grid=grid, cfg=strong), range(t-1)))
    frames.append(state["current"])
    links, audit = associate_backwards(frames, state["current"], state["velocities"], dt=.5)
    points = [[rigid_source_points_world(c["voxel_indices"], pose, grid=grid) for c in frame]
              for frame, pose in zip(frames, poses)]
    registrations = [[None]*t for _ in state["current"]]
    for i, comp in enumerate(state["current"]):
        registrations[i][-1] = (np.eye(4), np.asarray(comp["voxel_indices"], np.int64))
        for f, j in enumerate(links[i][:-1]):
            if j is None: continue
            p = points[f][j]
            result = register_history_shape(p, state["source_world_points"][i], allow_yaw=int(comp["class_id"]) != 7)
            audit["registration_accepted" if result.accepted else "registration_rejected"] = audit.get("registration_accepted" if result.accepted else "registration_rejected", 0)+1
            if result.accepted:
                r = pose_motion(np.zeros(3), np.zeros(3), result.yaw_rad)
                r[:2, 3] = result.points[:, :2].mean(0)-p[:, :2].mean(0)@r[:2, :2].T
                registrations[i][f] = (r, np.asarray(frames[f][j]["voxel_indices"], np.int64))
    return registrations, points, links, audit


@dataclass
class PreparedColumns:
    window: object
    raw: dict
    state: dict
    baseline: list
    owners: list
    fallbacks: list
    components: list
    targets: list
    yaws: list
    registrations: list
    footprints: np.ndarray
    memory: np.ndarray
    source_audit: dict
    outputs: dict | None = None
    aligned_history_points: list | None = None
    fixed_candidate_geometry: list | None = None


class FrozenColumns(FrozenXYV18):
    def load_raw_columns(self, source, record, *, include_gt):
        return load_nuscenes_window_raw(source, window_from_record(record), self.pcfg,
            include_gt=include_gt, io_workers=min(self.workers, 4),
            active_history_frames=getattr(getattr(getattr(getattr(self, 'joint', None), 'transport', self.model), 'config', None), 'history_frames', 6))

    def prepare_columns(self, source, record, *, include_gt, raw_window=None, outputs=None):
        started = time.perf_counter()
        if raw_window is None and outputs is None and getattr(getattr(self.model, 'config', None), 'history_frames', 6) == 6:
            window, raw, state, outputs = super().prepare(source, record, include_gt=include_gt)
        else:
            window = window_from_record(record)
            raw = self.load_raw_columns(source, record, include_gt=include_gt) if raw_window is None else raw_window
            if not include_gt and raw.get('future_gt_occ') is not None:
                raise RuntimeError('causal deployment must not request future occupancy')
            causal = raw.get('_column_causal_preparation')
            fixed_state = causal.get('prepared_state') if causal is not None else None
            if fixed_state is not None:
                state = {**fixed_state, 'rec': record, 'window': window, 'gpu': None}
                if (len(state['current']) != len(record['features'])
                        or [int(c['class_id']) for c in state['current']] != record['source_class_id'].tolist()):
                    raise RuntimeError('cached Strong/source identity mismatch')
            else:
                state = runtime._prepare_record(record, source, self.pcfg, self.strong, self.device, raw_window=raw)
            centers = world_points_to_t0(np.asarray([c['centroid_world'] for c in state['current']]).reshape(-1, 3),
                                        state['current_pose'])[:, :2]
            if not np.allclose(centers, numpy(record['source_centroid_xy_t0_m']), rtol=0, atol=2e-4):
                raise RuntimeError('cache/actual source-centre identity mismatch')
            outputs = self.encode_record(record) if outputs is None else outputs
        prepared_at = time.perf_counter()
        baseline, owners, fallbacks, components, targets, yaws = render_column_layers(state, record, outputs, self.pcfg.grid,
            capture_backgrounds=bool(raw.get('_causal_cache_deferred')))
        if not getattr(self, "columns_checked", False):
            runtime._stage_gpu_inputs(state, self.device)
            try:
                runtime._exactness_check(self.model, state, self.pcfg, self.strong, self.device)
                # A resumed joint run reaches this check on its FIRST training
                # batch (prior audit is skipped), so outputs are live autograd
                # tensors. Detach a diagnostic-only copy for the NumPy renderer;
                # PreparedColumns.outputs must retain the original source graph.
                diagnostic_outputs = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in outputs.items()}
                reference = runtime._forecast_once(self.model, state, self.pcfg, self.strong, self.device,
                                                   precomputed_out=diagnostic_outputs)
                if any(not np.array_equal(a, b) for a, b in zip(reference, baseline)):
                    raise RuntimeError("layered column preparation differs from V18 renderer")
            finally: runtime._release_gpu_inputs(state)
            self.columns_checked = True
        renderer_at = time.perf_counter()
        causal = raw.get('_column_causal_preparation')
        if causal is not None:
            # Worker caches ONLY raw-history-dependent evidence. Never reuse
            # learned target poses, owner/fallback, candidates, features or GT labels.
            from real_motion.runtime_fastpath import component_lists_equal
            if not component_lists_equal(state['current'], causal['current']):
                raise RuntimeError('prefetched causal source identity mismatch')
            registrations, audit = causal['registrations'], causal['audit']
        else:
            registrations, _, _, audit = causal_source_history(raw["history_occ"], raw["history_poses"], state,
                                                              self.pcfg.grid, self.strong, self.workers)
        history_at = time.perf_counter()
        if causal is not None:
            footprints, memory = causal['footprints'], causal['memory']
        else:
            footprints = history_grid_footprint_bev_sequence(raw["history_poses"], raw["future_poses"], self.pcfg.grid, workers=self.workers)
            memory = build_future_static_memory_only(raw["history_occ"], raw["history_observed"], raw["history_poses"], raw["future_poses"],
                grid=self.pcfg.grid, dynamic_class_ids=DYN, free_label=FREE, workers=self.workers)
        self.last_prepare_seconds = {"raw_and_v18": prepared_at-started, "layered_renderer": renderer_at-prepared_at,
            "source_history": history_at-renderer_at, "static_memory_and_footprint": time.perf_counter()-history_at}
        return PreparedColumns(window, raw, state, baseline, owners, fallbacks, components, targets, yaws,
                               registrations, footprints, memory, audit, outputs,
                               causal.get('aligned_history_points') if causal is not None else None,
                               causal.get('fixed_candidate_geometry') if causal is not None else None)


def render_column_layers(state, record, outputs, grid, *, capture_backgrounds=False, baseline_only=False):
    """Model-dependent geometry ONLY; immutable history/registration can be reused."""
    render = outputs.get('_column_render_numpy')
    res, yaw = ((numpy(outputs["residual_xy_m"]), numpy(outputs["yaw_delta_rad"])) if render is None
                else (render['residual_xy_m'], render['yaw_delta_rad']))
    anchors = numpy(record['anchors_xy_t0_m'])
    from real_motion.v18_two_wheel_diagnostic import renderer_yaw_delta
    baseline, owners, fallbacks, components, targets, yaws = [], [], [], [], [], []
    backgrounds = []
    for h in range(6):
        centers = [runtime._target_world_from_xy_cached(anchors[i, h]+res[i, h],
            state["source_z_t0"][i], state["current_pose"]) for i in range(len(state["current"]))]
        yy = [renderer_yaw_delta(int(c["class_id"]), yaw[i, h], zero_two_wheel_yaw=False) for i, c in enumerate(state["current"])]
        layers = runtime._rasterize_all_sources_horizon(state["current"], state["source_world_points"], state["source_rel_xy"],
                                                       centers, yy, state["world_to_future"][h], grid)
        if 'column_backgrounds' in state: background = state['column_backgrounds'][h]
        else:
            background = compose_component_replacements_fast_exact(state["anchors"][h], state["baseline_by_hi"][h], [],
                dynamic_class_ids=DYN, free_label=FREE, grid=grid,
                precomputed_clear_flat_indices=state["baseline_clear_flat_by_hi"][h])
        if capture_backgrounds: backgrounds.append(background)
        if baseline_only:
            b = background.copy(); own = fall = None
            for layer in layers:
                if len(layer.voxel_indices): b[tuple(np.asarray(layer.voxel_indices).T)] = layer.class_id
        else: b, own, fall = component_layers(background, layers)
        baseline.append(b); owners.append(own); fallbacks.append(fall); components.append(layers); targets.append(centers); yaws.append(yy)
    if capture_backgrounds: state['column_backgrounds'] = backgrounds
    return baseline, owners, fallbacks, components, targets, yaws


def prepare_warm_columns_cpu(record, raw, outputs, grid):
    """CPU-only warm preparation; live Tensor references are NEVER evaluated.

    Caller has already checked the frozen renderer on its CUDA-owning thread
    and supplied detached render arrays. Original outputs retain the graph.
    """
    causal = raw['_column_causal_preparation']; fixed = causal['prepared_state']
    if 'column_backgrounds' not in fixed or '_column_render_numpy' not in outputs:
        raise RuntimeError('CPU warm preparation requires complete geometry and detached render arrays')
    window = window_from_record(record)
    state = {**fixed, 'rec': record, 'window': window, 'gpu': None}
    if (len(state['current']) != len(record['features'])
            or [int(c['class_id']) for c in state['current']] != record['source_class_id'].tolist()):
        raise RuntimeError('cached Strong/source identity mismatch')
    centers = world_points_to_t0(np.asarray([c['centroid_world'] for c in state['current']]).reshape(-1, 3),
                                state['current_pose'])[:, :2]
    if not np.allclose(centers, numpy(record['source_centroid_xy_t0_m']), rtol=0, atol=2e-4):
        raise RuntimeError('cache/actual source-centre identity mismatch')
    from real_motion.runtime_fastpath import component_lists_equal
    if not component_lists_equal(state['current'], causal['current']):
        raise RuntimeError('prefetched causal source identity mismatch')
    baseline, owners, fallbacks, components, targets, yaws = render_column_layers(state, record, outputs, grid)
    return PreparedColumns(window, raw, state, baseline, owners, fallbacks, components, targets, yaws,
        causal['registrations'], causal['footprints'], causal['memory'], causal['audit'], outputs,
        causal.get('aligned_history_points'), causal.get('fixed_candidate_geometry'))


def fixed_candidate_geometry(memory, footprints, grid, config):
    """History/ego-only frontier, semantics and vertical support; never scores/GT."""
    out = []
    for m, footprint in zip(memory, footprints):
        frontier = build_surface_frontier(m, footprint, class_ids=(11, 13), widths_m=(config.entry_radius_m,),
                                         voxel_size_xy_m=grid.voxel_size[0], free_label=FREE)
        counts = np.stack([(m == c).sum(2) for c in (11, 13)], -1)
        historical = (m == 11)|(m == 13)
        out.append(dict(radius=config.entry_radius_m, frontier=frontier,
            dominant=np.asarray((11, 13), np.uint8)[counts.argmax(-1)], historical=historical,
            static_allowed=binary_dilation(historical, structure=np.ones((1, 1, 3)), iterations=1),
            generation_xy=np.argwhere(frontier.causal_by_width[config.entry_radius_m]).astype(np.int32),
            static_xy=np.argwhere(footprint & historical.any(2)).astype(np.int32)))
    return out


def padded_support_xy(xy, shape, padding=1, *, optimize=False):
    """Exact full-grid binary dilation/argwhere order on a source-local bbox."""
    lo = np.maximum(np.asarray(xy).min(0)-padding, 0)
    hi = np.minimum(np.asarray(xy).max(0)+padding+1, shape)
    support = np.zeros(tuple(hi-lo), bool); at = np.asarray(xy)-lo; support[tuple(at.T)] = True
    if optimize and padding == 1:
        # The exact default scipy cross stencil; NOT an 8-neighbor dilation.
        grown = support.copy()
        grown[1:] |= support[:-1]; grown[:-1] |= support[1:]
        grown[:, 1:] |= support[:, :-1]; grown[:, :-1] |= support[:, 1:]
        return np.argwhere(grown)+lo
    return np.argwhere(binary_dilation(support, iterations=padding))+lo


def _column_context(xy, legal, actor, ax, ay, age, h, rel, grid, config):
    """Reference arithmetic, materialized only for TRAIN's selected rows."""
    context = np.zeros((len(xy), CONTEXT_DIM), np.float32)
    context[:, :2] = (xy+.5)/np.asarray(grid.shape_hwd[:2])*2-1
    context[:, 2] = .5*(h+1)/3
    context[:, 3:6] = (rel[0, 3]/40, rel[1, 3]/40, np.arctan2(rel[1, 0], rel[0, 0])/np.pi)
    context[:, 6:8] = (xy-np.column_stack((ax, ay)))*np.asarray(grid.voxel_size[:2])/config.entry_radius_m
    context[:, 8] = np.linalg.norm(context[:, 6:8], axis=1)
    context[:, 9] = age/2.5
    context[:, 10] = legal[..., ADD].mean(1)
    context[:, 11] = actor >= 0
    return context


class _DeferredContextColumns:
    """TRAIN-only complete population. No model may consume this wrapper.

    Candidate fields, labels and sampling order stay dense/exact. Only context
    arithmetic is delayed; subset() always produces a normal ColumnPlan. The
    original full actor support centre is retained, never recomputed from the
    sampled subset. No geometry or scores survive the current window/update.
    """
    def __init__(self, plan, descriptors, h, rel, grid, config):
        self.plan, self.descriptors = plan, descriptors
        self.h, self.rel, self.grid, self.config = h, rel, grid, config
        self.ends = np.asarray([row[1] for row in descriptors], np.int64)

    def __getattr__(self, name):
        if name == 'context': raise RuntimeError('TRAIN context must be materialized through subset()')
        return getattr(self.plan, name)

    def __len__(self): return len(self.plan)

    def subset(self, indices):
        ids = np.arange(len(self))[indices] if isinstance(indices, slice) else np.asarray(indices)
        if ids.ndim != 1: raise ValueError('TRAIN subset indices must be one dimensional')
        if ids.dtype == bool:
            if len(ids) != len(self): raise ValueError('TRAIN boolean subset population mismatch')
            ids = np.flatnonzero(ids)
        if ids.size and ids.dtype.kind not in 'iu': raise IndexError('TRAIN subset requires integer indices')
        if ids.dtype.kind == 'u' and np.any(ids > np.iinfo(np.int64).max): raise IndexError('TRAIN subset index out of bounds')
        ids = ids.astype(np.int64, copy=False)
        ids = np.where(ids < 0, ids+len(self), ids)
        if np.any((ids < 0)|(ids >= len(self))): raise IndexError('TRAIN subset index out of bounds')
        small = self.plan.subset(ids)
        if not len(ids): return small
        groups = np.searchsorted(self.ends, ids, side='right')
        xs, ys, ages = (np.empty(len(ids), np.float64) for _ in range(3))
        for group in np.unique(groups):
            start, stop, actor, ax, ay, age = self.descriptors[group]
            take = np.flatnonzero(groups == group)
            local = ids[take]-start
            xs[take] = ax[local]; ys[take] = ay[local]; ages[take] = age
        # All integer anchors are bounded by this small grid, hence their
        # float64 conversion is exact. Keep the reference float32 norm/mean.
        small.context = _column_context(small.xy, small.legal, small.actor,
            xs, ys, ages, self.h, self.rel, self.grid, self.config)
        return small


def candidate_plan(prepared, h, grid, config=ColumnConfig(), *, defer_context=False):
    """GT-free candidates, complete population, no evaluation top-positive cap."""
    config.validate()
    b, m, footprint = prepared.baseline[h], prepared.memory[h], prepared.footprints[h]
    shape = tuple(grid.shape_hwd); z = shape[2]
    if z != config.z_bins: raise RuntimeError("Z lattice/checkpoint mismatch")
    if not np.isclose(grid.voxel_size[0], grid.voxel_size[1], rtol=0, atol=1e-10):
        raise RuntimeError("frontier distance contract requires an isotropic XY lattice")
    if defer_context:
        # Pathological scales must still fail through the original full-context
        # validation, even when the offending row would not have been sampled.
        with np.errstate(over='ignore', invalid='ignore'):
            extent = np.asarray(shape[:2])*np.asarray(grid.voxel_size[:2])/config.entry_radius_m
        if not np.isfinite(extent).all() or np.any(np.abs(extent) > np.sqrt(np.finfo(np.float32).max/4)):
            defer_context = False
    fixed = getattr(prepared, 'fixed_candidate_geometry', None)
    if fixed is not None and fixed[h]['radius'] == config.entry_radius_m:
        geometry = fixed[h]
    else:
        geometry = fixed_candidate_geometry([m], [footprint], grid, config)[0]
    frontier, dominant = geometry['frontier'], geometry['dominant']
    rows = []; descriptors = []; row_count = 0
    fast = getattr(prepared, 'cpu_pipeline_optimized', True)
    kernels = fast and getattr(prepared, 'cpu_kernels_optimized', True)
    from real_motion.native_column_cpu import get_native
    native = get_native() if kernels else None
    support_workspace = None
    relative_pose = None
    def append(xy, kind, actor, classes, allowed_z, ax, ay, age):
        nonlocal relative_pose, row_count
        if not len(xy): return
        if native is not None:
            flat, base, fall, legal, active = native.rows(xy, classes, allowed_z, b,
                prepared.owners[h], prepared.fallbacks[h], kind, actor)
        else:
            flat = ((xy[:, 0:1]*shape[1]+xy[:, 1:2])*z+np.arange(z)).astype(np.int64)
            base = b.reshape(-1)[flat]; fall = base.copy()
            legal = np.zeros((*base.shape, 3), bool); legal[..., KEEP] = True
            legal[..., ADD] = (base == FREE)&allowed_z
            if kind == REFINE:
                if actor >= 0:
                    own = prepared.owners[h].reshape(-1)[flat] == actor
                    restored = prepared.fallbacks[h].reshape(-1)[flat]
                else:
                    own = base == classes[:, None]; restored = np.full_like(base, FREE)
                fall[own] = restored[own]
                legal[..., REMOVE] = own & (base == classes[:, None]) & (fall != base)
            active = legal[..., 1:].any(axis=(1, 2))
        xy, flat, base, fall, legal, classes, ax, ay = (v[active] for v in (xy, flat, base, fall, legal, classes, ax, ay))
        if (fast or defer_context) and relative_pose is None:
            relative_pose = np.linalg.inv(prepared.state['current_pose'])@prepared.raw['future_poses'][h]
            if defer_context and (not np.isfinite(relative_pose).all()
                    or np.any(np.abs(relative_pose[:2, 3]) > float(np.finfo(np.float32).max)*40)):
                raise ValueError('invalid TRAIN context pose')
        rel = relative_pose if fast or defer_context else np.linalg.inv(prepared.state["current_pose"])@prepared.raw["future_poses"][h]
        if defer_context:
            # XY/anchors are bounded integer grid indices (dynamic anchors are
            # means of those indices); age is an integer history-frame count.
            # Pose finite-check is performed once above, not per source.
            context = np.broadcast_to(np.zeros(CONTEXT_DIM, np.float32), (len(xy), CONTEXT_DIM))
            descriptors.append((row_count, row_count+len(xy), actor, ax, ay, age))
            row_count += len(xy)
        else: context = _column_context(xy, legal, actor, ax, ay, age, h, rel, grid, config)
        rows.append(ColumnPlan(xy.astype(np.int32), np.full(len(xy), kind, np.uint8), np.full(len(xy), actor, np.int32),
                               classes.astype(np.uint8), flat, base, fall, legal, context,
                               np.column_stack((ax, ay)).astype(np.int32) if kind == GENERATE else xy.astype(np.int32)))
    if kernels:
        # Only history/ego defines these potential XY rows. Test the CURRENT
        # baseline on that support, not every voxel of the entire grid.
        potential = (geometry['generation_xy'] if 'generation_xy' in geometry
            else np.argwhere(frontier.causal_by_width[config.entry_radius_m]))
        xy = native.generation(potential, b) if native is not None else potential[(b[tuple(potential.T)] == FREE).any(1)]
        ax, ay = frontier.nearest_x[tuple(xy.T)], frontier.nearest_y[tuple(xy.T)]
    else:
        entry = frontier.causal_by_width[config.entry_radius_m] & (b == FREE).any(2)
        xy = np.argwhere(entry)
        ax, ay = frontier.nearest_x[entry], frontier.nearest_y[entry]
    append(xy, GENERATE, -3, dominant[ax, ay], np.ones((len(xy), z), bool), ax, ay, 0.)
    # Historical road/sidewalk residuals ONLY inside visited grid; not full-scene
    # refinement. Static occupied edits never touch dynamic/source-owned labels.
    historical = geometry['historical']
    if native is not None:
        from real_motion.local_supervision_fastpath import static_roi_enabled
        potential = geometry.get('static_xy')
        xy = (native.static_roi(potential, historical, m, b)
              if static_roi_enabled() and potential is not None and len(potential)*4 < shape[0]*shape[1]
              else native.static(footprint, historical, m, b))
        cls = dominant[tuple(xy.T)]; mask = geometry['static_allowed'][tuple(xy.T)]
    elif kernels:
        potential = geometry.get('static_xy')
        static_support = None if potential is not None else footprint & historical.any(2)
        count = len(potential) if potential is not None else np.count_nonzero(static_support)
        if count*4 < shape[0]*shape[1]:
            if potential is None: potential = np.argwhere(static_support)
            at = tuple(potential.T)
            xy = potential[(historical[at] & (m[at] != b[at])).any(1)]
        else:
            # For broad road support, contiguous full-grid reductions beat
            # copying most of the grid through advanced indexing.
            if static_support is None: static_support = footprint & historical.any(2)
            xy = np.argwhere(static_support & (historical & (m != b)).any(2))
        cls = dominant[tuple(xy.T)]; mask = geometry['static_allowed'][tuple(xy.T)]
    else:
        mismatch = (historical & (m != b)).any(2)
        static = footprint & mismatch & historical.any(2)
        xy = np.argwhere(static)
        cls = dominant[static]; mask = geometry['static_allowed'][static]
    append(xy, REFINE, -2, cls, mask, xy[:, 0], xy[:, 1], 0.)
    for i, comp in enumerate(prepared.state["current"]):
        registered = prepared.registrations[i]
        if not any(r is not None for r in registered[:-1]): continue
        all_flat = [np.ravel_multi_index(prepared.components[h][i].voxel_indices.T, shape)]
        for f, reg in enumerate(registered[:-1]):
            if reg is None: continue
            if getattr(prepared, 'aligned_history_points', None) is not None:
                aligned = prepared.aligned_history_points[i][f]
            else:
                points = rigid_source_points_world(reg[1], prepared.raw["history_poses"][f], grid=grid)
                aligned = transform_points(points, reg[0])
            moved = planar_move(aligned, comp["centroid_world"], prepared.targets[h][i], prepared.yaws[h][i])
            ids, _ = raster_flat(moved, prepared.state["world_to_future"][h],
                                (grid.x_min, grid.y_min, grid.z_min), grid.voxel_size, shape,
                                deduplicate=not fast, optimize=kernels)
            all_flat.append(ids)
        ids = np.concatenate(all_flat)
        if not fast: ids = np.unique(ids)
        if not len(ids): continue
        if native is not None and config.boundary_padding_cells == 1:
            if support_workspace is None:
                support_workspace = (np.empty(shape[:2], np.uint8),
                    np.empty((shape[0]*shape[1], 2), np.int64), np.empty(2, np.int64))
            xy, zlo, zhi = native.support(ids, shape, support_workspace)
        else:
            xyz = np.column_stack(np.unravel_index(ids, shape))
            xy = padded_support_xy(xyz[:, :2], shape[:2], config.boundary_padding_cells, optimize=kernels)
            zlo, zhi = xyz[:, 2].min(), xyz[:, 2].max()
        allowed = np.broadcast_to((np.arange(z) >= max(0, zlo-1))
                                  & (np.arange(z) <= min(z-1, zhi+1)), (len(xy), z)).copy()
        age = .5*(len(registered)-1-min(f for f, reg in enumerate(registered) if reg is not None))
        center = xy.mean(0)
        append(xy, REFINE, i, np.full(len(xy), int(comp["class_id"]), np.uint8), allowed,
               np.full(len(xy), center[0]), np.full(len(xy), center[1]), age)
    if rows:
        arrays = {k: np.concatenate([getattr(r, k) for r in rows]) for k in vars(rows[0]) if k != 'context' or not defer_context}
        if defer_context: arrays['context'] = np.broadcast_to(np.zeros(CONTEXT_DIM, np.float32), (row_count, CONTEXT_DIM))
        plan = ColumnPlan(**arrays)
    else:
        plan = ColumnPlan(np.empty((0, 2), np.int32), np.empty(0, np.uint8), np.empty(0, np.int32), np.empty(0, np.uint8),
            np.empty((0, z), np.int64), np.empty((0, z), np.uint8), np.empty((0, z), np.uint8),
            np.empty((0, z, 3), bool), np.empty((0, CONTEXT_DIM), np.float32))
    plan.validate(packed_keys=kernels)
    return _DeferredContextColumns(plan, descriptors, h, relative_pose, grid, config) if defer_context else plan


def sample_column_features(prepared, h, plan, grid, config=ColumnConfig()):
    """Inverse sample LOCAL patches, not 36 full dense feature volumes.

    Flags bit0 = lidar visibility; bit1 = assigned static class/source membership.
    Historical semantic input follows V18 full occupancy protocol. Unknown=18
    means outside grid/unmatched source, NEVER an observed free label.
    """
    n, p, z = len(plan), config.patch, config.z_bins
    t = len(prepared.raw['history_occ'])
    hist = np.full((n, t, p, p, z), UNKNOWN, np.uint8)
    flags = np.zeros_like(hist)
    offsets = np.stack(np.meshgrid(np.arange(p)-p//2, np.arange(p)-p//2, np.arange(z), indexing="ij"), -1)
    # A GRID_ENTRY query can be >patch_radius outside ALL historical grids.
    # Sampling only around that unseen centre gives no height/appearance evidence
    # at all. GEN therefore attends its causal frontier-anchor patch; query->
    # anchor displacement stays explicit in context. Refine samples its edit site.
    idx = offsets[None]+np.pad(plan.evidence_xy, ((0, 0), (0, 1)))[:, None, None, None, :]
    origin = np.asarray((grid.x_min, grid.y_min, grid.z_min))
    future_xyz = origin+(idx+.5)*np.asarray(grid.voxel_size)
    future_pose = prepared.raw["future_poses"][h]
    for actor in np.unique(plan.actor):
        take = np.flatnonzero(plan.actor == actor)
        for f in range(t):
            transform = np.linalg.inv(prepared.raw["history_poses"][f])@future_pose
            owned_ids = None
            if actor >= 0:
                reg = prepared.registrations[actor][f]
                if reg is None: continue
                motion = pose_motion(prepared.state["current"][actor]["centroid_world"], prepared.targets[h][actor], prepared.yaws[h][actor])
                transform = np.linalg.inv(prepared.raw["history_poses"][f])@np.linalg.inv(reg[0])@np.linalg.inv(motion)@future_pose
                owned_ids = np.unique(np.ravel_multi_index(reg[1].T, grid.shape_hwd))
            pts = transform_points(future_xyz[take].reshape(-1, 3), transform)
            ijk = np.floor((pts-origin)/np.asarray(grid.voxel_size)).astype(np.int64)
            valid = ((ijk >= 0)&(ijk < np.asarray(grid.shape_hwd))).all(1)
            labels = np.full(len(ijk), UNKNOWN, np.uint8); ff = np.zeros(len(ijk), np.uint8)
            at = tuple(ijk[valid].T)
            labels[valid] = np.asarray(prepared.raw["history_occ"])[f][at]
            ff[valid] = np.asarray(prepared.raw["history_observed"])[f][at].astype(np.uint8)
            if owned_ids is None:
                inherited = np.repeat(plan.classes[take], p*p*z)
                ff[valid] |= ((labels[valid] == inherited[valid]).astype(np.uint8)*2)
            else:
                flat = np.ravel_multi_index(ijk[valid].T, grid.shape_hwd)
                ff[valid] |= np.isin(flat, owned_ids).astype(np.uint8)*2
            hist[take, f] = labels.reshape(len(take), p, p, z)
            flags[take, f] = ff.reshape(len(take), p, p, z)
    return {"history": hist, "flags": flags, "base": plan.base, "fallback": plan.fallback,
            "context": plan.context, "kind": plan.kind, "classes": plan.classes}


def predict_probabilities(model, prepared, h, plan, grid, device, batch_size=256, *,
                          feature_backend='cpu', history_index=None, gpu_sampler=None, verify_features=False,
                          optimized=None):
    optimized = getattr(model, 'column_inference_optimized', False) if optimized is None else optimized
    verify = optimized and getattr(model, 'column_inference_verify_remaining', 0) > 0
    reference = None
    reference_seconds = None
    if verify:
        reference_started = time.perf_counter()
        reference = predict_probabilities(model, prepared, h, plan, grid, device, batch_size,
            feature_backend=feature_backend, history_index=history_index, gpu_sampler=gpu_sampler,
            verify_features=verify_features, optimized=False)
        reference_seconds = time.perf_counter()-reference_started
    model.eval(); parts = []
    started = time.perf_counter()
    from real_motion.column_inference_pipeline import InferenceFeatures
    sampler = InferenceFeatures(prepared, h, plan, grid, model.config, device, pose_motion,
        backend=feature_backend, workers=getattr(model, 'column_sampling_workers', 1),
        history_index=history_index, gpu_sampler=gpu_sampler, verify=verify_features)
    mapped_at = time.perf_counter(); sampled_seconds = sampling_wait = 0.
    pool = ThreadPoolExecutor(max_workers=1) if optimized and feature_backend == 'cpu' else None
    def sample(start):
        small = plan.subset(slice(start, start+batch_size))
        tick = time.perf_counter()
        arrays = sampler.sample(small, sample_column_features)
        return small, arrays, time.perf_counter()-tick
    pending = None
    from real_motion.causal_column_model import CausalColumnModel
    defer_checks = bool(optimized and isinstance(model, CausalColumnModel))
    try:
      with torch.inference_mode():
        finite = (torch.isfinite(model.generation_pos_weight).all() & torch.isfinite(model.refine_class_weights).all()
            & (model.generation_pos_weight > 0) & (model.refine_class_weights > 0).all()) if defer_checks else None
        if pool is not None and len(plan): pending = pool.submit(sample, 0)
        for start in range(0, len(plan), batch_size):
            waiting = time.perf_counter()
            small, arrays, sample_seconds = pending.result() if pending is not None else sample(start)
            sampling_wait += time.perf_counter()-waiting
            sampled_seconds += sample_seconds
            pending = (pool.submit(sample, start+batch_size)
                if pool is not None and start+batch_size < len(plan) else None)
            b = {k: torch.as_tensor(v, device=device) for k, v in arrays.items()}
            legal = torch.as_tensor(small.legal, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                if hasattr(model, 'source_features_for'):
                    b['source_features'] = model.source_features_for(prepared, h, small, device)
                if hasattr(model, 'extra_inputs_for'):
                    b.update(model.extra_inputs_for(prepared, h, small, grid, device))
                g, r = model(**b)
            if defer_checks:
                finite = finite & torch.isfinite(g).all() & torch.isfinite(r).all()
                probability = model.calibrated_probabilities(g, r, b['kind'], legal, validate=False)
            else: probability = model.calibrated_probabilities(g, r, b['kind'], legal)
            parts.append(probability.cpu().numpy())
        if defer_checks and not finite:
            raise RuntimeError('nonfinite prediction or invalid TRAIN calibration weights')
    finally:
        if pool is not None: pool.shutdown(wait=True, cancel_futures=True)
    model.last_prediction_profile = {'queries': len(plan), 'cache_mib': sampler.cache_bytes/2**20,
        'inverse_map_seconds': mapped_at-started, 'patch_gather_seconds': sampled_seconds,
        'sampling_wait_seconds': sampling_wait,
        'network_transfer_and_other_seconds': max(0., time.perf_counter()-mapped_at-sampling_wait),
        'sampling_seconds_are_overlapping_worker_time': pool is not None,
        'feature_backend': 'gpu' if sampler.gpu is not None else 'cpu', 'feature_audit': dict(sampler.audit)}
    result = np.concatenate(parts) if parts else np.empty((*plan.base.shape, 3), np.float32)
    if verify:
        if not np.array_equal(reference, result):
            raise RuntimeError('optimized inference probability exactness failed; use reference inference')
        model.column_inference_verify_remaining -= 1
        model.last_prediction_profile['probability_exactness_passed'] = True
        model.last_prediction_profile['verification_timing_seconds'] = {
            'reference': reference_seconds, 'optimized': time.perf_counter()-started,
            'note': 'first-use warm/cache effects; not a controlled throughput benchmark'}
    return result


def report_states(base, metrics, quality, scenes):
    b = base.compute(); reports = {}
    for name, current in metrics.items():
        by_scene = {s: row[name].compute()["mIoU"]-row["baseline"].compute()["mIoU"] for s, row in scenes.items()}
        q = dict(quality[name]); added = q.get("added", 0); removed = q.get("removed", 0)
        q["addition_semantic_precision"] = q.get("added_semantic_tp", 0)/added if added else None
        q["removal_false_occupancy_fraction"] = q.get("removed_false_occupancy", 0)/removed if removed else None
        reports[name] = {"metrics": current.compute(), "delta_vs_v18_pp": delta(current.compute(), b), "quality": q,
                        "scene_delta": {"scenes": len(by_scene), "positive": sum(v > 0 for v in by_scene.values()),
                            "negative": sum(v < 0 for v in by_scene.values()), "zero": sum(v == 0 for v in by_scene.values()), "by_scene": by_scene}}
    return {"baseline": b, "variants": reports}


def evaluate_columns(provider, source, records, model, thresholds, *, progress=None, batch_size=256, dev64_keys=None,
                     diagnostic_thresholds=(.50, .50, .50), stop_event=None, feature_backend='cpu'):
    """Bounded persistent CPU candidate pool; GPU and metrics stay caller-owned."""
    if feature_backend not in ('cpu', 'gpu'): raise ValueError('invalid evaluation feature backend')
    with ThreadPoolExecutor(max_workers=min(3,provider.workers)) as pool:
        return _evaluate_columns(provider, source, records, model, thresholds, progress=progress,
            batch_size=batch_size, dev64_keys=dev64_keys, diagnostic_thresholds=diagnostic_thresholds,
            stop_event=stop_event, feature_backend=feature_backend, candidate_pool=pool)


def _evaluate_columns(provider, source, records, model, thresholds, *, progress=None, batch_size=256, dev64_keys=None,
                     diagnostic_thresholds=(.50, .50, .50), stop_event=None, feature_backend='cpu', candidate_pool=None):
    iterator = evaluation_steps(provider, source, records, model, thresholds, progress=progress,
        batch_size=batch_size, dev64_keys=dev64_keys, diagnostic_thresholds=diagnostic_thresholds,
        stop_event=stop_event, feature_backend=feature_backend, candidate_pool=candidate_pool)
    try:
        while True: next(iterator)
    except StopIteration as done: return done.value
    finally: iterator.close()


def evaluation_steps(provider, source, records, model, thresholds, *, progress=None, batch_size=256, dev64_keys=None,
                     diagnostic_thresholds=(.50, .50, .50), stop_event=None, feature_backend='cpu', candidate_pool=None,
                     window_iter=None, resume_state=None, start_window=0, verbose=True, yield_initial_state=False):
    """Four-way ablation ONE raw/V18/model probability pass, sparse exact counts."""
    model.column_sampling_workers = provider.workers
    populations = {"all": None}
    if dev64_keys is not None: populations["dev64"] = set(tuple(k) for k in dev64_keys)
    variants = (*VARIANTS, *("diagnostic_"+n for n in VARIANTS)) if diagnostic_thresholds is not None else VARIANTS
    bases = {p: Metrics() for p in populations}
    metrics = {p: {n: Metrics() for n in variants} for p in populations}
    quality = {p: {n: defaultdict(int) for n in variants} for p in populations}
    scenes = {p: defaultdict(lambda: {n: Metrics() for n in ("baseline", *variants)}) for p in populations}
    counts_windows = defaultdict(int)
    reference_metrics = {p: {} for p in populations}
    audits = defaultdict(int)
    scores = {p: {k: {"legal_query_voxels": 0, "sum": 0., "max": 0., "ge_0_50": 0, "ge_0_75": 0, "ge_0_95": 0}
        for k in ("generation_ADD", "refine_ADD", "refine_REMOVE")} for p in populations}
    state = dict(bases=bases, metrics=metrics, quality=quality, scenes=scenes, counts_windows=counts_windows,
        reference_metrics=reference_metrics, audits=audits, scores=scores)
    if resume_state is not None: restore_evaluation_state(state, resume_state)
    if yield_initial_state: yield {'event': 'evaluation_state_ready'}, state
    previous_end = time.perf_counter()
    iterator = prefetch_raw_columns(provider, source, records[start_window:]) if window_iter is None else window_iter
    for wi, (record, raw_window) in enumerate(iterator, start_window+1):
        if stop_event is not None and stop_event.is_set(): raise InterruptedError('evaluation stopped at window boundary')
        started = time.perf_counter()
        input_wait = started-previous_end if window_iter is None else 0.
        if verbose: print(f"evaluate_columns={wi}/{len(records)} four_way_shared_pass", flush=True)
        prep = (provider.prepare_columns(source, record, include_gt=True, raw_window=raw_window)
                if raw_window is not None else provider.prepare_columns(source, record, include_gt=True))
        reference_started = time.perf_counter()
        cache = getattr(provider,'frozen_metric_counts',None) if getattr(provider,'reference_enabled',False) else None
        reference_key = (id(source),str(record['scene_name']),str(record['t0_token']))
        cached_reference = cache.get(reference_key) if cache is not None else None
        if cache is not None:
            references = provider.reference_predictions(prep, record, skip_frozen=cached_reference is not None)
        else: references = provider.reference_predictions(prep, record) if hasattr(provider, 'reference_predictions') else {}
        reference_seconds = time.perf_counter()-reference_started
        new_reference_counts = {}
        active = [p for p, keys in populations.items() if keys is None or (str(record["scene_name"]), str(record["t0_token"])) in keys]
        for p in active: counts_windows[p] += 1
        for k, v in prep.source_audit.items(): audits[k] += v
        moving = raw_window.get('_evaluation_moving_support') if raw_window is not None else None
        if moving is None:
            moving = gt_moving_support_sequence(source.nusc, prep.window.t0_token, prep.window.future_tokens,
                tuple(.5*(h+1) for h in range(6)), grid=provider.pcfg.grid, workers=provider.workers)
        prediction_profile = {}; metric_started = time.perf_counter()
        planning = {h: candidate_pool.submit(candidate_plan, prep, h, provider.pcfg.grid, model.config) for h in REPORT}
        from real_motion.causal_column_sampling import ColumnHistoryIndex
        from real_motion.column_inference_pipeline import inference_gpu
        index = raw_window.get('_evaluation_history_index') if raw_window is not None else None
        if index is None:
            index = ColumnHistoryIndex(prep, provider.pcfg.grid)
            if window_iter is not None: raw_window['_evaluation_history_index'] = index
        prep.column_history_index = index
        gpu, resident = inference_gpu(provider.device, feature_backend, prep, provider.pcfg.grid, model.config)
        with resident:
          for ri, h in enumerate(REPORT):
            plan = planning[h].result()
            layout = sparse_layout(plan)
            if feature_backend == 'gpu':
                probability = predict_probabilities(model, prep, h, plan, provider.pcfg.grid, provider.device, batch_size,
                    feature_backend=feature_backend, history_index=index, gpu_sampler=gpu,
                    verify_features=wi <= 3 or wi % 128 == 0)
            else:
                probability = predict_probabilities(model, prep, h, plan, provider.pcfg.grid, provider.device, batch_size)
            prediction_profile[str(.5*(h+1))] = getattr(model, 'last_prediction_profile', {})
            for label, kind, action in (("generation_ADD", GENERATE, ADD), ("refine_ADD", REFINE, ADD),
                                         ("refine_REMOVE", REFINE, REMOVE)):
                valid = (plan.kind == kind)[:, None] & plan.legal[..., action]
                values = probability[..., action][valid]
                for population in active:
                    row = scores[population][label]
                    row["legal_query_voxels"] += len(values); row["sum"] += float(values.astype(np.float64).sum())
                    row["max"] = max(row["max"], float(values.max(initial=0.)))
                    for cutoff in (.50, .75, .95): row[f"ge_{cutoff:.2f}".replace('.', '_')] += int((values >= cutoff).sum())
            actions = actions_from_probabilities(plan, probability, thresholds)
            diagnostic = actions_from_probabilities(plan, probability, diagnostic_thresholds) if diagnostic_thresholds is not None else None
            gt = np.asarray(prep.raw["future_gt_occ"][h]); mask = moving[h][0]
            before = Metrics.counts(prep.baseline[h], gt, mask, FREE)
            reference_counts = {name: Metrics.counts(predictions[h],gt,mask,FREE) for name,predictions in references.items()}
            if cached_reference is not None: reference_counts['frozen_E14'] = cached_reference[h]
            if 'frozen_E14' in reference_counts: new_reference_counts[h] = reference_counts['frozen_E14']
            for p in active:
                bases[p].update(ri, counts=before); scenes[p][prep.window.scene_name]["baseline"].update(ri, counts=before)
                for name, counts in reference_counts.items():
                    reference_metrics[p].setdefault(name, Metrics()).update(ri, counts=counts)
            choices = [(n, eg, er, actions) for n, eg, er in (("generation", True, False), ("refine", False, True), ("joint", True, True))]
            if diagnostic is not None:
                choices += [("diagnostic_"+n, eg, er, diagnostic) for n, eg, er, _ in choices.copy()]
            for name, eg, er, acts in choices:
                ids, b, after = compose_sparse(plan, acts, enable_generation=eg, enable_refine=er, layout=layout)
                target = gt.reshape(-1)[ids]; mm = mask.reshape(-1)[ids]
                current = sparse_counts(before, b, after, target, mm, DYN)
                if wi == 1:
                    full = prep.baseline[h].copy(); full.reshape(-1)[ids] = after
                    if any(not np.array_equal(a, z) for a, z in zip(current, Metrics.counts(full, gt, mask, FREE))):
                        raise RuntimeError("add/remove sparse metrics differ from frozen full counts")
                edits = edit_quality(b, after, target)
                edits["source_layer_REMOVE_decisions"] = int(((acts == REMOVE)&(plan.actor[:, None] >= 0)).sum()) if er else 0
                for p in active:
                    metrics[p][name].update(ri, counts=current); scenes[p][prep.window.scene_name][name].update(ri, counts=current)
                    for k, v in edits.items(): quality[p][name][k] += v
        if cache is not None and cached_reference is None and len(new_reference_counts) == len(REPORT):
            cache[reference_key] = new_reference_counts
            if len(cache) > 1024: cache.popitem(last=False)
        compute_elapsed = time.perf_counter()-started
        elapsed = compute_elapsed+input_wait
        if verbose: print(f"evaluate_columns={wi}/{len(records)} complete seconds={elapsed:.3f}", flush=True)
        event = {"event": "evaluation", "window": wi, "windows": len(records), "seconds": elapsed,
                               "compute_seconds": compute_elapsed, "input_wait_seconds": input_wait,
                               "prediction_seconds_by_horizon": prediction_profile,
                               "candidate_inference_metrics_seconds": time.perf_counter()-metric_started,
                               "reference_seconds": reference_seconds, "frozen_reference_count_cache_hit": cached_reference is not None,
                               "prepare_seconds": getattr(provider, "last_prepare_seconds", {})}
        if progress: progress(event)
        previous_end = time.perf_counter()
        if window_iter is not None:
            # A suspended per-checkpoint generator must retain COUNTS only,
            # not each model's six dense baselines/owners/features across the
            # group boundary. Shared immutable raw geometry is caller-owned.
            del prep, planning, plan, probability, actions, diagnostic, layout, choices, acts
            del references, gt, mask, ids, b, after, target, mm, index, gpu, resident, raw_window
            if wi == 1: del full
        yield event, state
        if stop_event is not None and stop_event.is_set(): raise InterruptedError('evaluation stopped at window boundary')
    result = {}
    for p in populations:
        if p == "dev64" and counts_windows[p] != len(populations[p]): raise RuntimeError("incomplete frozen dev64 evaluation")
        result[p] = report_states(bases[p], metrics[p], quality[p], scenes[p])
        result[p].update(windows=counts_windows[p], scenes=len(scenes[p]), gate=acceptance_gate(result[p]["variants"]))
        if reference_metrics[p]:
            result[p]['reference_metrics'] = {k: v.compute() for k, v in reference_metrics[p].items()}
            result[p]['joint_vs_reference_pp'] = {k: delta(result[p]['variants']['joint']['metrics'], v.compute())
                                                  for k, v in reference_metrics[p].items()}
        for row in scores[p].values():
            total = row.pop("sum"); row["mean"] = total/row["legal_query_voxels"] if row["legal_query_voxels"] else None
        result[p]["confidence_audit"] = scores[p]
    result["source_audit"] = dict(audits)
    return result


def pack_evaluation_state(state):
    """Small metric/count state ONLY: no raw/GT volumes, poses or features."""
    fields = ('oi', 'ou', 'si', 'su', 'mi', 'mu')
    def metric(value): return {key: getattr(value, key).tolist() for key in fields}
    return dict(bases={p: metric(v) for p, v in state['bases'].items()},
        metrics={p: {n: metric(v) for n, v in rows.items()} for p, rows in state['metrics'].items()},
        scenes={p: {s: {n: metric(v) for n, v in rows.items()} for s, rows in scenes.items()}
            for p, scenes in state['scenes'].items()},
        reference_metrics={p: {n: metric(v) for n, v in rows.items()} for p, rows in state['reference_metrics'].items()},
        quality={p: {n: dict(v) for n, v in rows.items()} for p, rows in state['quality'].items()},
        counts_windows=dict(state['counts_windows']), audits=dict(state['audits']),
        scores={p: {k: dict(v) for k, v in rows.items()} for p, rows in state['scores'].items()})


def restore_evaluation_state(state, saved):
    def metric(value):
        result = Metrics()
        for key in ('oi', 'ou', 'si', 'su', 'mi', 'mu'):
            array = np.asarray(value[key])
            if array.dtype.kind not in 'iu' or array.shape != getattr(result, key).shape or np.any(array < 0):
                raise RuntimeError('invalid saved metric counts')
            getattr(result, key)[:] = array
        return result
    for key in ('bases', 'metrics', 'scenes', 'reference_metrics', 'quality', 'scores'):
        if set(saved[key]) != set(state[key]): raise RuntimeError('saved evaluation population changed')
    for p in state['bases']:
        state['bases'][p] = metric(saved['bases'][p])
        if set(saved['metrics'][p]) != set(state['metrics'][p]): raise RuntimeError('saved variants changed')
        for n in state['metrics'][p]:
            state['metrics'][p][n] = metric(saved['metrics'][p][n])
            state['quality'][p][n].update(saved['quality'][p][n])
        for s, rows in saved['scenes'][p].items():
            for n, value in rows.items(): state['scenes'][p][s][n] = metric(value)
        state['reference_metrics'][p].update({n: metric(v) for n, v in saved['reference_metrics'][p].items()})
        state['scores'][p] = saved['scores'][p]
    state['counts_windows'].update(saved['counts_windows']); state['audits'].update(saved['audits'])


def forecast_columns(provider, source, record, model, thresholds, batch_size=256):
    """Six horizons, causal deployment: never requests future occupancy."""
    model.column_sampling_workers = provider.workers
    prep = provider.prepare_columns(source, record, include_gt=False)
    predictions = []
    for h in range(6):
        plan = candidate_plan(prep, h, provider.pcfg.grid, model.config)
        probability = predict_probabilities(model, prep, h, plan, provider.pcfg.grid, provider.device, batch_size)
        actions = actions_from_probabilities(plan, probability, thresholds)
        ids, _, after = compose_sparse(plan, actions)
        out = prep.baseline[h].copy(); out.reshape(-1)[ids] = after; predictions.append(out)
    return prep.window, predictions


CALIBRATION_LEVELS = (.50, .75, .95, None)


def safe_metrics(row, baseline):
    """No single aggregate may hide a degraded horizon or Moving metric."""
    d = delta(row, baseline)
    values = [d[k] for k in ("IoU", "mIoU", "MovingMacro", "MovingMicro")]
    values += [r[k] for r in d["per_horizon"].values() for k in ("IoU", "mIoU", "MovingMacro", "MovingMicro")]
    return all(v is not None and np.isfinite(v) and v >= -1e-10 for v in values)


def calibrate_columns(provider, source, records, model, *, progress=None, batch_size=256, stop_event=None,
                      feature_backend='cpu'):
    if feature_backend not in ('cpu', 'gpu'): raise ValueError('invalid calibration feature backend')
    with ThreadPoolExecutor(max_workers=min(3,provider.workers)) as pool:
        return _calibrate_columns(provider, source, records, model, progress=progress,batch_size=batch_size,
            stop_event=stop_event,feature_backend=feature_backend,candidate_pool=pool)


def _calibrate_columns(provider, source, records, model, *, progress=None, batch_size=256, stop_event=None,
                       feature_backend='cpu',candidate_pool=None):
    """ONE held-out TRAIN pass, predeclared finite grid, NO dev tuning.

    4 generation settings, 16 refine settings, 64 joint settings share the same
    probabilities/raw counts. No forward/render repeats per threshold setting.
    Selection evaluates actual composed voxel metrics, not calibrated-score
    confidence or correct-minus-wrong counts assumed to equal mIoU.
    """
    import itertools
    model.column_sampling_workers = provider.workers
    levels = CALIBRATION_LEVELS
    tuples = list(itertools.product(levels, repeat=3))
    jobs = [("g", (g, None, None), True, False) for g in levels]
    jobs += [("r", (None, a, r), False, True) for a, r in itertools.product(levels, repeat=2)]
    jobs += [("j", t, True, True) for t in tuples]
    base = Metrics(); metrics = [Metrics() for _ in jobs]; quality = [defaultdict(int) for _ in jobs]
    previous_end = time.perf_counter()
    for wi, (record, raw_window) in enumerate(prefetch_raw_columns(provider, source, records), 1):
        if stop_event is not None and stop_event.is_set(): raise InterruptedError('TRAIN calibration stopped at window boundary')
        started = time.perf_counter()
        input_wait = started-previous_end
        print(f"calibrate_TRAIN={wi}/{len(records)} thresholds_fixed_grid_dev_unseen", flush=True)
        prep = (provider.prepare_columns(source, record, include_gt=True, raw_window=raw_window)
                if raw_window is not None else provider.prepare_columns(source, record, include_gt=True))
        moving = gt_moving_support_sequence(source.nusc, prep.window.t0_token, prep.window.future_tokens,
                    tuple(.5*(h+1) for h in range(6)), grid=provider.pcfg.grid, workers=provider.workers)
        planning = {h:candidate_pool.submit(candidate_plan,prep,h,provider.pcfg.grid,model.config) for h in REPORT}
        from real_motion.causal_column_sampling import ColumnHistoryIndex
        from real_motion.column_inference_pipeline import inference_gpu
        index = ColumnHistoryIndex(prep,provider.pcfg.grid);prep.column_history_index = index
        gpu,resident = inference_gpu(provider.device,feature_backend,prep,provider.pcfg.grid,model.config)
        with resident:
            for ri, h in enumerate(REPORT):
                plan = planning[h].result()
                layout = sparse_layout(plan)
                if feature_backend == 'gpu':
                    p = predict_probabilities(model, prep, h, plan, provider.pcfg.grid, provider.device, batch_size,
                        feature_backend=feature_backend,history_index=index,gpu_sampler=gpu,verify_features=wi <= 3 or wi % 128 == 0)
                else: p = predict_probabilities(model, prep, h, plan, provider.pcfg.grid, provider.device, batch_size)
                gt, mask = np.asarray(prep.raw["future_gt_occ"][h]), moving[h][0]
                before = Metrics.counts(prep.baseline[h], gt, mask, FREE); base.update(ri, counts=before)
                for k, (_, gates, eg, er) in enumerate(jobs):
                    acts = actions_from_probabilities(plan, p, gates)
                    ids, b, after = compose_sparse(plan, acts, enable_generation=eg, enable_refine=er, layout=layout)
                    target, mm = gt.reshape(-1)[ids], mask.reshape(-1)[ids]
                    metrics[k].update(ri, counts=sparse_counts(before, b, after, target, mm, DYN))
                    for name, value in edit_quality(b, after, target).items(): quality[k][name] += value
        compute_elapsed = time.perf_counter()-started
        elapsed = compute_elapsed+input_wait
        print(f"calibrate_TRAIN={wi}/{len(records)} complete seconds={elapsed:.3f}", flush=True)
        if progress: progress({"event": "TRAIN_calibration", "window": wi, "windows": len(records), "seconds": elapsed,
                               "compute_seconds": compute_elapsed, "input_wait_seconds": input_wait,
                               "prepare_seconds": getattr(provider, "last_prepare_seconds", {})})
        previous_end = time.perf_counter()
        if stop_event is not None and stop_event.is_set(): raise InterruptedError('TRAIN calibration stopped at window boundary')
    baseline = base.compute()
    lookup = {(task, gates): (m.compute(), dict(q)) for (task, gates, _, _), m, q in zip(jobs, metrics, quality)}
    selected = (None, None, None); best_score = (False, 0., -np.inf); candidates = []
    for gates in tuples:
        g = lookup[("g", (gates[0], None, None))]
        r = lookup[("r", (None, gates[1], gates[2]))]
        j = lookup[("j", gates)]
        safe = all(safe_metrics(m, baseline) for m, _ in (g, r, j))
        gain = j[0]["mIoU"]-baseline["mIoU"]
        both = g[1].get("added_semantic_tp", 0) > 0 and r[1].get("corrected", 0) > 0
        conservative = sum(1.01 if t is None else t for t in gates)
        score = (both, gain, conservative)
        candidates.append({"thresholds": gates, "safe_all_variants_horizons": safe, "both_have_correct_edits": both,
                           "joint_delta_mIoU_pp": gain, "joint_quality": j[1]})
        if safe and np.isfinite(gain) and gain > 1e-10 and score > best_score:
            selected, best_score = gates, score
    return selected, {"population": "held_out_TRAIN_scenes_not_dev", "baseline": baseline,
        "levels": levels, "selected_thresholds": selected, "selection_uses_actual_layered_metrics": True,
        "both_branches_have_safe_correct_edits": bool(best_score[0]), "candidates": candidates}
