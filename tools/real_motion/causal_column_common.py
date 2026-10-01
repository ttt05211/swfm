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
    with ThreadPoolExecutor(max_workers=min(workers, 5)) as pool:
        frames = list(pool.map(lambda f: extract_instances_cropped_exact(raw_history[f], poses[f], grid=grid, cfg=strong), range(5)))
    frames.append(state["current"])
    links, audit = associate_backwards(frames, state["current"], state["velocities"], dt=.5)
    points = [[rigid_source_points_world(c["voxel_indices"], pose, grid=grid) for c in frame]
              for frame, pose in zip(frames, poses)]
    registrations = [[None]*6 for _ in state["current"]]
    for i, comp in enumerate(state["current"]):
        registrations[i][5] = (np.eye(4), np.asarray(comp["voxel_indices"], np.int64))
        for f, j in enumerate(links[i][:5]):
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


class FrozenColumns(FrozenXYV18):
    def load_raw_columns(self, source, record, *, include_gt):
        return load_nuscenes_window_raw(source, window_from_record(record), self.pcfg,
            include_gt=include_gt, io_workers=min(self.workers, 4))

    def prepare_columns(self, source, record, *, include_gt, raw_window=None, outputs=None):
        started = time.perf_counter()
        if raw_window is None and outputs is None:
            window, raw, state, outputs = super().prepare(source, record, include_gt=include_gt)
        else:
            window = window_from_record(record)
            raw = self.load_raw_columns(source, record, include_gt=include_gt) if raw_window is None else raw_window
            if not include_gt and raw.get('future_gt_occ') is not None:
                raise RuntimeError('causal deployment must not request future occupancy')
            state = runtime._prepare_record(record, source, self.pcfg, self.strong, self.device, raw_window=raw)
            centers = world_points_to_t0(np.asarray([c['centroid_world'] for c in state['current']]).reshape(-1, 3),
                                        state['current_pose'])[:, :2]
            if not np.allclose(centers, numpy(record['source_centroid_xy_t0_m']), rtol=0, atol=2e-4):
                raise RuntimeError('cache/actual source-centre identity mismatch')
            outputs = self.encode_record(record) if outputs is None else outputs
        prepared_at = time.perf_counter()
        baseline, owners, fallbacks, components, targets, yaws = render_column_layers(state, record, outputs, self.pcfg.grid)
        if not getattr(self, "columns_checked", False):
            runtime._stage_gpu_inputs(state, self.device)
            try:
                runtime._exactness_check(self.model, state, self.pcfg, self.strong, self.device)
                reference = runtime._forecast_once(self.model, state, self.pcfg, self.strong, self.device, precomputed_out=outputs)
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
                               registrations, footprints, memory, audit, outputs)


def render_column_layers(state, record, outputs, grid):
    """Model-dependent geometry ONLY; immutable history/registration can be reused."""
    res, yaw = numpy(outputs["residual_xy_m"]), numpy(outputs["yaw_delta_rad"])
    from real_motion.v18_two_wheel_diagnostic import renderer_yaw_delta
    baseline, owners, fallbacks, components, targets, yaws = [], [], [], [], [], []
    for h in range(6):
        centers = [runtime._target_world_from_xy_cached(numpy(record["anchors_xy_t0_m"])[i, h]+res[i, h],
            state["source_z_t0"][i], state["current_pose"]) for i in range(len(state["current"]))]
        yy = [renderer_yaw_delta(int(c["class_id"]), yaw[i, h], zero_two_wheel_yaw=False) for i, c in enumerate(state["current"])]
        layers = runtime._rasterize_all_sources_horizon(state["current"], state["source_world_points"], state["source_rel_xy"],
                                                       centers, yy, state["world_to_future"][h], grid)
        background = compose_component_replacements_fast_exact(state["anchors"][h], state["baseline_by_hi"][h], [],
            dynamic_class_ids=DYN, free_label=FREE, grid=grid,
            precomputed_clear_flat_indices=state["baseline_clear_flat_by_hi"][h])
        b, own, fall = component_layers(background, layers)
        baseline.append(b); owners.append(own); fallbacks.append(fall); components.append(layers); targets.append(centers); yaws.append(yy)
    return baseline, owners, fallbacks, components, targets, yaws


def candidate_plan(prepared, h, grid, config=ColumnConfig()):
    """GT-free candidates, complete population, no evaluation top-positive cap."""
    config.validate()
    b, m, footprint = prepared.baseline[h], prepared.memory[h], prepared.footprints[h]
    shape = tuple(grid.shape_hwd); z = shape[2]
    if z != config.z_bins: raise RuntimeError("Z lattice/checkpoint mismatch")
    if not np.isclose(grid.voxel_size[0], grid.voxel_size[1], rtol=0, atol=1e-10):
        raise RuntimeError("frontier distance contract requires an isotropic XY lattice")
    frontier = build_surface_frontier(m, footprint, class_ids=(11, 13), widths_m=(config.entry_radius_m,),
                                     voxel_size_xy_m=grid.voxel_size[0], free_label=FREE)
    counts = np.stack([(m == c).sum(2) for c in (11, 13)], -1)
    dominant = np.asarray((11, 13), np.uint8)[counts.argmax(-1)]
    rows = []
    def append(xy, kind, actor, classes, allowed_z, ax, ay, age):
        if not len(xy): return
        flat = ((xy[:, 0:1]*shape[1]+xy[:, 1:2])*z+np.arange(z)).astype(np.int64)
        base = b.reshape(-1)[flat].copy(); fall = base.copy()
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
        context = np.zeros((len(xy), CONTEXT_DIM), np.float32)
        context[:, :2] = (xy+.5)/np.asarray(shape[:2])*2-1
        context[:, 2] = .5*(h+1)/3
        rel = np.linalg.inv(prepared.state["current_pose"])@prepared.raw["future_poses"][h]
        context[:, 3:6] = (rel[0, 3]/40, rel[1, 3]/40, np.arctan2(rel[1, 0], rel[0, 0])/np.pi)
        context[:, 6:8] = (xy-np.column_stack((ax, ay)))*np.asarray(grid.voxel_size[:2])/config.entry_radius_m
        context[:, 8] = np.linalg.norm(context[:, 6:8], axis=1)
        context[:, 9] = age/2.5
        context[:, 10] = legal[..., ADD].mean(1)
        context[:, 11] = actor >= 0
        rows.append(ColumnPlan(xy.astype(np.int32), np.full(len(xy), kind, np.uint8), np.full(len(xy), actor, np.int32),
                               classes.astype(np.uint8), flat, base, fall, legal, context,
                               np.column_stack((ax, ay)).astype(np.int32) if kind == GENERATE else xy.astype(np.int32)))
    entry = frontier.causal_by_width[config.entry_radius_m] & (b == FREE).any(2)
    xy = np.argwhere(entry)
    ax, ay = frontier.nearest_x[entry], frontier.nearest_y[entry]
    append(xy, GENERATE, -3, dominant[ax, ay], np.ones((len(xy), z), bool), ax, ay, 0.)
    # Historical road/sidewalk residuals ONLY inside visited grid; not full-scene
    # refinement. Static occupied edits never touch dynamic/source-owned labels.
    historical = np.isin(m, (11, 13))
    mismatch = (historical & (m != b)).any(2)
    static = footprint & mismatch & historical.any(2)
    xy = np.argwhere(static)
    cls = dominant[static]; mask = historical[static]
    mask = binary_dilation(mask, structure=np.ones((1, 3)), iterations=1)
    append(xy, REFINE, -2, cls, mask, xy[:, 0], xy[:, 1], 0.)
    for i, comp in enumerate(prepared.state["current"]):
        registered = prepared.registrations[i]
        if not any(r is not None for r in registered[:5]): continue
        all_flat = [np.ravel_multi_index(prepared.components[h][i].voxel_indices.T, shape)]
        for f, reg in enumerate(registered[:5]):
            if reg is None: continue
            points = rigid_source_points_world(reg[1], prepared.raw["history_poses"][f], grid=grid)
            aligned = transform_points(points, reg[0])
            moved = planar_move(aligned, comp["centroid_world"], prepared.targets[h][i], prepared.yaws[h][i])
            ids, _ = raster_flat(moved, prepared.state["world_to_future"][h],
                                (grid.x_min, grid.y_min, grid.z_min), grid.voxel_size, shape)
            all_flat.append(ids)
        ids = np.unique(np.concatenate(all_flat))
        if not len(ids): continue
        xyz = np.column_stack(np.unravel_index(ids, shape))
        support = np.zeros(shape[:2], bool); support[xyz[:, 0], xyz[:, 1]] = True
        support = binary_dilation(support, iterations=config.boundary_padding_cells)
        xy = np.argwhere(support)
        allowed = np.broadcast_to((np.arange(z) >= max(0, xyz[:, 2].min()-1))
                                  & (np.arange(z) <= min(z-1, xyz[:, 2].max()+1)), (len(xy), z)).copy()
        age = .5*(5-min(f for f, reg in enumerate(registered) if reg is not None))
        center = xy.mean(0)
        append(xy, REFINE, i, np.full(len(xy), int(comp["class_id"]), np.uint8), allowed,
               np.full(len(xy), center[0]), np.full(len(xy), center[1]), age)
    if rows:
        plan = ColumnPlan(**{k: np.concatenate([getattr(r, k) for r in rows]) for k in vars(rows[0])})
    else:
        plan = ColumnPlan(np.empty((0, 2), np.int32), np.empty(0, np.uint8), np.empty(0, np.int32), np.empty(0, np.uint8),
            np.empty((0, z), np.int64), np.empty((0, z), np.uint8), np.empty((0, z), np.uint8),
            np.empty((0, z, 3), bool), np.empty((0, CONTEXT_DIM), np.float32))
    plan.validate(); return plan


def sample_column_features(prepared, h, plan, grid, config=ColumnConfig()):
    """Inverse sample LOCAL patches, not 36 full dense feature volumes.

    Flags bit0 = lidar visibility; bit1 = assigned static class/source membership.
    Historical semantic input follows V18 full occupancy protocol. Unknown=18
    means outside grid/unmatched source, NEVER an observed free label.
    """
    n, p, z = len(plan), config.patch, config.z_bins
    hist = np.full((n, 6, p, p, z), UNKNOWN, np.uint8)
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
        for f in range(6):
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


def predict_probabilities(model, prepared, h, plan, grid, device, batch_size=256):
    model.eval(); parts = []
    started = time.perf_counter()
    sampler = ColumnFeatureSampler(prepared, h, plan, grid, model.config, pose_motion,
                                   workers=getattr(model, 'column_sampling_workers', 1))
    mapped_at = time.perf_counter(); sampled_seconds = 0.
    with torch.inference_mode():
        for start in range(0, len(plan), batch_size):
            small = plan.subset(slice(start, start+batch_size))
            sampled_at = time.perf_counter()
            arrays = sampler.sample(small, sample_column_features)
            sampled_seconds += time.perf_counter()-sampled_at
            b = {k: torch.as_tensor(v, device=device) for k, v in arrays.items()}
            legal = torch.as_tensor(small.legal, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                if hasattr(model, 'source_features_for'):
                    b['source_features'] = model.source_features_for(prepared, h, small, device)
                g, r = model(**b)
            parts.append(model.calibrated_probabilities(g, r, b["kind"], legal).cpu().numpy())
    model.last_prediction_profile = {'queries': len(plan), 'cache_mib': sampler.cache_bytes/2**20,
        'inverse_map_seconds': mapped_at-started, 'patch_gather_seconds': sampled_seconds,
        'network_transfer_and_other_seconds': time.perf_counter()-mapped_at-sampled_seconds}
    return np.concatenate(parts) if parts else np.empty((*plan.base.shape, 3), np.float32)


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
                     diagnostic_thresholds=(.50, .50, .50)):
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
    previous_end = time.perf_counter()
    for wi, (record, raw_window) in enumerate(prefetch_raw_columns(provider, source, records), 1):
        started = time.perf_counter()
        input_wait = started-previous_end
        print(f"evaluate_columns={wi}/{len(records)} four_way_shared_pass", flush=True)
        prep = (provider.prepare_columns(source, record, include_gt=True, raw_window=raw_window)
                if raw_window is not None else provider.prepare_columns(source, record, include_gt=True))
        references = provider.reference_predictions(prep, record) if hasattr(provider, 'reference_predictions') else {}
        active = [p for p, keys in populations.items() if keys is None or (str(record["scene_name"]), str(record["t0_token"])) in keys]
        for p in active: counts_windows[p] += 1
        for k, v in prep.source_audit.items(): audits[k] += v
        moving = gt_moving_support_sequence(source.nusc, prep.window.t0_token, prep.window.future_tokens,
                    tuple(.5*(h+1) for h in range(6)), grid=provider.pcfg.grid, workers=provider.workers)
        prediction_profile = {}
        for ri, h in enumerate(REPORT):
            plan = candidate_plan(prep, h, provider.pcfg.grid, model.config)
            layout = sparse_layout(plan)
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
            for p in active:
                bases[p].update(ri, counts=before); scenes[p][prep.window.scene_name]["baseline"].update(ri, counts=before)
                for name, predictions in references.items():
                    reference_metrics[p].setdefault(name, Metrics()).update(ri, counts=Metrics.counts(predictions[h], gt, mask, FREE))
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
        compute_elapsed = time.perf_counter()-started
        elapsed = compute_elapsed+input_wait
        print(f"evaluate_columns={wi}/{len(records)} complete seconds={elapsed:.3f}", flush=True)
        if progress: progress({"event": "evaluation", "window": wi, "windows": len(records), "seconds": elapsed,
                               "compute_seconds": compute_elapsed, "input_wait_seconds": input_wait,
                               "prediction_seconds_by_horizon": prediction_profile,
                               "prepare_seconds": getattr(provider, "last_prepare_seconds", {})})
        previous_end = time.perf_counter()
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


def calibrate_columns(provider, source, records, model, *, progress=None, batch_size=256):
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
        started = time.perf_counter()
        input_wait = started-previous_end
        print(f"calibrate_TRAIN={wi}/{len(records)} thresholds_fixed_grid_dev_unseen", flush=True)
        prep = (provider.prepare_columns(source, record, include_gt=True, raw_window=raw_window)
                if raw_window is not None else provider.prepare_columns(source, record, include_gt=True))
        moving = gt_moving_support_sequence(source.nusc, prep.window.t0_token, prep.window.future_tokens,
                    tuple(.5*(h+1) for h in range(6)), grid=provider.pcfg.grid, workers=provider.workers)
        for ri, h in enumerate(REPORT):
            plan = candidate_plan(prep, h, provider.pcfg.grid, model.config)
            layout = sparse_layout(plan)
            p = predict_probabilities(model, prep, h, plan, provider.pcfg.grid, provider.device, batch_size)
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
