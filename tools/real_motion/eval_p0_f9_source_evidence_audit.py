#!/usr/bin/env python3
"""One raw-data/V18 pass: motion, historical geometry and causal alignment.

No training, prototype bank or persistent dense cache. Future labels are used
only for metrics and explicitly named GT diagnostics. See the contract document.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.nuscenes_adapter import NuScenesWindowSource, gt_moving_support_sequence
from real_motion.prepared import load_nuscenes_window_raw
from real_motion.rigid_transport import rigid_source_points_world
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.runtime_fastpath import extract_instances_cropped_exact, compose_component_replacements_fast_exact
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.source_evidence_audit import (
    associate_backwards, choose_candidate_gt_assisted, edit_quality, metric_count_delta,
    planar_move, protected_add_indices, raster_flat, register_history_shape, route_diagnostic,
)
from real_motion.v18_two_wheel_diagnostic import renderer_yaw_delta
from real_motion.v22_causal_emergence import build_future_static_memory_only
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
from tools.real_motion import eval_p0_f9_v18_full_validation as full
from tools.real_motion import eval_p0_f9_v18_se2 as base
from tools.real_motion.diagnose_p0_f9_v17_yaw_oracle import _match_components
from tools.real_motion.diagnose_p0_f9_v19_innovation_decomposition import _ann_map
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import (
    Metrics, align_records, assert_forward_exact, delta, load_manifest, sha256,
    validate_clean_e14_checkpoint,
)
from tools.real_motion.train_p0_f9_v18_se2_clean import PROTOCOL as CLEAN_PROTOCOL

PROTOCOL = "p0_f9_source_evidence_audit_v1"
REPORT = (1, 3, 5)
DYN = tuple(int(c) for c in DYNAMIC_CLASS_IDS)
VARIANTS = (
    "V18_BASE", "T0_GT_MOTION",
    "HISTORY_GT_ALIGN_GT_MOTION", "HISTORY_GT_ALIGN_PRED_MOTION",
    "HISTORY_CAUSAL_ALIGN_GT_MOTION", "HISTORY_CAUSAL_ALIGN_PRED_MOTION",
    "HISTORY_COMMON_GT_ALIGN_GT_MOTION", "HISTORY_COMMON_CAUSAL_ALIGN_GT_MOTION",
    "HISTORY_GT_SELECT_GT_MOTION", "HISTORY_GT_SELECT_PRED_MOTION",
    "HISTORY_CAUSAL_VOXEL_GT_FILTER", "STATIC_MEMORY",
    "STATIC_PATCH_GT_SELECT", "STATIC_VOXEL_GT_FILTER",
)


def _numpy(x):
    if torch.is_tensor(x):
        x = x.detach()
        if x.dtype == torch.bfloat16:
            x = x.float()
        return x.cpu().numpy()
    return np.asarray(x)


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _annotation_align(points, previous, current, yaw_enabled):
    yaw = float(current["yaw_world"] - previous["yaw_world"]) if yaw_enabled else 0.0
    return planar_move(points, previous["center_world"], current["center_world"], yaw)


def _registration_candidates(frames, points, current, velocity, gt_matches, ann_maps, workers, dt):
    links, audit = associate_backwards(frames, current, velocity, dt=dt)
    gt_links = {str(a["instance_token"]): i for i, a in gt_matches[-1].items()}
    jobs = [(i, f, j) for i, rows in enumerate(links) for f, j in enumerate(rows[:-1]) if j is not None]

    def register(job):
        i, f, j = job
        result = register_history_shape(points[f][j], points[-1][i],
                                        allow_yaw=int(current[i]["class_id"]) != 7)
        return i, f, j, result

    causal = [[] for _ in current]
    exact = [[] for _ in current]
    common_exact = [[] for _ in current]
    common_causal = [[] for _ in current]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, f, j, result in pool.map(register, jobs):
            audit["registration_accepted" if result.accepted else "registration_rejected"] = (
                audit.get("registration_accepted" if result.accepted else "registration_rejected", 0) + 1)
            if result.accepted:
                causal[i].append(result.points)
            a0, af = gt_matches[-1].get(i), gt_matches[f].get(j)
            if a0 is not None and af is not None:
                correct = str(a0["instance_token"]) == str(af["instance_token"])
                audit["certified_causal_identity_correct" if correct else "certified_causal_identity_wrong"] = (
                    audit.get("certified_causal_identity_correct" if correct else "certified_causal_identity_wrong", 0) + 1)
                if correct and result.accepted:
                    common_causal[i].append(result.points)
                    common_exact[i].append(_annotation_align(points[f][j], af, a0,
                                                            int(current[i]["class_id"]) != 7))
            else:
                audit["causal_identity_uncertified"] = audit.get("causal_identity_uncertified", 0) + 1
    for f in range(5):
        for j, annotation in gt_matches[f].items():
            token = str(annotation["instance_token"])
            if token not in gt_links:
                continue
            i = gt_links[token]
            exact[i].append(_annotation_align(points[f][j], annotation, ann_maps[-1][token],
                                             int(current[i]["class_id"]) != 7))
    audit["current_sources"] = len(current)
    audit["gt_linked_current_sources"] = len(gt_links)
    audit["causal_sources_with_history"] = sum(bool(x) for x in causal)
    audit["gt_sources_with_history"] = sum(bool(x) for x in exact)
    audit["history_only_annotation_tracks"] = len({str(a["instance_token"])
        for row in gt_matches[:-1] for a in row.values()} - set(gt_links))
    return {"CAUSAL_ALIGN": causal, "GT_ALIGN": exact,
            "COMMON_GT_ALIGN": common_exact, "COMMON_CAUSAL_ALIGN": common_causal}, audit


def _render_current(state, pcfg, centers, yaw, *, audit=None, condition=None):
    """Unmodified observed t0 geometry, complete A1 CLEAR/WRITE (not add-only)."""
    predictions, components = [], []
    for h in REPORT:
        if audit is not None:
            for i, comp in enumerate(state["current"]):
                moved = planar_move(state["source_world_points"][i], comp["centroid_world"],
                                    centers[h][i], yaw[h][i])
                _, oob = raster_flat(moved, state["world_to_future"][h],
                    (pcfg.grid.x_min, pcfg.grid.y_min, pcfg.grid.z_min),
                    pcfg.grid.voxel_size, pcfg.grid.shape_hwd)
                audit[f"{condition}_oob_points"] += oob
        repl = runtime._rasterize_all_sources_horizon(
            state["current"], state["source_world_points"], state["source_rel_xy"],
            centers[h], yaw[h], state["world_to_future"][h], pcfg.grid)
        predictions.append(compose_component_replacements_fast_exact(
            state["anchors"][h], state["baseline_by_hi"][h], repl,
            dynamic_class_ids=DYN, free_label=pcfg.free_label, grid=pcfg.grid,
            precomputed_clear_flat_indices=state["baseline_clear_flat_by_hi"][h]))
        components.append(repl)
    return predictions, components


def _history_predictions(start, families, state, grid, centers_pred, yaw_pred,
                         centers_gt, yaw_gt, truth, audit):
    origin = (grid.x_min, grid.y_min, grid.z_min)
    predictions = {name: start["T0_GT_MOTION" if name.endswith("GT_MOTION") else "V18_BASE"].copy()
                   for name in VARIANTS if name.startswith("HISTORY_")}
    for i, comp in enumerate(state["current"]):
        center = np.asarray(comp["centroid_world"], float)
        cid = int(comp["class_id"])
        by_condition = {}
        render_cache = {}
        for alignment, sources in families.items():
            candidates = sources[i]
            for motion, target, yaw in (("GT_MOTION", centers_gt[i], yaw_gt[i]),
                                        ("PRED_MOTION", centers_pred[i], yaw_pred[i])):
                name = f"HISTORY_{alignment}_{motion}"
                if name not in predictions:
                    continue
                rendered = []
                for points in candidates:
                    key = (id(points), motion)
                    if key not in render_cache:
                        moved = planar_move(points, center, target, yaw)
                        render_cache[key] = raster_flat(moved, state["world_to_future_current"], origin,
                                                       grid.voxel_size, grid.shape_hwd)
                    idx, oob = render_cache[key]
                    audit[f"{alignment}_{motion}_oob_points"] += oob
                    rendered.append(idx)
                by_condition[alignment, motion] = rendered
                if rendered:
                    protected_add_indices(predictions[name], np.unique(np.concatenate(rendered)), cid, copy=False)
        for motion in ("GT_MOTION", "PRED_MOTION"):
            name = f"HISTORY_GT_SELECT_{motion}"
            best, utility = choose_candidate_gt_assisted(predictions[name], truth,
                                    by_condition["CAUSAL_ALIGN", motion], cid)
            protected_add_indices(predictions[name], best, cid, copy=False)
            audit[f"candidate_select_{motion}_accepted"] += int(utility > 0)
        rendered = by_condition["CAUSAL_ALIGN", "PRED_MOTION"]
        if rendered:
            ids = np.unique(np.concatenate(rendered))
            ids = ids[truth.reshape(-1)[ids] == cid]
            protected_add_indices(predictions["HISTORY_CAUSAL_VOXEL_GT_FILTER"], ids, cid, copy=False)
    return predictions


def _static_predictions(baseline, memory, gt, patch_cells):
    free = baseline == 17
    eligible = free & (memory != 17)
    direct = baseline.copy()
    direct[eligible] = memory[eligible]
    filtered = baseline.copy()
    correct = eligible & (memory == gt)
    filtered[correct] = memory[correct]
    # Whole fixed BEV patch, including ALL Z and all static classes. No GT shape.
    selected = baseline.copy()
    x, y, _ = baseline.shape
    nx, ny = (x + patch_cells - 1) // patch_cells, (y + patch_cells - 1) // patch_cells
    ix, iy, iz = np.nonzero(eligible)
    patch_ids = (ix // patch_cells) * ny + iy // patch_cells
    utility = np.where(memory[ix, iy, iz] == gt[ix, iy, iz], 1, -1)
    scores = np.bincount(patch_ids, weights=utility, minlength=nx * ny)
    accepted = scores[patch_ids] > 0
    selected[ix[accepted], iy[accepted], iz[accepted]] = memory[ix[accepted], iy[accepted], iz[accepted]]
    return {"STATIC_MEMORY": direct, "STATIC_PATCH_GT_SELECT": selected,
            "STATIC_VOXEL_GT_FILTER": filtered}


def _merge(dst, src):
    for k, v in src.items():
        dst[k] += int(v)


def _scene_delta(scene_states, variant):
    values = {s: st[variant].compute()["mIoU"] - st["V18_BASE"].compute()["mIoU"]
              for s, st in scene_states.items()}
    finite = [v for v in values.values() if np.isfinite(v)]
    return {"scenes": len(values), "positive": sum(v > 0 for v in finite),
            "negative": sum(v < 0 for v in finite), "zero": sum(v == 0 for v in finite),
            "by_scene": values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    for name in ("val-cache", "population-manifest", "checkpoint", "expected-checkpoint-sha256",
                 "dataroot", "info-pkl", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument("--exactness-windows", type=int, default=1)
    parser.add_argument("--static-patch-cells", type=int, default=4)
    args = parser.parse_args()
    started = time.perf_counter()
    if args.cpu_workers < 1 or args.exactness_windows < 1 or args.static_patch_cells < 1:
        parser.error("workers, exactness windows and patch cells must be positive")
    for name in ("config", "val_cache", "population_manifest", "checkpoint", "info_pkl"):
        if not str(getattr(args, name) or "").strip() or not Path(getattr(args, name)).is_file():
            parser.error(f"--{name.replace('_', '-')} must name an existing file")
    if not Path(args.dataroot).is_dir():
        parser.error("--dataroot must name an existing directory")
    destination = Path(args.output)
    if destination.exists():
        parser.error("output already exists; choose a new run path (no silent overwrite)")
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    pcfg = make_prepare_config(load_runtime_config(args.config, args.override))
    if pcfg.free_label != 17:
        raise RuntimeError("frozen metric free label must be 17")
    manifest, keys, _ = load_manifest(args.population_manifest)
    print(f"inputs validated; loading V18 cache for {len(keys)} frozen keys", flush=True)
    _, records = base.load_cache(args.val_cache)
    records = align_records(records, keys)
    checkpoint, model, _ = full._load_model(args.checkpoint, CLEAN_PROTOCOL, device)
    checkpoint_sha = validate_clean_e14_checkpoint(checkpoint, args.checkpoint, args.expected_checkpoint_sha256)
    source = NuScenesWindowSource(args.dataroot, info_pkl=args.info_pkl, verbose=False)
    strong = StrongW2DetConfig(free_label=pcfg.free_label)
    states = {name: Metrics() for name in VARIANTS}
    scenes = defaultdict(lambda: {name: Metrics() for name in VARIANTS})
    edits = {name: defaultdict(int) for name in VARIANTS}
    audits, timings, error_mass = defaultdict(int), defaultdict(float), defaultdict(int)
    per_window = []
    timings["input_load_and_checkpoint_validation"] = time.perf_counter() - started
    destination.parent.mkdir(parents=True, exist_ok=True)
    progress_path = destination.with_suffix(".progress.jsonl")
    # Refuse old progress too, so partial attempts remain inspectable.
    with progress_path.open("x", encoding="utf-8") as progress, ThreadPoolExecutor(max_workers=args.cpu_workers) as pool:
        for wi, record in enumerate(records, 1):
            tick = time.perf_counter()
            print(f"source_evidence {wi}/{len(records)} stage=raw_load", flush=True)
            window = window_from_record(record)
            raw = load_nuscenes_window_raw(source, window, pcfg, include_gt=True, io_workers=args.cpu_workers)
            timings["raw_load"] += time.perf_counter() - tick
            tick = time.perf_counter()
            state = runtime._prepare_record(record, source, pcfg, strong, device, raw_window=raw)
            runtime._stage_gpu_inputs(state, device)
            try:
                if wi <= args.exactness_windows:
                    assert_forward_exact(model, state, device)
                    runtime._exactness_check(model, state, pcfg, strong, device)
                outputs = runtime._model_forward(model, state["gpu"], device)
                baseline_all = runtime._forecast_once(model, state, pcfg, strong, device, precomputed_out=outputs)
            finally:
                runtime._release_gpu_inputs(state)
            residual, predicted_yaw = _numpy(outputs["residual_xy_m"]), _numpy(outputs["yaw_delta_rad"])
            cached = {k: _numpy(record[k]) for k in ("anchors_xy_t0_m", "se2_target_valid",
                      "target_source_residual_xy_m", "yaw_label_valid", "target_yaw_rad")}
            pred_centers, gt_centers, pred_yaws, gt_yaws = {}, {}, {}, {}
            for h in REPORT:
                pred_centers[h], gt_centers[h], pred_yaws[h], gt_yaws[h] = [], [], [], []
                for i, comp in enumerate(state["current"]):
                    xy = cached["anchors_xy_t0_m"][i, h] + residual[i, h]
                    pc = runtime._target_world_from_xy_cached(xy, state["source_z_t0"][i], state["current_pose"])
                    py = renderer_yaw_delta(int(comp["class_id"]), predicted_yaw[i, h], zero_two_wheel_yaw=False)
                    gc, gy = pc, py
                    if bool(cached["se2_target_valid"][i, h]):
                        xy_gt = cached["anchors_xy_t0_m"][i, h] + cached["target_source_residual_xy_m"][i, h]
                        gc = runtime._target_world_from_xy_cached(xy_gt, state["source_z_t0"][i], state["current_pose"])
                        if bool(cached["yaw_label_valid"][i, h]):
                            gy = renderer_yaw_delta(int(comp["class_id"]), cached["target_yaw_rad"][i, h], zero_two_wheel_yaw=False)
                        else:
                            audits["gt_yaw_unavailable_kept_v18"] += 1
                        audits["gt_motion_valid_source_horizons"] += 1
                    else:
                        audits["gt_motion_unavailable_kept_v18"] += 1
                    pred_centers[h].append(pc); gt_centers[h].append(gc)
                    pred_yaws[h].append(py); gt_yaws[h].append(gy)
            gt_motion_all, gt_components = _render_current(state, pcfg, gt_centers, gt_yaws,
                                                          audit=audits, condition="t0_gt_motion")
            pred_reference, _ = _render_current(state, pcfg, pred_centers, pred_yaws,
                                                audit=audits, condition="t0_v18_motion")
            if any(not np.array_equal(pred_reference[r], baseline_all[h]) for r, h in enumerate(REPORT)):
                raise RuntimeError("zero-intervention renderer differs from frozen V18")
            timings["v18_and_motion_render"] += time.perf_counter() - tick
            print(f"source_evidence {wi}/{len(records)} stage=history_geometry", flush=True)
            tick = time.perf_counter()

            def extract(f):
                return extract_instances_cropped_exact(raw["history_occ"][f], raw["history_poses"][f], grid=pcfg.grid, cfg=strong)

            frames = list(pool.map(extract, range(5))) + [state["current"]]
            # Match V18's supplied semantic input protocol, not an implicit
            # mask_lidar-only input. Report mask support rather than claiming
            # every source voxel was directly lidar-observed.
            for f, rows in enumerate(frames):
                for comp in rows:
                    idx = np.asarray(comp["voxel_indices"], np.int64)
                    prefix = "t0" if f == 5 else "history"
                    audits[f"{prefix}_source_voxels_supplied"] += len(idx)
                    audits[f"{prefix}_source_voxels_lidar_observed"] += int(
                        np.asarray(raw["history_observed"][f])[tuple(idx.T)].sum())
            points = [[rigid_source_points_world(c["voxel_indices"], raw["history_poses"][f], grid=pcfg.grid)
                       for c in rows] for f, rows in enumerate(frames)]
            ann_maps = [_ann_map(source.nusc, token) for token in window.history_tokens]
            matches = [_match_components(rows, pts, list(anns.values()), 4.0)
                       for rows, pts, anns in zip(frames, points, ann_maps)]
            families, registration_audit = _registration_candidates(
                frames, points, state["current"], state["velocities"], matches, ann_maps,
                args.cpu_workers, float(pcfg.frame_dt_s))
            _merge(audits, registration_audit)
            memory = build_future_static_memory_only(raw["history_occ"], raw["history_observed"],
                raw["history_poses"], raw["future_poses"], grid=pcfg.grid, dynamic_class_ids=DYN,
                free_label=pcfg.free_label, workers=args.cpu_workers)
            timings["history_geometry_and_static"] += time.perf_counter() - tick
            tick = time.perf_counter()
            moving_rows = gt_moving_support_sequence(source.nusc, window.t0_token, window.future_tokens,
                tuple(0.5 * (i + 1) for i in range(6)), grid=pcfg.grid, workers=args.cpu_workers)
            timings["moving_support"] += time.perf_counter() - tick
            tick = time.perf_counter()
            window_metrics = {name: Metrics() for name in VARIANTS}
            for ri, h in enumerate(REPORT):
                baseline = np.asarray(baseline_all[h], np.uint8)
                truth = np.asarray(raw["future_gt_occ"][h], np.uint8)
                moving = moving_rows[h][0]
                rows = {"V18_BASE": baseline, "T0_GT_MOTION": gt_motion_all[ri]}
                state["world_to_future_current"] = state["world_to_future"][h]
                rows.update(_history_predictions(rows, families, state, pcfg.grid,
                    pred_centers[h], pred_yaws[h], gt_centers[h], gt_yaws[h], truth, audits))
                rows.update(_static_predictions(baseline, memory[h], truth, args.static_patch_cells))
                base_counts = Metrics.counts(baseline, truth, moving, pcfg.free_label)
                for name in VARIANTS:
                    counts = metric_count_delta(base_counts, baseline, rows[name], truth, moving, DYN)
                    if wi <= args.exactness_windows:
                        reference_counts = Metrics.counts(rows[name], truth, moving, pcfg.free_label)
                        if any(not np.array_equal(a, b) for a, b in zip(counts, reference_counts)):
                            raise RuntimeError(f"incremental metric mismatch: {name}")
                    states[name].update(ri, counts=counts)
                    scenes[window.scene_name][name].update(ri, counts=counts)
                    window_metrics[name].update(ri, counts=counts)
                    _merge(edits[name], edit_quality(baseline, rows[name], truth))
                # Disjoint coarse error mass, NOT an ancestry oracle: captures
                # static/dynamic FN, FP and semantic error without dropping FP.
                missing = (baseline == 17) & (truth != 17)
                dynamic = np.isin(truth, DYN)
                error_mass["dynamic_false_negative"] += int((missing & dynamic).sum())
                error_mass["static_false_negative"] += int((missing & ~dynamic).sum())
                error_mass["false_positive"] += int(((baseline != 17) & (truth == 17)).sum())
                error_mass["occupied_semantic_wrong"] += int(((baseline != 17) & (truth != 17) & (baseline != truth)).sum())
                transported = np.zeros(baseline.size, bool)
                for c in gt_components[ri]:
                    idx = np.asarray(c.voxel_indices)
                    if len(idx):
                        transported[np.ravel_multi_index(idx.T, baseline.shape)] = True
                error_mass["dynamic_fn_inside_t0_gt_motion_occupied_support"] += int(
                    (missing & dynamic & transported.reshape(baseline.shape)).sum())
                for cid in range(17):
                    error_mass[f"class_{cid}_fn"] += int((missing & (truth == cid)).sum())
            timings["variants_render_and_metrics"] += time.perf_counter() - tick
            b = window_metrics["V18_BASE"].compute()
            row = {"window": wi, "key": [window.scene_name, window.t0_token],
                   "delta_mIoU_pp": {name: st.compute()["mIoU"] - b["mIoU"] for name, st in window_metrics.items()},
                   "elapsed_seconds": time.perf_counter() - started}
            per_window.append(row)
            row = finite_json(row)
            progress.write(json.dumps(row, allow_nan=False) + "\n"); progress.flush()
            print(json.dumps(row, allow_nan=False), flush=True)
    baseline = states["V18_BASE"].compute()
    reports = {}
    for name in VARIANTS:
        metrics = states[name].compute()
        q = dict(edits[name]); additions = q.get("added", 0)
        q["addition_occ_precision"] = q.get("added_occ_tp", 0) / additions if additions else None
        q["addition_semantic_precision"] = q.get("added_semantic_tp", 0) / additions if additions else None
        reports[name] = {"metrics": metrics, "delta_vs_v18_pp": delta(metrics, baseline),
                         "edit_quality": q, "scene_delta": _scene_delta(scenes, name)}
    gains = {name: report["delta_vs_v18_pp"]["mIoU"] for name, report in reports.items()}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    result = {"protocol": PROTOCOL, "analysis_only": True, "windows": len(records),
        "report_horizons_s": [1.0, 2.0, 3.0], "git_commit": commit,
        "checkpoint_sha256": checkpoint_sha, "population_manifest_fingerprint": manifest["manifest_fingerprint"],
        "implementation_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
            (Path(__file__), ROOT / "real_motion/source_evidence_audit.py")},
        "selected_key_fingerprint": manifest["selected_key_fingerprint"],
        "config_sha256": sha256(args.config), "arguments": vars(args),
        "contracts": {"gt_motion": "cached valid source-centred SE2 labels; invalid labels retain V18; no GT survival",
            "gt_alignment": "oriented-box >=80%/second<=20% one-to-one history identity; no future shape",
            "causal_alignment": "same-class backwards association plus <=t0-only trimmed planar ICP",
            "common_alignment_subset": "same GT-certified identity, same accepted causal frames/shapes; GT-conditioned comparison ONLY",
            "historical_geometry": "only current-t0-associated sources; history-only/dormant not rendered",
            "history_input": "supplied historical Occ3D semantics as in V18; not mask_lidar-filtered; mask support audited separately",
            "history_composition": "protected add-only, original source order; retains ALL proposal false positives",
            "candidate_selection": "GT-assisted sequential whole-observation choice/abstain, correct-minus-wrong voxel utility; NOT global oracle",
            "static_selection": f"GT-assisted fixed {args.static_patch_cells}x{args.static_patch_cells} BEV patches/all Z; NOT voxel filter",
            "voxel_gt_filters": "hindsight perfect precision diagnostics ONLY; not deployable",
            "metrics": "frozen horizon-first raw counts, no new support mask or class exclusions"},
        "variants": reports, "error_mass": dict(error_mass), "geometry_audit": dict(audits),
        "resource_triage": {"route": route_diagnostic(gains), "automatic_training_approved": False,
            "history_increment_over_t0_gt_motion_pp": gains["HISTORY_GT_ALIGN_GT_MOTION"] - gains["T0_GT_MOTION"],
            "warning": "dev64 exploratory diagnostic; GT candidate selection is not a realizable gain guarantee"},
        "performance": {"seconds_total": time.perf_counter() - started, "seconds_by_stage": dict(timings)},
        "progress_path": str(progress_path), "per_window": per_window}
    # Metrics for absent classes contain NaN in the frozen API; JSON null is
    # used in the artifact, without changing any metric aggregation.
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(finite_json(result), handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"completed: {destination}\nroute: {result['resource_triage']['route']}", flush=True)


if __name__ == "__main__":
    main()
