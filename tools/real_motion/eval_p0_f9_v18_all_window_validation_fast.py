#!/usr/bin/env python3
"""Fast, exact all-window validation for frozen V18 motion checkpoints.

This is an optimized implementation of ``eval_p0_f9_v18_all_window_validation``.
It preserves the same Strong/KTA, Y600, Clean-E14, Moving-mIoU-v2, hard-A1 and
scene-bootstrap contracts while removing the expensive redundancies that make
4369-window validation CPU-bound:

- raw semantics/poses are LRU cached across overlapping windows;
- only the three reported horizons (1s/2s/3s) are reconstructed;
- Strong connected components are cached per frame and reused by adjacent windows;
- Strong report-horizon anchors reuse one component extraction/matching pass;
- Y600/Clean model inference is source-batched per scene rather than one tiny
  forward per model per window;
- rigid source points are transformed to world coordinates once per component;
- semantic/moving IoU counts use confusion-matrix bincounts instead of repeated
  full-grid per-class scans;
- scene-level partial state is atomically checkpointed and can be resumed.

The first non-empty window performs exactness checks against the historical slow
implementations for Strong anchors, rigid rasterization, and metric raw counts.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.geometry import relative_transform
from real_motion.metrics.moving_miou_v2 import (
    DYNAMIC_CLASS_IDS,
    NUSCENES_LABELS,
    REPORT_HORIZONS_S,
)
from real_motion.nuscenes_adapter import gt_moving_support_for_horizon
from real_motion.rigid_transport import (
    RasterizedRigidComponent,
    _deduplicate_indices,
    _indices_to_ego_xyz,
    _metric_to_indices,
    _transform_points as rigid_transform_points,
    compose_component_replacements_in_input_order,
    rasterize_rigid_component,
)
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.strong_w2det import (
    StrongW2DetConfig,
    _in_grid,
    _metric_to_voxel,
    _transform_points as strong_transform_points,
    extract_instances,
    inverse_warp,
    majority_fill,
    match_instances,
    strong_w2det_sequence,
)
from real_motion.v18_two_wheel_diagnostic import renderer_yaw_delta
from tools.real_motion import eval_p0_f9_v18_full_validation as mid
from tools.real_motion.eval_p0_f9_v17_local_stwm import (
    t0_xy_to_world_preserve_source_z,
    window_from_record,
)


PROTOCOL = "p0_f9_v18_all_window_validation_fast_v1"
SEMANTIC_CLASSES = tuple(range(17))


class CachedSource(mid.base.NuScenesWindowSource):
    """Small scene-local-ish LRUs; enough to exploit stride-1 overlap without GBs of RAM."""

    @lru_cache(maxsize=192)
    def load_semantics(self, scene_name, token):
        return super().load_semantics(scene_name, token)

    @lru_cache(maxsize=768)
    def pose(self, token):
        return super().pose(token)


def _accumulate_count_row(dst: dict, src: dict) -> None:
    dst["occ_inter"] += int(src["occ_inter"])
    dst["occ_union"] += int(src["occ_union"])
    dst["sem_inter"] += src["sem_inter"]
    dst["sem_union"] += src["sem_union"]
    dst["mov_inter"] += src["mov_inter"]
    dst["mov_union"] += src["mov_union"]


def _scene_record_counts(records):
    out = {}
    for r in records:
        s = str(window_from_record(r).scene_name)
        out[s] = out.get(s, 0) + 1
    return out


def _fast_count_row(pred, gt, moving, free_label: int):
    """Exact raw IoU counts using one confusion matrix per semantic support."""
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    moving = np.asarray(moving, dtype=bool)
    if pred.shape != gt.shape or moving.shape != gt.shape:
        raise ValueError("metric shape mismatch")

    p = pred.reshape(-1).astype(np.int16, copy=False)
    g = gt.reshape(-1).astype(np.int16, copy=False)
    if p.size and (p.min() < 0 or p.max() > 17 or g.min() < 0 or g.max() > 17):
        raise RuntimeError("semantic labels outside frozen 0..17 Occ3D range")

    conf = np.bincount(g * 18 + p, minlength=18 * 18).reshape(18, 18)
    diag = np.diag(conf)
    row_sum = conf.sum(axis=1)
    col_sum = conf.sum(axis=0)

    out = mid._empty_scene_row()
    po = p != int(free_label)
    go = g != int(free_label)
    out["occ_inter"] = int(np.count_nonzero(po & go))
    out["occ_union"] = int(np.count_nonzero(po | go))
    for j, cid in enumerate(SEMANTIC_CLASSES):
        out["sem_inter"][j] = int(diag[cid])
        out["sem_union"][j] = int(row_sum[cid] + col_sum[cid] - diag[cid])

    ms = moving.reshape(-1)
    if bool(ms.any()):
        pm = p[ms]
        gm = g[ms]
        mconf = np.bincount(gm * 18 + pm, minlength=18 * 18).reshape(18, 18)
        mdiag = np.diag(mconf)
        mrow = mconf.sum(axis=1)
        mcol = mconf.sum(axis=0)
        for j, cid in enumerate(DYNAMIC_CLASS_IDS):
            out["mov_inter"][j] = int(mdiag[int(cid)])
            out["mov_union"][j] = int(
                mrow[int(cid)] + mcol[int(cid)] - mdiag[int(cid)]
            )
    return out


def _assert_count_rows_equal(a: dict, b: dict) -> None:
    for key in ("occ_inter", "occ_union"):
        if int(a[key]) != int(b[key]):
            raise RuntimeError(f"fast metric mismatch for {key}: {a[key]} != {b[key]}")
    for key in ("sem_inter", "sem_union", "mov_inter", "mov_union"):
        if not np.array_equal(np.asarray(a[key]), np.asarray(b[key])):
            raise RuntimeError(f"fast metric mismatch for {key}")


def _precompute_source_world(current, current_pose, grid):
    rows = []
    for comp in current:
        idx = np.asarray(comp["voxel_indices"], dtype=np.int64)
        pts_ego = _indices_to_ego_xyz(idx, grid)
        pts_world = rigid_transform_points(current_pose, pts_ego)
        rows.append(pts_world)
    return rows


def _fast_rasterize_from_world(
    source_points_world: np.ndarray,
    class_id: int,
    source_voxel_count: int,
    source_center_world: np.ndarray,
    target_center_world: np.ndarray,
    yaw_delta_rad: float,
    world_to_future: np.ndarray,
    grid,
) -> RasterizedRigidComponent:
    pts_world = np.asarray(source_points_world, dtype=np.float64)
    if len(pts_world) == 0:
        return RasterizedRigidComponent(
            int(class_id), np.zeros((0, 3), dtype=np.int64), int(source_voxel_count)
        )
    source_center = np.asarray(source_center_world, dtype=np.float64)
    target_center = np.asarray(target_center_world, dtype=np.float64)
    theta = float(yaw_delta_rad)
    c, s = math.cos(theta), math.sin(theta)
    rel = pts_world[:, :2] - source_center[None, :2]
    moved_world = pts_world.copy()
    moved_world[:, 0] = target_center[0] + c * rel[:, 0] - s * rel[:, 1]
    moved_world[:, 1] = target_center[1] + s * rel[:, 0] + c * rel[:, 1]
    moved_future = rigid_transform_points(world_to_future, moved_world)
    dst_idx, valid = _metric_to_indices(moved_future, grid)
    dst_idx = _deduplicate_indices(dst_idx[valid], grid)
    return RasterizedRigidComponent(int(class_id), dst_idx, int(source_voxel_count))


def _fast_strong_report_anchors(
    current_semantics,
    current_pose,
    future_pose_by_hi,
    current,
    velocities,
    source_world_points,
    *,
    frame_dt_s: float,
    grid,
    cfg: StrongW2DetConfig,
):
    """Exact Strong-W2Det at only report horizons, reusing extracted components."""
    sem0 = np.asarray(current_semantics)
    dyn = np.isin(sem0, np.asarray(DYNAMIC_CLASS_IDS, dtype=sem0.dtype))
    static_src = sem0.copy()
    static_src[dyn] = int(cfg.free_label)

    covered = np.zeros_like(dyn, dtype=bool)
    for comp in current:
        idx = np.asarray(comp["voxel_indices"], dtype=np.int64)
        covered[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    rest = dyn & ~covered
    if bool(rest.any()):
        ridx = np.argwhere(rest)
        origin = np.asarray([grid.x_min, grid.y_min, grid.z_min], dtype=np.float64)
        step = np.asarray(grid.voxel_size, dtype=np.float64)
        rest_ego = origin + (ridx.astype(np.float64) + 0.5) * step
        rest_world = strong_transform_points(current_pose, rest_ego)
        rest_labels = sem0[ridx[:, 0], ridx[:, 1], ridx[:, 2]]
    else:
        rest_world = np.zeros((0, 3), dtype=np.float64)
        rest_labels = np.zeros((0,), dtype=sem0.dtype)

    outputs = {}
    baselines = {}
    for horizon, hi in mid.REPORT.items():
        future_pose = np.asarray(future_pose_by_hi[int(hi)], dtype=np.float64)
        t_future_from_current = relative_transform(current_pose, future_pose)
        static_dst, known = inverse_warp(
            static_src, t_future_from_current, grid, int(cfg.free_label)
        )
        out = majority_fill(
            static_dst,
            ~known,
            kernel=cfg.fill_kernel,
            min_fraction=cfg.fill_min_fraction,
        )

        dt = (int(hi) + 1) * float(frame_dt_s)
        world_parts = []
        label_parts = []
        component_slices = []
        cursor = 0
        for j, comp in enumerate(current):
            pts = np.asarray(source_world_points[j], dtype=np.float64)
            v = np.asarray(velocities.get(j, np.zeros(3)), dtype=np.float64)
            moved = pts + v[None] * float(dt)
            world_parts.append(moved)
            label_parts.append(
                np.full(len(moved), int(comp["class_id"]), dtype=sem0.dtype)
            )
            component_slices.append(slice(cursor, cursor + len(moved)))
            cursor += len(moved)
        if len(rest_world):
            world_parts.append(rest_world)
            label_parts.append(rest_labels)

        baseline_components = []
        if world_parts:
            moved_world = np.concatenate(world_parts, axis=0)
            labels = np.concatenate(label_parts, axis=0)
            world_to_future = np.linalg.inv(future_pose)
            moved_future = strong_transform_points(world_to_future, moved_world)
            idx_all = _metric_to_voxel(moved_future, grid)
            valid_all = _in_grid(idx_all, grid)
            idx = idx_all[valid_all]
            labels_valid = labels[valid_all]
            out[idx[:, 0], idx[:, 1], idx[:, 2]] = labels_valid

            for comp, sl in zip(current, component_slices):
                cidx = idx_all[sl][valid_all[sl]]
                baseline_components.append(
                    RasterizedRigidComponent(
                        int(comp["class_id"]),
                        cidx,
                        int(len(comp["voxel_indices"])),
                    )
                )
        outputs[int(hi)] = out.astype(np.uint8, copy=False)
        baselines[int(hi)] = baseline_components
    return outputs, baselines


def _predict_scene(records, y_model, c_model, device, *, source_batch_size: int):
    """Run both frozen models in large source batches and split back by window."""
    counts = [int(r["features"].shape[0]) for r in records]
    total = int(sum(counts))
    empty_xy = lambda n: np.zeros((n, 6, 2), dtype=np.float32)
    empty_yaw = lambda n: np.zeros((n, 6), dtype=np.float32)
    if total == 0:
        return {
            str(r["sample_id"]): (empty_xy(0), empty_yaw(0), empty_xy(0), empty_yaw(0))
            for r in records
        }

    fields = (
        "features",
        "local_semantic_tube",
        "kta_displacement_xy_m",
        "frame_motion_features",
        "target_source_mask_tube",
    )
    cat = {k: torch.cat([r[k] for r in records if int(r["features"].shape[0])], dim=0)
           for k in fields}
    ys, yy, cs, cy = [], [], [], []
    bs = max(1, int(source_batch_size))
    with torch.inference_mode():
        for start in range(0, total, bs):
            stop = min(total, start + bs)
            f = cat["features"][start:stop].float().to(device, non_blocking=True)
            tube = cat["local_semantic_tube"][start:stop].to(device, non_blocking=True)
            kta = cat["kta_displacement_xy_m"][start:stop].float().to(device, non_blocking=True)
            fm = cat["frame_motion_features"][start:stop].float().to(device, non_blocking=True)
            sm = cat["target_source_mask_tube"][start:stop].to(device, non_blocking=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                yo = y_model(f, tube, kta, fm, sm)
                co = c_model(f, tube, kta, fm, sm)
            ys.append(yo["residual_xy_m"].float().cpu())
            yy.append(yo["yaw_delta_rad"].float().cpu())
            cs.append(co["residual_xy_m"].float().cpu())
            cy.append(co["yaw_delta_rad"].float().cpu())
    yres = torch.cat(ys, dim=0).numpy()
    yyaw = torch.cat(yy, dim=0).numpy()
    cres = torch.cat(cs, dim=0).numpy()
    cyaw = torch.cat(cy, dim=0).numpy()

    out = {}
    cursor = 0
    for rec, n in zip(records, counts):
        sid = str(rec["sample_id"])
        out[sid] = (
            yres[cursor:cursor+n], yyaw[cursor:cursor+n],
            cres[cursor:cursor+n], cyaw[cursor:cursor+n],
        )
        cursor += n
    if cursor != total:
        raise AssertionError("source split mismatch")
    return out


def _empty_support_counts():
    return {
        str(cid): {
            str(float(h)): {
                "window_instance_occurrences": 0,
                "instance_tokens": set(),
                "scenes": set(),
            }
            for h in REPORT_HORIZONS_S
        }
        for cid in DYNAMIC_CLASS_IDS
    }


def _partial_contract(a):
    return {
        "protocol": PROTOCOL,
        "full_val_cache": str(Path(a.full_val_cache).resolve()),
        "development_cache": str(Path(a.development_cache).resolve()) if a.development_cache else "",
        "y600_checkpoint": str(Path(a.y600_checkpoint).resolve()),
        "clean_checkpoint": str(Path(a.clean_checkpoint).resolve()),
        "expected_windows": int(a.expected_windows),
        "expected_scenes": int(a.expected_scenes),
    }


def _atomic_torch_save(obj, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _save_partial(path, contract, completed_scenes, scene_rows, support_counts):
    _atomic_torch_save(
        {
            "contract": contract,
            "completed_scenes": sorted(completed_scenes),
            "scene_rows": scene_rows,
            "support_counts": support_counts,
        },
        path,
    )


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--full-val-cache", required=True)
    p.add_argument("--development-cache", default="")
    p.add_argument("--y600-checkpoint", required=True)
    p.add_argument("--clean-checkpoint", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--expected-windows", type=int, default=0)
    p.add_argument("--expected-scenes", type=int, default=0)
    p.add_argument("--model-source-batch-size", type=int, default=1024)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-exactness-check", action="store_true")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    cache_meta, records = mid.base.load_cache(a.full_val_cache)
    if not records:
        raise RuntimeError("full validation cache is empty")
    if int(a.model_source_batch_size) <= 0:
        raise ValueError("model-source-batch-size must be positive")

    sample_ids = [str(r["sample_id"]) for r in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("all-window validation cache contains duplicate sample IDs")
    scene_stream = [str(window_from_record(r).scene_name) for r in records]
    all_scenes = tuple(dict.fromkeys(scene_stream))
    if int(a.expected_windows) > 0 and len(records) != int(a.expected_windows):
        raise RuntimeError(f"validation cache windows {len(records)} != expected {a.expected_windows}")
    if int(a.expected_scenes) > 0 and len(all_scenes) != int(a.expected_scenes):
        raise RuntimeError(f"validation cache scenes {len(all_scenes)} != expected {a.expected_scenes}")
    if len(records) <= len(all_scenes):
        raise RuntimeError("fast all-window evaluator requires overlapping validation windows")

    dev_ids, dev_scenes = mid._load_dev_subset(a.development_cache)
    if dev_ids:
        missing = sorted(dev_ids - set(sample_ids))
        if missing:
            raise RuntimeError(
                f"all-window cache does not contain previous development midpoint samples: {missing[:5]}"
            )
    full_scene_set = set(all_scenes)
    heldout_scenes = full_scene_set - dev_scenes if dev_scenes else set()

    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    y_ck, y_model, y_cfg = mid._load_model(a.y600_checkpoint, mid.PAIR_PROTOCOL, device)
    c_ck, c_model, c_cfg = mid._load_model(a.clean_checkpoint, mid.CLEAN_PROTOCOL, device)
    if y_cfg != c_cfg:
        raise RuntimeError("Y600 and Clean model_config differ")

    source = CachedSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))

    @lru_cache(maxsize=192)
    def cached_instances(scene_name: str, token: str):
        sem = source.load_semantics(scene_name, token)
        pose = np.asarray(source.pose(token), dtype=np.float64)
        return extract_instances(sem, pose, grid=pcfg.grid, cfg=strong_cfg)

    records_by_scene = {s: [] for s in all_scenes}
    for rec in records:
        records_by_scene[str(window_from_record(rec).scene_name)].append(rec)

    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(op) + ".partial.pt")
    contract = _partial_contract(a)
    scene_rows = {}
    support_counts = _empty_support_counts()
    completed_scenes = set()
    if bool(a.resume) and partial.exists():
        state = torch.load(partial, map_location="cpu", weights_only=False)
        if state.get("contract") != contract:
            raise RuntimeError("partial-state contract differs from current evaluation arguments")
        scene_rows = state.get("scene_rows") or {}
        support_counts = state.get("support_counts") or _empty_support_counts()
        completed_scenes = set(str(x) for x in state.get("completed_scenes", []))
        print(
            f"RESUME partial={partial} completed_scenes={len(completed_scenes)}/{len(all_scenes)}",
            flush=True,
        )
    elif partial.exists() and not bool(a.resume):
        raise FileExistsError(f"partial state already exists: {partial}; use --resume or remove it")

    exactness_pending = not bool(a.no_exactness_check) and not completed_scenes
    total_done = sum(len(records_by_scene[s]) for s in completed_scenes if s in records_by_scene)
    started = time.perf_counter()

    for si, scene in enumerate(all_scenes, start=1):
        if scene in completed_scenes:
            continue
        scene_records = records_by_scene[scene]
        predictions = _predict_scene(
            scene_records, y_model, c_model, device,
            source_batch_size=int(a.model_source_batch_size),
        )

        for rec in scene_records:
            sid = str(rec["sample_id"])
            w = window_from_record(rec)
            current_token = str(w.t0_token)
            previous_token = str(w.history_tokens[-2])
            current_sem = np.asarray(source.load_semantics(scene, current_token), dtype=np.uint8)
            previous_sem = np.asarray(source.load_semantics(scene, previous_token), dtype=np.uint8)
            current_pose = np.asarray(source.pose(current_token), dtype=np.float64)
            previous_pose = np.asarray(source.pose(previous_token), dtype=np.float64)
            future_pose_by_hi = {
                int(hi): np.asarray(source.pose(str(w.future_tokens[int(hi)])), dtype=np.float64)
                for hi in mid.REPORT.values()
            }
            future_gt_by_hi = {
                int(hi): np.asarray(source.load_semantics(scene, str(w.future_tokens[int(hi)])), dtype=np.uint8)
                for hi in mid.REPORT.values()
            }

            current = cached_instances(scene, current_token)
            previous = cached_instances(scene, previous_token)
            velocities = match_instances(
                previous,
                current,
                float(pcfg.frame_dt_s),
                max_speed_mps=strong_cfg.max_match_speed_mps,
            )
            if len(current) != int(rec["features"].shape[0]):
                raise RuntimeError(f"{sid}: Strong/source count mismatch")
            current_classes = [int(c["class_id"]) for c in current]
            cached_classes = [int(x) for x in rec["source_class_id"].tolist()]
            if current_classes != cached_classes:
                raise RuntimeError(f"{sid}: Strong/source class order mismatch")

            source_world_points = _precompute_source_world(current, current_pose, pcfg.grid)
            anchors, baseline_by_hi = _fast_strong_report_anchors(
                current_sem,
                current_pose,
                future_pose_by_hi,
                current,
                velocities,
                source_world_points,
                frame_dt_s=float(pcfg.frame_dt_s),
                grid=pcfg.grid,
                cfg=strong_cfg,
            )

            y_res, y_yaw, c_res, c_yaw = predictions[sid]
            if y_res.shape[0] != len(current) or c_res.shape[0] != len(current):
                raise RuntimeError(f"{sid}: batched model/source count mismatch")

            if exactness_pending:
                hist2 = np.stack([previous_sem, current_sem], axis=0)
                hist2poses = [previous_pose, current_pose]
                future_all = [np.asarray(source.pose(str(t)), dtype=np.float64) for t in w.future_tokens]
                ref_anchor = strong_w2det_sequence(
                    hist2,
                    hist2poses,
                    future_all,
                    frame_dt_s=float(pcfg.frame_dt_s),
                    grid=pcfg.grid,
                    cfg=strong_cfg,
                )
                for hi in mid.REPORT.values():
                    if not np.array_equal(anchors[int(hi)], ref_anchor[int(hi)]):
                        neq = int(np.count_nonzero(anchors[int(hi)] != ref_anchor[int(hi)]))
                        raise RuntimeError(f"fast Strong anchor mismatch horizon_index={hi} voxels={neq}")
                print("EXACTNESS Strong report-horizon anchors: PASS", flush=True)

            for horizon, hi in mid.REPORT.items():
                hi = int(hi)
                gt = future_gt_by_hi[hi]
                anchor = anchors[hi]
                moving, moving_records, _ = gt_moving_support_for_horizon(
                    source.nusc,
                    w.t0_token,
                    w.future_tokens[hi],
                    float(horizon),
                    grid=pcfg.grid,
                )
                for mr in moving_records:
                    cid = int(mr["class_id"])
                    if cid in DYNAMIC_CLASS_IDS:
                        e = support_counts[str(cid)][str(float(horizon))]
                        e["window_instance_occurrences"] += 1
                        e["instance_tokens"].add(str(mr["instance_token"]))
                        e["scenes"].add(scene)

                world_to_future = np.linalg.inv(future_pose_by_hi[hi])
                y_components = []
                c_components = []
                for i, comp in enumerate(current):
                    cid = int(comp["class_id"])
                    src_center = np.asarray(comp["centroid_world"], dtype=np.float64)
                    y_xy = rec["anchors_xy_t0_m"][i, hi].numpy() + y_res[i, hi]
                    c_xy = rec["anchors_xy_t0_m"][i, hi].numpy() + c_res[i, hi]
                    y_center = t0_xy_to_world_preserve_source_z(y_xy, src_center, current_pose)
                    c_center = t0_xy_to_world_preserve_source_z(c_xy, src_center, current_pose)
                    y_comp = _fast_rasterize_from_world(
                        source_world_points[i], cid, len(comp["voxel_indices"]),
                        src_center, y_center,
                        renderer_yaw_delta(cid, y_yaw[i, hi], zero_two_wheel_yaw=False),
                        world_to_future, pcfg.grid,
                    )
                    c_comp = _fast_rasterize_from_world(
                        source_world_points[i], cid, len(comp["voxel_indices"]),
                        src_center, c_center,
                        renderer_yaw_delta(cid, c_yaw[i, hi], zero_two_wheel_yaw=False),
                        world_to_future, pcfg.grid,
                    )
                    if exactness_pending:
                        ref_y = rasterize_rigid_component(
                            comp["voxel_indices"], cid, current_pose, future_pose_by_hi[hi],
                            source_center_world=src_center,
                            target_center_world=y_center,
                            yaw_delta_rad=renderer_yaw_delta(
                                cid, y_yaw[i, hi], zero_two_wheel_yaw=False
                            ),
                            grid=pcfg.grid,
                        )
                        ref_c = rasterize_rigid_component(
                            comp["voxel_indices"], cid, current_pose, future_pose_by_hi[hi],
                            source_center_world=src_center,
                            target_center_world=c_center,
                            yaw_delta_rad=renderer_yaw_delta(
                                cid, c_yaw[i, hi], zero_two_wheel_yaw=False
                            ),
                            grid=pcfg.grid,
                        )
                        if not np.array_equal(y_comp.voxel_indices, ref_y.voxel_indices):
                            raise RuntimeError("fast Y600 rigid raster mismatch")
                        if not np.array_equal(c_comp.voxel_indices, ref_c.voxel_indices):
                            raise RuntimeError("fast Clean rigid raster mismatch")
                    y_components.append(y_comp)
                    c_components.append(c_comp)

                y_pred = compose_component_replacements_in_input_order(
                    anchor, baseline_by_hi[hi], y_components,
                    dynamic_class_ids=DYNAMIC_CLASS_IDS,
                    free_label=int(pcfg.free_label), grid=pcfg.grid,
                )
                c_pred = compose_component_replacements_in_input_order(
                    anchor, baseline_by_hi[hi], c_components,
                    dynamic_class_ids=DYNAMIC_CLASS_IDS,
                    free_label=int(pcfg.free_label), grid=pcfg.grid,
                )
                preds = {
                    "strong_anchor": anchor,
                    "y600_pred": y_pred,
                    "clean_pred": c_pred,
                }
                for name, pred in preds.items():
                    key = (scene, name, float(horizon))
                    row = _fast_count_row(pred, gt, moving, int(pcfg.free_label))
                    if exactness_pending:
                        ref_row = mid._count_row(pred, gt, moving, int(pcfg.free_label))
                        _assert_count_rows_equal(row, ref_row)
                    if key not in scene_rows:
                        scene_rows[key] = row
                    else:
                        _accumulate_count_row(scene_rows[key], row)

            if exactness_pending:
                print("EXACTNESS rigid raster + raw metric counts: PASS", flush=True)
                exactness_pending = False
            total_done += 1

        completed_scenes.add(scene)
        _save_partial(partial, contract, completed_scenes, scene_rows, support_counts)
        elapsed = max(time.perf_counter() - started, 1e-9)
        fresh_done = max(1, total_done - sum(
            len(records_by_scene[s]) for s in completed_scenes if s in records_by_scene
        ) + len(scene_records))
        # Use actual newly completed windows since process start for ETA.
        completed_now = sum(
            len(records_by_scene[s]) for s in completed_scenes
        )
        resumed_before = completed_now - sum(
            len(records_by_scene[s]) for s in completed_scenes
            if s == scene or False
        )
        # Simpler stable rate: total windows completed in this process is tracked separately below.
        cache_info = source.load_semantics.cache_info()
        print(
            f"FAST_VAL scene={si}/{len(all_scenes)} {scene} "
            f"windows={total_done}/{len(records)} elapsed={elapsed/60:.1f}m "
            f"sem_cache={cache_info}",
            flush=True,
        )

    raw_by_variant = mid._stack_scene_rows(scene_rows, all_scenes)
    subsets = {"full": set(all_scenes)}
    if dev_scenes:
        subsets["development_scenes_all_windows"] = set(dev_scenes)
        subsets["heldout_scenes_all_windows"] = set(heldout_scenes)

    report_subsets = {}
    bootstrap_subsets = {}
    for subset_name, subset_scene_set in subsets.items():
        idx = mid._subset_indices(all_scenes, subset_scene_set)
        report_subsets[subset_name] = {
            "num_scenes": int(idx.size),
            "num_windows": int(sum(1 for s in scene_stream if s in subset_scene_set)),
            "scene_names": sorted(str(x) for x in subset_scene_set),
            "variants": {
                name: mid._metric_from_raw(raw_by_variant[name], idx)
                for name in mid.VARIANTS
            },
        }
        bootstrap_subsets[subset_name] = mid._bootstrap_delta(
            raw_by_variant["y600_pred"],
            raw_by_variant["clean_pred"],
            idx,
            samples=int(a.bootstrap_samples),
            seed=mid.BOOTSTRAP_SEED,
        )

    support_json = {}
    for cid in DYNAMIC_CLASS_IDS:
        support_json[str(cid)] = {}
        for h in REPORT_HORIZONS_S:
            e = support_counts[str(cid)][str(float(h))]
            support_json[str(cid)][str(float(h))] = {
                "class_name": NUSCENES_LABELS[int(cid)],
                "window_instance_occurrences": int(e["window_instance_occurrences"]),
                "unique_instance_tokens": int(len(e["instance_tokens"])),
                "scenes": int(len(e["scenes"])),
            }

    result = {
        "protocol": PROTOCOL,
        "status": "completed_evaluation",
        "full_val_cache": str(Path(a.full_val_cache).resolve()),
        "development_cache": str(Path(a.development_cache).resolve()) if a.development_cache else None,
        "cache_metadata": cache_meta,
        "num_windows": len(records),
        "num_scenes": len(all_scenes),
        "windows_per_scene": _scene_record_counts(records),
        "all_eligible_overlapping_windows": True,
        "report_horizons_s": list(REPORT_HORIZONS_S),
        "dynamic_class_ids": list(DYNAMIC_CLASS_IDS),
        "y600_checkpoint": str(Path(a.y600_checkpoint).resolve()),
        "clean_checkpoint": str(Path(a.clean_checkpoint).resolve()),
        "y600_checkpoint_protocol": y_ck.get("protocol"),
        "clean_checkpoint_protocol": c_ck.get("protocol"),
        "y600_step": int(y_ck.get("continuation_step", -1)),
        "clean_epoch": int(c_ck.get("epoch", -1)),
        "clean_global_step": int(c_ck.get("global_step", -1)),
        "hard_compositor": "legacy_clear_plus_original_strong_source_write_order_v1",
        "yaw_mode": "pred",
        "moving_support": "interval_displacement_v2_gt_moving_support_for_horizon",
        "subsets": report_subsets,
        "moving_support_counts_full": support_json,
        "paired_bootstrap_clean_minus_y600": bootstrap_subsets,
        "optimization": {
            "cached_semantics": True,
            "cached_poses": True,
            "cached_strong_components": True,
            "report_horizons_only": True,
            "scene_batched_model_inference": True,
            "model_source_batch_size": int(a.model_source_batch_size),
            "confusion_matrix_metric_counts": True,
            "scene_partial_resume": True,
            "exactness_check": not bool(a.no_exactness_check),
        },
        "notes": {
            "bootstrap_unit": "scene",
            "bootstrap_reaggregation": "all overlapping windows are first summed inside scene; paired scene resample then sums raw intersections/unions",
            "overlapping_windows_are_not_independent_bootstrap_units": True,
            "scene_iou_is_not_averaged": True,
            "no_training": True,
            "development_subset_definition": "all full-validation windows from scenes represented in the supplied 128-scene development cache",
            "heldout_subset_definition": "all full-validation windows from scenes absent from the supplied development cache",
            "moving_support_count_note": "window_instance_occurrences counts repeated appearances across overlapping windows; unique_instance_tokens removes those repeats",
        },
    }
    op.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if partial.exists():
        partial.unlink()

    print("\n=== OFFICIAL-STYLE ALL-WINDOW VALIDATION (FAST EXACT) ===")
    for subset_name in ("full", "development_scenes_all_windows", "heldout_scenes_all_windows"):
        if subset_name not in report_subsets:
            continue
        rr = report_subsets[subset_name]
        print(f"\n[{subset_name}] scenes={rr['num_scenes']} windows={rr['num_windows']}")
        print(f"{'variant':18s} {'IoU':>9s} {'mIoU':>9s} {'MacroMov':>10s} {'MicroMov':>10s}")
        for name in mid.VARIANTS:
            m = rr["variants"][name]
            print(
                f"{name:18s} {m['IoU']:9.4f} {m['mIoU']:9.4f} "
                f"{m['MovingMacro']:10.4f} {m['MovingMicro']:10.4f}"
            )
        boot = bootstrap_subsets.get(subset_name)
        if boot:
            print("Clean-Y600 paired scene bootstrap:")
            for k in ("IoU", "mIoU", "MovingMacro", "MovingMicro"):
                b = boot["metrics"][k]
                print(
                    f"  {k:11s} point={b['point_delta_pp']:+.4f} "
                    f"95%CI=[{b['p2_5_pp']:+.4f},{b['p97_5_pp']:+.4f}]"
                )

    print("\n=== ALL-WINDOW MOVING PER CLASS x HORIZON ===")
    full = report_subsets["full"]["variants"]
    for cid in DYNAMIC_CLASS_IDS:
        name = NUSCENES_LABELS[int(cid)]
        meta = support_json[str(cid)]
        for h in REPORT_HORIZONS_S:
            hs = str(float(h))
            y = full["y600_pred"]["per_horizon"][hs]["moving_per_class"][str(cid)]
            c = full["clean_pred"]["per_horizon"][hs]["moving_per_class"][str(cid)]
            b = bootstrap_subsets["full"]["per_class_horizon"][str(cid)][hs]
            m = meta[hs]
            print(
                f"{name:22s} {h:.1f}s Y600={y:8.3f} Clean={c:8.3f} "
                f"d={c-y:+8.3f} occ={m['window_instance_occurrences']:5d} "
                f"uniq={m['unique_instance_tokens']:4d} scenes={m['scenes']:3d} "
                f"CI=[{b['p2_5_pp']:+.3f},{b['p97_5_pp']:+.3f}]"
            )
    print(f"saved {op}")


if __name__ == "__main__":
    main()
