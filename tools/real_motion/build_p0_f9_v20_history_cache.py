#!/usr/bin/env python3
"""Build V20 Stage-1 cache with t0-canonical 3D history evidence.

Storage is intentionally hybrid:
* global history evidence is aligned once at a coarse 3D lattice;
* future transforms are stored, not six dense render-index tensors;
* future observed static supervision is sparse;
* GT instance identity appears only in supervision metadata.

No V18/V19 artifact is overwritten.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from real_motion.geometry import quaternion_yaw
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.motion_transport import (
    dynamic_annotations,
    match_sources_to_annotations,
)
from real_motion.nuscenes_adapter import category_to_dynamic_class
from real_motion.prepared import load_nuscenes_window_raw
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.runtime_fastpath import extract_instances_cropped_exact
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.v20_history_world import (
    CanonicalLattice,
    DynamicResponsibility,
    align_history_once_to_canonical,
    future_union_query_mask,
    poses_to_t0_canonical,
)
from tools.real_motion import eval_p0_f9_v18_se2 as base
from tools.real_motion.diagnose_p0_f9_v19_innovation_decomposition import CachedSource
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record

PROTOCOL = "p0_f9_v20_stage1_history_cache_v1"
DYNAMIC_IDS = tuple(int(x) for x in DYNAMIC_CLASS_IDS)


def _load_v20(path):
    obj = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if obj.get("protocol") != "p0_f9_v20_3d_history_world_model_v1":
        raise RuntimeError("unexpected V20 config protocol")
    return obj


def _coarse_lattice(cfg):
    lc = cfg["canonical_lattice"]
    if not bool(lc.get("extent_scan_complete", False)):
        raise RuntimeError(
            "V20 Ωmax is not frozen: run extent scan and set extent_scan_complete=true"
        )
    high = CanonicalLattice(
        tuple(float(x) for x in lc["origin_xyz_m"]),
        tuple(float(x) for x in lc["voxel_size_xyz_m"]),
        tuple(int(x) for x in lc["shape_xyz"]),
    )
    factor = int(cfg["scene_encoder"].get("downsample_xy", 4))
    factor_z = int(cfg["scene_encoder"].get("downsample_z", factor))
    hs = np.asarray(high.shape_xyz, dtype=np.int64)
    coarse_shape = (
        int(math.ceil(hs[0] / factor)),
        int(math.ceil(hs[1] / factor)),
        int(math.ceil(hs[2] / factor_z)),
    )
    vs = high.voxel_size_xyz_m
    coarse = CanonicalLattice(
        high.origin_xyz_m,
        (float(vs[0]) * factor, float(vs[1]) * factor, float(vs[2]) * factor_z),
        coarse_shape,
    )
    return high, coarse


def _pack_bool(x):
    arr = np.asarray(x, dtype=np.uint8)
    return np.packbits(arr.reshape(-1), bitorder="little")


def _ann_by_instance(nusc, token):
    sample = nusc.get("sample", str(token))
    out = {}
    for atok in sample["anns"]:
        ann = nusc.get("sample_annotation", atok)
        cid = category_to_dynamic_class(ann["category_name"])
        if cid is not None:
            out[str(ann["instance_token"])] = (int(cid), ann)
    return out


def _history_source_evidence(source, w, raw, pcfg, strong_cfg, match_max_distance_m):
    matched_sets = []
    ambiguous_sets = []
    components_by_frame = []
    for ti, token in enumerate(w.history_tokens):
        sem_raw = np.asarray(raw["history_occ"][ti], dtype=np.uint8)
        obs = np.asarray(raw["history_observed"][ti], dtype=bool)
        sem = np.where(obs, sem_raw, int(pcfg.free_label)).astype(np.uint8)
        pose = np.asarray(raw["history_poses"][ti], dtype=np.float64)
        comps = extract_instances_cropped_exact(
            sem, pose, grid=pcfg.grid, cfg=strong_cfg
        )
        components_by_frame.append(comps)
        anns = dynamic_annotations(source.nusc, token)
        matched = match_sources_to_annotations(
            comps, anns, max_distance_m=float(match_max_distance_m)
        )
        matched_sets.append({str(x) for x in matched if x is not None})

        # Ambiguity is reserved for a GT ancestor with nearby same-class source
        # evidence that failed deterministic one-to-one matching.
        matched_now = matched_sets[-1]
        amb = set()
        for ann in anns:
            tok = str(ann["instance_token"])
            if tok in matched_now:
                continue
            ac = np.asarray(ann["center_world"], dtype=np.float64)
            same = [
                c for c in comps
                if int(c["class_id"]) == int(ann["class_id"])
                and float(np.linalg.norm(np.asarray(c["centroid_world"])[:2] - ac[:2]))
                <= 1.5 * float(match_max_distance_m)
            ]
            if same:
                amb.add(tok)
        ambiguous_sets.append(amb)
    return components_by_frame, matched_sets, ambiguous_sets


def _future_dynamic_targets(source, w, raw, matched_sets, ambiguous_sets):
    t0_pose = np.asarray(raw["history_poses"][-1], dtype=np.float64)
    inv_t0 = np.linalg.inv(t0_pose)
    t0_yaw = math.atan2(float(t0_pose[1, 0]), float(t0_pose[0, 0]))
    current = set(matched_sets[-1])
    earlier = set().union(*matched_sets[:-1])
    ambiguous_hist = set().union(*ambiguous_sets)

    future_maps = [_ann_by_instance(source.nusc, tok) for tok in w.future_tokens]
    tokens = sorted(set().union(*(set(x) for x in future_maps)))
    records = []
    counts = {x.name: 0 for x in DynamicResponsibility}
    for token in tokens:
        appearances = [fi for fi, amap in enumerate(future_maps) if token in amap]
        first = min(appearances)
        cid = int(future_maps[first][token][0])
        if token in ambiguous_hist:
            responsibility = DynamicResponsibility.IGNORE
        elif token in current:
            responsibility = DynamicResponsibility.CURRENT_ANCESTRAL
        elif token in earlier:
            responsibility = DynamicResponsibility.DORMANT_ANCESTRAL
        else:
            responsibility = DynamicResponsibility.BIRTH
        counts[responsibility.name] += 1

        existence = np.zeros(6, dtype=np.uint8)
        traj = np.zeros((6, 4), dtype=np.float32)
        size_lwh = None
        for fi in appearances:
            _, ann = future_maps[fi][token]
            existence[fi] = 1
            cw = np.asarray(ann["translation"], dtype=np.float64)
            ct0 = (inv_t0 @ np.r_[cw, 1.0])[:3]
            yaw = quaternion_yaw(ann["rotation"]) - t0_yaw
            yaw = (float(yaw) + math.pi) % (2.0 * math.pi) - math.pi
            traj[fi] = np.asarray([ct0[0], ct0[1], ct0[2], yaw], dtype=np.float32)
            w_m, l_m, h_m = ann["size"]
            size_lwh = [float(l_m), float(w_m), float(h_m)]
        records.append({
            "instance_token": token,
            "class_id": cid,
            "responsibility": int(responsibility),
            "responsibility_name": responsibility.name,
            "first_horizon": int(first),
            "existence": existence.tolist(),
            "trajectory_xyz_yaw_t0": traj.tolist(),
            "size_lwh_m": size_lwh,
        })
    return records, counts


def _static_sparse_supervision(source, w, raw, free_label):
    rows = []
    for fi, token in enumerate(w.future_tokens):
        gt, obs = source.load_occ3d(str(w.scene_name), str(token), require_lidar_mask=True)
        gt = np.asarray(gt, dtype=np.uint8)
        obs = np.asarray(obs, dtype=bool)
        dyn = np.isin(gt, np.asarray(DYNAMIC_IDS, dtype=np.uint8))
        valid = obs & ~dyn
        idx = np.argwhere(valid).astype(np.int16)
        lab = gt[valid].astype(np.uint8)
        rows.append({
            "indices_xyz": torch.from_numpy(idx),
            "semantic": torch.from_numpy(lab),
            "observed_count": int(valid.sum()),
            "observed_free_count": int((valid & (gt == int(free_label))).sum()),
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--v20-config", required=True)
    p.add_argument("--source-cache", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--shard-size", type=int, default=16)
    p.add_argument("--match-max-distance-m", type=float, default=4.0)
    a = p.parse_args()

    v20 = _load_v20(a.v20_config)
    high, coarse = _coarse_lattice(v20)
    pcfg = make_prepare_config(load_runtime_config(a.config, a.override))
    _, records = base.load_cache(a.source_cache)
    if int(a.max_windows) > 0:
        records = records[:min(len(records), int(a.max_windows))]
    if not records:
        raise RuntimeError("empty V20 Stage-1 population")
    source = CachedSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))
    out = Path(a.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing non-empty output dir: {out}")
    out.mkdir(parents=True, exist_ok=True)

    shards = []
    rows = []
    shard_write_seconds = []
    shard_bytes = 0
    window_compute_seconds = []
    started = time.perf_counter()
    oob_query = oob_history = 0
    dyn_totals = {x.name: 0 for x in DynamicResponsibility}
    scene_names = set()
    for wi, rec in enumerate(records, start=1):
        window_started = time.perf_counter()
        w = window_from_record(rec)
        scene_names.add(str(w.scene_name))
        raw = load_nuscenes_window_raw(source, w, pcfg, include_gt=False)
        history_occ = np.asarray(raw["history_occ"], dtype=np.uint8)
        history_obs = np.asarray(raw["history_observed"], dtype=bool)
        history_poses = np.asarray(raw["history_poses"], dtype=np.float64)
        future_poses_world = np.asarray(raw["future_poses"], dtype=np.float64)
        t0_pose = history_poses[-1]
        future_rel = poses_to_t0_canonical(future_poses_world, t0_pose)

        aligned = align_history_once_to_canonical(
            coarse,
            history_semantic=history_occ,
            history_observed=history_obs,
            history_ego_to_world=history_poses,
            t0_ego_to_world=t0_pose,
            native_origin_xyz_m=(pcfg.grid.x_min, pcfg.grid.y_min, pcfg.grid.z_min),
            native_voxel_size_xyz_m=pcfg.grid.voxel_size,
            free_label=int(pcfg.free_label),
        )
        q = future_union_query_mask(
            coarse,
            future_ego_to_canonical=future_rel,
            native_shape_xyz=pcfg.grid.shape_hwd,
            native_origin_xyz_m=(pcfg.grid.x_min, pcfg.grid.y_min, pcfg.grid.z_min),
            native_voxel_size_xyz_m=pcfg.grid.voxel_size,
        )
        _, matched, ambiguous = _history_source_evidence(
            source, w, raw, pcfg, strong_cfg, float(a.match_max_distance_m)
        )
        dyn, counts = _future_dynamic_targets(source, w, raw, matched, ambiguous)
        for k, v in counts.items():
            dyn_totals[k] += int(v)

        static_sup = _static_sparse_supervision(source, w, raw, int(pcfg.free_label))
        oob_query += int(q.out_of_bounds_voxels)
        oob_history += int(aligned.out_of_bounds_samples)
        rows.append({
            "scene_name": str(w.scene_name),
            "t0_token": str(w.t0_token),
            "history_tokens": tuple(str(x) for x in w.history_tokens),
            "future_tokens": tuple(str(x) for x in w.future_tokens),
            "history_semantic_coarse": torch.from_numpy(aligned.semantic),
            "history_observed_bits": torch.from_numpy(_pack_bool(aligned.observed)),
            "history_observed_free_bits": torch.from_numpy(_pack_bool(aligned.observed_free)),
            "history_conflict_bits": torch.from_numpy(_pack_bool(aligned.conflict)),
            "coarse_shape_xyz": tuple(int(x) for x in coarse.shape_xyz),
            "query_mask_bits": torch.from_numpy(_pack_bool(q.mask)),
            "future_ego_to_t0": torch.from_numpy(future_rel.astype(np.float32)),
            "query_oob_voxels": int(q.out_of_bounds_voxels),
            "history_oob_observed_samples": int(aligned.out_of_bounds_samples),
            "static_supervision": static_sup,
            "dynamic_supervision": dyn,
        })

        window_compute_seconds.append(time.perf_counter() - window_started)
        if len(rows) >= int(a.shard_size) or wi == len(records):
            name = f"shard_{len(shards):05d}.pt"
            save_started = time.perf_counter()
            torch.save({"protocol": PROTOCOL, "rows": rows}, out / name)
            write_s = time.perf_counter() - save_started
            nbytes = int((out / name).stat().st_size)
            shard_write_seconds.append(write_s)
            shard_bytes += nbytes
            shards.append({
                "file": name,
                "count": len(rows),
                "bytes": nbytes,
                "write_seconds": float(write_s),
            })
            rows = []
        if wi == 1 or wi % 25 == 0 or wi == len(records):
            print(f"v20_stage1_cache {wi}/{len(records)}", flush=True)

    index = {
        "protocol": PROTOCOL,
        "v20_config": str(Path(a.v20_config).resolve()),
        "source_cache": str(Path(a.source_cache).resolve()),
        "num_windows": len(records),
        "num_scenes": len(scene_names),
        "scene_names": sorted(scene_names),
        "canonical_frame": "per_window_t0_ego",
        "highres_lattice": {
            "origin_xyz_m": list(high.origin_xyz_m),
            "voxel_size_xyz_m": list(high.voxel_size_xyz_m),
            "shape_xyz": list(high.shape_xyz),
        },
        "coarse_lattice": {
            "origin_xyz_m": list(coarse.origin_xyz_m),
            "voxel_size_xyz_m": list(coarse.voxel_size_xyz_m),
            "shape_xyz": list(coarse.shape_xyz),
        },
        "inference_fields_use_future_semantics": False,
        "dynamic_identity_is_supervision_only": True,
        "query_out_of_bounds_voxels": int(oob_query),
        "history_out_of_bounds_observed_samples": int(oob_history),
        "dynamic_partition_counts": dyn_totals,
        "build_profile": {
            "elapsed_seconds": float(time.perf_counter() - started),
            "window_compute_mean_seconds": float(
                np.mean(window_compute_seconds)
            ),
            "window_compute_p50_seconds": float(
                np.quantile(window_compute_seconds, 0.50)
            ),
            "window_compute_p95_seconds": float(
                np.quantile(window_compute_seconds, 0.95)
            ),
            "shard_write_seconds": float(sum(shard_write_seconds)),
            "cache_bytes": int(shard_bytes),
            "bytes_per_window": float(shard_bytes / max(len(records), 1)),
            "process_max_rss_mib": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            ),
            "note": (
                "Linux ru_maxrss is process peak resident memory. "
                "Run --max-windows smoke before full cache construction."
            ),
        },
        "shards": shards,
    }
    (out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
