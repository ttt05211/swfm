#!/usr/bin/env python3
"""Train V20 Stage-3 Dormant-source adapter.

Current t0 sources never enter this optimizer path. The frozen V18 model is
used only to expose a causal source token for memory-only tracks.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.geometry import quaternion_yaw
from real_motion.local_st_world_model_v18_se2 import YAW_ENABLED_CLASS_IDS
from real_motion.nuscenes_adapter import category_to_dynamic_class
from real_motion.prepared import load_nuscenes_window_raw
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.v19_scene_memory import (
    build_dynamic_source_memory,
    prepare_causal_arrays_from_tracks,
)
from real_motion.v20_dormant import (
    recompute_synthetic_t0_occlusion,
    split_current_and_dormant_tracks,
)
from real_motion.v20_history_world import CanonicalLattice, align_history_once_to_canonical
from real_motion.v20_runtime import sample_scene_features_at_t0_points
from real_motion.v20_training import dormant_source_loss, load_v20_checkpoint, checkpoint_payload
from tools.real_motion import eval_p0_f9_v18_se2 as base
from tools.real_motion import eval_p0_f9_v18_full_validation as full
from tools.real_motion.diagnose_p0_f9_v19_innovation_decomposition import CachedSource
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.train_p0_f9_v18_se2_clean import PROTOCOL as CLEAN_PROTOCOL

PROTOCOL = "p0_f9_v20_dormant_train_v1"
YAW_SET = set(int(x) for x in YAW_ENABLED_CLASS_IDS)


def _lattice(d):
    return CanonicalLattice(
        tuple(float(x) for x in d["origin_xyz_m"]),
        tuple(float(x) for x in d["voxel_size_xyz_m"]),
        tuple(int(x) for x in d["shape_xyz"]),
    )


def _dynamic_ann_map(nusc, token):
    sample = nusc.get("sample", str(token))
    out = {}
    for atok in sample["anns"]:
        ann = nusc.get("sample_annotation", atok)
        cid = category_to_dynamic_class(ann["category_name"])
        if cid is not None:
            out[str(ann["instance_token"])] = (int(cid), ann)
    return out


def _match_tracks_to_history_gt(tracks, source, history_tokens, max_distance_m=4.0):
    """Training-only identity assignment at each track's last real observation."""
    by_frame = {}
    for i, tr in enumerate(tracks):
        by_frame.setdefault(int(tr.last_observed_frame), []).append((i, tr))
    tokens = [None] * len(tracks)
    for ti, rows in by_frame.items():
        amap = _dynamic_ann_map(source.nusc, history_tokens[ti])
        candidates = []
        for local_i, (global_i, tr) in enumerate(rows):
            c = np.asarray(tr.centers_world[ti], dtype=np.float64)
            for tok, (cid, ann) in amap.items():
                if int(cid) != int(tr.class_id):
                    continue
                p = np.asarray(ann["translation"], dtype=np.float64)
                d = float(np.linalg.norm(c[:2] - p[:2]))
                if d <= float(max_distance_m):
                    candidates.append((d, global_i, tok))
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        used_i, used_t = set(), set()
        for _, gi, tok in candidates:
            if gi in used_i or tok in used_t:
                continue
            used_i.add(gi); used_t.add(tok); tokens[gi] = tok
    return tokens


def _targets_for_tracks(tracks, tokens, arrays, source, w, t0_pose):
    n = len(tracks)
    xy = np.zeros((n, 6, 2), dtype=np.float32)
    yaw = np.zeros((n, 6), dtype=np.float32)
    exists = np.zeros((n, 6), dtype=np.float32)
    yaw_valid = np.zeros((n, 6), dtype=bool)
    supervised = np.zeros(n, dtype=bool)
    inv_t0 = np.linalg.inv(np.asarray(t0_pose, dtype=np.float64))
    future = [_dynamic_ann_map(source.nusc, tok) for tok in w.future_tokens]
    history_maps = [_dynamic_ann_map(source.nusc, tok) for tok in w.history_tokens]
    kta = arrays["kta_displacement_xy_m"].numpy()

    for i, (tr, token) in enumerate(zip(tracks, tokens)):
        if token is None:
            continue
        last_map = history_maps[int(tr.last_observed_frame)]
        if token not in last_map:
            continue
        _, last_ann = last_map[token]
        last_yaw = float(quaternion_yaw(last_ann["rotation"]))
        anchor_world = tr.anchor_center_world(0.5)
        anchor_t0 = (inv_t0 @ np.r_[anchor_world, 1.0])[:3]
        any_future = False
        for h, fmap in enumerate(future):
            if token not in fmap:
                continue
            cid, ann = fmap[token]
            if int(cid) != int(tr.class_id):
                continue
            p = np.asarray(ann["translation"], dtype=np.float64)
            pt0 = (inv_t0 @ np.r_[p, 1.0])[:3]
            desired = pt0[:2] - anchor_t0[:2]
            xy[i, h] = (desired - kta[i, h]).astype(np.float32)
            exists[i, h] = 1.0
            any_future = True
            if int(tr.class_id) in YAW_SET:
                d = float(quaternion_yaw(ann["rotation"])) - last_yaw
                yaw[i, h] = float((d + math.pi) % (2 * math.pi) - math.pi)
                yaw_valid[i, h] = True
        supervised[i] = any_future
    return {
        "target_xy": torch.from_numpy(xy),
        "target_yaw": torch.from_numpy(yaw),
        "target_exists": torch.from_numpy(exists),
        "yaw_valid": torch.from_numpy(yaw_valid),
        "supervised": torch.from_numpy(supervised),
    }


def _scene_and_local(model, sem, obs, poses, coarse, pcfg, tracks, device):
    t0 = np.asarray(poses[-1], dtype=np.float64)
    aligned = align_history_once_to_canonical(
        coarse,
        history_semantic=np.asarray(sem, dtype=np.uint8),
        history_observed=np.asarray(obs, dtype=bool),
        history_ego_to_world=np.asarray(poses, dtype=np.float64),
        t0_ego_to_world=t0,
        native_origin_xyz_m=(pcfg.grid.x_min, pcfg.grid.y_min, pcfg.grid.z_min),
        native_voxel_size_xyz_m=pcfg.grid.voxel_size,
        free_label=int(pcfg.free_label),
    )
    hs = torch.from_numpy(aligned.semantic).to(device).unsqueeze(0)
    ho = torch.from_numpy(aligned.observed).to(device).unsqueeze(0)
    hf = torch.from_numpy(aligned.observed_free).to(device).unsqueeze(0)
    with torch.no_grad():
        scene = model.encode_history(hs, ho, hf)
    inv_t0 = np.linalg.inv(t0)
    pts = []
    for tr in tracks:
        pw = tr.anchor_center_world(float(pcfg.frame_dt_s))
        pts.append((inv_t0 @ np.r_[pw, 1.0])[:3])
    p = torch.as_tensor(np.asarray(pts), dtype=torch.float32, device=device)
    local = sample_scene_features_at_t0_points(scene, p, coarse)
    return scene, local


def _v18_tokens(v18, arrays, ids, device, amp):
    def mv(name, dtype=None):
        x = arrays[name].index_select(0, ids).to(device)
        return x.to(dtype) if dtype is not None else x
    with torch.no_grad(), (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if amp and device.type == "cuda" else nullcontext()
    ):
        out = v18(
            mv("features", torch.float32),
            mv("local_semantic_tube"),
            mv("kta_displacement_xy_m", torch.float32),
            mv("frame_motion_features", torch.float32),
            mv("target_source_mask_tube"),
            return_latents=True,
        )
    return out["history_source_context"].float()


def _train_window(
    model, v18, source, w, raw, pcfg, strong_cfg, coarse, device, amp,
    optimizer, *, synthetic_current_index=None,
):
    history_sem = [np.asarray(x, dtype=np.uint8) for x in raw["history_occ"]]
    history_obs = [np.asarray(x, dtype=bool) for x in raw["history_observed"]]
    poses = [np.asarray(x, dtype=np.float64) for x in raw["history_poses"]]

    if synthetic_current_index is None:
        tracks, _ = build_dynamic_source_memory(
            history_sem, poses, grid=pcfg.grid, strong_cfg=strong_cfg,
            frame_dt_s=float(pcfg.frame_dt_s),
        )
        _, dormant = split_current_and_dormant_tracks(tracks)
        tracks_use = list(dormant)
        sem_use, obs_use = history_sem, history_obs
        if not tracks_use:
            return None
        arrays = prepare_causal_arrays_from_tracks(
            tracks_use, sem_use, poses, grid=pcfg.grid,
            free_label=int(pcfg.free_label), frame_dt_s=float(pcfg.frame_dt_s)
        )
    else:
        synth = recompute_synthetic_t0_occlusion(
            history_sem, history_obs, poses,
            current_source_index=int(synthetic_current_index),
            grid=pcfg.grid, free_label=int(pcfg.free_label),
            frame_dt_s=float(pcfg.frame_dt_s), strong_cfg=strong_cfg,
        )
        tracks_use = [synth.dormant_track]
        arrays = synth.dormant_inputs
        sem_use, obs_use = list(synth.history_semantics), list(synth.history_observed)

    tokens = _match_tracks_to_history_gt(tracks_use, source, w.history_tokens)
    targets = _targets_for_tracks(tracks_use, tokens, arrays, source, w, poses[-1])
    ids = torch.nonzero(targets["supervised"], as_tuple=False).flatten()
    if ids.numel() == 0:
        return None

    tracks_sup = [tracks_use[int(i)] for i in ids.tolist()]
    _, local = _scene_and_local(
        model, sem_use, obs_use, poses, coarse, pcfg, tracks_sup, device
    )
    source_tok = _v18_tokens(v18, arrays, ids, device, amp)
    with (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if amp and device.type == "cuda" else nullcontext()
    ):
        out = model.dormant(source_tok, local)
        loss, stats = dormant_source_loss(
            out,
            target_xy_m=targets["target_xy"].index_select(0, ids).to(device),
            target_yaw_rad=targets["target_yaw"].index_select(0, ids).to(device),
            target_exists=targets["target_exists"].index_select(0, ids).to(device),
            yaw_valid=targets["yaw_valid"].index_select(0, ids).to(device),
        )
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.dormant.parameters(), 5.0)
        optimizer.step()
    stats["sources"] = int(ids.numel())
    stats["synthetic"] = bool(synthetic_current_index is not None)
    return stats


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--static-checkpoint", required=True)
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--synthetic-per-window", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    a = p.parse_args()

    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not bool(a.no_amp)
    model, sck = load_v20_checkpoint(a.static_checkpoint, map_location="cpu")
    if str(sck.get("stage")) != "static":
        raise RuntimeError("Dormant training requires frozen Stage-2 Static checkpoint")
    model.to(device)
    for p0 in model.parameters(): p0.requires_grad = False
    for p0 in model.dormant.parameters(): p0.requires_grad = True
    model.eval(); model.dormant.train()
    extra = dict(sck.get("extra") or {})
    coarse = _lattice(extra["coarse_lattice"])

    _, v18, _ = full._load_model(a.base_checkpoint, CLEAN_PROTOCOL, device)
    v18.eval()
    for p0 in v18.parameters(): p0.requires_grad = False
    if int(model.cfg.source_dim) != int(v18.config.d_model):
        raise RuntimeError(
            f"V20 source_dim={model.cfg.source_dim} != Clean-E14 d_model={v18.config.d_model}"
        )

    pcfg = make_prepare_config(load_runtime_config(a.config, a.override))
    source = CachedSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))
    _, train_records = base.load_cache(a.train_cache)
    _, val_records = base.load_cache(a.val_cache)
    train_scenes = {str(r["scene_name"]) for r in train_records}
    val_scenes = {str(r["scene_name"]) for r in val_records}
    if train_scenes & val_scenes:
        raise RuntimeError("Dormant train/val scene overlap")
    optimizer = torch.optim.AdamW(model.dormant.parameters(), lr=float(a.lr), weight_decay=1e-4)
    rng = random.Random(int(a.seed))
    outdir = Path(a.output_dir)
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"refusing non-empty output dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(1, int(a.epochs) + 1):
        order = list(range(len(train_records))); rng.shuffle(order)
        stats = []
        for ri in order:
            rec = train_records[ri]; w = window_from_record(rec)
            raw = load_nuscenes_window_raw(source, w, pcfg, include_gt=False)
            s = _train_window(
                model, v18, source, w, raw, pcfg, strong_cfg, coarse, device, amp,
                optimizer,
            )
            if s is not None: stats.append(s)

            if int(a.synthetic_per_window) > 0:
                # Select current sources from a clean causal rebuild, then each
                # synthetic sample goes through the full recomputation helper.
                clean_tracks, clean_comps = build_dynamic_source_memory(
                    [np.asarray(x) for x in raw["history_occ"]],
                    [np.asarray(x) for x in raw["history_poses"]],
                    grid=pcfg.grid, strong_cfg=strong_cfg,
                    frame_dt_s=float(pcfg.frame_dt_s),
                )
                ncur = len(clean_comps[-1])
                ids = list(range(ncur)); rng.shuffle(ids)
                for ci in ids[:min(ncur, int(a.synthetic_per_window))]:
                    ss = _train_window(
                        model, v18, source, w, raw, pcfg, strong_cfg, coarse,
                        device, amp, optimizer, synthetic_current_index=ci,
                    )
                    if ss is not None: stats.append(ss)

        model.dormant.eval()
        val_stats = []
        with torch.no_grad():
            for rec in val_records:
                w = window_from_record(rec)
                raw = load_nuscenes_window_raw(source, w, pcfg, include_gt=False)
                s = _train_window(
                    model, v18, source, w, raw, pcfg, strong_cfg, coarse,
                    device, amp, None,
                )
                if s is not None: val_stats.append(s)
        model.dormant.train()

        def mean(rows, key):
            vals = [float(x[key]) for x in rows if key in x]
            return float(np.mean(vals)) if vals else float("nan")
        row = {
            "epoch": epoch,
            "train": {
                "loss": mean(stats, "loss"),
                "exist_bce": mean(stats, "exist_bce"),
                "translation": mean(stats, "translation"),
                "yaw": mean(stats, "yaw"),
                "source_batches": len(stats),
            },
            "val": {
                "loss": mean(val_stats, "loss"),
                "exist_bce": mean(val_stats, "exist_bce"),
                "translation": mean(val_stats, "translation"),
                "yaw": mean(val_stats, "yaw"),
                "source_batches": len(val_stats),
            },
        }
        history.append(row); print(json.dumps(row))
        payload = checkpoint_payload(
            model,
            stage="dormant",
            v18_checkpoint=str(Path(a.base_checkpoint).resolve()),
            thresholds={},
            extra={
                **extra,
                "train_protocol": PROTOCOL,
                "epoch": epoch,
                "parent_static_checkpoint": str(Path(a.static_checkpoint).resolve()),
                "history": history,
                "checkpoint_selection": "formal Dormant composed Moving/semantic metrics on dev split",
            },
        )
        torch.save(payload, outdir / f"epoch_{epoch:04d}.pt")
        torch.save(payload, outdir / "latest.pt")
    (outdir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
