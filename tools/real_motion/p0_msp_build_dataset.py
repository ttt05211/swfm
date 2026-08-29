#!/usr/bin/env python3
"""Build small GT-supervised MSP probe caches from causal occupancy features.

Train mode uses a deterministic round-robin over scenes to avoid taking hundreds
of adjacent windows from one scene. Validation mode takes exactly one midpoint
window per scene. Future instance GT is stored only as supervision labels; the
MSP feature tensor is built exclusively from causal occupancy components.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.msp import (
    FEATURE_DIM,
    FEATURE_NAMES,
    MSP_CACHE_VERSION,
    build_probe_record,
    validate_probe_record,
)
from real_motion.nuscenes_adapter import NuScenesWindowSource, WindowTokens
from real_motion.runtime_config import (
    add_config_args,
    config_fingerprint,
    load_runtime_config,
    make_prepare_config,
    save_resolved_config,
)

DEFAULT_SEED = 20260829
FEATURE_CONTRACT = "causal_occ_components_only_no_gt_instance_features"
TARGET_CONTRACT = "future_gt_instance_motion_labels_training_only"


def _eligible_windows_for_scene(source, scene, history, future, stride):
    tokens = source.scene_tokens(scene)
    out = []
    for i in range(history - 1, len(tokens) - future, stride):
        out.append(WindowTokens(
            scene_name=scene["name"],
            history_tokens=tuple(tokens[i-history+1:i+1]),
            t0_token=tokens[i],
            future_tokens=tuple(tokens[i+1:i+future+1]),
        ))
    return out


def select_windows(source, *, mode, history=6, future=6, stride=1,
                   max_windows=1024, seed=DEFAULT_SEED):
    if mode not in {"train", "val"}:
        raise ValueError("mode must be train or val")
    if history <= 0 or future <= 0 or stride <= 0:
        raise ValueError("history/future/stride must be positive")
    if max_windows is not None and max_windows <= 0:
        raise ValueError("max_windows must be positive or None")

    groups = []
    for scene in source.nusc.scene:
        if source.allowed_scenes is not None and scene["name"] not in source.allowed_scenes:
            continue
        ws = _eligible_windows_for_scene(source, scene, history, future, stride)
        if ws:
            groups.append((scene["name"], ws))
    groups.sort(key=lambda x: x[0])
    if not groups:
        return []

    rng = np.random.default_rng(int(seed))
    scene_order = rng.permutation(len(groups)).tolist()
    groups = [groups[i] for i in scene_order]

    if mode == "val":
        selected = [ws[len(ws)//2] for _, ws in groups]
        return selected if max_windows is None else selected[:int(max_windows)]

    # Scene-balanced round robin. Shuffle temporal indices independently per
    # scene, then take one window from each scene before taking a second one.
    queues = []
    for _, ws in groups:
        order = rng.permutation(len(ws)).tolist()
        queues.append([ws[i] for i in order])
    selected = []
    round_idx = 0
    while True:
        added = False
        for q in queues:
            if round_idx < len(q):
                selected.append(q[round_idx])
                added = True
                if max_windows is not None and len(selected) >= int(max_windows):
                    return selected
        if not added:
            break
        round_idx += 1
    return selected


def summarize(records):
    num_candidates = sum(int(r["num_candidates"]) for r in records)
    num_matched = sum(int(r["num_matched_candidates"]) for r in records)
    valid = sum(int(r["activation_valid"].sum()) for r in records)
    positive = sum(int(((r["activation"] > 0.5) & r["activation_valid"]).sum()) for r in records)
    moving = sum(int((r["candidate_state"] == 0).sum()) for r in records)
    dormant = sum(int((r["candidate_state"] == 1).sum()) for r in records)
    return {
        "num_records": len(records),
        "num_candidates": num_candidates,
        "num_observed_moving_candidates": moving,
        "num_dormant_candidates": dormant,
        "matched_candidate_ratio": float(num_matched / max(num_candidates, 1)),
        "positive_activation_ratio": float(positive / max(valid, 1)),
        "num_activation_labels": valid,
    }


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--mode", choices=("train", "val"), required=True)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--match-max-distance-m", type=float, default=4.0)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    default_cap = 1024 if a.mode == "train" else 128
    cap = int(a.max_windows if a.max_windows is not None else default_cap)
    source = NuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    windows = select_windows(
        source,
        mode=a.mode,
        history=pcfg.history_frames,
        future=pcfg.future_frames,
        stride=a.stride,
        max_windows=cap,
        seed=a.seed,
    )
    if not windows:
        raise RuntimeError("no MSP probe windows selected")
    if a.mode == "val" and len({w.scene_name for w in windows}) != len(windows):
        raise AssertionError("validation selection is not scene-disjoint")

    records = []
    for i, w in enumerate(windows):
        rec = build_probe_record(
            source,
            w,
            pcfg,
            match_max_distance_m=float(a.match_max_distance_m),
        )
        validate_probe_record(rec, future_frames=pcfg.future_frames)
        records.append(rec)
        if i % 25 == 0:
            print("built", i, rec["sample_id"], "candidates", rec["num_candidates"],
                  "matched", rec["num_matched_candidates"])

    scenes = sorted({str(r["scene_name"]) for r in records})
    metadata = {
        "version": MSP_CACHE_VERSION,
        "mode": a.mode,
        "selection": (
            "scene_balanced_round_robin_v1" if a.mode == "train"
            else "scene_disjoint_midpoint_one_window_per_scene_v1"
        ),
        "seed": int(a.seed),
        "stride": int(a.stride),
        "match_max_distance_m": float(a.match_max_distance_m),
        "feature_dim": FEATURE_DIM,
        "feature_names": list(FEATURE_NAMES),
        "feature_contract": FEATURE_CONTRACT,
        "target_contract": TARGET_CONTRACT,
        "config_contract_sha256": config_fingerprint(cfg, "cache"),
        "num_windows": len(records),
        "num_unique_scenes": len(scenes),
        "scene_names": scenes,
        "summary": summarize(records),
        "resolved_config": cfg,
    }
    payload = {"version": MSP_CACHE_VERSION, "metadata": metadata, "records": records}
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, op)
    save_resolved_config(cfg, op.with_suffix(".resolved.yaml"))
    op.with_suffix(".summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
