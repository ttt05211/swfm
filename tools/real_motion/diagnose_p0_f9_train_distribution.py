#!/usr/bin/env python3
"""Measure how P0-F9's MSP/Top-2 training population differs from native OccFM.

This is a CPU / I/O diagnostic; it does not run a neural network or train
anything. It compares four semantic populations under the exact 6-history +
6-future OccFM temporal protocol:

1. ``native_full_future``: every eligible training window from the supplied
   temporal-info split, full future occupancy. This is the native raw-window
   reference population.
2. ``selected_full_future``: the exact P0-F9/MSP selected windows, still counted
   on the full future occupancy grid. This isolates window-selection bias.
3. ``selected_top2_union``: only cells covered by valid Top-2 latent windows,
   with overlap counted once per selected sample. This isolates routing bias.
4. ``selected_top2_effective``: the actual valid crop slots seen by the sparse
   WM; overlap is counted twice, matching crop-slot exposure.

A selected MSP sample is allowed to have zero valid routed slots. That is a real
property of the frozen cache/training protocol, not a malformed sample. Such a
sample contributes to ``selected_full_future`` but contributes no Top-2 crop
population; the zero-route rate is reported explicitly.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "upstream_occfm"
sys.path[:0] = [str(ROOT), str(UP)]

import numpy as np
import torch

from real_motion.edit_repair import EditTargetCache
from real_motion.edit_repair_v2 import full_edit_supervision
from real_motion.msp_wm_cache import MSPWorldModelCacheDataset, MSP_WM_CACHE_VERSION_V2
from real_motion.native_forecast import class_weights_from_edit_cache
from real_motion.occfm_io import file_sha256
from real_motion.runtime_config import make_prepare_config
from real_motion.training_diagnostics import (
    OCC3D_CLASS_NAMES,
    OCC3D_FREE_ID,
    class_histogram,
    enrichment_ratio,
    jensen_shannon_divergence,
    summarize_class_histogram,
)
from tools.real_motion.build_p0_f5_cache_direct import (
    CachedNuScenesWindowSource,
    _load_probe,
    _window_from_record,
)
from tools.real_motion.build_p0_f9_cache_fast import P0_F9_CACHE_PROTOCOL


COMPACT_NAMES = (
    "background_non_dynamic",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "trailer",
    "truck",
)


def _add_hist(dst: np.ndarray, labels, multiplier: int = 1) -> None:
    if int(multiplier) > 0:
        dst += class_histogram(labels) * int(multiplier)


def _occupied_only(counts: np.ndarray) -> np.ndarray:
    x = np.asarray(counts, dtype=np.int64).copy()
    x[OCC3D_FREE_ID] = 0
    return x


def _comparison(target: np.ndarray, reference: np.ndarray) -> dict:
    all_enrich = enrichment_ratio(target, reference)
    target_occ = _occupied_only(target)
    ref_occ = _occupied_only(reference)
    occ_enrich = enrichment_ratio(target_occ, ref_occ)
    return {
        "js_divergence_all_voxels_nats": jensen_shannon_divergence(target, reference),
        "js_divergence_occupied_only_nats": jensen_shannon_divergence(target_occ, ref_occ),
        "enrichment_all_voxels": {
            OCC3D_CLASS_NAMES[i]: float(all_enrich[i]) for i in range(18)
        },
        "enrichment_occupied_only": {
            OCC3D_CLASS_NAMES[i]: float(occ_enrich[i]) for i in range(17)
        },
    }


def _compact_semantic_report(edit: EditTargetCache) -> dict:
    counts = torch.zeros(len(COMPACT_NAMES), dtype=torch.int64)
    for rec in edit.records.values():
        labels = full_edit_supervision(rec)["result_slots"].long()
        if labels.numel():
            counts += torch.bincount(labels, minlength=len(COMPACT_NAMES)).cpu()
    weights = class_weights_from_edit_cache(edit).cpu().double()
    total = int(counts.sum())
    frac = counts.double() / max(total, 1)
    return {
        "num_records": len(edit.records),
        "total_supervised_voxels": total,
        "counts": {COMPACT_NAMES[i]: int(counts[i]) for i in range(len(COMPACT_NAMES))},
        "fractions": {COMPACT_NAMES[i]: float(frac[i]) for i in range(len(COMPACT_NAMES))},
        "current_inverse_sqrt_clamped_weights": {
            COMPACT_NAMES[i]: float(weights[i]) for i in range(len(COMPACT_NAMES))
        },
    }


def _latent_to_occ_crop(occ: np.ndarray, origin, *, latent_hw, window_hw):
    if occ.ndim != 3:
        raise ValueError(f"Occ3D semantics must be [H,W,D], got {occ.shape}")
    lh, lw = (int(x) for x in latent_hw)
    wh, ww = (int(x) for x in window_hw)
    if occ.shape[0] % lh or occ.shape[1] % lw:
        raise RuntimeError(
            f"occupancy {occ.shape[:2]} is not an integer multiple of latent grid {(lh, lw)}"
        )
    sy, sx = occ.shape[0] // lh, occ.shape[1] // lw
    y0, x0 = int(origin[0]), int(origin[1])
    if y0 < 0 or x0 < 0 or y0 + wh > lh or x0 + ww > lw:
        raise ValueError(f"latent crop origin {(y0, x0)} out of range")
    return occ[y0 * sy:(y0 + wh) * sy, x0 * sx:(x0 + ww) * sx, :]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--msp-cache", required=True)
    p.add_argument("--semantic-targets", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-selected", type=int, default=None)
    p.add_argument("--reference-stride", type=int, default=1)
    a = p.parse_args()
    if a.max_selected is not None and a.max_selected <= 0:
        raise ValueError("max-selected must be positive or omitted")
    if a.reference_stride <= 0:
        raise ValueError("reference-stride must be positive")

    ds = MSPWorldModelCacheDataset(a.train_cache)
    if ds.version != MSP_WM_CACHE_VERSION_V2:
        raise RuntimeError("distribution diagnostic requires the audited P0-F9 v2 cache")
    if ds.metadata.get("protocol") != P0_F9_CACHE_PROTOCOL:
        raise RuntimeError("train cache is not the audited P0-F9 native-future cache")
    if list(ds.metadata.get("latent_hw", [])) != [50, 50]:
        raise RuntimeError("diagnostic expects the released OccFM 50x50 latent grid")
    if list(ds.metadata.get("window_hw", [])) != [20, 20] or int(ds.metadata.get("topk", -1)) != 2:
        raise RuntimeError("diagnostic expects frozen Top-2 20x20 routing")

    _, records, cfg = _load_probe(a.msp_cache)
    pcfg = make_prepare_config(cfg)
    if int(pcfg.history_frames) != 6 or int(pcfg.future_frames) != 6:
        raise RuntimeError("formal OccFM distribution diagnostic requires 6+6 windows")
    record_map = {str(r["sample_id"]): r for r in records}
    if len(record_map) != len(records):
        raise RuntimeError("MSP cache contains duplicate sample IDs")

    source = CachedNuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    selected_n = len(ds) if a.max_selected is None else min(len(ds), int(a.max_selected))

    # key=(scene_name, future_token) -> one origin-list per selected sample that
    # sees that frame. Empty origin-lists are deliberately retained so the full
    # selected-window population remains exact even for zero-route samples.
    selected_frame_instances = defaultdict(list)
    selected_scenes = set()
    route_slot_hist = Counter()
    zero_route_sample_ids = []
    for i in range(selected_n):
        sample = ds[i]
        sid = str(sample["sample_id"])
        if sid not in record_map:
            raise RuntimeError(f"selected cache sample {sid} missing from MSP cache")
        w = _window_from_record(record_map[sid], pcfg.history_frames, pcfg.future_frames)
        if str(sample["scene_name"]) != w.scene_name:
            raise RuntimeError(f"{sid}: P0-F9/MSP scene mismatch")
        origins = sample["window_origins"].long().cpu().numpy()
        valid = sample["window_valid"].bool().cpu().numpy()
        valid_origins = [
            tuple(int(x) for x in origins[k]) for k in range(len(valid)) if bool(valid[k])
        ]
        route_slot_hist[len(valid_origins)] += 1
        if not valid_origins:
            zero_route_sample_ids.append(sid)
        selected_scenes.add(w.scene_name)
        for tok in w.future_tokens:
            selected_frame_instances[(w.scene_name, str(tok))].append(valid_origins)

    # Native OccFM raw-window reference. Count token multiplicity first so each
    # Occ3D frame is read only once even though chronological windows overlap.
    native_frame_multiplicity = Counter()
    native_windows = 0
    native_scenes = set()
    for w in source.iter_windows(
        history=pcfg.history_frames,
        future=pcfg.future_frames,
        stride=int(a.reference_stride),
        max_windows=None,
    ):
        native_windows += 1
        native_scenes.add(w.scene_name)
        for tok in w.future_tokens:
            native_frame_multiplicity[(w.scene_name, str(tok))] += 1
    if native_windows <= 0:
        raise RuntimeError("native reference split contains no eligible 6+6 windows")

    h_native = np.zeros(18, dtype=np.int64)
    h_selected_full = np.zeros(18, dtype=np.int64)
    h_top2_union = np.zeros(18, dtype=np.int64)
    h_top2_effective = np.zeros(18, dtype=np.int64)

    all_frame_keys = sorted(set(native_frame_multiplicity) | set(selected_frame_instances))
    lh, lw = (int(x) for x in ds.metadata["latent_hw"])
    wh, ww = (int(x) for x in ds.metadata["window_hw"])
    for n, key in enumerate(all_frame_keys, 1):
        scene, token = key
        occ = np.asarray(source.load_semantics(scene, token), dtype=np.int64)
        _add_hist(h_native, occ, int(native_frame_multiplicity.get(key, 0)))

        instances = selected_frame_instances.get(key, ())
        if instances:
            # Includes zero-route selected samples by design: selection bias and
            # routing bias are separate populations.
            _add_hist(h_selected_full, occ, len(instances))
            sy, sx = occ.shape[0] // lh, occ.shape[1] // lw
            for origins in instances:
                for origin in origins:
                    _add_hist(
                        h_top2_effective,
                        _latent_to_occ_crop(
                            occ, origin, latent_hw=(lh, lw), window_hw=(wh, ww)
                        ),
                    )
                if origins:
                    mask = np.zeros(occ.shape[:2], dtype=bool)
                    for y0, x0 in origins:
                        mask[y0 * sy:(y0 + wh) * sy, x0 * sx:(x0 + ww) * sx] = True
                    _add_hist(h_top2_union, occ[mask])
        if n == 1 or n % 500 == 0 or n == len(all_frame_keys):
            print(f"semantic frames {n}/{len(all_frame_keys)}")

    populations = {
        "native_full_future": h_native,
        "selected_full_future": h_selected_full,
        "selected_top2_union": h_top2_union,
        "selected_top2_effective": h_top2_effective,
    }
    for name, hist in populations.items():
        if int(hist.sum()) <= 0:
            raise RuntimeError(f"population {name} is empty; cannot compute distribution statistics")

    zero_n = len(zero_route_sample_ids)
    report = {
        "protocol": "p0_f9_training_distribution_diagnostic_v2",
        "provenance": {
            "train_cache": str(Path(a.train_cache).resolve()),
            "train_cache_index_sha256": file_sha256(Path(a.train_cache) / "index.json"),
            "msp_cache": str(Path(a.msp_cache).resolve()),
            "msp_cache_sha256": file_sha256(a.msp_cache),
            "semantic_targets": str(Path(a.semantic_targets).resolve()),
            "semantic_targets_sha256": file_sha256(a.semantic_targets),
            "info_pkl": str(Path(a.info_pkl).resolve()),
            "reference_contract": (
                "all eligible chronological 6-history+6-future windows from the supplied "
                "temporal train info; full Occ3D semantics; stride=" + str(a.reference_stride)
            ),
            "selected_contract": "exact P0-F9 MSP-selected cache and frozen Top-2 20x20 latent routes",
        },
        "population_sizes": {
            "native_eligible_windows": native_windows,
            "native_unique_scenes": len(native_scenes),
            "selected_windows": selected_n,
            "selected_unique_scenes": len(selected_scenes),
            "selected_routed_windows": selected_n - zero_n,
            "selected_zero_route_windows": zero_n,
            "selected_zero_route_fraction": float(zero_n / selected_n) if selected_n else float("nan"),
            "valid_route_slot_histogram": {str(k): int(v) for k, v in sorted(route_slot_hist.items())},
            "zero_route_sample_ids_preview": zero_route_sample_ids[:20],
            "selected_unique_future_frames": len(selected_frame_instances),
        },
        "populations": {
            name: summarize_class_histogram(hist) for name, hist in populations.items()
        },
        "vs_native_full_future": {
            name: _comparison(hist, h_native)
            for name, hist in populations.items()
            if name != "native_full_future"
        },
        "current_compact_semantic_objective": _compact_semantic_report(
            EditTargetCache(a.semantic_targets)
        ),
        "notes": [
            "selected_full_future vs native_full_future isolates MSP/window-selection bias",
            "selected_top2_union vs selected_full_future adds spatial-routing bias",
            "selected_top2_effective additionally counts overlap twice, matching valid crop-slot exposure",
            "zero-route MSP-selected samples are retained in selected_full_future but contribute no Top-2 crop voxels",
            "the native reference is a raw-window population comparison, not a claim about optimizer sampling weights",
        ],
    }

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== P0-F9 TRAINING DISTRIBUTION ===")
    print(
        f"native_windows={native_windows} selected_windows={selected_n} "
        f"routed={selected_n-zero_n} zero_route={zero_n} "
        f"({100.0*zero_n/max(selected_n,1):.3f}%) slots={dict(sorted(route_slot_hist.items()))}"
    )
    print(f"{'population':30s} {'occ%':>9s} {'dyn/all%':>10s} {'dyn/occ%':>10s} {'JS(native)':>11s}")
    for name, hist in populations.items():
        s = report["populations"][name]
        js = 0.0 if name == "native_full_future" else report["vs_native_full_future"][name]["js_divergence_all_voxels_nats"]
        print(
            f"{name:30s} {100*s['occupied_fraction']:9.3f} "
            f"{100*s['dynamic_fraction_all_voxels']:10.4f} "
            f"{100*s['dynamic_fraction_occupied_only']:10.3f} {js:11.6f}"
        )

    enrich = report["vs_native_full_future"]["selected_top2_effective"]["enrichment_occupied_only"]
    print("\n=== TOP-2 OCCUPIED-CLASS ENRICHMENT VS NATIVE ===")
    for name in OCC3D_CLASS_NAMES[:-1]:
        print(f"{name:24s} x{enrich[name]:.4f}")

    compact = report["current_compact_semantic_objective"]
    print("\n=== CURRENT P0-F9 COMPACT SEMANTIC POOL ===")
    for name in COMPACT_NAMES:
        print(
            f"{name:24s} frac={compact['fractions'][name]:.6f} "
            f"weight={compact['current_inverse_sqrt_clamped_weights'][name]:.4f}"
        )
    print("saved", out)


if __name__ == "__main__":
    main()
