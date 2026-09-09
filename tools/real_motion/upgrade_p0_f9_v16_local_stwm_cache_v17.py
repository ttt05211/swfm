#!/usr/bin/env python3
"""Upgrade an existing V16 Local-STWM cache to the V17 representation contract.

No nuScenes files are re-read.  The expensive six-frame semantic tube is reused
verbatim.  V17 adds only quantities derivable from that causal cache:
  - explicit per-frame motion coordinates from the frozen 46-D v13 features;
  - a target-source mask derived from the source-centered semantic tube, class,
    track validity and Strong source extent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from real_motion.local_st_world_model import LOCAL_STWM_CACHE_VERSION, LOCAL_TUBE_CONTRACT
from real_motion.local_st_world_model_v17 import (
    FRAME_MOTION_CONTRACT,
    LOCAL_STWM_V17_CACHE_VERSION,
    REPRESENTATION_CONTRACT,
    SOURCE_MASK_CONTRACT,
    frame_motion_features_from_flat,
    target_source_mask_from_tube,
)
from real_motion.motion_transport_v2 import TARGET_CONTRACT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="existing V16 local-STWM cache")
    p.add_argument("--output", required=True, help="new V17 cache path")
    a = p.parse_args()

    src = torch.load(a.input, map_location="cpu", weights_only=False)
    if src.get("version") != LOCAL_STWM_CACHE_VERSION:
        raise RuntimeError(f"expected V16 cache version {LOCAL_STWM_CACHE_VERSION}")
    meta = dict(src.get("metadata") or {})
    if meta.get("target_contract") != TARGET_CONTRACT:
        raise RuntimeError("target contract mismatch")
    if meta.get("local_tube_contract") != LOCAL_TUBE_CONTRACT:
        raise RuntimeError("local tube contract mismatch")
    records = src.get("records") or []
    if not records:
        raise RuntimeError("input cache has no records")

    resolution = float(meta.get("patch_resolution_m", 0.8))
    out_records = []
    source_count = 0
    valid_track_frames = mask_present_frames = 0
    t0_sources = t0_mask_present = 0

    for wi, rec in enumerate(records, start=1):
        r = dict(rec)
        features = r["features"].float()
        tube = r["local_semantic_tube"].to(torch.uint8)
        track_valid = r["track_valid"].bool()
        class_id = r["source_class_id"].long()
        frame_motion = frame_motion_features_from_flat(features)
        source_mask = target_source_mask_from_tube(
            tube,
            class_id,
            track_valid,
            features,
            patch_resolution_m=resolution,
        )
        r["frame_motion_features"] = frame_motion.to(torch.float32)
        r["target_source_mask_tube"] = source_mask.to(torch.uint8)
        out_records.append(r)

        n = int(features.shape[0]); source_count += n; t0_sources += n
        valid_track_frames += int(track_valid.sum().item())
        present = source_mask.flatten(2).any(dim=-1)
        mask_present_frames += int((present & track_valid).sum().item())
        if n:
            t0_mask_present += int(present[:, -1].sum().item())
        if wi == 1 or wi % 500 == 0 or wi == len(records):
            print(
                f"v17_cache_upgrade {wi}/{len(records)} sources={source_count} "
                f"t0_mask={t0_mask_present}/{max(t0_sources,1)}"
            )

    new_meta = dict(meta)
    new_meta.update({
        "version": LOCAL_STWM_V17_CACHE_VERSION,
        "upgraded_from_version": LOCAL_STWM_CACHE_VERSION,
        "upgraded_from_path": str(Path(a.input).resolve()),
        "representation_contract": REPRESENTATION_CONTRACT,
        "frame_motion_contract": FRAME_MOTION_CONTRACT,
        "source_mask_contract": SOURCE_MASK_CONTRACT,
        "frame_motion_dim": 5,
        "target_source_mask_dtype": "uint8",
        "target_source_mask_is_causal": True,
        "target_source_mask_reads_future": False,
        "num_windows": len(out_records),
        "num_sources": source_count,
        "source_mask_valid_track_frame_fraction": mask_present_frames / max(valid_track_frames, 1),
        "source_mask_t0_fraction": t0_mask_present / max(t0_sources, 1),
    })
    payload = {
        "version": LOCAL_STWM_V17_CACHE_VERSION,
        "metadata": new_meta,
        "records": out_records,
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, op)
    op.with_suffix(".summary.json").write_text(json.dumps(new_meta, indent=2), encoding="utf-8")
    print(json.dumps(new_meta, indent=2))
    print(f"saved {op}")


if __name__ == "__main__":
    main()
