#!/usr/bin/env python3
"""Merge independently generated V19 Innovation cache shards without reprocessing data."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from real_motion.v19_innovation_training import INNOVATION_CACHE_PROTOCOL


def _same(indexes, key):
    vals = [json.dumps(x.get(key), sort_keys=True) for x in indexes]
    if len(set(vals)) != 1:
        raise RuntimeError(f"cache shard mismatch for {key}")
    return indexes[0].get(key)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()

    roots = [Path(x) for x in a.inputs]
    indexes = [
        json.loads((r / "index.json").read_text(encoding="utf-8"))
        for r in roots
    ]
    if any(x.get("protocol") != INNOVATION_CACHE_PROTOCOL for x in indexes):
        raise RuntimeError("unexpected innovation cache protocol")

    for key in (
        "source_cache",
        "base_checkpoint",
        "base_checkpoint_epoch",
        "future_frames",
        "history_frames",
        "grid_shape_hwd",
        "free_label",
        "positive_mode",
        "positive_categories",
        "explained_state",
    ):
        _same(indexes, key)

    declared = {int(x.get("num_shards", 1)) for x in indexes}
    if len(declared) != 1:
        raise RuntimeError("num_shards mismatch")
    num_shards = declared.pop()
    shard_ids = [int(x.get("shard_index", 0)) for x in indexes]
    if num_shards > 1 and sorted(shard_ids) != list(range(num_shards)):
        raise RuntimeError(
            f"expected shard indexes 0..{num_shards-1}, got {sorted(shard_ids)}"
        )

    out = Path(a.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing non-empty output dir: {out}")
    out.mkdir(parents=True, exist_ok=True)

    merged_shards = []
    next_id = 0
    for root, idx in zip(roots, indexes):
        for row in idx["shards"]:
            src = root / row["file"]
            dst_name = f"shard_{next_id:05d}.pt"
            dst = out / dst_name
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            merged_shards.append(
                {"file": dst_name, "count": int(row["count"])}
            )
            next_id += 1

    total_keys = set().union(
        *(set((x.get("totals") or {}).keys()) for x in indexes)
    )
    totals = {
        k: int(sum(int((x.get("totals") or {}).get(k, 0)) for x in indexes))
        for k in sorted(total_keys)
    }
    cat_keys = set().union(
        *(set((x.get("category_voxels") or {}).keys()) for x in indexes)
    )
    category_voxels = {
        k: int(
            sum(int((x.get("category_voxels") or {}).get(k, 0)) for x in indexes)
        )
        for k in sorted(cat_keys)
    }
    scenes = sorted(
        set().union(*(set(x.get("scene_names", [])) for x in indexes))
    )

    merged = dict(indexes[0])
    merged.update(
        {
            "num_windows": int(sum(int(x["num_windows"]) for x in indexes)),
            "num_scenes": int(len(scenes)),
            "scene_names": scenes,
            "num_shards": 1,
            "shard_index": 0,
            "merged_parallel_shards": len(indexes),
            "merged_from": [str(r.resolve()) for r in roots],
            "totals": totals,
            "category_voxels": category_voxels,
            "shards": merged_shards,
        }
    )
    expected_global = {
        int(x.get("global_num_windows_before_shard", merged["num_windows"]))
        for x in indexes
    }
    if len(expected_global) == 1:
        expected = expected_global.pop()
        if int(merged["num_windows"]) != expected:
            raise RuntimeError(
                f"merged windows {merged['num_windows']} != expected {expected}"
            )
        merged["global_num_windows_before_shard"] = expected

    (out / "index.json").write_text(
        json.dumps(merged, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(out),
                "num_windows": merged["num_windows"],
                "num_scenes": merged["num_scenes"],
                "num_tensor_shards": len(merged_shards),
                "totals": totals,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
