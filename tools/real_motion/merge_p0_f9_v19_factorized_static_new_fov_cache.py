#!/usr/bin/env python3
"""Merge independently built factorized Static New-FOV cache shards.

Designed for multi-GPU cache construction with --num-shards/--shard-index.
Shard files are hard-linked by default to avoid copying large tensor payloads.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.real_motion.build_p0_f9_v19_factorized_static_new_fov_cache import (
    PROTOCOL,
)


IDENTICAL_KEYS = (
    "protocol",
    "source_cache",
    "base_checkpoint",
    "base_checkpoint_epoch",
    "future_frames",
    "history_frames",
    "grid_shape_hwd",
    "free_label",
    "target",
    "representation",
    "anchor_condition",
    "future_gt_used_for_inference_input",
    "anchor_distance_max_m",
)


def _link(src: Path, dst: Path, mode: str) -> None:
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)
            return
    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    raise ValueError(f"unknown mode: {mode}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
    )
    a = p.parse_args()

    roots = [Path(x).resolve() for x in a.inputs]
    indices = []
    for root in roots:
        idx_path = root / "index.json"
        if not idx_path.is_file():
            raise FileNotFoundError(idx_path)
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
        if idx.get("protocol") != PROTOCOL:
            raise RuntimeError(
                f"unexpected cache protocol in {idx_path}: "
                f"{idx.get('protocol')}"
            )
        indices.append(idx)

    ref = indices[0]
    for i, idx in enumerate(indices[1:], start=1):
        for key in IDENTICAL_KEYS:
            if idx.get(key) != ref.get(key):
                raise RuntimeError(
                    f"cache part {i} mismatch for {key}: "
                    f"{idx.get(key)!r} != {ref.get(key)!r}"
                )

    out = Path(a.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing non-empty output dir: {out}")
    out.mkdir(parents=True, exist_ok=True)

    merged_shards = []
    merged_scenes = set()
    totals = {}
    semantic_hist = {}
    total_windows = 0
    elapsed_sum = 0.0

    next_sid = 0
    for part_i, (root, idx) in enumerate(zip(roots, indices)):
        total_windows += int(idx["num_windows"])
        merged_scenes.update(str(x) for x in idx.get("scene_names", []))
        elapsed_sum += float(idx.get("timing", {}).get("elapsed_s", 0.0))

        for k, v in idx.get("totals", {}).items():
            totals[k] = int(totals.get(k, 0)) + int(v)
        for k, v in idx.get("semantic_positive_column_histogram", {}).items():
            semantic_hist[str(k)] = int(
                semantic_hist.get(str(k), 0)
            ) + int(v)

        for row in idx["shards"]:
            src = root / row["file"]
            if not src.is_file():
                raise FileNotFoundError(src)
            name = f"shard_{next_sid:05d}.pt"
            dst = out / name
            _link(src, dst, a.mode)
            merged_shards.append(
                {
                    "file": name,
                    "count": int(row["count"]),
                    "source_part": int(part_i),
                }
            )
            next_sid += 1

    shard_count_windows = sum(int(x["count"]) for x in merged_shards)
    if shard_count_windows != total_windows:
        raise RuntimeError(
            f"window count mismatch: index={total_windows}, "
            f"shards={shard_count_windows}"
        )

    pos_bev = int(totals.get("positive_bev_columns", 0))
    cand_bev = int(totals.get("candidate_bev_columns", 0))
    pos_vert = int(totals.get("positive_vertical_voxels", 0))
    sup_vert = int(totals.get("vertical_supervised_voxels", 0))
    inter = int(totals.get("baseline_occ_inter", 0))
    union = int(totals.get("baseline_occ_union", 0))

    merged = dict(ref)
    merged.update(
        {
            "num_windows": int(total_windows),
            "num_scenes": int(len(merged_scenes)),
            "scene_names": sorted(merged_scenes),
            "num_shards": 1,
            "shard_index": 0,
            "merged_from": [str(x) for x in roots],
            "totals": totals,
            "presence_positive_fraction": float(
                pos_bev / max(cand_bev, 1)
            ),
            "presence_neg_pos_ratio": float(
                (cand_bev - pos_bev) / max(pos_bev, 1)
            ),
            "vertical_positive_fraction_on_positive_columns": float(
                pos_vert / max(sup_vert, 1)
            ),
            "vertical_neg_pos_ratio_on_positive_columns": float(
                (sup_vert - pos_vert) / max(pos_vert, 1)
            ),
            "baseline_occ_iou": float(inter / max(union, 1)),
            "semantic_positive_column_histogram": semantic_hist,
            "shards": merged_shards,
            "timing": {
                "merged_part_elapsed_s_sum": float(elapsed_sum),
                "construction_mode": str(a.mode),
            },
        }
    )
    (out / "index.json").write_text(
        json.dumps(merged, indent=2),
        encoding="utf-8",
    )

    print("=== MERGED FACTORIZED STATIC NEW-FOV CACHE ===")
    print(
        json.dumps(
            {
                "parts": len(roots),
                "num_windows": total_windows,
                "num_scenes": len(merged_scenes),
                "tensor_shards": len(merged_shards),
                "presence_positive_fraction": merged[
                    "presence_positive_fraction"
                ],
                "vertical_positive_fraction_on_positive_columns": merged[
                    "vertical_positive_fraction_on_positive_columns"
                ],
                "baseline_occ_iou": merged["baseline_occ_iou"],
                "mode": a.mode,
            },
            indent=2,
        )
    )
    print(f"saved {out / 'index.json'}")


if __name__ == "__main__":
    main()
