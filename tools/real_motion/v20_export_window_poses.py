#!/usr/bin/env python3
"""Export exact V20 train/dev pose windows for Omega-max scanning.

This is intentionally metadata-only: no semantic occupancy or observation
volumes are loaded. Pose lookup is memoized because adjacent V18 windows share
most history/future tokens.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from real_motion.runtime_config import add_config_args
from tools.real_motion import eval_p0_f9_v18_se2 as base
from tools.real_motion.diagnose_p0_f9_v19_innovation_decomposition import CachedSource
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record

PROTOCOL = "p0_f9_v20_pose_windows_v1"


def main():
    p = argparse.ArgumentParser()
    # Keep runtime config args for command compatibility, but pose export does
    # not consume the runtime grid or any semantic data.
    add_config_args(p)
    p.add_argument("--source-cache", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-windows", type=int, default=0)
    a = p.parse_args()

    _, records = base.load_cache(a.source_cache)
    if int(a.max_windows) > 0:
        records = records[: min(len(records), int(a.max_windows))]
    if not records:
        raise RuntimeError("empty source cache")

    source = CachedSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    pose_cache = {}

    def pose(token):
        token = str(token)
        cached = pose_cache.get(token)
        if cached is None:
            cached = np.asarray(source.pose(token), dtype=np.float64)
            if cached.shape != (4, 4):
                raise RuntimeError(
                    f"{token}: expected ego pose [4,4], got {cached.shape}"
                )
            pose_cache[token] = cached
        return cached

    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    scenes = set()
    total = len(records)
    with op.open("w", encoding="utf-8") as f:
        for wi, rec in enumerate(records, start=1):
            w = window_from_record(rec)
            history = np.stack(
                [pose(tok) for tok in w.history_tokens],
                axis=0,
            )
            future = np.stack(
                [pose(tok) for tok in w.future_tokens],
                axis=0,
            )
            if history.shape != (6, 4, 4) or future.shape != (6, 4, 4):
                raise RuntimeError(f"{w.t0_token}: unexpected pose shape")
            scenes.add(str(w.scene_name))
            row = {
                "protocol": PROTOCOL,
                "scene": str(w.scene_name),
                "t0_token": str(w.t0_token),
                "t0_ego_to_world": history[-1].tolist(),
                "history_ego_to_world": history.tolist(),
                "future_ego_to_world": future.tolist(),
            }
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
            if wi == 1 or wi % 1000 == 0 or wi == total:
                print(
                    f"v20_pose_export {wi}/{total} "
                    f"unique_pose_tokens={len(pose_cache)}",
                    flush=True,
                )

    print(json.dumps({
        "protocol": PROTOCOL,
        "windows": total,
        "scenes": len(scenes),
        "unique_pose_tokens": len(pose_cache),
        "pose_source": "CachedSource.pose(token) with token-level memoization",
        "semantic_or_occ_io": False,
        "output": str(op.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
