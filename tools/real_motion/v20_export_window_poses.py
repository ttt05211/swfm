#!/usr/bin/env python3
"""Export exact V20 train/dev pose windows for Ωmax scanning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from real_motion.prepared import load_nuscenes_window_raw
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from tools.real_motion import eval_p0_f9_v18_se2 as base
from tools.real_motion.diagnose_p0_f9_v19_innovation_decomposition import CachedSource
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record

PROTOCOL = "p0_f9_v20_pose_windows_v1"


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--source-cache", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-windows", type=int, default=0)
    a = p.parse_args()

    pcfg = make_prepare_config(load_runtime_config(a.config, a.override))
    _, records = base.load_cache(a.source_cache)
    if int(a.max_windows) > 0:
        records = records[: min(len(records), int(a.max_windows))]
    if not records:
        raise RuntimeError("empty source cache")
    source = CachedSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    scenes = set()
    with op.open("w", encoding="utf-8") as f:
        for rec in records:
            w = window_from_record(rec)
            raw = load_nuscenes_window_raw(source, w, pcfg, include_gt=False)
            history = np.asarray(raw["history_poses"], dtype=np.float64)
            future = np.asarray(raw["future_poses"], dtype=np.float64)
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
    print(json.dumps({
        "protocol": PROTOCOL,
        "windows": len(records),
        "scenes": len(scenes),
        "output": str(op.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
