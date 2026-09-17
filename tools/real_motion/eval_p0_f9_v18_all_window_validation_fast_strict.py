#!/usr/bin/env python3
"""Strict-model wrapper for the optimized V18 all-window evaluator.

All CPU/geometry/metric caching optimizations from the fast evaluator are kept,
but Y600/Clean inference is executed one window at a time exactly like the
historical slow evaluator.  This avoids any possible BF16 batch-shape numerical
change from concatenating sources across neighboring windows.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tools.real_motion import eval_p0_f9_v18_all_window_validation_fast as fast


def _predict_scene_strict(records, y_model, c_model, device, *, source_batch_size: int):
    del source_batch_size
    out = {}
    with torch.inference_mode():
        for rec in records:
            sid = str(rec["sample_id"])
            n = int(rec["features"].shape[0])
            if n == 0:
                zxy = np.zeros((0, 6, 2), dtype=np.float32)
                zyaw = np.zeros((0, 6), dtype=np.float32)
                out[sid] = (zxy, zyaw, zxy.copy(), zyaw.copy())
                continue
            f = rec["features"].float().to(device)
            tube = rec["local_semantic_tube"].to(device)
            kta = rec["kta_displacement_xy_m"].float().to(device)
            fm = rec["frame_motion_features"].float().to(device)
            sm = rec["target_source_mask_tube"].to(device)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                yo = y_model(f, tube, kta, fm, sm)
                co = c_model(f, tube, kta, fm, sm)
            out[sid] = (
                yo["residual_xy_m"].float().cpu().numpy(),
                yo["yaw_delta_rad"].float().cpu().numpy(),
                co["residual_xy_m"].float().cpu().numpy(),
                co["yaw_delta_rad"].float().cpu().numpy(),
            )
    return out


def _output_path_from_argv() -> Path | None:
    for i, arg in enumerate(sys.argv[1:], start=1):
        if arg == "--output" and i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1])
        if arg.startswith("--output="):
            return Path(arg.split("=", 1)[1])
    return None


fast._predict_scene = _predict_scene_strict

if __name__ == "__main__":
    output = _output_path_from_argv()
    fast.main()
    if output is not None and output.exists():
        report = json.loads(output.read_text(encoding="utf-8"))
        opt = report.setdefault("optimization", {})
        opt["scene_batched_model_inference"] = False
        opt["model_inference_mode"] = "strict_per_window_historical_bf16"
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
