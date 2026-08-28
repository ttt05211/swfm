#!/usr/bin/env python3
"""P0-D route scan: compare causal support schedules by oracle quality and window cost.

This diagnostic never trains a model.  It rebuilds generation support from the
prepared v5 moving-KTA and uncertain-zero branches, composes GT dynamic
semantics only inside that causal support, then reports the oracle upper bound.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.prepared import PreparedShardDataset
from real_motion.support import MotionTubeConfig, build_motion_tube, downsample_support
from real_motion.windows import WindowPlanner, window_coverage
from real_motion.composition import static_protected_compose
from real_motion.metrics.moving_miou_v2 import (
    DYNAMIC_CLASS_IDS,
    MovingMIoUV2MultiHorizon,
    SemanticIoUAccumulator,
    REPORT_HORIZONS_S,
)
from real_motion.nuscenes_adapter import (
    gt_true_static_mask,
    dynamic_only_semantics,
    causal_dynamic_target_semantics,
)
from real_motion.runtime_config import (
    add_config_args,
    load_runtime_config,
    get_cfg,
    save_resolved_config,
)

REPORT = {1.0: 1, 2.0: 3, 3.0: 5}

DEFAULT_ROUTES = {
    "current": {
        "moving_radii": [1, 2, 3, 4, 5, 6],
        "uncertain_radii": [0, 0, 0, 0, 0, 0],
    },
    "balanced": {
        "moving_radii": [1, 2, 3, 4, 4, 5],
        "uncertain_radii": [0, 0, 0, 1, 2, 3],
    },
    "efficient": {
        "moving_radii": [1, 1, 1, 3, 3, 4],
        "uncertain_radii": [0, 0, 0, 0, 1, 1],
    },
}


def _mean_h(acc):
    per = {h: acc[h].compute() for h in REPORT_HORIZONS_S}
    return {
        "mIoU": float(np.nanmean([per[h]["mIoU"] for h in REPORT_HORIZONS_S])),
        "per_horizon": per,
    }


def _parse_routes(text):
    if text is None:
        return DEFAULT_ROUTES
    payload = json.loads(Path(text).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("route JSON must be a non-empty object")
    for name, cfg in payload.items():
        for key in ("moving_radii", "uncertain_radii"):
            vals = list(cfg.get(key, []))
            if len(vals) != 6 or any(int(v) < 0 for v in vals):
                raise ValueError(f"{name}.{key} must contain six non-negative radii")
            cfg[key] = [int(v) for v in vals]
    return payload


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--prepared", required=True)
    p.add_argument("--routes-json", default=None,
                   help="optional JSON file overriding the built-in current/balanced/efficient routes")
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    cfg = load_runtime_config(a.config, a.override)
    routes = _parse_routes(a.routes_json)
    ds = PreparedShardDataset(a.prepared)
    n = len(ds) if a.max_windows is None else min(len(ds), a.max_windows)

    latent_hw = tuple(int(v) for v in get_cfg(cfg, "UPSTREAM.LATENT_HW", [50, 50]))
    window_hw = get_cfg(cfg, "MODEL.WINDOW_HW", [20, 20])
    if int(window_hw[0]) != int(window_hw[1]):
        raise ValueError("P0 route scan currently expects square windows")
    window = int(window_hw[0])
    max_slots = int(get_cfg(cfg, "MODEL.MAX_WINDOWS", 8))
    planner = WindowPlanner((window, window), max_slots)

    # Full decomposition oracle is common to every route.
    full_overall = {h: SemanticIoUAccumulator() for h in REPORT_HORIZONS_S}
    full_moving = MovingMIoUV2MultiHorizon()

    state = {}
    for name in routes:
        state[name] = {
            "overall": {h: SemanticIoUAccumulator() for h in REPORT_HORIZONS_S},
            "moving": MovingMIoUV2MultiHorizon(),
            "active_bev": np.zeros(6, dtype=np.float64),
            "active_latent": np.zeros(6, dtype=np.float64),
            "window_count": [],
            "slot_compute_ratio": [],
            "window_coverage": [],
        }

    for i in range(n):
        s = ds[i]
        moving = torch.from_numpy(np.asarray(s["moving_kta_support"])).bool()
        uncertain = torch.from_numpy(np.asarray(s["uncertain_zero_support"])).bool()
        history_lat = downsample_support(
            torch.from_numpy(np.asarray(s["history_candidate_support"])).bool(),
            latent_hw,
            extra_radius=0,
        )

        # Route-independent decomposition oracle.
        for h, fi in REPORT.items():
            static = torch.from_numpy(np.asarray(s["static_future_occ"])[fi])
            gt = np.asarray(s["future_gt_occ"])[fi]
            all_dyn = torch.from_numpy(dynamic_only_semantics(gt))
            prot = torch.from_numpy(np.asarray(s["confident_static_future_mask"])[fi])
            oracle = static_protected_compose(
                static, all_dyn, prot, DYNAMIC_CLASS_IDS, write_support=None
            ).numpy()
            full_overall[h].update(oracle, gt)
            full_moving.update(h, oracle, gt, np.asarray(s["gt_moving_support"])[fi])

        for name, rcfg in routes.items():
            mtube = build_motion_tube(
                moving,
                MotionTubeConfig(tuple(rcfg["moving_radii"]), 0),
            )
            utube = build_motion_tube(
                uncertain,
                MotionTubeConfig(tuple(rcfg["uncertain_radii"]), 0),
            )
            support = mtube | utube
            latent = downsample_support(support, latent_hw, extra_radius=0)

            state[name]["active_bev"] += support.float().mean(dim=(1, 2)).numpy()
            state[name]["active_latent"] += latent.float().mean(dim=(1, 2)).numpy()

            req = latent.unsqueeze(0)
            ctx = torch.cat([history_lat, latent], dim=0).unsqueeze(0)
            plan = planner.plan(req, context_support=ctx)
            nw = int(plan.valid.sum())
            state[name]["window_count"].append(nw)
            state[name]["slot_compute_ratio"].append(
                nw * window * window / float(latent_hw[0] * latent_hw[1])
            )
            state[name]["window_coverage"].append(float(window_coverage(req, plan)[0]))

            for h, fi in REPORT.items():
                gt = np.asarray(s["future_gt_occ"])[fi]
                static = torch.from_numpy(np.asarray(s["static_future_occ"])[fi])
                prot = torch.from_numpy(np.asarray(s["confident_static_future_mask"])[fi])
                write = support[fi]
                causal = torch.from_numpy(
                    causal_dynamic_target_semantics(gt, write.numpy())
                )
                oracle = static_protected_compose(
                    static, causal, prot, DYNAMIC_CLASS_IDS, write_support=write
                ).numpy()
                state[name]["overall"][h].update(oracle, gt)
                state[name]["moving"].update(
                    h, oracle, gt, np.asarray(s["gt_moving_support"])[fi]
                )

        if i % 4 == 0:
            print("processed", i, "/", n)

    report = {
        "num_windows": n,
        "decomposition_oracle_overall": _mean_h(full_overall),
        "decomposition_oracle_Moving-mIoU_v2": full_moving.compute(),
        "routes": {},
        "note": (
            "Compare each causal-support oracle against the common decomposition oracle. "
            "The gap is support/reachability loss only; no learned WM is involved."
        ),
    }

    for name, rcfg in routes.items():
        st = state[name]
        report["routes"][name] = {
            "moving_radii": rcfg["moving_radii"],
            "uncertain_radii": rcfg["uncertain_radii"],
            "causal_support_oracle_overall": _mean_h(st["overall"]),
            "causal_support_oracle_Moving-mIoU_v2": st["moving"].compute(),
            "active_bev_per_horizon": (st["active_bev"] / n).tolist(),
            "active_latent_per_horizon": (st["active_latent"] / n).tolist(),
            "window_backend": {
                "mean_num_windows": float(np.mean(st["window_count"])),
                "min_num_windows": int(np.min(st["window_count"])),
                "max_num_windows": int(np.max(st["window_count"])),
                "mean_slot_compute_ratio": float(np.mean(st["slot_compute_ratio"])),
                "mean_window_coverage": float(np.mean(st["window_coverage"])),
            },
        }

    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_resolved_config(cfg, op.with_suffix(".resolved.yaml"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
