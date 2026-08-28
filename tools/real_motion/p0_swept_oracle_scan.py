#!/usr/bin/env python3
"""P0-D2 oracle: endpoint squares vs swept moving corridors.

This script intentionally operates from raw nuScenes/Occ3D windows rather than
changing the prepared-cache contract. It never trains a model. For each raw
window it rebuilds the causal t0 motion split, constructs a current->KTA
swept corridor for MOVING components in the correct common ego frame, then
warps that corridor into each future ego frame before oracle composition.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.composition import static_protected_compose
from real_motion.geometry import ego_compensate_sequence
from real_motion.kta import causal_kta
from real_motion.metrics.moving_miou_v2 import (
    DYNAMIC_CLASS_IDS,
    MovingMIoUV2MultiHorizon,
    SemanticIoUAccumulator,
    REPORT_HORIZONS_S,
)
from real_motion.motion import decompose_masks
from real_motion.nuscenes_adapter import (
    NuScenesWindowSource,
    causal_dynamic_target_semantics,
    dynamic_only_semantics,
)
from real_motion.prepared import (
    _align_observation_sequence,
    load_nuscenes_window_raw,
    prepare_nuscenes_window,
)
from real_motion.runtime_config import (
    add_config_args,
    get_cfg,
    load_runtime_config,
    make_prepare_config,
    save_resolved_config,
)
from real_motion.support import MotionTubeConfig, build_motion_tube, downsample_support
from real_motion.swept_support import swept_support_in_future_ego
from real_motion.windows import WindowPlanner, window_coverage

REPORT = {1.0: 1, 2.0: 3, 3.0: 5}

DEFAULT_ROUTES = {
    "current_endpoint": {
        "geometry": "endpoint",
        "moving_radii": [1, 2, 3, 4, 5, 6],
        "uncertain_radii": [0, 0, 0, 0, 0, 0],
    },
    "balanced_endpoint": {
        "geometry": "endpoint",
        "moving_radii": [1, 2, 3, 4, 4, 5],
        "uncertain_radii": [0, 0, 0, 1, 2, 3],
    },
    "swept_r0_balanced_uncertain": {
        "geometry": "swept",
        "moving_radii": [0, 0, 0, 0, 0, 0],
        "uncertain_radii": [0, 0, 0, 1, 2, 3],
    },
    "swept_r1_balanced_uncertain": {
        "geometry": "swept",
        "moving_radii": [1, 1, 1, 1, 1, 1],
        "uncertain_radii": [0, 0, 0, 1, 2, 3],
    },
}


def _mean_h(acc):
    per = {h: acc[h].compute() for h in REPORT_HORIZONS_S}
    return {
        "mIoU": float(np.nanmean([per[h]["mIoU"] for h in REPORT_HORIZONS_S])),
        "per_horizon": per,
    }


def _new_state():
    return {
        "overall": {h: SemanticIoUAccumulator() for h in REPORT_HORIZONS_S},
        "moving": MovingMIoUV2MultiHorizon(),
        "active_bev": np.zeros(6, dtype=np.float64),
        "active_latent": np.zeros(6, dtype=np.float64),
        "window_count": [],
        "slot_compute_ratio": [],
        "window_coverage": [],
        "gt_total": np.zeros(6, dtype=np.float64),
        "moving_hit": np.zeros(6, dtype=np.float64),
        "uncertain_hit_only": np.zeros(6, dtype=np.float64),
        "missed": np.zeros(6, dtype=np.float64),
    }


def _parse_routes(path):
    if path is None:
        return DEFAULT_ROUTES
    routes = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(routes, dict) or not routes:
        raise ValueError("route JSON must be a non-empty object")
    for name, cfg in routes.items():
        if cfg.get("geometry") not in {"endpoint", "swept"}:
            raise ValueError(f"{name}.geometry must be endpoint or swept")
        for key in ("moving_radii", "uncertain_radii"):
            vals = [int(v) for v in cfg.get(key, [])]
            if len(vals) != 6 or any(v < 0 for v in vals):
                raise ValueError(f"{name}.{key} must contain six non-negative radii")
            cfg[key] = vals
    return routes


def _accumulate_support_hits(st, gt_support, mtube, utube):
    """Count GT moving voxels covered by 2-D MOVING/UNCERTAIN supports.

    gt_support is [F,X,Y,Z], while mtube/utube are BEV [F,X,Y]. The BEV write
    permission applies to every height cell at that XY location, so expand only
    the support masks over Z before intersecting. This keeps the attribution a
    true voxel count and avoids accidentally leaving the Z dimension unsummed.
    """
    gt_support = gt_support.bool()
    mtube = mtube.bool()
    utube = utube.bool()
    if gt_support.ndim != 4:
        raise ValueError(f"gt_moving_support must be [F,X,Y,Z], got {tuple(gt_support.shape)}")
    if tuple(gt_support.shape[:3]) != tuple(mtube.shape) or tuple(mtube.shape) != tuple(utube.shape):
        raise ValueError(
            f"support shape mismatch: gt={tuple(gt_support.shape)}, "
            f"moving={tuple(mtube.shape)}, uncertain={tuple(utube.shape)}"
        )

    mtube_vox = mtube.unsqueeze(-1)
    utube_vox = utube.unsqueeze(-1)
    reduce_dims = (1, 2, 3)
    st["gt_total"] += gt_support.sum(dim=reduce_dims, dtype=torch.float64).numpy()
    st["moving_hit"] += (
        gt_support & mtube_vox
    ).sum(dim=reduce_dims, dtype=torch.float64).numpy()
    st["uncertain_hit_only"] += (
        gt_support & ~mtube_vox & utube_vox
    ).sum(dim=reduce_dims, dtype=torch.float64).numpy()
    st["missed"] += (
        gt_support & ~(mtube_vox | utube_vox)
    ).sum(dim=reduce_dims, dtype=torch.float64).numpy()


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--max-windows", type=int, default=16)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--routes-json", default=None)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    routes = _parse_routes(a.routes_json)
    source = NuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)

    latent_hw = tuple(int(v) for v in get_cfg(cfg, "UPSTREAM.LATENT_HW", [50, 50]))
    window_hw = get_cfg(cfg, "MODEL.WINDOW_HW", [20, 20])
    if int(window_hw[0]) != int(window_hw[1]):
        raise ValueError("P0-D2 currently expects square windows")
    window = int(window_hw[0])
    max_slots = int(get_cfg(cfg, "MODEL.MAX_WINDOWS", 8))
    planner = WindowPlanner((window, window), max_slots)

    full_overall = {h: SemanticIoUAccumulator() for h in REPORT_HORIZONS_S}
    full_moving = MovingMIoUV2MultiHorizon()
    state = {name: _new_state() for name in routes}
    processed = 0

    windows = source.iter_windows(
        history=pcfg.history_frames,
        future=pcfg.future_frames,
        stride=a.stride,
        max_windows=a.max_windows,
    )
    for i, w in enumerate(windows):
        raw = load_nuscenes_window_raw(source, w, pcfg, include_gt=True)
        base = prepare_nuscenes_window(source, w, pcfg, include_gt=True, raw=raw)

        aligned_hist = ego_compensate_sequence(
            raw["history_occ"], raw["history_poses"], -1, pcfg.grid, pcfg.free_label
        )
        aligned_obs = _align_observation_sequence(
            raw["history_observation_mask"] if "history_observation_mask" in raw else raw["history_observed"],
            raw["history_poses"], -1, pcfg.grid,
        )
        t0_masks = decompose_masks(aligned_hist, pcfg.motion, history_observed=aligned_obs)
        horizons = [(j + 1) * pcfg.frame_dt_s for j in range(pcfg.future_frames)]
        _, _, components = causal_kta(
            aligned_hist, t0_masks.moving, horizons, pcfg.grid, pcfg.kta
        )
        swept = torch.from_numpy(
            swept_support_in_future_ego(
                components,
                horizons,
                raw["history_poses"][-1],
                raw["future_poses"],
                pcfg.grid,
            )
        ).bool()
        endpoint = torch.from_numpy(np.asarray(base["moving_kta_support"])).bool()
        uncertain = torch.from_numpy(np.asarray(base["uncertain_zero_support"])).bool()
        history_lat = downsample_support(
            torch.from_numpy(np.asarray(base["history_candidate_support"])).bool(),
            latent_hw,
            extra_radius=0,
        )

        for h, fi in REPORT.items():
            static = torch.from_numpy(np.asarray(base["static_future_occ"])[fi])
            gt = np.asarray(base["future_gt_occ"])[fi]
            all_dyn = torch.from_numpy(dynamic_only_semantics(gt))
            prot = torch.from_numpy(np.asarray(base["confident_static_future_mask"])[fi])
            oracle = static_protected_compose(
                static, all_dyn, prot, DYNAMIC_CLASS_IDS, write_support=None
            ).numpy()
            full_overall[h].update(oracle, gt)
            full_moving.update(h, oracle, gt, np.asarray(base["gt_moving_support"])[fi])

        for name, rcfg in routes.items():
            moving_base = swept if rcfg["geometry"] == "swept" else endpoint
            mtube = build_motion_tube(
                moving_base,
                MotionTubeConfig(tuple(rcfg["moving_radii"]), 0),
            )
            utube = build_motion_tube(
                uncertain,
                MotionTubeConfig(tuple(rcfg["uncertain_radii"]), 0),
            )
            support = mtube | utube
            latent = downsample_support(support, latent_hw, extra_radius=0)
            st = state[name]
            st["active_bev"] += support.float().mean(dim=(1, 2)).numpy()
            st["active_latent"] += latent.float().mean(dim=(1, 2)).numpy()

            req = latent.unsqueeze(0)
            ctx = torch.cat([history_lat, latent], dim=0).unsqueeze(0)
            plan = planner.plan(req, context_support=ctx)
            nw = int(plan.valid.sum())
            st["window_count"].append(nw)
            st["slot_compute_ratio"].append(
                nw * window * window / float(latent_hw[0] * latent_hw[1])
            )
            st["window_coverage"].append(float(window_coverage(req, plan)[0]))

            gt_support = torch.from_numpy(np.asarray(base["gt_moving_support"])).bool()
            _accumulate_support_hits(st, gt_support, mtube, utube)

            for h, fi in REPORT.items():
                gt = np.asarray(base["future_gt_occ"])[fi]
                static = torch.from_numpy(np.asarray(base["static_future_occ"])[fi])
                prot = torch.from_numpy(np.asarray(base["confident_static_future_mask"])[fi])
                write = support[fi]
                causal = torch.from_numpy(causal_dynamic_target_semantics(gt, write.numpy()))
                oracle = static_protected_compose(
                    static, causal, prot, DYNAMIC_CLASS_IDS, write_support=write
                ).numpy()
                st["overall"][h].update(oracle, gt)
                st["moving"].update(h, oracle, gt, np.asarray(base["gt_moving_support"])[fi])

        processed += 1
        if i % 4 == 0:
            print("processed", i, "sample", base["sample_id"])

    if processed == 0:
        raise RuntimeError("no windows were processed")

    report = {
        "num_windows": processed,
        "decomposition_oracle_overall": _mean_h(full_overall),
        "decomposition_oracle_Moving-mIoU_v2": full_moving.compute(),
        "routes": {},
        "notes": [
            "No learned world model is involved; decomposition-to-route gaps are support/reachability loss only.",
            "support_hit_attribution is voxel accounting: moving_hit is covered by the MOVING branch, uncertain_hit_only is additionally recovered by the UNCERTAIN branch, and missed is outside both. It does not claim instance identity attribution.",
            "Swept support is built in the common t0 ego frame per moving component, then warped to each future ego frame; future-frame masks are never directly stacked.",
        ],
    }

    for name, rcfg in routes.items():
        st = state[name]
        denom = np.maximum(st["gt_total"], 1.0)
        report["routes"][name] = {
            "geometry": rcfg["geometry"],
            "moving_radii": rcfg["moving_radii"],
            "uncertain_radii": rcfg["uncertain_radii"],
            "causal_support_oracle_overall": _mean_h(st["overall"]),
            "causal_support_oracle_Moving-mIoU_v2": st["moving"].compute(),
            "active_bev_per_horizon": (st["active_bev"] / processed).tolist(),
            "active_latent_per_horizon": (st["active_latent"] / processed).tolist(),
            "support_hit_attribution": {
                "moving_hit_ratio_per_horizon": (st["moving_hit"] / denom).tolist(),
                "uncertain_hit_only_ratio_per_horizon": (st["uncertain_hit_only"] / denom).tolist(),
                "missed_ratio_per_horizon": (st["missed"] / denom).tolist(),
                "gt_moving_voxels_per_horizon": st["gt_total"].tolist(),
            },
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
