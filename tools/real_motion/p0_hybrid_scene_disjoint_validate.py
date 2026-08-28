#!/usr/bin/env python3
"""Large P0-D2 validation on one deterministic window per validation scene.

This is the pre-training support freeze check. It intentionally samples at most
one window from each eligible nuScenes validation scene so that the result is not
dominated by many highly correlated windows from the same scene. For each scene,
the temporal midpoint window is used; eligible scenes are then deterministically
shuffled with a frozen seed before taking the requested cap.

No learned world model is involved. The script compares only the three routes
that survived the 16-window smoke study:
- balanced_endpoint
- hybrid_balanced_r1
- hybrid_compact_r1
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
    WindowTokens,
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
from tools.real_motion.p0_hybrid_oracle_scan import (
    DEFAULT_ROUTES,
    _accumulate_partition,
    _build_moving_support,
    _mean_h,
    _new_state,
    _partition_report,
)

REPORT = {1.0: 1, 2.0: 3, 3.0: 5}
VALIDATION_ROUTE_NAMES = (
    "balanced_endpoint",
    "hybrid_balanced_r1",
    "hybrid_compact_r1",
)
DEFAULT_SEED = 20260828


def select_scene_disjoint_windows(source, history=6, future=6, stride=1,
                                  max_windows=128, seed=DEFAULT_SEED):
    """Select <=1 midpoint window per eligible scene, deterministically.

    Returns a list of ``WindowTokens``. Scene eligibility follows the temporal
    info pickle through ``source.allowed_scenes`` when present.
    """
    if history <= 0 or future <= 0 or stride <= 0:
        raise ValueError("history, future, and stride must be positive")
    if max_windows is not None and max_windows <= 0:
        raise ValueError("max_windows must be positive or None")

    candidates = []
    for scene in source.nusc.scene:
        scene_name = scene["name"]
        if source.allowed_scenes is not None and scene_name not in source.allowed_scenes:
            continue
        tokens = source.scene_tokens(scene)
        eligible = list(range(history - 1, len(tokens) - future, stride))
        if not eligible:
            continue
        center_i = eligible[len(eligible) // 2]
        candidates.append(
            WindowTokens(
                scene_name=scene_name,
                history_tokens=tuple(tokens[center_i - history + 1:center_i + 1]),
                t0_token=tokens[center_i],
                future_tokens=tuple(tokens[center_i + 1:center_i + future + 1]),
            )
        )

    candidates.sort(key=lambda w: w.scene_name)
    if candidates:
        rng = np.random.default_rng(int(seed))
        order = rng.permutation(len(candidates)).tolist()
        candidates = [candidates[i] for i in order]
    if max_windows is not None:
        candidates = candidates[:int(max_windows)]
    return candidates


def _validation_routes():
    return {
        name: {
            "geometry": DEFAULT_ROUTES[name]["geometry"],
            "endpoint_radii": list(DEFAULT_ROUTES[name]["endpoint_radii"]),
            "swept_radii": list(DEFAULT_ROUTES[name]["swept_radii"]),
            "uncertain_radii": list(DEFAULT_ROUTES[name]["uncertain_radii"]),
        }
        for name in VALIDATION_ROUTE_NAMES
    }


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--max-windows", type=int, default=128,
                   help="Maximum number of distinct validation scenes/windows")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--scene-seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    routes = _validation_routes()
    source = NuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)

    windows = select_scene_disjoint_windows(
        source,
        history=pcfg.history_frames,
        future=pcfg.future_frames,
        stride=a.stride,
        max_windows=a.max_windows,
        seed=a.scene_seed,
    )
    if not windows:
        raise RuntimeError("no eligible scene-disjoint validation windows")
    if len({w.scene_name for w in windows}) != len(windows):
        raise AssertionError("scene-disjoint selector produced duplicate scenes")

    latent_hw = tuple(int(v) for v in get_cfg(cfg, "UPSTREAM.LATENT_HW", [50, 50]))
    window_hw = get_cfg(cfg, "MODEL.WINDOW_HW", [20, 20])
    if int(window_hw[0]) != int(window_hw[1]):
        raise ValueError("scene-disjoint hybrid validation expects square windows")
    window = int(window_hw[0])
    max_slots = int(get_cfg(cfg, "MODEL.MAX_WINDOWS", 8))
    planner = WindowPlanner((window, window), max_slots)

    full_overall = {h: SemanticIoUAccumulator() for h in REPORT_HORIZONS_S}
    full_moving = MovingMIoUV2MultiHorizon()
    state = {name: _new_state() for name in routes}
    processed = 0
    sample_ids = []

    for i, w in enumerate(windows):
        raw = load_nuscenes_window_raw(source, w, pcfg, include_gt=True)
        base = prepare_nuscenes_window(source, w, pcfg, include_gt=True, raw=raw)
        sample_ids.append(str(base["sample_id"]))

        aligned_hist = ego_compensate_sequence(
            raw["history_occ"], raw["history_poses"], -1, pcfg.grid, pcfg.free_label
        )
        aligned_obs = _align_observation_sequence(
            raw["history_observed"], raw["history_poses"], -1, pcfg.grid
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
        metric_support = torch.from_numpy(np.asarray(base["gt_moving_support"])).bool()
        future_arrival = torch.from_numpy(
            np.asarray(base["future_moving_occ"]) != pcfg.free_label
        ).bool()

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
            mtube = _build_moving_support(endpoint, swept, rcfg)
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

            _accumulate_partition(st, "metric", metric_support, mtube, utube)
            _accumulate_partition(st, "arrival", future_arrival, mtube, utube)

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
        if i % 8 == 0:
            print("processed", i, "scene", w.scene_name, "sample", base["sample_id"])

    report = {
        "validation_protocol": {
            "name": "scene_disjoint_midpoint_one_window_per_scene_v1",
            "requested_max_windows": int(a.max_windows),
            "num_windows": processed,
            "num_unique_scenes": len({w.scene_name for w in windows}),
            "scene_seed": int(a.scene_seed),
            "one_window_per_scene": True,
            "window_choice_within_scene": "middle eligible temporal window",
            "scene_names": [w.scene_name for w in windows],
            "sample_ids": sample_ids,
        },
        "decomposition_oracle_overall": _mean_h(full_overall),
        "decomposition_oracle_Moving-mIoU_v2": full_moving.compute(),
        "routes": {},
        "notes": [
            "Pre-training support-freeze validation; no learned world model is involved.",
            "Each selected window comes from a different validation scene.",
            "future_arrival_hit_attribution is the generation-reachability diagnostic used for route selection.",
            "balanced_endpoint, hybrid_balanced_r1, and hybrid_compact_r1 are the only retained routes from the 16-window smoke study.",
        ],
    }

    for name, rcfg in routes.items():
        st = state[name]
        report["routes"][name] = {
            "geometry": rcfg["geometry"],
            "endpoint_radii": rcfg["endpoint_radii"],
            "swept_radii": rcfg["swept_radii"],
            "uncertain_radii": rcfg["uncertain_radii"],
            "causal_support_oracle_overall": _mean_h(st["overall"]),
            "causal_support_oracle_Moving-mIoU_v2": st["moving"].compute(),
            "active_bev_per_horizon": (st["active_bev"] / processed).tolist(),
            "active_latent_per_horizon": (st["active_latent"] / processed).tolist(),
            "future_arrival_hit_attribution": _partition_report(st, "arrival"),
            "moving_metric_support_hit_attribution": _partition_report(st, "metric"),
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
