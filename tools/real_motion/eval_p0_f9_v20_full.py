#!/usr/bin/env python3
"""Formal same-window Stage-5 evaluation for V20 Static/Dormant/Birth."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.v19_innovation import (
    base_explained_bev,
    build_future_aligned_history_and_static_memory,
    decode_innovation,
)
from real_motion.v19_scene_memory import protected_add_only as v19_protected_add_only
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.v20_birth import birth_query_match_counts
from real_motion.v20_dormant import (
    dormant_future_existence_targets,
    match_dormant_tracks_to_history_gt,
)
from real_motion.v20_evaluation import (
    AdditionAccumulator,
    BirthMatchAccumulator,
    DormantExistenceAccumulator,
    SemanticMetricAccumulator,
    StaticSubsetAccumulator,
    metric_delta,
    population_fingerprint,
)
from real_motion.v20_history_world import (
    CanonicalLattice,
    assert_zero_contribution_identity,
    poses_to_t0_canonical,
    protected_add_only,
)
from real_motion.v20_pipeline import run_v20_modules
from real_motion.v20_runtime import static_subset_masks
from real_motion.v20_training import load_v20_checkpoint
from real_motion.prepared import load_nuscenes_window_raw
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.strong_w2det import StrongW2DetConfig
from tools.real_motion import eval_p0_f9_v18_se2 as base
from tools.real_motion import eval_p0_f9_v18_full_validation as full
from tools.real_motion.benchmark_p0_f9_v18_runtime import (
    _forecast_once,
    _model_forward,
    _release_gpu_inputs,
    _stage_gpu_inputs,
)
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.eval_p0_f9_v19_factorized_static_new_fov import (
    CachedSource,
    _ComponentLRU,
    _moving_support_sequence,
    _prepare_record_from_raw,
)
from tools.real_motion.eval_p0_f9_v19_innovation import (
    PROTOCOL as V19_REFERENCE_EVAL_PROTOCOL,
    _load_innovation as _load_v19_innovation,
)
from tools.real_motion.train_p0_f9_v18_se2_clean import PROTOCOL as CLEAN_PROTOCOL
from tools.real_motion.build_p0_f9_v20_history_cache import PROTOCOL as STAGE1_PROTOCOL

PROTOCOL = "p0_f9_v20_stage5_full_eval_v1"
BASE_VARIANTS = (
    "v18",
    "v20_static",
    "v20_dormant",
    "v20_birth",
    "v20_full",
)


def _lattice(d):
    return CanonicalLattice(
        tuple(float(x) for x in d["origin_xyz_m"]),
        tuple(float(x) for x in d["voxel_size_xyz_m"]),
        tuple(int(x) for x in d["shape_xyz"]),
    )


def _load_stage1_dynamic(cache_dir):
    root = Path(cache_dir)
    idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
    if idx.get("protocol") != STAGE1_PROTOCOL:
        raise RuntimeError("Stage-5 requires a V20 Stage-1 cache for strict dynamic labels")
    out = {}
    for shard in idx["shards"]:
        obj = torch.load(root / shard["file"], map_location="cpu", weights_only=False)
        if obj.get("protocol") != STAGE1_PROTOCOL:
            raise RuntimeError(f"bad Stage-1 shard: {shard['file']}")
        for row in obj["rows"]:
            key = (str(row["scene_name"]), str(row["t0_token"]))
            if key in out:
                raise RuntimeError(f"duplicate Stage-1 row: {key}")
            out[key] = {"dynamic_supervision": row["dynamic_supervision"]}
    return idx, out


def _compose(base_np, *, dormant=None, birth=None, static=None, free_label=17):
    kwargs = {"free_label": int(free_label)}
    if dormant is not None:
        kwargs["dormant"] = torch.from_numpy(np.asarray(dormant))
    if birth is not None:
        kwargs["birth"] = torch.from_numpy(np.asarray(birth))
    if static is not None:
        kwargs["static_world"] = torch.from_numpy(np.asarray(static))
    return protected_add_only(torch.from_numpy(np.asarray(base_np)), **kwargs).numpy()


def _mean_timing(rows):
    keys = sorted(set().union(*(r.keys() for r in rows))) if rows else []
    return {
        k: {
            "mean_ms": float(np.mean([float(r.get(k, 0.0)) for r in rows])),
            "p95_ms": float(np.quantile([float(r.get(k, 0.0)) for r in rows], 0.95)),
        }
        for k in keys
    }


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--stage1-cache", required=True)
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--v20-checkpoint", required=True)
    p.add_argument(
        "--v19-innovation-checkpoint",
        default="",
        help=(
            "Optional exact formal V19 Static+Innovation reference. "
            "It is reported under its real protocol name, not relabeled context-only."
        ),
    )
    p.add_argument("--v19-add-threshold", type=float, default=0.5)
    p.add_argument("--v19-vertical-threshold", type=float, default=0.5)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--alignment-workers", type=int, default=6)
    p.add_argument("--dormant-existence-threshold", type=float, default=0.5)
    p.add_argument("--birth-existence-threshold", type=float, default=0.5)
    p.add_argument("--birth-shape-threshold", type=float, default=0.5)
    p.add_argument("--birth-duplicate-distance-m", type=float, default=2.0)
    p.add_argument("--birth-match-distance-m", type=float, default=4.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    a = p.parse_args()

    pcfg = make_prepare_config(load_runtime_config(a.config, a.override))
    _, records = base.load_cache(a.val_cache)
    if int(a.max_windows) > 0:
        records = records[: min(len(records), int(a.max_windows))]
    if not records:
        raise RuntimeError("empty validation population")
    stage1_idx, dynamic_rows = _load_stage1_dynamic(a.stage1_cache)
    missing = [
        (str(r["scene_name"]), str(r["t0_token"]))
        for r in records
        if (str(r["scene_name"]), str(r["t0_token"])) not in dynamic_rows
    ]
    if missing:
        raise RuntimeError(
            f"Stage-1 cache misses selected validation rows: {missing[:5]}"
        )

    device = torch.device(
        a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    amp = device.type == "cuda" and not bool(a.no_amp)
    base_ck, v18, _ = full._load_model(
        a.base_checkpoint, CLEAN_PROTOCOL, device
    )
    v18.eval()
    model, vck = load_v20_checkpoint(a.v20_checkpoint, map_location="cpu")
    if str(vck.get("stage")) not in {"birth", "full"}:
        raise RuntimeError(
            f"Stage-5 full eval requires Birth-capable checkpoint, got {vck.get('stage')}"
        )
    if str(Path(vck["v18_checkpoint"]).name) != str(Path(a.base_checkpoint).name):
        raise RuntimeError("V20 checkpoint references a different V18 checkpoint")
    extra = dict(vck.get("extra") or {})
    high = _lattice(extra["highres_lattice"])
    coarse = _lattice(extra["coarse_lattice"])
    tile_size = tuple(int(x) for x in extra.get("tile_size_xyz", [32, 32, 16]))
    model.to(device).eval()
    v19_ck = v19_model = None
    variants = list(BASE_VARIANTS)
    if str(a.v19_innovation_checkpoint):
        v19_ck, v19_model = _load_v19_innovation(
            a.v19_innovation_checkpoint, device
        )
        variants.append("v19_static_innovation_reference")

    source = CachedSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))
    component_cache = _ComponentLRU(maxsize=1024)
    metrics = {name: SemanticMetricAccumulator() for name in variants}
    per_scene = {}
    additions = {
        "static_standalone": AdditionAccumulator(),
        "dormant_standalone": AdditionAccumulator(),
        "birth_standalone": AdditionAccumulator(),
        "full_dormant_priority": AdditionAccumulator(),
        "full_birth_priority": AdditionAccumulator(),
        "full_static_priority": AdditionAccumulator(),
    }
    static_subsets = {
        "history_seen_t0_missing": StaticSubsetAccumulator(),
        "never_seen_static_domain": StaticSubsetAccumulator(),
    }
    birth_match = BirthMatchAccumulator()
    dormant_existence = DormantExistenceAccumulator()
    strict_dynamic_counts = {"DORMANT_ANCESTRAL": 0, "BIRTH": 0}
    audit = {
        "windows": 0,
        "dormant_tracks": 0,
        "static_query_voxels": 0,
        "static_active_tiles": 0,
        "static_render_oob": 0,
        "history_observed_oob": 0,
        "dormant_render_oob": 0,
        "birth_render_oob": 0,
        "birth_duplicate_suppressed_query_horizons": 0,
    }
    timing_rows = []
    started = time.perf_counter()

    for wi, rec in enumerate(records, start=1):
        w = window_from_record(rec)
        key = (str(w.scene_name), str(w.t0_token))
        drow = dynamic_rows[key]
        for x in drow["dynamic_supervision"]:
            name = str(x["responsibility_name"])
            if name in strict_dynamic_counts:
                strict_dynamic_counts[name] += 1

        raw = load_nuscenes_window_raw(source, w, pcfg, include_gt=True)
        state = _prepare_record_from_raw(
            rec, raw, source, pcfg, strong_cfg, device, component_cache
        )
        _stage_gpu_inputs(state, device)
        try:
            with torch.inference_mode(), (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if amp and device.type == "cuda" else nullcontext()
            ):
                v18_out = _model_forward(
                    v18, state["gpu"], device, return_latents=True
                )
            pred_stack = np.asarray(
                _forecast_once(
                    v18, state, pcfg, strong_cfg, device,
                    precomputed_out=v18_out,
                ),
                dtype=np.uint8,
            )
        finally:
            _release_gpu_inputs(state)

        vp = run_v20_modules(
            model,
            v18,
            rec,
            raw,
            pcfg=pcfg,
            strong_cfg=strong_cfg,
            high_lattice=high,
            coarse_lattice=coarse,
            tile_size_xyz=tile_size,
            device=device,
            amp=amp,
            v18_output_with_latents=v18_out,
            enable_static=True,
            enable_dormant=True,
            enable_birth=True,
            dormant_existence_threshold=float(a.dormant_existence_threshold),
            birth_existence_threshold=float(a.birth_existence_threshold),
            birth_shape_threshold=float(a.birth_shape_threshold),
            birth_duplicate_distance_m=float(a.birth_duplicate_distance_m),
        )
        p_static = _compose(
            pred_stack, static=vp.static_future, free_label=pcfg.free_label
        )
        p_dormant = _compose(
            pred_stack, dormant=vp.dormant_future, free_label=pcfg.free_label
        )
        p_birth = _compose(
            pred_stack, birth=vp.birth_future, free_label=pcfg.free_label
        )
        p_full = _compose(
            pred_stack,
            dormant=vp.dormant_future,
            birth=vp.birth_future,
            static=vp.static_future,
            free_label=pcfg.free_label,
        )
        # Explicit zero-contribution invariant for this exact V18 population.
        assert_zero_contribution_identity(
            torch.from_numpy(pred_stack),
            protected_add_only(
                torch.from_numpy(pred_stack),
                free_label=int(pcfg.free_label),
            ),
        )

        scene_name = str(w.scene_name)
        if scene_name not in per_scene:
            per_scene[scene_name] = {
                name: SemanticMetricAccumulator() for name in variants
            }
        moving_rows = _moving_support_sequence(
            source, w, grid=pcfg.grid, workers=int(a.alignment_workers)
        )
        pred_by_variant = {
            "v18": pred_stack,
            "v20_static": p_static,
            "v20_dormant": p_dormant,
            "v20_birth": p_birth,
            "v20_full": p_full,
        }
        if v19_model is not None:
            sem19, geo19, _, static19 = (
                build_future_aligned_history_and_static_memory(
                    raw["history_occ"],
                    raw["history_observed"],
                    raw["history_poses"],
                    raw["future_poses"],
                    grid=pcfg.grid,
                    free_label=int(pcfg.free_label),
                    dynamic_class_ids=tuple(int(x) for x in DYNAMIC_CLASS_IDS),
                    workers=int(a.alignment_workers),
                )
            )
            explained19 = np.stack([
                v19_protected_add_only(
                    pred_stack[fi],
                    static19[fi],
                    free_label=int(pcfg.free_label),
                )
                for fi in range(6)
            ]).astype(np.uint8)
            sem19_t = torch.from_numpy(sem19[None]).to(device)
            geo19_t = torch.from_numpy(geo19[None]).to(device)
            base19_t = torch.from_numpy(
                base_explained_bev(
                    explained19,
                    free_label=int(pcfg.free_label),
                )[None]
            ).to(device)
            with torch.inference_mode(), (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if amp and device.type == "cuda" else nullcontext()
            ):
                v19_out = v19_model(sem19_t, geo19_t, base19_t)
            v19_prop = decode_innovation(
                v19_out,
                free_label=int(pcfg.free_label),
                add_threshold=float(a.v19_add_threshold),
                vertical_threshold=float(a.v19_vertical_threshold),
            )[0].cpu().numpy().astype(np.uint8)
            pred_by_variant["v19_static_innovation_reference"] = np.stack([
                v19_protected_add_only(
                    explained19[fi],
                    v19_prop[fi],
                    free_label=int(pcfg.free_label),
                )
                for fi in range(6)
            ]).astype(np.uint8)

        gt_all = np.asarray(raw["future_gt_occ"], dtype=np.uint8)
        for hi in range(6):
            gt = gt_all[hi]
            moving = moving_rows[hi]
            for name, pred in pred_by_variant.items():
                metrics[name].update(
                    hi, pred[hi], gt, moving, free_label=pcfg.free_label
                )
                per_scene[scene_name][name].update(
                    hi, pred[hi], gt, moving, free_label=pcfg.free_label
                )

            additions["static_standalone"].update(
                pred_stack[hi], vp.static_future[hi], gt,
                free_label=pcfg.free_label,
            )
            additions["dormant_standalone"].update(
                pred_stack[hi], vp.dormant_future[hi], gt,
                free_label=pcfg.free_label,
            )
            additions["birth_standalone"].update(
                pred_stack[hi], vp.birth_future[hi], gt,
                free_label=pcfg.free_label,
            )
            after_d = _compose(
                pred_stack[hi], dormant=vp.dormant_future[hi],
                free_label=pcfg.free_label,
            )
            additions["full_dormant_priority"].update(
                pred_stack[hi], vp.dormant_future[hi], gt,
                free_label=pcfg.free_label,
            )
            additions["full_birth_priority"].update(
                after_d, vp.birth_future[hi], gt,
                free_label=pcfg.free_label,
            )
            after_db = _compose(
                pred_stack[hi],
                dormant=vp.dormant_future[hi],
                birth=vp.birth_future[hi],
                free_label=pcfg.free_label,
            )
            additions["full_static_priority"].update(
                after_db, vp.static_future[hi], gt,
                free_label=pcfg.free_label,
            )

        future_obs = np.stack([
            source.load_lidar_observation(str(w.scene_name), str(tok))
            for tok in w.future_tokens
        ])
        hist_pose = np.asarray(raw["history_poses"], dtype=np.float64)
        future_pose = np.asarray(raw["future_poses"], dtype=np.float64)
        t0 = hist_pose[-1]
        masks = static_subset_masks(
            high_lattice=high,
            history_observed=np.asarray(raw["history_observed"], dtype=bool),
            history_ego_to_canonical=poses_to_t0_canonical(hist_pose, t0),
            future_ego_to_canonical=poses_to_t0_canonical(future_pose, t0),
            future_observed=future_obs,
            future_gt_semantic=gt_all,
            native_origin_xyz_m=(
                pcfg.grid.x_min, pcfg.grid.y_min, pcfg.grid.z_min
            ),
            native_voxel_size_xyz_m=pcfg.grid.voxel_size,
            free_label=int(pcfg.free_label),
        )
        for hi in range(6):
            static_subsets["history_seen_t0_missing"].update(
                masks["history_seen_t0_missing"][hi],
                pred_stack[hi], p_static[hi], gt_all[hi],
                free_label=pcfg.free_label,
            )
            static_subsets["never_seen_static_domain"].update(
                masks["never_seen_static_domain"][hi],
                pred_stack[hi], p_static[hi], gt_all[hi],
                free_label=pcfg.free_label,
            )

        if vp.dormant_outputs is not None and vp.dormant_track_objects:
            identity = match_dormant_tracks_to_history_gt(
                vp.dormant_track_objects,
                source.nusc,
                w.history_tokens,
            )
            supervised, target_exists = dormant_future_existence_targets(
                vp.dormant_track_objects,
                identity,
                source.nusc,
                w.future_tokens,
            )
            pred_active = (
                torch.sigmoid(
                    vp.dormant_outputs["existence_logits"].float()
                ).numpy()
                >= float(a.dormant_existence_threshold)
            )
            dormant_existence.update(
                pred_active, target_exists, supervised
            )

        if vp.birth_outputs is not None:
            birth_match.update(
                birth_query_match_counts(
                    vp.birth_outputs,
                    drow,
                    existence_threshold=float(a.birth_existence_threshold),
                    distance_threshold_m=float(a.birth_match_distance_m),
                )
            )
        audit["windows"] += 1
        audit["dormant_tracks"] += int(vp.dormant_tracks)
        audit["history_observed_oob"] += int(
            vp.history_out_of_bounds_observed_samples
        )
        if vp.static_report is not None:
            audit["static_query_voxels"] += int(vp.static_report.query_voxels)
            audit["static_active_tiles"] += int(vp.static_report.active_tiles)
            audit["static_render_oob"] += int(
                vp.static_report.out_of_bounds_voxels
            )
        if vp.dormant_report is not None:
            audit["dormant_render_oob"] += int(
                vp.dormant_report.out_of_bounds_voxels
            )
        if vp.birth_report is not None:
            audit["birth_render_oob"] += int(
                vp.birth_report.out_of_bounds_voxels
            )
            audit["birth_duplicate_suppressed_query_horizons"] += int(
                vp.birth_report.duplicate_suppressed_query_horizons
            )
        timing_rows.append(vp.timings_ms)

        if wi == 1 or wi % 25 == 0 or wi == len(records):
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(
                f"v20_stage5_eval {wi}/{len(records)} "
                f"rate={wi/elapsed:.3f} win/s",
                flush=True,
            )

    final = {name: acc.finalize() for name, acc in metrics.items()}
    result = {
        "protocol": PROTOCOL,
        "num_windows": int(len(records)),
        "population_fingerprint_sha256": population_fingerprint(records),
        "num_scenes": int(len(per_scene)),
        "base_checkpoint": str(Path(a.base_checkpoint).resolve()),
        "base_checkpoint_epoch": int(base_ck.get("epoch", -1)),
        "v20_checkpoint": str(Path(a.v20_checkpoint).resolve()),
        "v20_stage": str(vck.get("stage")),
        "v19_reference": (
            {
                "label": "v19_static_innovation_reference",
                "eval_protocol": V19_REFERENCE_EVAL_PROTOCOL,
                "checkpoint": str(Path(a.v19_innovation_checkpoint).resolve()),
                "checkpoint_epoch": int(v19_ck.get("epoch", -1)),
                "add_threshold": float(a.v19_add_threshold),
                "vertical_threshold": float(a.v19_vertical_threshold),
                "explicitly_not_renamed_context_only": True,
            }
            if v19_ck is not None else None
        ),
        "stage1_cache": str(Path(a.stage1_cache).resolve()),
        "future_gt_used_for_prediction": False,
        "composition_priority": ["v18", "dormant", "birth", "static"],
        "thresholds": {
            "dormant_existence": float(a.dormant_existence_threshold),
            "birth_existence": float(a.birth_existence_threshold),
            "birth_shape": float(a.birth_shape_threshold),
            "birth_duplicate_distance_m": float(a.birth_duplicate_distance_m),
            "birth_match_distance_m": float(a.birth_match_distance_m),
        },
        "metrics": final,
        "delta_vs_v18": {
            name: metric_delta(final[name], final["v18"])
            for name in variants if name != "v18"
        },
        "per_scene_metrics": {
            scene: {name: acc.finalize() for name, acc in rows.items()}
            for scene, rows in sorted(per_scene.items())
        },
        "branch_addition_quality": {
            k: v.finalize() for k, v in additions.items()
        },
        "static_capability_subsets": {
            k: v.finalize() for k, v in static_subsets.items()
        },
        "strict_dynamic_gt_counts": strict_dynamic_counts,
        "dormant_existence": dormant_existence.finalize(),
        "birth_matching": birth_match.finalize(),
        "audit": audit,
        "v20_component_timing_diagnostic": _mean_timing(timing_rows),
        "raw_counts": {name: acc.raw() for name, acc in metrics.items()},
        "notes": {
            "birth_shape": (
                "Birth local shape is still the annotation-cuboid initial "
                "approximation. Claims must use rendered OCC semantic TP/FP "
                "and mIoU, not Birth center matching alone."
            ),
            "checkpoint_selection": (
                "Use scene-disjoint development composed semantic metrics; "
                "do not select by training loss."
            ),
            "v19_reference": (
                "The optional reference is the repository's exact formal "
                "V18 + deterministic Static Memory + trained V19 Innovation "
                "path. No separate formally named context-only protocol was "
                "found, so this result is not relabeled."
            ),
            "full_4369": (
                "Run only after structure and thresholds are locked on dev."
            ),
        },
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "metrics": final,
        "branch_addition_quality": result["branch_addition_quality"],
        "dormant_existence": result["dormant_existence"],
        "birth_matching": result["birth_matching"],
        "audit": audit,
    }, indent=2))
    print(f"saved {op}")


if __name__ == "__main__":
    main()
