#!/usr/bin/env python3
"""Evaluate the three mandatory V20 unified variants on identical windows."""
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

from real_motion.local_st_world_model_v17 import config_from_mapping_v17
from real_motion.local_st_world_model_v18_se2 import LocalSpatialTemporalWorldModelV18SE2
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.v20_unified_training import file_sha256, load_model_checkpoint
from tools.real_motion.v20_unified_common import (
    CachedSource,
    ComponentLRU,
    align_v18_records_to_stage1,
    first_stage_forward,
    full_completion_prediction,
    hard_render_transport,
    load_stage1_rows,
    load_v18_cache,
    moving_support_sequence,
    prepare_unified_window,
)

PROTOCOL = "p0_f9_v20_unified_transport_completion_eval_v1"
VARIANTS = (
    "frozen_v18_reference",
    "current_transport_only",
    "transport_plus_completion",
)
HORIZONS = tuple(0.5 * (i + 1) for i in range(6))
SEMANTIC_CLASSES = tuple(range(17))
CLEAN_PROTOCOL = "p0_f9_v18_se2_clean_train_v1"


def load_clean_v18(path: str | Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != CLEAN_PROTOCOL or checkpoint.get("arm") != "Y":
        raise RuntimeError("expected the V18 Clean-E14 arm=Y checkpoint")
    model = LocalSpatialTemporalWorldModelV18SE2(
        config_from_mapping_v17(checkpoint.get("model_config"))
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return checkpoint, model


def _new_raw():
    return {
        "occ_inter": np.zeros(6, dtype=np.int64),
        "occ_union": np.zeros(6, dtype=np.int64),
        "sem_inter": np.zeros((6, len(SEMANTIC_CLASSES)), dtype=np.int64),
        "sem_union": np.zeros((6, len(SEMANTIC_CLASSES)), dtype=np.int64),
        "mov_inter": np.zeros((6, len(DYNAMIC_CLASS_IDS)), dtype=np.int64),
        "mov_union": np.zeros((6, len(DYNAMIC_CLASS_IDS)), dtype=np.int64),
    }


def _update_many(raw_by_name, horizon, pred_by_name, gt, moving, free_label):
    label_count = max(int(free_label) + 1, max(DYNAMIC_CLASS_IDS) + 1)
    gt_flat = np.asarray(gt).reshape(-1).astype(np.int16, copy=False)
    moving_flat = np.asarray(moving, dtype=bool).reshape(-1)
    for name, prediction in pred_by_name.items():
        pred_flat = np.asarray(prediction).reshape(-1).astype(np.int16, copy=False)
        pair = gt_flat * label_count + pred_flat
        confusion = np.bincount(
            pair, minlength=label_count * label_count
        ).reshape(label_count, label_count)
        moving_confusion = (
            np.bincount(
                pair[moving_flat], minlength=label_count * label_count
            ).reshape(label_count, label_count)
            if bool(moving_flat.any())
            else np.zeros_like(confusion)
        )
        raw = raw_by_name[name]
        free = int(free_label)
        raw["occ_inter"][horizon] += int(confusion[:free, :free].sum())
        raw["occ_union"][horizon] += int(confusion.sum() - confusion[free, free])
        diagonal = np.diag(confusion)[: len(SEMANTIC_CLASSES)]
        gt_count = confusion[: len(SEMANTIC_CLASSES), :].sum(axis=1)
        pred_count = confusion[:, : len(SEMANTIC_CLASSES)].sum(axis=0)
        raw["sem_inter"][horizon] += diagonal
        raw["sem_union"][horizon] += gt_count + pred_count - diagonal
        for index, class_id in enumerate(DYNAMIC_CLASS_IDS):
            class_id = int(class_id)
            intersection = int(moving_confusion[class_id, class_id])
            raw["mov_inter"][horizon, index] += intersection
            raw["mov_union"][horizon, index] += int(
                moving_confusion[class_id, :].sum()
                + moving_confusion[:, class_id].sum()
                - intersection
            )


def _safe_ratio(intersection, union):
    intersection = np.asarray(intersection, dtype=np.float64)
    union = np.asarray(union, dtype=np.float64)
    out = np.full(intersection.shape, np.nan, dtype=np.float64)
    np.divide(intersection, union, out=out, where=union > 0)
    return 100.0 * out


def _finalize(raw):
    occupancy = _safe_ratio(raw["occ_inter"], raw["occ_union"])
    semantic = _safe_ratio(raw["sem_inter"], raw["sem_union"])
    moving = _safe_ratio(raw["mov_inter"], raw["mov_union"])
    semantic_mean = np.nanmean(semantic, axis=1)
    moving_macro = np.nanmean(moving, axis=1)
    moving_micro = _safe_ratio(
        raw["mov_inter"].sum(axis=1), raw["mov_union"].sum(axis=1)
    )
    main = [1, 3, 5]
    result = {
        "IoU": float(np.nanmean(occupancy)),
        "mIoU": float(np.nanmean(semantic_mean)),
        "MovingMacro": float(np.nanmean(moving_macro)),
        "MovingMicro": float(np.nanmean(moving_micro)),
        "main_1_2_3s": {},
        "per_horizon": {},
    }
    rows = {
        "IoU": occupancy,
        "mIoU": semantic_mean,
        "MovingMacro": moving_macro,
        "MovingMicro": moving_micro,
    }
    result["main_1_2_3s"] = {
        name: float(np.nanmean(values[main])) for name, values in rows.items()
    }
    for horizon, index in zip(HORIZONS, range(6)):
        result["per_horizon"][str(horizon)] = {
            name: float(values[index]) for name, values in rows.items()
        }
    return result


def _delta(left, right):
    keys = ("IoU", "mIoU", "MovingMacro", "MovingMicro")
    return {
        **{key: float(left[key]) - float(right[key]) for key in keys},
        "main_1_2_3s": {
            key: float(left["main_1_2_3s"][key])
            - float(right["main_1_2_3s"][key])
            for key in keys
        },
    }


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _dynamic_diagnostic_group_counts(row: dict) -> dict[str, int] | None:
    if "dynamic_supervision" in row:
        names = (
            item.get("responsibility_name", "IGNORE")
            for item in row["dynamic_supervision"]
        )
    elif "dynamic_diagnostic_counts" in row:
        return {
            str(name): int(count)
            for name, count in row["dynamic_diagnostic_counts"].items()
        }
    elif "dynamic_diagnostic_groups" in row:
        names = row["dynamic_diagnostic_groups"]
    else:
        return None
    counts: dict[str, int] = {}
    for name in names:
        key = str(name)
        counts[key] = counts.get(key, 0) + 1
    return counts


def evaluate_model(
    *,
    model,
    frozen_v18,
    records,
    stage1_rows,
    source,
    pcfg,
    native_grid,
    device: torch.device,
    amp: bool,
    alignment_workers: int = 6,
    progress: bool = True,
    ablate_source_latents: bool = False,
    include_per_scene: bool = False,
    frozen_reference_raw: dict | None = None,
) -> dict:
    model.eval()
    if frozen_v18 is not None:
        frozen_v18.eval()
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))
    component_cache = ComponentLRU(maxsize=1024)
    raw_by_variant = {name: _new_raw() for name in VARIANTS}
    if frozen_reference_raw is not None:
        cached = raw_by_variant["frozen_v18_reference"]
        for key in cached:
            cached[key][...] = np.asarray(frozen_reference_raw[key], dtype=np.int64)
    raw_by_scene = {}
    total_runtime_oob = total_scatter_oob = total_completion_support = 0
    diagnostic_groups = {
        "STATIC_target_voxels": 0,
        "CURRENT_ANCESTRAL_instances": 0,
        "DORMANT_ANCESTRAL_instances": 0,
        "BIRTH_instances": 0,
        "IGNORE_instances": 0,
    }
    dynamic_diagnostic_windows = 0
    addition = {
        "support_valid_voxels": np.zeros(6, dtype=np.int64),
        "target_positive_voxels": np.zeros(6, dtype=np.int64),
        "predicted_add_voxels": np.zeros(6, dtype=np.int64),
        "added_occ_tp": np.zeros(6, dtype=np.int64),
        "added_occ_fp": np.zeros(6, dtype=np.int64),
        "added_semantic_correct": np.zeros(6, dtype=np.int64),
        "added_semantic_wrong_occupied": np.zeros(6, dtype=np.int64),
        "missed_positive_voxels": np.zeros(6, dtype=np.int64),
        "full_confusion_matrix": np.zeros((18, 18), dtype=np.int64),
    }
    started = time.perf_counter()
    for wi, record in enumerate(records, start=1):
        key = (str(record["scene_name"]), str(record["t0_token"]))
        if key not in stage1_rows:
            raise RuntimeError(f"Stage1 cache misses evaluation row {key}")
        prepared = prepare_unified_window(
            record,
            stage1_rows[key],
            source=source,
            pcfg=pcfg,
            strong_cfg=strong_cfg,
            component_cache=component_cache,
            device=device,
        )
        try:
            with torch.inference_mode(), _autocast(device, amp):
                gpu = prepared.state["gpu"]
                reference_out = (
                    frozen_v18(
                        gpu["features"],
                        gpu["tube"],
                        gpu["kta"],
                        gpu["frame_motion"],
                        gpu["source_mask"],
                    )
                    if frozen_reference_raw is None
                    else None
                )
                first = first_stage_forward(model, prepared, adapter_enabled=True)
            reference = (
                hard_render_transport(
                    model,
                    prepared,
                    reference_out,
                    pcfg=pcfg,
                    strong_cfg=strong_cfg,
                    device=device,
                )[0].cpu().numpy().astype(np.uint8)
                if reference_out is not None
                else None
            )
            current_t = hard_render_transport(
                model,
                prepared,
                first["transport"],
                pcfg=pcfg,
                strong_cfg=strong_cfg,
                device=device,
            )
            with torch.inference_mode(), _autocast(device, amp):
                final_t, reports = full_completion_prediction(
                    model,
                    prepared,
                    first,
                    current_t,
                    native_grid=native_grid,
                    ablate_source_latents=bool(ablate_source_latents),
                )
            current = current_t[0].cpu().numpy().astype(np.uint8)
            final = final_t[0].cpu().numpy().astype(np.uint8)
            total_runtime_oob += int(reports["runtime"].out_of_bounds_voxels)
            total_completion_support += int(reports["runtime"].eligible_voxels)
            total_scatter_oob += int(reports["scatter"].out_of_bounds_points)
            moving = moving_support_sequence(
                source, record, grid=pcfg.grid, workers=int(alignment_workers)
            )
            gt_all = prepared.future_semantic[0].cpu().numpy().astype(np.uint8)
            support_all = reports["support"][0].cpu().numpy().astype(bool)
            scene_name = str(record["scene_name"])
            if include_per_scene and scene_name not in raw_by_scene:
                raw_by_scene[scene_name] = {
                    name: _new_raw() for name in VARIANTS
                }
            dynamic_ids = np.asarray(DYNAMIC_CLASS_IDS, dtype=gt_all.dtype)
            diagnostic_groups["STATIC_target_voxels"] += int(
                ((gt_all != int(pcfg.free_label)) & ~np.isin(gt_all, dynamic_ids)).sum()
            )
            dynamic_counts = _dynamic_diagnostic_group_counts(prepared.row)
            if dynamic_counts is not None:
                dynamic_diagnostic_windows += 1
            for group_name, count in (dynamic_counts or {}).items():
                name = f"{group_name}_instances"
                if name in diagnostic_groups:
                    diagnostic_groups[name] += int(count)
            for hi, _ in enumerate(HORIZONS):
                pred_by_name = {
                    "current_transport_only": current[hi],
                    "transport_plus_completion": final[hi],
                }
                if reference is not None:
                    pred_by_name["frozen_v18_reference"] = reference[hi]
                _update_many(
                    raw_by_variant,
                    hi,
                    pred_by_name,
                    gt_all[hi],
                    moving[hi],
                    int(pcfg.free_label),
                )
                if include_per_scene:
                    _update_many(
                        raw_by_scene[scene_name],
                        hi,
                        pred_by_name,
                        gt_all[hi],
                        moving[hi],
                        int(pcfg.free_label),
                    )

                free = int(pcfg.free_label)
                support = support_all[hi]
                target_positive = support & (gt_all[hi] != free)
                written = support & (current[hi] == free) & (final[hi] != free)
                addition["support_valid_voxels"][hi] += int(support.sum())
                addition["target_positive_voxels"][hi] += int(
                    target_positive.sum()
                )
                addition["predicted_add_voxels"][hi] += int(written.sum())
                addition["added_occ_tp"][hi] += int(
                    (written & (gt_all[hi] != free)).sum()
                )
                addition["added_occ_fp"][hi] += int(
                    (written & (gt_all[hi] == free)).sum()
                )
                addition["added_semantic_correct"][hi] += int(
                    (written & (final[hi] == gt_all[hi])).sum()
                )
                addition["added_semantic_wrong_occupied"][hi] += int(
                    (
                        written
                        & (gt_all[hi] != free)
                        & (final[hi] != gt_all[hi])
                    ).sum()
                )
                addition["missed_positive_voxels"][hi] += int(
                    (target_positive & (final[hi] == free)).sum()
                )
                pair = (
                    gt_all[hi].reshape(-1).astype(np.int64) * 18
                    + final[hi].reshape(-1).astype(np.int64)
                )
                addition["full_confusion_matrix"] += np.bincount(
                    pair, minlength=18 * 18
                ).reshape(18, 18)
        finally:
            prepared.release()
        if progress and (wi == 1 or wi % 25 == 0 or wi == len(records)):
            print(f"v20_unified_eval {wi}/{len(records)}", flush=True)

    metrics = {name: _finalize(raw_by_variant[name]) for name in VARIANTS}
    addition_json = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in addition.items()
    }
    addition_json["totals"] = {
        key: int(value.sum())
        for key, value in addition.items()
        if isinstance(value, np.ndarray) and value.ndim == 1
    }
    per_scene = (
        {
            scene: {
                name: _finalize(raw[name]) for name in VARIANTS
            }
            for scene, raw in raw_by_scene.items()
        }
        if include_per_scene
        else None
    )
    raw_counts = {
        name: {
            key: value.tolist()
            for key, value in raw.items()
        }
        for name, raw in raw_by_variant.items()
    }
    dynamic_diagnostics_available = dynamic_diagnostic_windows == len(records)
    if not dynamic_diagnostics_available:
        for key in (
            "CURRENT_ANCESTRAL_instances",
            "DORMANT_ANCESTRAL_instances",
            "BIRTH_instances",
            "IGNORE_instances",
        ):
            diagnostic_groups[key] = None
    return {
        "protocol": PROTOCOL,
        "windows": len(records),
        "variants": metrics,
        "delta_current_vs_frozen": _delta(
            metrics["current_transport_only"], metrics["frozen_v18_reference"]
        ),
        "delta_completion_vs_current": _delta(
            metrics["transport_plus_completion"], metrics["current_transport_only"]
        ),
        "runtime": {
            "query_out_of_bounds_voxels": total_runtime_oob,
            "source_scatter_out_of_bounds_points": total_scatter_oob,
            "completion_support_voxels": total_completion_support,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "diagnostic_groups_only": diagnostic_groups,
        "diagnostic_groups_availability": {
            "static_target_voxels": True,
            "dynamic_responsibility": dynamic_diagnostics_available,
            "dynamic_windows_with_metadata": dynamic_diagnostic_windows,
            "windows": len(records),
        },
        "addition_quality": addition_json,
        "raw_metric_counts": raw_counts,
        "per_scene": per_scene,
        "future_gt_used_for_prediction": False,
        "static_dormant_birth_are_diagnostics_only": True,
        "completion_source_latents_ablated": bool(ablate_source_latents),
        "frozen_reference_reused": frozen_reference_raw is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--stage1-cache", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--info-pkl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--alignment-workers", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--ablate-completion-source-latents",
        action="store_true",
        help="Ablate only completion scatter latents; keep current transport unchanged.",
    )
    args = parser.parse_args()

    pcfg = make_prepare_config(load_runtime_config(args.config, args.override))
    _, records = load_v18_cache(args.val_cache)
    _, stage1_index, stage1_rows = load_stage1_rows(args.stage1_cache)
    records, population_alignment = align_v18_records_to_stage1(
        records,
        stage1_rows,
        population_name="evaluation selection",
    )
    if int(args.max_windows) > 0:
        if int(args.max_windows) > len(records):
            raise RuntimeError(
                f"max-windows={args.max_windows} exceeds aligned evaluation "
                f"population={len(records)}"
            )
        records = records[: int(args.max_windows)]
    if not records:
        raise RuntimeError("empty evaluation population")
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model, checkpoint = load_model_checkpoint(args.checkpoint, map_location="cpu")
    if checkpoint["base_checkpoint_sha256"] != file_sha256(args.base_checkpoint):
        raise RuntimeError("checkpoint/base-checkpoint hash mismatch")
    _, frozen_v18 = load_clean_v18(args.base_checkpoint, device)
    model.to(device)
    source = CachedSource(args.dataroot, info_pkl=args.info_pkl, verbose=False)
    result = evaluate_model(
        model=model,
        frozen_v18=frozen_v18,
        records=records,
        stage1_rows=stage1_rows,
        source=source,
        pcfg=pcfg,
        native_grid=stage1_index["native_grid"],
        device=device,
        amp=device.type == "cuda" and not bool(args.no_amp),
        alignment_workers=int(args.alignment_workers),
        ablate_source_latents=bool(args.ablate_completion_source_latents),
        include_per_scene=True,
    )
    result.update(
        {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
            "stage1_cache": str(Path(args.stage1_cache).resolve()),
            "population_alignment": population_alignment,
            "evaluation_population_truncated": bool(
                int(args.max_windows) > 0
            ),
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
