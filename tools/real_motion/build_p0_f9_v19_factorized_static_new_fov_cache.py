#!/usr/bin/env python3
"""Build factorized Static New-FOV training cache.

Supervision is restricted to ancestor-free never-seen static occupancy inside
geometric New-FOV support.  The cache separates:
  * BEV presence target over causal New-FOV candidate columns;
  * one semantic target per GT-positive column;
  * direct Z-bin profile target, supervised only on GT-positive columns.

Nearest deterministic Static-Memory columns are cached only as conditioning:
semantic, vertical profile and distance.  Future GT never enters an inference
input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.motion_transport import (
    FUTURE_FRAMES,
    HISTORY_FRAMES,
    dynamic_annotations,
    match_sources_to_annotations,
)
from real_motion.prepared import load_nuscenes_window_raw
from real_motion.runtime_config import (
    add_config_args,
    load_runtime_config,
    make_prepare_config,
)
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.v19_innovation import (
    base_explained_bev,
    build_future_aligned_history_and_static_memory,
)
from real_motion.v19_innovation_training import (
    pack_vertical_occupancy,
    quantize_geometry,
)
from real_motion.v19_scene_memory import protected_add_only
from real_motion.v19_static_novelty import (
    history_grid_footprint_bev,
    majority_semantic_per_column,
    nearest_static_anchor_map,
)
from real_motion.v19_static_novelty_factorized import (
    ANCHOR_DISTANCE_MAX_M,
    quantize_anchor_distance_m,
)
from tools.real_motion import eval_p0_f9_v18_se2 as base
from tools.real_motion import eval_p0_f9_v18_full_validation as full
from tools.real_motion.benchmark_p0_f9_v18_runtime import (
    _forecast_once,
    _prepare_record,
    _release_gpu_inputs,
    _stage_gpu_inputs,
)
from tools.real_motion.build_p0_f9_v19_innovation_cache import (
    _category_masks_for_future,
)
from tools.real_motion.diagnose_p0_f9_v19_innovation_decomposition import (
    CachedSource,
    _ann_map,
    _grid_spec,
)
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.train_p0_f9_v18_se2_clean import (
    PROTOCOL as CLEAN_PROTOCOL,
)


PROTOCOL = "p0_f9_v19_factorized_static_new_fov_cache_v1"
DYNAMIC_IDS = tuple(int(x) for x in DYNAMIC_CLASS_IDS)


TENSOR_KEYS = (
    "future_aligned_semantic",
    "future_aligned_geometry_q",
    "base_explained",
    "base_free_bits",
    "new_fov_mask",
    "presence_target",
    "vertical_target_bits",
    "semantic_target",
    "anchor_semantic",
    "anchor_profile_bits",
    "anchor_distance_q",
    "gt_occupied_bits",
)


def _flush_shard(out_dir: Path, shard_id: int, rows: list[dict]) -> dict:
    """Atomically write one shard for deterministic crash recovery."""
    if not rows:
        raise ValueError("cannot flush empty factorized New-FOV shard")
    name = f"shard_{int(shard_id):05d}.pt"
    final = out_dir / name
    tmp = out_dir / f".{name}.tmp"
    payload = {
        "protocol": PROTOCOL,
        **{
            k: torch.stack([r[k] for r in rows])
            for k in TENSOR_KEYS
        },
        "scene_name": [r["scene_name"] for r in rows],
        "t0_token": [r["t0_token"] for r in rows],
    }
    try:
        if tmp.exists():
            tmp.unlink()
        torch.save(payload, tmp)
        tmp.replace(final)
    finally:
        if tmp.exists():
            tmp.unlink()
    return {"file": name, "count": len(rows)}


def _unpack_bits_numpy(bits: torch.Tensor, z: int) -> np.ndarray:
    x = bits.detach().cpu().numpy().astype(np.uint16, copy=False)
    shifts = np.arange(int(z), dtype=np.uint16)
    return ((x[..., None] >> shifts) & np.uint16(1)).astype(bool)


def _load_shard_for_resume(path: Path) -> dict:
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )


def _recover_existing_shards(
    out_dir: Path,
    records: list,
    *,
    shard_size: int,
    free_label: int,
    vertical_bins: int,
) -> tuple[list[dict], dict, dict[str, int], set[str], int]:
    """Validate existing shards, drop a corrupt tail, and rebuild counts."""
    for tmp in out_dir.glob(".shard_*.pt.tmp"):
        tmp.unlink(missing_ok=True)

    shard_paths = sorted(out_dir.glob("shard_*.pt"))
    shards: list[dict] = []
    totals = {
        "candidate_bev_columns": 0,
        "positive_bev_columns": 0,
        "vertical_supervised_voxels": 0,
        "positive_vertical_voxels": 0,
        "target_voxels": 0,
        "anchor_valid_candidate_columns": 0,
        "baseline_occ_inter": 0,
        "baseline_occ_union": 0,
    }
    semantic_hist = {str(i): 0 for i in range(17)}
    scenes: set[str] = set()
    processed = 0
    invalid_from: int | None = None

    for file_i, path in enumerate(shard_paths):
        expected_name = f"shard_{file_i:05d}.pt"
        if path.name != expected_name:
            invalid_from = file_i
            print(
                f"resume: non-contiguous shard sequence at {path.name}; "
                f"expected {expected_name}",
                flush=True,
            )
            break
        try:
            obj = _load_shard_for_resume(path)
            if obj.get("protocol") != PROTOCOL:
                raise RuntimeError(
                    f"protocol={obj.get('protocol')!r}"
                )
            n = len(obj.get("t0_token", []))
            if n <= 0 or n > int(shard_size):
                raise RuntimeError(f"invalid shard count={n}")
            if processed + n > len(records):
                raise RuntimeError("shard exceeds selected record population")
            if file_i < len(shard_paths) - 1 and n != int(shard_size):
                raise RuntimeError(
                    f"non-tail shard count {n} != shard_size {shard_size}"
                )
            for k in TENSOR_KEYS:
                if k not in obj or int(obj[k].shape[0]) != n:
                    raise RuntimeError(f"bad tensor batch for {k}")

            expected_tokens = [
                str(window_from_record(records[processed + j]).t0_token)
                for j in range(n)
            ]
            got_tokens = [str(x) for x in obj["t0_token"]]
            if got_tokens != expected_tokens:
                raise RuntimeError(
                    "cached t0 tokens do not match current selected record order"
                )

            base_free = _unpack_bits_numpy(
                obj["base_free_bits"],
                vertical_bins,
            )
            vertical = _unpack_bits_numpy(
                obj["vertical_target_bits"],
                vertical_bins,
            )
            gt_occ = _unpack_bits_numpy(
                obj["gt_occupied_bits"],
                vertical_bins,
            )
            new_fov = (
                obj["new_fov_mask"]
                .detach()
                .cpu()
                .numpy()
                .astype(bool)
            )
            presence = (
                obj["presence_target"]
                .detach()
                .cpu()
                .numpy()
                .astype(bool)
            )
            semantic = obj["semantic_target"].detach().cpu().numpy()
            anchor_sem = obj["anchor_semantic"].detach().cpu().numpy()

            candidate = new_fov & base_free.any(axis=-1)
            supervised_vertical = presence[..., None] & base_free
            base_occ = ~base_free

            totals["candidate_bev_columns"] += int(candidate.sum())
            totals["positive_bev_columns"] += int(presence.sum())
            totals["vertical_supervised_voxels"] += int(
                supervised_vertical.sum()
            )
            totals["positive_vertical_voxels"] += int(vertical.sum())
            totals["target_voxels"] += int(vertical.sum())
            totals["anchor_valid_candidate_columns"] += int(
                ((anchor_sem != int(free_label)) & candidate).sum()
            )
            totals["baseline_occ_inter"] += int(
                (base_occ & gt_occ).sum()
            )
            totals["baseline_occ_union"] += int(
                (base_occ | gt_occ).sum()
            )
            for cid in range(17):
                semantic_hist[str(cid)] += int(
                    (presence & (semantic == int(cid))).sum()
                )

            scenes.update(str(x) for x in obj.get("scene_name", []))
            shards.append({"file": path.name, "count": int(n)})
            processed += int(n)
            del obj
        except Exception as exc:
            invalid_from = file_i
            print(
                f"resume: dropping invalid tail from {path.name}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            break

    if invalid_from is not None:
        for path in shard_paths[invalid_from:]:
            path.unlink(missing_ok=True)

    print(
        f"resume: recovered {processed}/{len(records)} windows "
        f"from {len(shards)} valid shards",
        flush=True,
    )
    return shards, totals, semantic_hist, scenes, processed


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--source-cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--alignment-workers", type=int, default=6)
    p.add_argument("--preserve-record-order", action="store_true")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-size", type=int, default=8)
    p.add_argument("--match-max-distance-m", type=float, default=4.0)
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "resume a partial cache after validating existing shards; "
            "a corrupt/incomplete tail shard is deleted and recomputed"
        ),
    )
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    if int(a.shard_size) <= 0:
        raise ValueError("shard-size must be positive")
    if int(a.alignment_workers) <= 0:
        raise ValueError("alignment-workers must be positive")
    if int(a.num_shards) <= 0:
        raise ValueError("num-shards must be positive")
    if not 0 <= int(a.shard_index) < int(a.num_shards):
        raise ValueError("shard-index must be in [0,num-shards)")

    out_dir = Path(a.output_dir)
    if (
        out_dir.exists()
        and any(out_dir.iterdir())
        and not bool(a.resume)
    ):
        raise FileExistsError(
            f"refusing non-empty output dir without --resume: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    pcfg = make_prepare_config(load_runtime_config(a.config, a.override))
    source_meta, records = base.load_cache(a.source_cache)
    if int(a.max_windows) > 0:
        records = records[: min(len(records), int(a.max_windows))]
    global_num_windows = int(len(records))
    if not bool(a.preserve_record_order):
        records = sorted(
            records,
            key=lambda r: str(window_from_record(r).scene_name),
        )
    if int(a.num_shards) > 1:
        n = len(records)
        lo = n * int(a.shard_index) // int(a.num_shards)
        hi = n * (int(a.shard_index) + 1) // int(a.num_shards)
        records = records[lo:hi]
    if not records:
        raise RuntimeError("empty source cache shard")
    selected_num_windows = int(len(records))

    if bool(a.resume):
        (
            shards,
            totals,
            semantic_column_hist,
            scenes,
            processed_windows,
        ) = _recover_existing_shards(
            out_dir,
            records,
            shard_size=int(a.shard_size),
            free_label=int(pcfg.free_label),
            vertical_bins=int(pcfg.grid.shape_hwd[2]),
        )
    else:
        totals = {
            "candidate_bev_columns": 0,
            "positive_bev_columns": 0,
            "vertical_supervised_voxels": 0,
            "positive_vertical_voxels": 0,
            "target_voxels": 0,
            "anchor_valid_candidate_columns": 0,
            "baseline_occ_inter": 0,
            "baseline_occ_union": 0,
        }
        semantic_column_hist = {str(i): 0 for i in range(17)}
        shards = []
        scenes = set()
        processed_windows = 0

    device = torch.device(
        a.device
        if a.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    ck, model, _ = full._load_model(a.checkpoint, CLEAN_PROTOCOL, device)
    source = CachedSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))
    future_component_cfg = StrongW2DetConfig(
        free_label=int(pcfg.free_label),
        min_component_voxels=1,
        max_match_speed_mps=float(strong_cfg.max_match_speed_mps),
        connectivity=int(strong_cfg.connectivity),
        fill_kernel=tuple(strong_cfg.fill_kernel),
        fill_min_fraction=float(strong_cfg.fill_min_fraction),
    )
    metric_grid = _grid_spec(pcfg.grid)

    shard_rows = []
    started = time.perf_counter()

    remaining_records = records[processed_windows:]
    for wi, rec in enumerate(
        remaining_records,
        start=processed_windows + 1,
    ):
        w = window_from_record(rec)
        scenes.add(str(w.scene_name))
        raw = load_nuscenes_window_raw(source, w, pcfg, include_gt=True)
        state = _prepare_record(rec, source, pcfg, strong_cfg, device)
        _stage_gpu_inputs(state, device)
        try:
            pred_all = _forecast_once(model, state, pcfg, strong_cfg, device)
        finally:
            _release_gpu_inputs(state)

        history_occ = np.asarray(raw["history_occ"], dtype=np.uint8)
        history_obs = np.asarray(raw["history_observed"], dtype=bool)
        history_poses = np.asarray(raw["history_poses"], dtype=np.float64)
        future_poses = np.asarray(raw["future_poses"], dtype=np.float64)

        ann_hist = [_ann_map(source.nusc, tok) for tok in w.history_tokens]
        ann0 = ann_hist[-1]
        t0_tokens = set(ann0)
        source_tokens = match_sources_to_annotations(
            state["current"],
            dynamic_annotations(source.nusc, w.t0_token),
            max_distance_m=float(a.match_max_distance_m),
        )
        represented = {str(x) for x in source_tokens if x is not None}

        (
            aligned_sem,
            aligned_geo,
            history_coverage_all,
            static_render_all,
        ) = build_future_aligned_history_and_static_memory(
            history_occ,
            history_obs,
            history_poses,
            future_poses,
            grid=pcfg.grid,
            free_label=int(pcfg.free_label),
            dynamic_class_ids=DYNAMIC_IDS,
            workers=int(a.alignment_workers),
        )

        explained_all = []
        base_free_all = []
        new_fov_all = []
        presence_all = []
        vertical_all = []
        semantic_all = []
        anchor_sem_all = []
        anchor_profile_all = []
        anchor_distance_q_all = []
        gt_occ_all = []

        for fi in range(len(future_poses)):
            gt = np.asarray(raw["future_gt_occ"][fi], dtype=np.uint8)
            pred_v18 = np.asarray(pred_all[fi], dtype=np.uint8)
            masks, static_render, _ = _category_masks_for_future(
                gt=gt,
                pred_v18=pred_v18,
                future_pose=future_poses[fi],
                future_token=w.future_tokens[fi],
                state=state,
                source_tokens=source_tokens,
                represented=represented,
                history_occ=history_occ,
                history_obs=history_obs,
                history_poses=history_poses,
                history_tokens=tuple(w.history_tokens),
                ann_hist=ann_hist,
                ann0=ann0,
                t0_tokens=t0_tokens,
                source=source,
                pcfg=pcfg,
                future_component_cfg=future_component_cfg,
                metric_grid=metric_grid,
                match_max_distance_m=float(a.match_max_distance_m),
                history_coverage=history_coverage_all[fi],
                static_render=static_render_all[fi],
                moving_tokens=None,
            )

            explained = protected_add_only(
                pred_v18,
                static_render,
                free_label=int(pcfg.free_label),
            )
            base_free = explained == int(pcfg.free_label)
            footprint = history_grid_footprint_bev(
                history_poses,
                future_poses[fi],
                pcfg.grid,
            )
            new_fov = ~footprint
            new_fov_3d = np.broadcast_to(new_fov[..., None], gt.shape)
            target = (
                np.asarray(masks["never_seen_static"], dtype=bool)
                & new_fov_3d
                & base_free
            )
            presence = target.any(axis=2)
            candidate_bev = new_fov & base_free.any(axis=2)
            semantic_target = majority_semantic_per_column(
                target,
                gt,
                num_classes=17,
                ignore_label=255,
            )

            (
                dist_cells,
                anchor_x,
                anchor_y,
                anchor_valid,
            ) = nearest_static_anchor_map(
                static_render,
                footprint,
                free_label=int(pcfg.free_label),
            )
            static_occ = (
                np.asarray(static_render, dtype=np.uint8)
                != int(pcfg.free_label)
            )
            anchor_sem_source = majority_semantic_per_column(
                static_occ,
                static_render,
                num_classes=17,
                ignore_label=int(pcfg.free_label),
            )
            anchor_sem = np.full(
                new_fov.shape,
                int(pcfg.free_label),
                dtype=np.uint8,
            )
            anchor_profile = np.zeros(
                gt.shape,
                dtype=bool,
            )
            if bool(anchor_valid.any()):
                anchor_sem[anchor_valid] = anchor_sem_source[
                    anchor_x[anchor_valid],
                    anchor_y[anchor_valid],
                ]
                copied = static_occ[anchor_x, anchor_y]
                anchor_profile[anchor_valid] = copied[anchor_valid]
            anchor_distance_m = (
                np.asarray(dist_cells, dtype=np.float32)
                * float(pcfg.grid.voxel_size[0])
            )

            po = explained != int(pcfg.free_label)
            go = gt != int(pcfg.free_label)
            totals["baseline_occ_inter"] += int((po & go).sum())
            totals["baseline_occ_union"] += int((po | go).sum())
            totals["candidate_bev_columns"] += int(candidate_bev.sum())
            totals["positive_bev_columns"] += int(presence.sum())
            supervised_vertical = presence[..., None] & base_free
            totals["vertical_supervised_voxels"] += int(
                supervised_vertical.sum()
            )
            totals["positive_vertical_voxels"] += int(target.sum())
            totals["target_voxels"] += int(target.sum())
            totals["anchor_valid_candidate_columns"] += int(
                (anchor_valid & candidate_bev).sum()
            )
            for cid in range(17):
                semantic_column_hist[str(cid)] += int(
                    (presence & (semantic_target == int(cid))).sum()
                )

            explained_all.append(explained)
            base_free_all.append(base_free)
            new_fov_all.append(new_fov)
            presence_all.append(presence)
            vertical_all.append(target)
            semantic_all.append(semantic_target)
            anchor_sem_all.append(anchor_sem)
            anchor_profile_all.append(anchor_profile)
            anchor_distance_q_all.append(
                quantize_anchor_distance_m(
                    anchor_distance_m,
                    ANCHOR_DISTANCE_MAX_M,
                )
            )
            gt_occ_all.append(go)

        explained_all = np.stack(explained_all, axis=0).astype(np.uint8)
        base_free_all = np.stack(base_free_all, axis=0)
        vertical_all = np.stack(vertical_all, axis=0)
        anchor_profile_all = np.stack(anchor_profile_all, axis=0)
        gt_occ_all = np.stack(gt_occ_all, axis=0)

        row = {
            "future_aligned_semantic": torch.from_numpy(
                aligned_sem.astype(np.uint8)
            ),
            "future_aligned_geometry_q": torch.from_numpy(
                quantize_geometry(aligned_geo)
            ),
            "base_explained": torch.from_numpy(
                base_explained_bev(
                    explained_all,
                    free_label=int(pcfg.free_label),
                ).astype(np.uint8)
            ),
            "base_free_bits": torch.from_numpy(
                pack_vertical_occupancy(base_free_all)
            ),
            "new_fov_mask": torch.from_numpy(
                np.stack(new_fov_all, axis=0).astype(np.uint8)
            ),
            "presence_target": torch.from_numpy(
                np.stack(presence_all, axis=0).astype(np.uint8)
            ),
            "vertical_target_bits": torch.from_numpy(
                pack_vertical_occupancy(vertical_all)
            ),
            "semantic_target": torch.from_numpy(
                np.stack(semantic_all, axis=0).astype(np.uint8)
            ),
            "anchor_semantic": torch.from_numpy(
                np.stack(anchor_sem_all, axis=0).astype(np.uint8)
            ),
            "anchor_profile_bits": torch.from_numpy(
                pack_vertical_occupancy(anchor_profile_all)
            ),
            "anchor_distance_q": torch.from_numpy(
                np.stack(anchor_distance_q_all, axis=0).astype(np.uint8)
            ),
            "gt_occupied_bits": torch.from_numpy(
                pack_vertical_occupancy(gt_occ_all)
            ),
            "scene_name": str(w.scene_name),
            "t0_token": str(w.t0_token),
        }
        shard_rows.append(row)

        if len(shard_rows) >= int(a.shard_size):
            shards.append(_flush_shard(out_dir, len(shards), shard_rows))
            shard_rows = []

        if wi == 1 or wi % 25 == 0 or wi == selected_num_windows:
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(
                f"v19_factorized_new_fov_cache {wi}/{selected_num_windows} "
                f"rate={(wi-processed_windows)/elapsed:.3f} new-win/s",
                flush=True,
            )

    if shard_rows:
        shards.append(_flush_shard(out_dir, len(shards), shard_rows))

    pos_bev = int(totals["positive_bev_columns"])
    cand_bev = int(totals["candidate_bev_columns"])
    pos_vert = int(totals["positive_vertical_voxels"])
    sup_vert = int(totals["vertical_supervised_voxels"])
    inter = int(totals["baseline_occ_inter"])
    union = int(totals["baseline_occ_union"])
    index = {
        "protocol": PROTOCOL,
        "source_cache": str(Path(a.source_cache).resolve()),
        "source_cache_metadata_keys": sorted(str(k) for k in source_meta.keys()),
        "base_checkpoint": str(Path(a.checkpoint).resolve()),
        "base_checkpoint_epoch": int(ck.get("epoch", -1)),
        "num_windows": int(selected_num_windows),
        "global_num_windows_before_shard": int(global_num_windows),
        "num_scenes": int(len(scenes)),
        "scene_names": sorted(scenes),
        "future_frames": int(FUTURE_FRAMES),
        "history_frames": int(HISTORY_FRAMES),
        "grid_shape_hwd": [int(x) for x in pcfg.grid.shape_hwd],
        "free_label": int(pcfg.free_label),
        "target": "never_seen_static_intersect_geometric_new_fov",
        "representation": (
            "factorized_bev_presence_column_semantic_conditional_direct_z"
        ),
        "anchor_condition": (
            "nearest occupied deterministic Static-Memory column inside "
            "historical geometric footprint; semantic/profile/distance only"
        ),
        "future_gt_used_for_inference_input": False,
        "anchor_distance_max_m": float(ANCHOR_DISTANCE_MAX_M),
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
        "semantic_positive_column_histogram": semantic_column_hist,
        "shards": shards,
        "resume": {
            "enabled": bool(a.resume),
            "recovered_windows": int(processed_windows),
            "new_windows": int(len(remaining_records)),
        },
        "timing": {
            "elapsed_s": float(max(time.perf_counter() - started, 1e-9)),
            "windows_per_s": float(
                len(remaining_records)
                / max(time.perf_counter() - started, 1e-9)
            ),
            "alignment_workers": int(a.alignment_workers),
            "scene_grouped": not bool(a.preserve_record_order),
        },
    }
    (out_dir / "index.json").write_text(
        json.dumps(index, indent=2),
        encoding="utf-8",
    )
    print("\n=== V19 FACTORIZED STATIC NEW-FOV CACHE ===")
    print(
        json.dumps(
            {
                "num_windows": index["num_windows"],
                "num_scenes": index["num_scenes"],
                "presence_positive_fraction": index[
                    "presence_positive_fraction"
                ],
                "vertical_positive_fraction_on_positive_columns": index[
                    "vertical_positive_fraction_on_positive_columns"
                ],
                "baseline_occ_iou": index["baseline_occ_iou"],
                "totals": totals,
            },
            indent=2,
        )
    )
    print(f"saved {out_dir / 'index.json'}")


if __name__ == "__main__":
    main()
