#!/usr/bin/env python3
"""Build cache for frozen-base Static New-FOV Novelty training.

The cache is generated from the exact frozen Clean-E14/V18 forecast plus
Deterministic Static Memory.  The learned target is only ancestor-free static
occupancy in geometric New-FOV support.

Future GT and annotation identity are used only to construct supervision.
Inference inputs remain causal under the same future-ego-conditioning contract
as V18.
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


PROTOCOL = "p0_f9_v19_static_new_fov_cache_v1"
DYNAMIC_IDS = tuple(int(x) for x in DYNAMIC_CLASS_IDS)


def _flush_shard(
    out_dir: Path,
    shard_id: int,
    rows: list[dict],
) -> dict:
    if not rows:
        raise ValueError("cannot flush empty Static New-FOV shard")
    name = f"shard_{int(shard_id):05d}.pt"
    tensor_keys = (
        "future_aligned_semantic",
        "future_aligned_geometry_q",
        "base_explained",
        "base_free_bits",
        "new_fov_mask",
        "occupancy_target_bits",
        "semantic_target",
    )
    payload = {
        "protocol": PROTOCOL,
        **{
            k: torch.stack([r[k] for r in rows])
            for k in tensor_keys
        },
        "scene_name": [r["scene_name"] for r in rows],
        "t0_token": [r["t0_token"] for r in rows],
    }
    torch.save(payload, out_dir / name)
    return {"file": name, "count": len(rows)}


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
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"refusing non-empty output dir: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    pcfg = make_prepare_config(
        load_runtime_config(a.config, a.override)
    )
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

    device = torch.device(
        a.device
        if a.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    ck, model, _ = full._load_model(
        a.checkpoint,
        CLEAN_PROTOCOL,
        device,
    )
    source = CachedSource(
        a.dataroot,
        info_pkl=a.info_pkl,
        verbose=False,
    )
    strong_cfg = StrongW2DetConfig(
        free_label=int(pcfg.free_label)
    )
    future_component_cfg = StrongW2DetConfig(
        free_label=int(pcfg.free_label),
        min_component_voxels=1,
        max_match_speed_mps=float(
            strong_cfg.max_match_speed_mps
        ),
        connectivity=int(strong_cfg.connectivity),
        fill_kernel=tuple(strong_cfg.fill_kernel),
        fill_min_fraction=float(strong_cfg.fill_min_fraction),
    )
    metric_grid = _grid_spec(pcfg.grid)

    totals = {
        "candidate_voxels": 0,
        "positive_voxels": 0,
        "candidate_bev_columns": 0,
        "positive_bev_columns": 0,
        "mixed_semantic_positive_columns": 0,
    }
    class_hist = {str(i): 0 for i in range(17)}
    shards = []
    shard_rows = []
    scenes = set()
    started = time.perf_counter()

    for wi, rec in enumerate(records, start=1):
        w = window_from_record(rec)
        scenes.add(str(w.scene_name))
        raw = load_nuscenes_window_raw(
            source,
            w,
            pcfg,
            include_gt=True,
        )

        state = _prepare_record(
            rec,
            source,
            pcfg,
            strong_cfg,
            device,
        )
        _stage_gpu_inputs(state, device)
        try:
            pred_all = _forecast_once(
                model,
                state,
                pcfg,
                strong_cfg,
                device,
            )
        finally:
            _release_gpu_inputs(state)

        history_occ = np.asarray(
            raw["history_occ"],
            dtype=np.uint8,
        )
        history_obs = np.asarray(
            raw["history_observed"],
            dtype=bool,
        )
        history_poses = np.asarray(
            raw["history_poses"],
            dtype=np.float64,
        )
        future_poses = np.asarray(
            raw["future_poses"],
            dtype=np.float64,
        )
        ann_hist = [
            _ann_map(source.nusc, tok)
            for tok in w.history_tokens
        ]
        ann0 = ann_hist[-1]
        t0_tokens = set(ann0)
        source_tokens = match_sources_to_annotations(
            state["current"],
            dynamic_annotations(source.nusc, w.t0_token),
            max_distance_m=float(a.match_max_distance_m),
        )
        represented = {
            str(x)
            for x in source_tokens
            if x is not None
        }

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
        target_occ_all = []
        semantic_target_all = []

        for fi in range(len(future_poses)):
            gt = np.asarray(
                raw["future_gt_occ"][fi],
                dtype=np.uint8,
            )
            pred_v18 = np.asarray(
                pred_all[fi],
                dtype=np.uint8,
            )
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
                match_max_distance_m=float(
                    a.match_max_distance_m
                ),
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
            candidate = (
                new_fov[..., None]
                & base_free
            )
            target = (
                np.asarray(
                    masks["never_seen_static"],
                    dtype=bool,
                )
                & candidate
            )
            if bool((target & ~candidate).any()):
                raise RuntimeError(
                    "Static New-FOV target escaped causal candidate"
                )

            semantic_target = majority_semantic_per_column(
                target,
                gt,
                num_classes=17,
                ignore_label=255,
            )
            positive_bev = target.any(axis=2)
            candidate_bev = candidate.any(axis=2)

            totals["candidate_voxels"] += int(candidate.sum())
            totals["positive_voxels"] += int(target.sum())
            totals["candidate_bev_columns"] += int(
                candidate_bev.sum()
            )
            totals["positive_bev_columns"] += int(
                positive_bev.sum()
            )

            for x, y in np.argwhere(positive_bev):
                vals = gt[x, y][target[x, y]]
                uniq, counts = np.unique(
                    vals,
                    return_counts=True,
                )
                totals["mixed_semantic_positive_columns"] += int(
                    len(uniq) > 1
                )
                for cid, n in zip(uniq, counts):
                    if int(cid) < 17:
                        class_hist[str(int(cid))] += int(n)

            explained_all.append(explained)
            base_free_all.append(base_free)
            new_fov_all.append(new_fov)
            target_occ_all.append(target)
            semantic_target_all.append(semantic_target)

        explained_all = np.stack(
            explained_all,
            axis=0,
        ).astype(np.uint8)
        base_free_all = np.stack(
            base_free_all,
            axis=0,
        )
        target_occ_all = np.stack(
            target_occ_all,
            axis=0,
        )

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
            "occupancy_target_bits": torch.from_numpy(
                pack_vertical_occupancy(target_occ_all)
            ),
            "semantic_target": torch.from_numpy(
                np.stack(
                    semantic_target_all,
                    axis=0,
                ).astype(np.uint8)
            ),
            "scene_name": str(w.scene_name),
            "t0_token": str(w.t0_token),
        }
        shard_rows.append(row)

        if len(shard_rows) >= int(a.shard_size):
            shards.append(
                _flush_shard(
                    out_dir,
                    len(shards),
                    shard_rows,
                )
            )
            shard_rows = []

        if (
            wi == 1
            or wi % 25 == 0
            or wi == len(records)
        ):
            elapsed = max(
                time.perf_counter() - started,
                1e-9,
            )
            print(
                f"v19_static_new_fov_cache "
                f"{wi}/{len(records)} "
                f"rate={wi/elapsed:.3f} win/s",
                flush=True,
            )

    if shard_rows:
        shards.append(
            _flush_shard(
                out_dir,
                len(shards),
                shard_rows,
            )
        )

    pos = int(totals["positive_voxels"])
    cand = int(totals["candidate_voxels"])
    index = {
        "protocol": PROTOCOL,
        "source_cache": str(Path(a.source_cache).resolve()),
        "source_cache_metadata_keys": sorted(
            str(k) for k in source_meta.keys()
        ),
        "base_checkpoint": str(
            Path(a.checkpoint).resolve()
        ),
        "base_checkpoint_epoch": int(ck.get("epoch", -1)),
        "num_windows": int(len(records)),
        "global_num_windows_before_shard": int(global_num_windows),
        "num_scenes": int(len(scenes)),
        "scene_names": sorted(scenes),
        "future_frames": int(len(future_poses)),
        "history_frames": int(len(history_poses)),
        "grid_shape_hwd": [
            int(x) for x in pcfg.grid.shape_hwd
        ],
        "free_label": int(pcfg.free_label),
        "target": "never_seen_static_intersect_geometric_new_fov",
        "representation": (
            "direct_z_occupancy_plus_majority_column_semantic"
        ),
        "candidate_contract": (
            "geometric new-FOV BEV x base-free voxel; "
            "causal under frozen V18 future-ego contract"
        ),
        "future_gt_used_for_inference_input": False,
        "totals": totals,
        "occupancy_positive_fraction": float(
            pos / max(cand, 1)
        ),
        "occupancy_neg_pos_ratio": float(
            (cand - pos) / max(pos, 1)
        ),
        "semantic_voxel_class_histogram": class_hist,
        "shards": shards,
        "timing": {
            "elapsed_s": float(
                max(time.perf_counter() - started, 1e-9)
            ),
            "windows_per_s": float(
                len(records)
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
    print("\n=== V19 STATIC NEW-FOV CACHE ===")
    print(
        json.dumps(
            {
                "num_windows": index["num_windows"],
                "num_scenes": index["num_scenes"],
                "totals": totals,
                "occupancy_positive_fraction": index[
                    "occupancy_positive_fraction"
                ],
                "occupancy_neg_pos_ratio": index[
                    "occupancy_neg_pos_ratio"
                ],
            },
            indent=2,
        )
    )
    print(f"saved {out_dir / 'index.json'}")


if __name__ == "__main__":
    main()
