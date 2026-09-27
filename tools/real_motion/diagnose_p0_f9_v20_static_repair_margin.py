#!/usr/bin/env python3
"""Zero-training margin calibration for V20 Static Repair.

This diagnostic answers two questions with one frozen checkpoint:
1) Does the Static head already rank useful additions below the default
   argmax boundary (max_static_logit - free_logit >= 0)?
2) Is low add recall also present on a fixed train subset, or only on dev?

Every support voxel is decoded once.  Thresholds are evaluated from activation
buckets, so scanning many tau values does not rerun the network or multiply the
full-grid metric cost by the number of thresholds.

Selection contract:
* tau is one global Static abstention threshold;
* dev tau is selected only by composed semantic mIoU;
* binary occupancy IoU / precision are diagnostics, never selection criteria;
* train512 is diagnostic only and never selects tau.
"""
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

from real_motion.v20_history_world import FREE_LABEL
from real_motion.v20_static_repair import (
    STATIC_ALLOWED_IDS,
    SUPPORT_CACHE_PROTOCOL,
    TRAIN_PROTOCOL,
    full_grid_metrics_from_confusion,
    repair_diagnostics_from_confusion,
)
from real_motion.v20_training import load_v20_checkpoint
from tools.real_motion.build_p0_f9_v20_history_cache import (
    PROTOCOL as STAGE1_PROTOCOL,
)
from tools.real_motion.eval_p0_f9_v19_factorized_static_new_fov import CachedSource
from tools.real_motion.train_p0_f9_v20_static_repair import (
    _Geometry,
    _decode_query_logits_sparse,
    _iter_prepared,
    _lattice,
    _load_index,
)


DEFAULT_THRESHOLDS = (
    2.0,
    1.0,
    0.5,
    0.25,
    0.0,
    -0.25,
    -0.5,
    -0.75,
    -1.0,
    -1.5,
    -2.0,
    -2.5,
    -3.0,
    -4.0,
    -5.0,
    -6.0,
    -8.0,
)


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _parse_thresholds(raw: str) -> list[float]:
    if not str(raw).strip():
        vals = list(DEFAULT_THRESHOLDS)
    else:
        vals = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise ValueError("threshold list is empty")
    vals.append(0.0)
    vals = sorted(set(float(x) for x in vals))
    if not all(np.isfinite(x) for x in vals):
        raise ValueError("thresholds must be finite")
    return vals


def _check_cache_pair(stage_root: Path, stage_idx: dict, repair_root: Path, repair_idx: dict):
    if str(Path(repair_idx["stage1_cache"]).resolve()) != str(stage_root.resolve()):
        raise RuntimeError(
            f"{repair_root}: repair cache references different Stage1 cache"
        )
    if tuple(int(x) for x in repair_idx["native_shape_xyz"]) != tuple(
        int(x) for x in stage_idx["native_grid"]["shape_xyz"]
    ):
        raise RuntimeError(f"{repair_root}: native shape mismatch")


def _metric_row(tau, base_metrics, final_metrics, diag):
    delta = {
        "IoU": float(final_metrics["IoU"] - base_metrics["IoU"]),
        "mIoU": float(final_metrics["mIoU"] - base_metrics["mIoU"]),
        "main_1_2_3s": {
            "IoU": float(
                final_metrics["main_1_2_3s"]["IoU"]
                - base_metrics["main_1_2_3s"]["IoU"]
            ),
            "mIoU": float(
                final_metrics["main_1_2_3s"]["mIoU"]
                - base_metrics["main_1_2_3s"]["mIoU"]
            ),
        },
    }
    return {
        "tau": float(tau),
        "repair_diagnostics": diag,
        "base_metrics": base_metrics,
        "composed_metrics": final_metrics,
        "delta": delta,
    }


def _finalize_thresholds(
    thresholds: list[float],
    bucket_target: np.ndarray,
    bucket_gt: np.ndarray,
    base_full: np.ndarray,
):
    """Convert activation buckets to exact confusion/metrics for every tau.

    thresholds are ascending. torch.bucketize(..., right=True) gives b =
    number of thresholds <= margin. A sample is active for threshold index i
    iff b > i.
    """
    T = len(thresholds)
    free = int(FREE_LABEL)
    all_target = bucket_target.sum(axis=(0, 2))
    base_metrics = full_grid_metrics_from_confusion(base_full)

    # suffix[b] = contribution of samples whose activation bucket >= b.
    target_suffix = np.cumsum(bucket_target[::-1], axis=0)[::-1]
    gt_suffix = np.cumsum(bucket_gt[:, ::-1], axis=1)[:, ::-1]

    rows = []
    for i, tau in enumerate(thresholds):
        # Active iff bucket index > i.
        active_target = target_suffix[i + 1]
        conf = np.zeros((18, 18), dtype=np.int64)
        conf[:, free] = all_target
        active_by_target = active_target.sum(axis=1)
        conf[:, free] -= active_by_target
        conf += active_target
        diag = repair_diagnostics_from_confusion(conf)

        final_conf = np.array(base_full, dtype=np.int64, copy=True)
        for hi in range(6):
            active_gt = gt_suffix[hi, i + 1]
            active_by_gt = active_gt.sum(axis=1)
            final_conf[hi, :, free] -= active_by_gt
            final_conf[hi] += active_gt

        final_metrics = full_grid_metrics_from_confusion(final_conf)
        rows.append(
            _metric_row(tau, base_metrics, final_metrics, diag)
        )
    return rows


def _margin_quantiles(samples: list[np.ndarray]):
    if not samples:
        return {}
    x = np.concatenate(samples)
    qs = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
    vals = np.quantile(x, qs)
    return {
        f"{q:.2f}": float(v)
        for q, v in zip(qs, vals)
    }


@torch.inference_mode()
def _scan_split(
    *,
    name: str,
    model,
    stage_root: Path,
    repair_root: Path,
    repair_idx: dict,
    source,
    geom: _Geometry,
    thresholds: list[float],
    device: torch.device,
    amp: bool,
    max_windows: int,
    workers: int,
    prefetch: int,
    tile_batch_size: int,
    tile_batch_pad_multiple: int,
    progress_every: int,
):
    T = len(thresholds)
    threshold_t = torch.as_tensor(
        thresholds, dtype=torch.float32, device=device
    )
    allowed = torch.as_tensor(
        STATIC_ALLOWED_IDS, dtype=torch.long, device=device
    )
    free_pos = torch.nonzero(
        allowed == int(FREE_LABEL), as_tuple=False
    ).reshape(-1)
    if int(free_pos.numel()) != 1:
        raise RuntimeError("Static allowed taxonomy must contain free exactly once")
    free_local = int(free_pos.item())
    static_local = torch.nonzero(
        allowed != int(FREE_LABEL), as_tuple=False
    ).reshape(-1)
    static_global = allowed[static_local]

    bucket_target_gpu = torch.zeros(
        (T + 1, 18, 18), dtype=torch.int64, device=device
    )
    bucket_gt_gpu = torch.zeros(
        (6, T + 1, 18, 18), dtype=torch.int64, device=device
    )
    base_full = np.zeros((6, 18, 18), dtype=np.int64)
    margin_samples = []

    prepared = _iter_prepared(
        stage_root,
        repair_root,
        repair_idx,
        source,
        native_shape=geom.native_shape,
        free_label=FREE_LABEL,
        shuffle=False,
        seed=0,
        max_windows=int(max_windows),
        skip_windows=0,
        workers=int(workers),
        prefetch=int(prefetch),
        fixed_identities=None,
        need_metrics=True,
    )

    total = (
        min(int(repair_idx["num_windows"]), int(max_windows))
        if int(max_windows) > 0
        else int(repair_idx["num_windows"])
    )
    started = time.perf_counter()
    windows = 0
    tiles = 0

    for item in prepared:
        windows += 1
        base_full += np.asarray(item["base_full_conf"], dtype=np.int64)

        sem = torch.from_numpy(item["sem"]).to(
            device, non_blocking=True
        ).unsqueeze(0)
        obs = torch.from_numpy(item["obs"]).to(
            device, non_blocking=True
        ).unsqueeze(0)
        obsfree = torch.from_numpy(item["obsfree"]).to(
            device, non_blocking=True
        ).unsqueeze(0)

        with _autocast(device, amp):
            scene = model.encode_history(sem, obs, obsfree)
            linear, q = geom.future_linear_and_query(item["future_rel"])
            query_logits, row_map, ntiles = _decode_query_logits_sparse(
                model,
                scene,
                q,
                item["obs"],
                geom,
                tile_batch_size=int(tile_batch_size),
                tile_batch_pad_multiple=int(tile_batch_pad_multiple),
            )
        tiles += int(ntiles)

        support_t = torch.from_numpy(
            np.asarray(item["support"], dtype=bool)
        ).to(device, non_blocking=True)
        target_t = torch.from_numpy(
            np.asarray(item["target"], dtype=np.uint8)
        ).to(device, non_blocking=True)
        gt_t = torch.from_numpy(
            np.asarray(item["gt"], dtype=np.uint8)
        ).to(device, non_blocking=True)

        for hi in range(6):
            s = support_t[hi].reshape(-1)
            qrow = row_map[linear[hi][s]].long()
            if device.type == "cuda" and hasattr(torch, "_assert_async"):
                torch._assert_async(
                    (qrow >= 0).all(),
                    "margin diagnostic support escaped query union",
                )
            elif not bool((qrow >= 0).all().item()):
                raise RuntimeError(
                    "margin diagnostic support escaped query union"
                )

            rows = query_logits[qrow].float()
            static_scores, static_choice = rows.index_select(
                1, static_local
            ).max(dim=1)
            pred_static = static_global[static_choice]
            margin = static_scores - rows[:, free_local]

            # b = number of thresholds <= margin. The candidate is active at
            # threshold index i iff b > i.
            b = torch.bucketize(margin, threshold_t, right=True).long()
            target = target_t[hi].reshape(-1)[s].long()
            gt = gt_t[hi].reshape(-1)[s].long()

            code_target = (
                b * (18 * 18) + target * 18 + pred_static
            )
            bucket_target_gpu += torch.bincount(
                code_target,
                minlength=(T + 1) * 18 * 18,
            ).reshape(T + 1, 18, 18)

            code_gt = b * (18 * 18) + gt * 18 + pred_static
            bucket_gt_gpu[hi] += torch.bincount(
                code_gt,
                minlength=(T + 1) * 18 * 18,
            ).reshape(T + 1, 18, 18)

            # Small deterministic sample for score-scale diagnostics only.
            n = int(margin.numel())
            if n > 0:
                take = min(n, 64)
                if take == n:
                    sample = margin
                else:
                    idx = torch.linspace(
                        0, n - 1, steps=take,
                        device=device,
                    ).long()
                    sample = margin[idx]
                margin_samples.append(
                    sample.detach().cpu().numpy().astype(np.float32)
                )

        del sem, obs, obsfree, scene, linear, q, query_logits, row_map
        if progress_every > 0 and (
            windows == 1
            or windows % int(progress_every) == 0
            or windows == total
        ):
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(
                f"v20_static_margin_{name} {windows}/{total} "
                f"rate={windows/elapsed:.3f} win/s "
                f"tiles={tiles/max(windows,1):.1f}/win",
                flush=True,
            )

    if windows != total:
        raise RuntimeError(
            f"{name}: expected {total} windows, scanned {windows}"
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    bucket_target = bucket_target_gpu.cpu().numpy()
    bucket_gt = bucket_gt_gpu.cpu().numpy()
    rows = _finalize_thresholds(
        thresholds,
        bucket_target,
        bucket_gt,
        base_full,
    )
    # User-facing order: conservative -> permissive.
    rows = sorted(rows, key=lambda x: float(x["tau"]), reverse=True)

    best = max(
        rows,
        key=lambda x: (
            float(x["composed_metrics"]["mIoU"]),
            float(x["tau"]),
        ),
    )
    tau0 = min(rows, key=lambda x: abs(float(x["tau"])))
    return {
        "name": str(name),
        "windows": int(windows),
        "elapsed_seconds": float(time.perf_counter() - started),
        "mean_tiles_per_window": float(tiles / max(windows, 1)),
        "margin_sample_quantiles": _margin_quantiles(margin_samples),
        "thresholds": rows,
        "tau0": tau0,
        "best_by_composed_miou": best,
    }


def _history_reference(checkpoint_obj: dict):
    hist = list((checkpoint_obj.get("extra") or {}).get("history") or [])
    if not hist:
        return None
    val = dict(hist[-1].get("val") or {})
    if not val:
        return None
    return {
        "epoch": int(hist[-1].get("epoch", 0)),
        "repair_diagnostics": dict(val.get("repair_diagnostics") or {}),
        "full_grid_delta": dict(val.get("full_grid_delta") or {}),
        "full_grid_v18_metrics": dict(val.get("full_grid_v18_metrics") or {}),
        "full_grid_v18_plus_static_metrics": dict(
            val.get("full_grid_v18_plus_static_metrics") or {}
        ),
        "windows": val.get("windows"),
    }


def _assert_tau0_matches_history(dev_result: dict, checkpoint_obj: dict):
    ref = _history_reference(checkpoint_obj)
    if ref is None:
        return {"checked": False, "reason": "checkpoint has no val history"}
    tau0 = dev_result["tau0"]
    if int(ref.get("windows") or -1) != int(dev_result["windows"]):
        return {
            "checked": False,
            "reason": (
                "checkpoint val history population differs from diagnostic dev "
                f"({ref.get('windows')} != {dev_result['windows']})"
            ),
        }

    got_diag = tau0["repair_diagnostics"]
    ref_diag = ref["repair_diagnostics"]
    for key in ("added_tp", "added_fp", "predicted_add_voxels"):
        if int(got_diag.get(key, -1)) != int(ref_diag.get(key, -2)):
            raise RuntimeError(
                f"tau=0 failed checkpoint-history reproduction for {key}: "
                f"{got_diag.get(key)} != {ref_diag.get(key)}"
            )
    got_delta = tau0["delta"]
    ref_delta = ref["full_grid_delta"]
    for key in ("IoU", "mIoU"):
        if abs(float(got_delta[key]) - float(ref_delta[key])) > 1e-9:
            raise RuntimeError(
                f"tau=0 failed checkpoint-history reproduction for {key}: "
                f"{got_delta[key]} != {ref_delta[key]}"
            )
    for key in ("IoU", "mIoU"):
        if abs(
            float(got_delta["main_1_2_3s"][key])
            - float(ref_delta["main_1_2_3s"][key])
        ) > 1e-9:
            raise RuntimeError(
                "tau=0 failed checkpoint-history reproduction for "
                f"main_1_2_3s.{key}"
            )
    return {
        "checked": True,
        "passed": True,
        "reference_epoch": int(ref["epoch"]),
    }


def _compact(row):
    d = row["repair_diagnostics"]
    m = row["composed_metrics"]
    delta = row["delta"]
    return {
        "tau": float(row["tau"]),
        "addP": float(d["addition_precision"]),
        "addR": float(d["static_positive_recall"]),
        "semAcc": float(d["semantic_accuracy_on_static_positive"]),
        "predicted_add_voxels": int(d["predicted_add_voxels"]),
        "IoU": float(m["IoU"]),
        "mIoU": float(m["mIoU"]),
        "delta_IoU": float(delta["IoU"]),
        "delta_mIoU": float(delta["mIoU"]),
        "main_1_2_3s_mIoU": float(m["main_1_2_3s"]["mIoU"]),
        "delta_main_1_2_3s_mIoU": float(
            delta["main_1_2_3s"]["mIoU"]
        ),
    }


def _print_split(result: dict):
    print(
        f"\n=== {result['name'].upper()} THRESHOLD SWEEP "
        f"({result['windows']} windows) ==="
    )
    print(
        " tau      addP      addR    semAcc   adds        "
        "mIoU    dmIoU   d123mIoU"
    )
    print("-" * 86)
    for row in result["thresholds"]:
        x = _compact(row)
        print(
            f"{x['tau']:>6.2f} "
            f"{x['addP']:>9.4f} "
            f"{x['addR']:>9.4f} "
            f"{x['semAcc']:>9.4f} "
            f"{x['predicted_add_voxels']:>8d} "
            f"{x['mIoU']:>9.4f} "
            f"{x['delta_mIoU']:>8.4f} "
            f"{x['delta_main_1_2_3s_mIoU']:>11.4f}"
        )
    print("\nmargin sample quantiles:")
    print(json.dumps(result["margin_sample_quantiles"], indent=2))
    print("\ntau=0:")
    print(json.dumps(_compact(result["tau0"]), indent=2))
    print("\nbest by composed semantic mIoU:")
    print(json.dumps(_compact(result["best_by_composed_miou"]), indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train-stage1-cache", required=True)
    p.add_argument("--train-repair-cache", required=True)
    p.add_argument("--dev-stage1-cache", required=True)
    p.add_argument("--dev-repair-cache", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--train-info-pkl", required=True)
    p.add_argument("--dev-info-pkl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--train-windows", type=int, default=512)
    p.add_argument("--dev-windows", type=int, default=0)
    p.add_argument(
        "--thresholds",
        default=",".join(str(x) for x in DEFAULT_THRESHOLDS),
    )
    p.add_argument("--tile-batch-size", type=int, default=256)
    p.add_argument(
        "--tile-batch-pad-multiple",
        type=int,
        default=0,
        help="0 = reuse checkpoint training value.",
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--prefetch", type=int, default=32)
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-channels-last-3d", action="store_true")
    p.add_argument(
        "--allow-diagnostic-checkpoint",
        action="store_true",
        help="Allow a checkpoint explicitly marked diagnostic_only.",
    )
    a = p.parse_args()

    thresholds = _parse_thresholds(a.thresholds)
    device = torch.device(
        a.device
        if a.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    amp = device.type == "cuda" and not bool(a.no_amp)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    train_stage_root, train_stage_idx = _load_index(
        a.train_stage1_cache, STAGE1_PROTOCOL
    )
    train_repair_root, train_repair_idx = _load_index(
        a.train_repair_cache, SUPPORT_CACHE_PROTOCOL
    )
    dev_stage_root, dev_stage_idx = _load_index(
        a.dev_stage1_cache, STAGE1_PROTOCOL
    )
    dev_repair_root, dev_repair_idx = _load_index(
        a.dev_repair_cache, SUPPORT_CACHE_PROTOCOL
    )
    _check_cache_pair(
        train_stage_root, train_stage_idx,
        train_repair_root, train_repair_idx,
    )
    _check_cache_pair(
        dev_stage_root, dev_stage_idx,
        dev_repair_root, dev_repair_idx,
    )

    model, ck = load_v20_checkpoint(a.checkpoint, map_location="cpu")
    if str(ck.get("stage")) != "static":
        raise RuntimeError(
            f"margin diagnostic requires stage='static', got {ck.get('stage')!r}"
        )
    extra = dict(ck.get("extra") or {})
    if extra.get("train_protocol") != TRAIN_PROTOCOL:
        raise RuntimeError(
            "checkpoint is not Static Repair v2"
        )
    if bool(extra.get("diagnostic_only", True)) and not bool(
        a.allow_diagnostic_checkpoint
    ):
        raise RuntimeError(
            "checkpoint is marked diagnostic_only; pass "
            "--allow-diagnostic-checkpoint only for an intentional diagnosis"
        )

    expected_base = str(Path(train_repair_idx["base_checkpoint"]).resolve())
    ck_base = str(Path(ck["v18_checkpoint"]).resolve())
    if ck_base != expected_base:
        raise RuntimeError(
            f"checkpoint/support V18 mismatch: {ck_base} != {expected_base}"
        )
    if str(Path(dev_repair_idx["base_checkpoint"]).resolve()) != expected_base:
        raise RuntimeError("train/dev repair caches use different V18 checkpoints")

    if train_stage_idx["highres_lattice"] != dev_stage_idx["highres_lattice"]:
        raise RuntimeError("train/dev highres lattice mismatch")
    if train_stage_idx["coarse_lattice"] != dev_stage_idx["coarse_lattice"]:
        raise RuntimeError("train/dev coarse lattice mismatch")
    if train_stage_idx["native_grid"] != dev_stage_idx["native_grid"]:
        raise RuntimeError("train/dev native grid mismatch")
    if extra.get("highres_lattice") != train_stage_idx["highres_lattice"]:
        raise RuntimeError("checkpoint/highres lattice mismatch")
    if extra.get("coarse_lattice") != train_stage_idx["coarse_lattice"]:
        raise RuntimeError("checkpoint/coarse lattice mismatch")

    high = _lattice(train_stage_idx["highres_lattice"])
    coarse = _lattice(train_stage_idx["coarse_lattice"])
    native = dict(train_stage_idx["native_grid"])
    tile_size = tuple(
        int(x) for x in extra.get("tile_size_xyz", [32, 32, 16])
    )
    geom = _Geometry(
        high,
        coarse,
        tuple(int(x) for x in native["shape_xyz"]),
        tuple(float(x) for x in native["origin_xyz_m"]),
        tuple(float(x) for x in native["voxel_size_xyz_m"]),
        tile_size,
        device,
    )

    model.to(device).eval()
    channels_last = (
        device.type == "cuda"
        and bool(extra.get("channels_last_3d", True))
        and not bool(a.no_channels_last_3d)
    )
    model.encoder.set_channels_last_3d(channels_last)
    model.static.set_tile_channels_last_3d(channels_last)

    pad_multiple = int(a.tile_batch_pad_multiple)
    if pad_multiple <= 0:
        pad_multiple = int(extra.get("tile_batch_pad_multiple", 1))
    if pad_multiple < 1:
        raise ValueError("tile batch pad multiple must be >=1")

    train_total = min(
        int(train_repair_idx["num_windows"]),
        max(int(a.train_windows), 0),
    )
    if int(a.train_windows) <= 0:
        raise ValueError("--train-windows must be >0 for the fixed train diagnostic")
    dev_total = (
        min(int(dev_repair_idx["num_windows"]), int(a.dev_windows))
        if int(a.dev_windows) > 0
        else int(dev_repair_idx["num_windows"])
    )

    print(json.dumps({
        "protocol": "p0_f9_v20_static_repair_margin_diagnostic_v1",
        "checkpoint": str(Path(a.checkpoint).resolve()),
        "checkpoint_stage": ck.get("stage"),
        "checkpoint_formal": not bool(extra.get("diagnostic_only", True)),
        "train_windows": int(train_total),
        "dev_windows": int(dev_total),
        "thresholds_ascending": thresholds,
        "selection_metric": "composed semantic mIoU on dev only",
        "train_subset": "first N canonical train-cache identities; diagnostic only",
        "amp_bfloat16": bool(amp),
        "channels_last_3d": bool(channels_last),
        "tile_batch_size": int(a.tile_batch_size),
        "tile_batch_pad_multiple": int(pad_multiple),
    }, indent=2), flush=True)

    train_source = CachedSource(
        a.dataroot, info_pkl=a.train_info_pkl, verbose=False
    )
    dev_source = CachedSource(
        a.dataroot, info_pkl=a.dev_info_pkl, verbose=False
    )

    # Scan dev first because it is the only calibration population.
    dev = _scan_split(
        name="dev",
        model=model,
        stage_root=dev_stage_root,
        repair_root=dev_repair_root,
        repair_idx=dev_repair_idx,
        source=dev_source,
        geom=geom,
        thresholds=thresholds,
        device=device,
        amp=amp,
        max_windows=int(a.dev_windows),
        workers=int(a.workers),
        prefetch=int(a.prefetch),
        tile_batch_size=int(a.tile_batch_size),
        tile_batch_pad_multiple=pad_multiple,
        progress_every=int(a.progress_every),
    )
    tau0_reproduction = _assert_tau0_matches_history(dev, ck)
    _print_split(dev)

    train = _scan_split(
        name="train_fixed",
        model=model,
        stage_root=train_stage_root,
        repair_root=train_repair_root,
        repair_idx=train_repair_idx,
        source=train_source,
        geom=geom,
        thresholds=thresholds,
        device=device,
        amp=amp,
        max_windows=int(train_total),
        workers=int(a.workers),
        prefetch=int(a.prefetch),
        tile_batch_size=int(a.tile_batch_size),
        tile_batch_pad_multiple=pad_multiple,
        progress_every=int(a.progress_every),
    )
    _print_split(train)

    dev_best_tau = float(dev["best_by_composed_miou"]["tau"])
    train_at_dev_tau = min(
        train["thresholds"],
        key=lambda x: abs(float(x["tau"]) - dev_best_tau),
    )

    result = {
        "protocol": "p0_f9_v20_static_repair_margin_diagnostic_v1",
        "checkpoint": str(Path(a.checkpoint).resolve()),
        "checkpoint_extra": {
            "diagnostic_only": bool(extra.get("diagnostic_only", True)),
            "checkpoint_eligible_for_formal_selection": bool(
                extra.get("checkpoint_eligible_for_formal_selection", False)
            ),
            "train_protocol": extra.get("train_protocol"),
        },
        "selection_contract": {
            "selected_on": "dev",
            "metric": "composed semantic mIoU",
            "one_global_tau": True,
            "binary_IoU_used_for_selection": False,
            "train_used_for_selection": False,
        },
        "tau0_reproduction": tau0_reproduction,
        "dev": dev,
        "train_fixed": train,
        "cross_split_summary": {
            "dev_tau0": _compact(dev["tau0"]),
            "train_tau0": _compact(train["tau0"]),
            "dev_best": _compact(dev["best_by_composed_miou"]),
            "train_at_dev_best_tau": _compact(train_at_dev_tau),
            "dev_best_tau": dev_best_tau,
            "dev_threshold_gain_over_tau0_mIoU": float(
                dev["best_by_composed_miou"]["composed_metrics"]["mIoU"]
                - dev["tau0"]["composed_metrics"]["mIoU"]
            ),
            "tau0_recall_train_minus_dev": float(
                train["tau0"]["repair_diagnostics"]["static_positive_recall"]
                - dev["tau0"]["repair_diagnostics"]["static_positive_recall"]
            ),
        },
    }

    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== CROSS-SPLIT SUMMARY ===")
    print(json.dumps(result["cross_split_summary"], indent=2))
    print("\n=== TAU=0 REPRODUCTION ===")
    print(json.dumps(tau0_reproduction, indent=2))
    print(f"saved {op}")


if __name__ == "__main__":
    main()
