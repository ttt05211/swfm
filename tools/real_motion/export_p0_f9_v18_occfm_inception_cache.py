#!/usr/bin/env python3
"""Export frozen Clean-E14 six-frame occupancy clips for inception-style metrics.

Default output is one compressed NPZ per validation window:
  clip_00000.npz: pred[6,200,200,16], gt[6,200,200,16], uint8

This avoids the ~30+ GB footprint of duplicating uncompressed pred/gt NPY files.
For strict compatibility with OccFM's released FVDEval loader, optionally pass
--official-npy-dir; the script will additionally write pred_<i>.npy and
gt_<i>.npy with exactly the [6,200,200,16] clip layout documented by OccFM.

Prediction is strictly causal under the frozen V18 contract.  Future GT is read
only after prediction and is used only as an evaluation target.
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

from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.strong_w2det import StrongW2DetConfig
from tools.real_motion import eval_p0_f9_v18_full_validation as mid
from tools.real_motion.benchmark_p0_f9_v18_runtime import (
    CachedSource,
    _exactness_check,
    _forecast_once,
    _prepare_record,
)
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.train_p0_f9_v18_se2_clean import PROTOCOL as CLEAN_PROTOCOL

PROTOCOL = "p0_f9_v18_occfm_inception_clip_export_v1"


def _valid_clip_file(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        x = np.load(path)
        pred = np.asarray(x["pred"])
        gt = np.asarray(x["gt"])
        return (
            pred.shape == (6, 200, 200, 16)
            and gt.shape == pred.shape
            and pred.dtype == np.uint8
            and gt.dtype == np.uint8
        )
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--official-npy-dir",
        default="",
        help="optional extra directory for OccFM-compatible pred_i.npy/gt_i.npy files",
    )
    p.add_argument("--expected-windows", type=int, default=0)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    _, records = mid.base.load_cache(a.val_cache)
    if int(a.expected_windows) > 0 and len(records) != int(a.expected_windows):
        raise RuntimeError(
            f"validation windows {len(records)} != expected {a.expected_windows}"
        )
    if int(a.max_windows) > 0:
        records = records[: min(len(records), int(a.max_windows))]
    if not records:
        raise RuntimeError("no validation records selected")

    device = torch.device(
        a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    ck, model, _ = mid._load_model(a.checkpoint, CLEAN_PROTOCOL, device)
    training_mode = str(ck.get("training_mode") or "")
    variant = str(ck.get("variant") or "")
    if "balanced" in training_mode.lower() or "balanced" in variant.lower():
        raise RuntimeError(
            f"export refuses balanced checkpoint: "
            f"training_mode={training_mode!r} variant={variant!r}"
        )
    allowed_main_modes = {
        "",
        "clean_one_stage_from_scratch_v1",
        "clean_one_stage_from_scratch_v1_tail_continuation",
    }
    if training_mode not in allowed_main_modes:
        raise RuntimeError(
            f"unexpected frozen-main checkpoint training_mode={training_mode!r}"
        )
    if training_mode.endswith("_tail_continuation") and int(ck.get("epoch", -1)) != 14:
        raise RuntimeError(
            "formal inception export expects the frozen Clean-E14 tail checkpoint; "
            f"got epoch={ck.get('epoch')!r}"
        )
    print(
        "EXPORT CHECKPOINT "
        + json.dumps({
            "protocol": ck.get("protocol"),
            "training_mode": ck.get("training_mode"),
            "variant": ck.get("variant"),
            "epoch": ck.get("epoch"),
            "global_step": ck.get("global_step"),
        }),
        flush=True,
    )

    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_dir = Path(a.official_npy_dir) if a.official_npy_dir else None
    if npy_dir is not None:
        npy_dir.mkdir(parents=True, exist_ok=True)

    source = CachedSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong_cfg = StrongW2DetConfig(free_label=int(pcfg.free_label))
    manifest = []
    started = time.perf_counter()
    built = skipped = 0
    exactness_done = False

    for i, rec in enumerate(records):
        sid = str(rec["sample_id"])
        scene = str(rec["scene_name"])
        clip_path = out_dir / f"clip_{i:05d}.npz"
        pred_npy = npy_dir / f"pred_{i}.npy" if npy_dir is not None else None
        gt_npy = npy_dir / f"gt_{i}.npy" if npy_dir is not None else None

        can_skip = bool(a.resume) and _valid_clip_file(clip_path)
        if can_skip and npy_dir is not None:
            can_skip = pred_npy.exists() and gt_npy.exists()
        if can_skip:
            skipped += 1
            manifest.append({"index": i, "sample_id": sid, "scene_name": scene})
            continue

        state = _prepare_record(rec, source, pcfg, strong_cfg, device)
        if not exactness_done:
            _exactness_check(model, state, pcfg, strong_cfg, device)
            exactness_done = True

        # Causal prediction first.
        pred = np.stack(
            _forecast_once(model, state, pcfg, strong_cfg, device), axis=0
        ).astype(np.uint8, copy=False)

        # Evaluation target is loaded only after the prediction is complete.
        w = window_from_record(rec)
        gt = np.stack(
            [
                np.asarray(source.load_semantics(str(w.scene_name), str(tok)), dtype=np.uint8)
                for tok in w.future_tokens
            ],
            axis=0,
        )

        if pred.shape != (6, 200, 200, 16) or gt.shape != pred.shape:
            raise RuntimeError(
                f"{sid}: unexpected clip shapes pred={pred.shape} gt={gt.shape}"
            )
        np.savez_compressed(
            clip_path,
            pred=pred,
            gt=gt,
            sample_id=np.asarray(sid),
            scene_name=np.asarray(scene),
        )
        if npy_dir is not None:
            np.save(pred_npy, pred, allow_pickle=False)
            np.save(gt_npy, gt, allow_pickle=False)

        built += 1
        manifest.append({"index": i, "sample_id": sid, "scene_name": scene})
        done = built + skipped
        if built == 1 or done % 50 == 0 or done == len(records):
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(
                f"inception_export {done}/{len(records)} "
                f"built={built} skipped={skipped} "
                f"new_rate={built/elapsed:.3f} win/s",
                flush=True,
            )

    meta = {
        "protocol": PROTOCOL,
        "checkpoint": str(Path(a.checkpoint).resolve()),
        "checkpoint_epoch": int(ck.get("epoch", -1)),
        "checkpoint_global_step": int(ck.get("global_step", -1)),
        "val_cache": str(Path(a.val_cache).resolve()),
        "num_windows": len(records),
        "clip_shape": [6, 200, 200, 16],
        "dtype": "uint8",
        "future_times_s": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "compressed_npz_dir": str(out_dir.resolve()),
        "official_npy_dir": str(npy_dir.resolve()) if npy_dir is not None else None,
        "official_occfm_layout": (
            "pred_<index>.npy / gt_<index>.npy, each [6,200,200,16]"
        ),
        "future_gt_used_for_prediction": False,
        "prediction_contract": (
            "frozen Clean-E14 + Strong/KTA + predicted SE2 + hard-A1"
        ),
        "manifest": manifest,
    }
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "manifest"}, indent=2))
    print(f"saved {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
