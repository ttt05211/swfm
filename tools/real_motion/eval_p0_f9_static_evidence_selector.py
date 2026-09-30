#!/usr/bin/env python3
"""Evaluate a frozen selector, or export six-horizon predictions WITHOUT GT."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.v21_source_induction import stable_json_fingerprint
from real_motion.static_evidence_selector import PROTOCOL
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import load_manifest, align_records, sha256
from tools.real_motion.static_evidence_selector_common import (CLEAN_SHA256, FrozenV18, prepare_dev,
    load_selector, evaluate_selector, forecast_with_selector, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    for name in ("val-cache", "population-manifest", "base-checkpoint", "selector-checkpoint", "dataroot", "info-pkl", "out-dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--expected-base-sha256", default=CLEAN_SHA256)
    parser.add_argument("--predict-only", action="store_true", help="no future GT or metrics; writes compressed six-frame predictions")
    parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.cpu_workers < 1 or args.batch_size < 1: parser.error("invalid resource budget")
    for name in ("config", "val_cache", "population_manifest", "base_checkpoint", "selector_checkpoint", "info_pkl"):
        if not str(getattr(args, name) or "").strip() or not Path(getattr(args, name)).is_file():
            parser.error(f"{name} must name an existing file")
    if not Path(args.dataroot).is_dir(): parser.error("dataroot must name an existing directory")
    out = Path(args.out_dir)
    if out.exists(): parser.error("out-dir exists; choose a new path")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(1)
    cfg = load_runtime_config(args.config, args.override)
    pcfg = make_prepare_config(cfg)
    if pcfg.free_label != 17: raise RuntimeError("frozen free label must be 17")
    config_fp = stable_json_fingerprint(cfg)
    manifest, keys, _ = load_manifest(args.population_manifest)
    _, all_records = load_cache(args.val_cache)
    records = align_records(all_records, keys); del all_records
    provider = FrozenV18(args.base_checkpoint, args.expected_base_sha256, pcfg, device, args.cpu_workers)
    ck, selector = load_selector(args.selector_checkpoint, device, base_sha=provider.sha, config_sha=config_fp)
    source = NuScenesWindowSource(args.dataroot, info_pkl=args.info_pkl, verbose=False)
    out.mkdir(parents=True)
    started = time.perf_counter()
    result = {"protocol": PROTOCOL, "checkpoint_sha256": sha256(args.selector_checkpoint),
              "base_checkpoint_sha256": provider.sha, "windows": len(keys),
              "selected_key_fingerprint": manifest["selected_key_fingerprint"], "predict_only": args.predict_only,
              "selector_update": ck["successful_updates"], "runtime_config_fingerprint": config_fp}
    if args.predict_only:
        for i, record in enumerate(records, 1):
            window, predictions = forecast_with_selector(provider, source, record, selector, args.batch_size)
            path = out / f"{i:04d}_{window.t0_token}.npz"
            # This opt-in export stores predictions, not a new training cache.
            np.savez_compressed(path, predictions=np.stack(predictions).astype(np.uint8),
                                scene_name=window.scene_name, t0_token=window.t0_token,
                                future_tokens=np.asarray(window.future_tokens),
                                horizon_seconds=np.arange(1, 7, dtype=np.float32) * .5)
            print(f"prediction={i}/{len(records)} path={path} future_gt_read=False", flush=True)
        result["future_gt_read"] = False
    else:
        rows = prepare_dev(provider, source, records)
        result["report"] = evaluate_selector(selector, rows, device, args.batch_size)
        result["gate"] = result["report"]["gate"]
    result["elapsed_seconds"] = time.perf_counter() - started
    write_json(out / "summary.json", result)
    print(f"completed: {out / 'summary.json'}", flush=True)


if __name__ == "__main__": main()
