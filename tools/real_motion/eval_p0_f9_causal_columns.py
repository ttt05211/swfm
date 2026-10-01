#!/usr/bin/env python3
"""Evaluate a frozen candidate without TRAIN recalibration or checkpoint writes."""
from __future__ import annotations
import sys
from pathlib import Path
if __package__ in (None, ""): sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import time
import torch
from real_motion.column_runtime_pipeline import CachedColumnSource
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion.causal_column_common import FrozenColumns, evaluate_columns
from tools.real_motion.train_p0_f9_causal_columns import load_columns, record_keys
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256, write_json
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import load_manifest, align_records, sha256
from tools.real_motion.train_p0_f9_v18_xy_trajectory import DEV64_FP


def main():
    parser = argparse.ArgumentParser(description=__doc__); add_config_args(parser)
    for name in ("checkpoint", "dev-cache", "population-manifest", "base-checkpoint", "dataroot", "dev-info", "out-dir"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--population", choices=("dev64", "dev512", "full4369"), default="dev512")
    parser.add_argument("--allow-diagnostic", action="store_true", help="explicitly evaluate FAILED/SMOKE candidate; never promotes it")
    parser.add_argument("--device", default="cuda"); parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(); started = time.perf_counter()
    out = Path(args.out_dir)
    if out.exists(): parser.error("NEW output directory required")
    for name in ("config", "checkpoint", "dev_cache", "population_manifest", "base_checkpoint", "dev_info"):
        if not str(getattr(args, name) or "").strip() or not Path(getattr(args, name)).is_file(): parser.error(f"missing {name}")
    if not Path(args.dataroot).is_dir() or min(args.cpu_workers, args.batch_size) < 1: parser.error("invalid paths/budgets")
    device = torch.device(args.device)
    if device.type == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()): raise RuntimeError("CUDA/BF16 unavailable")
    torch.set_num_threads(1)
    cfg = load_runtime_config(args.config, args.override); pcfg = make_prepare_config(cfg)
    digest = sha256(args.checkpoint)
    ck, model = load_columns(args.checkpoint, device, base_sha=CLEAN_SHA256, config_sha=stable_json_fingerprint(cfg),
                             allow_diagnostic=args.allow_diagnostic)
    if ck["checkpoint_role"] != "calibrated_candidate":
        raise RuntimeError("evaluation requires candidate.pt with frozen TRAIN thresholds, not last.pt")
    if sha256(args.dev_info) != ck["info_fingerprints"]["dev"]:
        raise RuntimeError("dev info differs from original screen; no silent input protocol change")
    manifest, dev64, _ = load_manifest(args.population_manifest)
    if manifest["selected_key_fingerprint"] != DEV64_FP or ck["dev_manifest_fingerprint"] != manifest["manifest_fingerprint"]:
        raise RuntimeError("frozen population manifest mismatch")
    _, records = load_cache(args.dev_cache); keys = record_keys(records)
    chosen = dev64 if args.population == "dev64" else manifest["parent_keys"] if args.population == "dev512" else keys
    if args.population == "full4369" and (len(keys) != 4369 or len({s for s, _ in keys}) != 150):
        raise RuntimeError("full validation requires 4369 windows / 150 scenes")
    if {str(s) for s, _ in chosen} & {str(s) for s, _ in (*ck["train_keys"], *ck["calibration_keys"])}:
        raise RuntimeError("TRAIN/expanded-dev scene overlap")
    records = align_records(records, chosen)
    provider = FrozenColumns(args.base_checkpoint, CLEAN_SHA256, pcfg, device, args.cpu_workers)
    source = CachedColumnSource(NuScenesWindowSource(args.dataroot, info_pkl=args.dev_info, verbose=False))
    result = evaluate_columns(provider, source, records, model, tuple(ck["thresholds"]), batch_size=args.batch_size,
                              dev64_keys=dev64 if args.population != "dev64" else None)
    if sha256(args.checkpoint) != digest: raise RuntimeError("checkpoint changed during read-only evaluation")
    out.mkdir(parents=True)
    write_json(out/"evaluation.json", {"checkpoint": args.checkpoint, "sha256": digest, "population": args.population,
        "original_screen_pass_unchanged": ck["screen_pass"], "thresholds_from_original_TRAIN": ck["thresholds"],
        "seconds": time.perf_counter()-started, "reports": result})
    print(f"Evaluation only; no threshold/checkpoint selection. Report: {out/'evaluation.json'}", flush=True)


if __name__ == "__main__": main()
