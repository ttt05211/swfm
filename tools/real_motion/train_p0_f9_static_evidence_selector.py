#!/usr/bin/env python3
"""One bounded selector run: prepare once, train, monitor, select and report.

V18 is frozen. No oracle sweep, dense disk cache or automatic retry/expansion.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.static_evidence_selector import (PROTOCOL, FEATURE_PROTOCOL, THRESHOLD,
    supervision_counts, sample_training_patches)
from real_motion.static_evidence_selector_model import SelectorConfig, StaticEvidenceSelector, utility_loss
from real_motion.v21_source_induction import select_scene_balanced_round_robin, stable_json_fingerprint
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import load_manifest, align_records, sha256
from tools.real_motion.static_evidence_selector_common import (CLEAN_SHA256, FrozenV18, prepare_dev,
    evaluate_selector, bank_fingerprint, atomic_checkpoint, write_json, finite_json, load_selector)

TRAINING_CONTRACT = {"objective": "importance_corrected_patch_semantic_utility_bce_v1",
    "threshold": THRESHOLD, "samples_per_window_horizon": 16, "learning_rate": 3e-4,
    "weight_decay": .01, "gradient_clip": 1., "patch_cells": 4,
    "candidate_classes": "all_static_classes", "v18_frozen": True,
    "no_gt_features": True, "no_dense_cache": True}


def record_keys(records):
    keys = tuple((str(r["scene_name"]), str(r["t0_token"])) for r in records)
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate V18 identities")
    return keys


def prepare_bank(provider, source, records, seed, samples_per_horizon=16, max_mib=4096, progress=None):
    rng = np.random.default_rng(seed)
    parts = {k: [] for k in ("history", "context", "correct", "wrong", "weight")}
    size = 0
    for wi, record in enumerate(records, 1):
        tick = time.perf_counter()
        print(f"prepare_train={wi}/{len(records)} stage=load_forecast_features", flush=True)
        _, raw, _, inputs, _ = provider.prepare(source, record, include_gt=True)
        for h, x in enumerate(inputs):
            if not len(x.patch_ids): continue
            good, bad = supervision_counts(x, raw["future_gt_occ"][h])
            ids, weight = sample_training_patches(good, bad, samples_per_horizon, rng)
            values = {"history": x.history[ids], "context": x.context[ids],
                      "correct": good[ids], "wrong": bad[ids], "weight": weight}
            for key, value in values.items():
                parts[key].append(value); size += value.nbytes
        # Concatenation temporarily doubles storage; do not exhaust RAM silently.
        if size * 2 > max_mib * 1024 ** 2:
            raise RuntimeError(f"compact feature bank exceeds RAM budget ({size / 1024**2:.1f} MiB; concatenation needs ~2x)")
        row = {"event": "prepare_train", "window": wi, "windows": len(records),
               "bank_mib": size / 1024 ** 2, "seconds": time.perf_counter() - tick}
        if progress: progress(row)
        if wi % 8 == 0 or wi == len(records): print(json.dumps(row), flush=True)
    if not parts["history"]:
        raise RuntimeError("training population has no addable static evidence")
    return {key: np.concatenate(value) for key, value in parts.items()}


def training_step(model, optimizer, bank, ids, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    batch = {k: torch.from_numpy(v[ids]).to(device) for k, v in bank.items()}
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(batch["history"], batch["context"])
    loss = utility_loss(logits, batch["correct"], batch["wrong"], batch["weight"])
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    with torch.no_grad():
        p = torch.sigmoid(logits.float())
    return {"loss": float(loss.detach()), "grad_norm": float(grad), "keep_fraction": float((p >= THRESHOLD).float().mean()),
            "mean_probability": float(p.mean())}


def restore_training_checkpoint(path, device, identity, mode):
    ck, model = load_selector(path, device, base_sha=identity["base_checkpoint_sha256"],
                              config_sha=identity["runtime_config_fingerprint"])
    if ck.get("checkpoint_role") != "resume_last":
        raise RuntimeError("resume requires last.pt, not an inference best.pt")
    if any(ck.get(k) != v for k, v in identity.items()):
        raise RuntimeError("resume data/feature/population/budget contract mismatch")
    if ck.get("mode") != mode:
        raise RuntimeError("resume mode mismatch")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    optimizer.load_state_dict(ck["optimizer"])
    rng = np.random.default_rng()
    rng.bit_generator.state = ck["sampling_rng_state"]
    torch.set_rng_state(ck["torch_rng_state"])
    if device.type == "cuda": torch.cuda.set_rng_state_all(ck["cuda_rng_states"])
    return ck, model, optimizer, rng


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    for name in ("train-cache", "dev-cache", "population-manifest", "base-checkpoint", "dataroot",
                 "train-info", "dev-info", "out-dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--expected-base-sha256", default=CLEAN_SHA256)
    parser.add_argument("--mode", choices=("smoke", "screen", "main"), default="screen")
    parser.add_argument("--screen-summary", help="main requires an accepted screen result; no automatic expansion")
    parser.add_argument("--resume", help="resume one compatible last.pt into a NEW output directory")
    parser.add_argument("--train-windows", type=int)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--batch-size", type=int, default=512, help="PATCHES per optimizer update, not windows")
    parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument("--eval-every", type=int, help="default: screen 128, main 1024")
    parser.add_argument("--bank-max-mib", type=int, help="peak compact-bank RAM budget: screen 4096, main 8192 MiB")
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.perf_counter()
    if args.train_windows is None: args.train_windows = {"smoke": 4, "screen": 1024, "main": 0}[args.mode]
    if args.updates is None: args.updates = {"smoke": 2, "screen": 1024, "main": 8192}[args.mode]
    if args.eval_every is None: args.eval_every = 1024 if args.mode == "main" else 128
    if args.bank_max_mib is None: args.bank_max_mib = 8192 if args.mode == "main" else 4096
    if args.mode != "smoke" and (args.train_windows, args.updates) != ((1024, 1024) if args.mode == "screen" else (0, 8192)):
        parser.error("screen/main population and update budgets are frozen; custom budgets are smoke-only")
    if min(args.updates, args.batch_size, args.cpu_workers, args.eval_every, args.bank_max_mib) < 1 or args.train_windows < 0:
        parser.error("invalid resource budget")
    for name in ("config", "train_cache", "dev_cache", "population_manifest", "base_checkpoint", "train_info", "dev_info"):
        if not str(getattr(args, name) or "").strip() or not Path(getattr(args, name)).is_file():
            parser.error(f"{name} must name an existing file")
    if not Path(args.dataroot).is_dir(): parser.error("dataroot must name an existing directory")
    out = Path(args.out_dir)
    if out.exists(): parser.error("out-dir already exists; choose a NEW directory, even when resuming")
    if args.resume:
        if not Path(args.resume).is_file(): parser.error("resume must name an existing last.pt")
        if torch.load(args.resume, map_location="cpu", weights_only=False).get("checkpoint_role") != "resume_last":
            parser.error("resume requires last.pt; best.pt is deployment-only")
    torch.set_num_threads(1)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    cfg = load_runtime_config(args.config, args.override)
    pcfg = make_prepare_config(cfg)
    if pcfg.free_label != 17: raise RuntimeError("frozen free label must be 17")
    config_fp = stable_json_fingerprint(cfg)
    manifest, dev_keys, _ = load_manifest(args.population_manifest)
    if len(manifest["parent_keys"]) != 512:
        raise RuntimeError("selection manifest must descend from the frozen Stage-1 dev512 population")
    expected_dev = {"smoke": 64, "screen": 64, "main": 512}[args.mode]
    if len(dev_keys) != expected_dev: raise RuntimeError(f"{args.mode} requires the frozen dev{expected_dev} manifest")
    if args.mode == "main":
        if not args.screen_summary: parser.error("main requires --screen-summary with a passing gate")
        screen = json.loads(Path(args.screen_summary).read_text(encoding="utf-8"))
        if (screen.get("protocol") != PROTOCOL or screen.get("mode") != "screen" or not screen.get("gate", {}).get("pass")
                or screen.get("runtime_config_fingerprint") != config_fp or screen.get("training_contract") != TRAINING_CONTRACT
                or screen.get("base_checkpoint_sha256") != args.expected_base_sha256
                or screen.get("model_config") != asdict(SelectorConfig())
                or (screen.get("successful_updates"), screen.get("train_windows"), screen.get("dev_windows")) != (1024, 1024, 64)
                or screen.get("execution_budget", {}).get("batch_size_patches") != args.batch_size
                or screen.get("execution_budget", {}).get("seed") != args.seed
                or screen.get("dev_parent_key_fingerprint") != manifest["parent_key_fingerprint"]):
            raise RuntimeError("screen gate or frozen execution contract mismatch")
    print("loading frozen V18 caches and selecting scene-balanced train population", flush=True)
    _, all_records = load_cache(args.train_cache)
    all_keys = record_keys(all_records)
    if args.train_windows > len(all_keys): raise RuntimeError("train-windows exceeds source population")
    train_keys = all_keys if args.train_windows == 0 else select_scene_balanced_round_robin(all_keys, args.train_windows)
    records = align_records(all_records, train_keys); del all_records
    _, all_dev = load_cache(args.dev_cache)
    dev_records = align_records(all_dev, dev_keys); del all_dev
    if args.mode == "smoke": dev_records = dev_records[:2]
    train_scenes = {k[0] for k in train_keys}
    dev_scenes = {str(k[0]) for k in manifest["parent_keys"]}
    if train_scenes & dev_scenes: raise RuntimeError("train/dev scenes overlap")
    provider = FrozenV18(args.base_checkpoint, args.expected_base_sha256, pcfg, device, args.cpu_workers)
    train_source = NuScenesWindowSource(args.dataroot, info_pkl=args.train_info, verbose=False)
    out.mkdir(parents=True)
    with (out / "progress.jsonl").open("x", encoding="utf-8") as handle:
        def progress(row):
            handle.write(json.dumps(finite_json(row), ensure_ascii=False, allow_nan=False) + "\n"); handle.flush()
        write_json(out / "execution_contract.json", {"protocol": PROTOCOL, "feature_protocol": FEATURE_PROTOCOL,
            "arguments": vars(args), "training_contract": TRAINING_CONTRACT, "model_config": asdict(SelectorConfig()),
            "base_checkpoint_sha256": provider.sha, "runtime_config_fingerprint": config_fp,
            "dev_manifest_fingerprint": manifest["manifest_fingerprint"], "train_keys": train_keys,
            "training_key_fingerprint": stable_json_fingerprint(train_keys), "scene_overlap": 0})
        bank = prepare_bank(provider, train_source, records, args.seed, max_mib=args.bank_max_mib, progress=progress)
        del records, train_source
        bank_sha = bank_fingerprint(bank)
        print(f"feature_bank complete patches={len(bank['history'])} mib={sum(v.nbytes for v in bank.values())/1024**2:.1f}; no dense disk cache", flush=True)
        dev_source = NuScenesWindowSource(args.dataroot, info_pkl=args.dev_info, verbose=False)
        dev = prepare_dev(provider, dev_source, dev_records, progress); del dev_records, dev_source
        model = StaticEvidenceSelector().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
        rng = np.random.default_rng(args.seed + 1)
        identity = {"base_checkpoint_sha256": provider.sha, "runtime_config_fingerprint": config_fp,
            "training_key_fingerprint": stable_json_fingerprint(train_keys), "dev_manifest_fingerprint": manifest["manifest_fingerprint"],
            "dev_parent_key_fingerprint": manifest["parent_key_fingerprint"],
            "bank_fingerprint": bank_sha, "training_contract": TRAINING_CONTRACT,
            "execution_budget": {"batch_size_patches": args.batch_size, "eval_every": args.eval_every,
                                  "seed": args.seed, "target_updates": args.updates}}
        report = evaluate_selector(model, dev, device, args.batch_size)
        if report["quality"].get("added", 0) or abs(report["delta_vs_v18_pp"]["mIoU"]) > 1e-10:
            raise RuntimeError("conservative initialization is not V18 identity")
        best_report, best_update, best_score = report, 0, (False, 0.)
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        start_update = 0
        if args.resume:
            ck, model, optimizer, rng = restore_training_checkpoint(args.resume, device, identity, args.mode)
            start_update = int(ck["successful_updates"])
            if start_update > args.updates: raise RuntimeError("resume step exceeds requested budget")
            best_state, best_report, best_update = ck["best_state_dict"], ck["best_report"], int(ck["best_update"])
            best_score = (bool(best_report["gate"]["pass"]), float(best_report["delta_vs_v18_pp"]["mIoU"]))
        def checkpoint(update):
            return {**model.contract(), **identity, "threshold": THRESHOLD, "mode": args.mode,
                "checkpoint_role": "resume_last",
                "successful_updates": update, "state_dict": {k: v.detach().cpu() for k,v in model.state_dict().items()},
                "optimizer": optimizer.state_dict(), "sampling_rng_state": rng.bit_generator.state,
                "torch_rng_state": torch.get_rng_state(), "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "best_state_dict": best_state, "best_report": best_report, "best_update": best_update}
        def save_best():
            atomic_checkpoint(out / "best.pt", {**model.contract(), **identity,
                "threshold": THRESHOLD, "mode": args.mode, "checkpoint_role": "inference_best",
                "successful_updates": best_update, "state_dict": best_state, "report": best_report})
        save_best()
        atomic_checkpoint(out / "last.pt", checkpoint(start_update))
        progress({"event": "initial_or_resumed_best", "update": best_update, "report": best_report})
        last_report = evaluate_selector(model, dev, device, args.batch_size) if args.resume else report
        for update in range(start_update + 1, args.updates + 1):
            tick = time.perf_counter()
            ids = rng.integers(0, len(bank["history"]), size=args.batch_size)
            stats = training_step(model, optimizer, bank, ids, device)
            progress({"event": "train", "update": update, **stats, "seconds": time.perf_counter() - tick})
            if update == 1 or update % 16 == 0 or update == args.updates:
                print(f"update={update}/{args.updates} loss={stats['loss']:.6f} keep={stats['keep_fraction']:.3%} grad_norm={stats['grad_norm']:.4f} seconds={time.perf_counter()-tick:.3f}", flush=True)
            if update % args.eval_every == 0 or update == args.updates:
                last_report = evaluate_selector(model, dev, device, args.batch_size)
                gain = float(last_report["delta_vs_v18_pp"]["mIoU"])
                score = (bool(last_report["gate"]["pass"]), gain)
                if np.isfinite(gain) and score > best_score:
                    best_report, best_update, best_score = last_report, update, score
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    save_best()
                atomic_checkpoint(out / "last.pt", checkpoint(update))
                progress({"event": "monitor", "update": update, "report": last_report})
                print(f"monitor update={update} dMiOU={gain:+.6f} add={last_report['quality'].get('added',0)} gate={last_report['gate']['pass']} best_update={best_update}", flush=True)
        atomic_checkpoint(out / "last.pt", checkpoint(args.updates))
        # Read the persisted best through the actual deployment loader.
        _, best_model = load_selector(out / "best.pt", device, base_sha=provider.sha, config_sha=config_fp)
        final = evaluate_selector(best_model, dev, device, args.batch_size)
        if final["delta_vs_v18_pp"] != best_report["delta_vs_v18_pp"]:
            # NaNs for absent classes are not comparable; raw top-level metrics
            # are the required serialization equivalence here.
            for k in ("IoU", "mIoU", "MovingMacro", "MovingMicro"):
                if not np.isclose(final["selected"][k], best_report["selected"][k], rtol=0, atol=1e-10, equal_nan=True):
                    raise RuntimeError("persisted selector changed evaluation")
        try: commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        except (OSError, subprocess.CalledProcessError): commit = None
        summary = {"protocol": PROTOCOL, "feature_protocol": FEATURE_PROTOCOL, "mode": args.mode,
            "real_data_run": True, "git_commit": commit, **identity, "model_config": asdict(model.config),
            "successful_updates": args.updates, "attempted_updates": args.updates, "resumed_from": args.resume,
            "train_windows": len(train_keys), "dev_windows": len(dev) // 3, "train_scenes": len(train_scenes), "scene_overlap": 0,
            "batch_size_patches": args.batch_size, "bank_patches": len(bank["history"]), "bank_mib": sum(v.nbytes for v in bank.values())/1024**2,
            "best_update": best_update, "best_checkpoint": str((out / "best.pt").resolve()),
            "last_checkpoint": str((out / "last.pt").resolve()), "best_checkpoint_sha256": sha256(out / "best.pt"),
            "report": final, "last_candidate_report": last_report, "gate": final["gate"],
            "route": "smoke_complete_not_a_scientific_gate" if args.mode == "smoke" else
                ("freeze_and_expand" if final["gate"]["pass"] else "stop_this_version_no_automatic_retry"),
            "elapsed_seconds": time.perf_counter() - started, "automatic_expansion": False}
        write_json(out / "summary.json", summary)
        print(json.dumps(finite_json({k: summary[k] for k in ("route", "best_update", "gate", "elapsed_seconds", "best_checkpoint")}), ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
