#!/usr/bin/env python3
"""One bounded paired experiment: end-to-end V18 continuation vs interaction."""
from __future__ import annotations
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from dataclasses import asdict
import json
import subprocess
import time
import numpy as np
import torch

from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.v21_source_induction import stable_json_fingerprint
from real_motion.v18_source_interaction import PROTOCOL, ARMS, InteractionConfig, causal_forward
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import load_manifest, align_records, delta
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256, finite_json, write_json, atomic_checkpoint
from tools.real_motion.v18_xy_trajectory_common import FrozenXYV18
from tools.real_motion.train_p0_f9_v18_xy_trajectory import DEV64_FP
from tools.real_motion.v18_source_interaction_common import (
    select_population, window_batches, make_batch, original_objective, build_pair, predict, evaluate_models, validate_records,
)


def cpu_weights(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def checkpoint_payload(identity, model, *, role, update, screen_pass=False, optimizer=None):
    payload = {**identity, "checkpoint_role": role, "selected_update": update,
               "screen_pass": screen_pass, "model_config": asdict(model.v17_config),
               "state_dict": cpu_weights(model)}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    return payload


def select_checkpoint(previous, report, model, update):
    """TRAIN-scene held-out REAL compositor mIoU, not dev or surrogate ADE.

    Update-0 is the explicit no-change control. No retry or threshold sweep.
    """
    score = report["metrics"]["mIoU"]
    if score is None or not np.isfinite(score):
        raise RuntimeError("nonfinite calibration mIoU")
    if score > previous["score"] + 1e-9:
        return {"score": score, "update": update, "report": report, "state_dict": cpu_weights(model)}
    return previous


def train_epoch(models, optimizers, records, config, device, *, seed, epoch, source_budget,
                window_budget, updates, progress=None):
    batches = list(window_batches(records, seed=seed, epoch=epoch,
                                  source_budget=source_budget, window_budget=window_budget))
    totals = {arm: {} for arm in ARMS}
    successful = 0
    for step, ids in enumerate(batches, 1):
        start = time.perf_counter()
        batch = make_batch([records[i] for i in ids], device, config)
        n = len(batch["features"])
        supervised = bool(batch["supervised_source"].bool().any())
        if not supervised:
            if progress:
                progress({"event": "empty_supervision_batch", "epoch": epoch, "windows": len(ids), "sources": n})
            continue
        for arm in ARMS:
            model, optimizer = models[arm], optimizers[arm]
            model.train(); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = causal_forward(model, batch)
            # All loss geometry/labels FP32 in BOTH arms, BF16 encoder on CUDA.
            loss, parts = original_objective(outputs, batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite paired loss arm={arm} epoch={epoch} step={step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            row = {"event": "train", "arm": arm, "epoch": epoch, "step": step, "steps": len(batches),
                   "update": updates+successful+1, "windows": len(ids), "sources": n,
                   "loss": float(loss.detach()), "grad_norm": float(norm), **parts}
            for key in ("objective_loss", "translation_smooth_l1", "existence_bce", "yaw_periodic_loss", "se2_shape_loss"):
                totals[arm][key] = totals[arm].get(key, 0.) + row[key]
            if progress:
                progress(row)
            if step == 1 or step % 32 == 0 or step == len(batches):
                print(f"arm={arm} epoch={epoch} step={step}/{len(batches)} update={row['update']} "
                      f"loss={row['loss']:.6f} xy={parts['translation_smooth_l1']:.6f} "
                      f"yaw={parts['yaw_periodic_loss']:.6f} windows={len(ids)} sources={n} "
                      f"pair_seconds={time.perf_counter()-start:.3f}", flush=True)
        successful += 1
    if not successful:
        raise RuntimeError("epoch had no supervised updates")
    return updates+successful, {arm: {key: value/successful for key, value in row.items()} for arm, row in totals.items()}


def summary_text(summary):
    lines = ["===== PAIRED END-TO-END V18 / SOURCE INTERACTION =====",
             f"protocol: {PROTOCOL}", f"mode: {summary['mode']}",
             f"train_windows: {summary['train_windows']} / {summary['full_train_windows']}",
             f"epochs_per_arm: {summary['epochs_per_arm']}", f"updates_per_arm: {summary['updates_per_arm']}",
             f"TRAIN_calibration_windows: {summary['calibration_windows']}", f"dev_windows: {summary['dev_windows']}",
             "selection: TRAIN-scene-disjoint true-renderer mIoU; dev never selects checkpoint",
             f"V18 baseline: {summary['dev']['baseline']}"]
    for arm in ARMS:
        info = summary["arms"][arm]
        lines += [f"\n===== {arm} =====", f"selected_update: {info['selected_update']}"]
        for role in ("selected", "last"):
            r = summary["dev"]["variants"][arm+"_"+role]
            lines += [f"{role}_delta_vs_V18_pp: {r['delta_vs_v18_pp']}",
                      f"{role}_motion_errors_all: {r['motion_errors'].get('all', {})}",
                      f"{role}_scene_delta: {r['scene_delta']}", f"{role}_gate: {r['gate']}"]
        lines += [f"screen_pass: {info['screen_pass']}", f"selected_checkpoint: {info['selected_checkpoint']}"]
    lines += [f"\ninteraction_minus_continuation_pp: {summary['interaction_vs_control_pp']}",
              f"route: {summary['route']}", f"seconds_by_stage: {summary['seconds_by_stage']}",
              f"elapsed_seconds: {summary['elapsed_seconds']:.2f}",
              "Update=0 / smoke / last diagnostics are NOT evidence of a learned improvement.",
              "This is ONE fixed experiment. No automatic extra oracles/retry/expansion/generation stage."]
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__); add_config_args(parser)
    for name in ("train-cache", "dev-cache", "population-manifest", "base-checkpoint", "dataroot", "train-info", "dev-info", "out-dir"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--expected-base-sha256", default=CLEAN_SHA256)
    parser.add_argument("--mode", choices=("smoke", "screen"), default="screen")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--source-budget", type=int, default=256)
    parser.add_argument("--window-budget", type=int, default=8)
    parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); start = time.perf_counter()
    if min(args.epochs, args.source_budget, args.window_budget, args.cpu_workers) < 1:
        parser.error("invalid resource/training budget")
    for name in ("config", "train_cache", "dev_cache", "population_manifest", "base_checkpoint", "train_info", "dev_info"):
        if not str(getattr(args, name) or "").strip() or not Path(getattr(args, name)).is_file():
            parser.error(f"{name} must be an existing file")
    if not Path(args.dataroot).is_dir():
        parser.error("dataroot must be an existing directory")
    out = Path(args.out_dir)
    if out.exists():
        parser.error("out-dir exists; choose a NEW directory (no checkpoint overwrite)")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    torch.set_num_threads(1); torch.manual_seed(args.seed); np.random.seed(args.seed)
    cfg = load_runtime_config(args.config, args.override); pcfg = make_prepare_config(cfg)
    manifest, dev_keys, _ = load_manifest(args.population_manifest)
    if len(dev_keys) != 64 or len(manifest["parent_keys"]) != 512 or manifest["selected_key_fingerprint"] != DEV64_FP:
        raise RuntimeError("requires frozen Stage-1 dev512-derived dev64 identity/order")
    train_meta, all_train = load_cache(args.train_cache)
    if args.mode == "screen" and len(all_train) != 20430:
        raise RuntimeError("formal 20% experiment requires confirmed full20430 TRAIN cache")
    train_keys, cal_keys = select_population([(r["scene_name"], r["t0_token"]) for r in all_train],
           {k[0] for k in manifest["parent_keys"]}, calibration_scenes=32 if args.mode == "screen" else 2, seed=args.seed)
    if args.mode == "smoke":
        train_keys, cal_keys = train_keys[:8], cal_keys[:2]
    train_records, cal_records = align_records(all_train, train_keys), align_records(all_train, cal_keys)
    full_train_count = len(all_train); del all_train
    dev_meta, all_dev = load_cache(args.dev_cache)
    use_dev_keys = dev_keys if args.mode == "screen" else dev_keys[:2]
    dev_records = align_records(all_dev, use_dev_keys); del all_dev
    validate_records(train_records+cal_records+dev_records)
    for meta in (train_meta, dev_meta):
        if float(meta.get("patch_resolution_m", .8)) != .8:
            raise RuntimeError("frozen V18 patch resolution mismatch")
    provider = FrozenXYV18(args.base_checkpoint, args.expected_base_sha256, pcfg, device, args.cpu_workers)
    checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    models, optimizers = build_pair(checkpoint, device); del checkpoint
    config = InteractionConfig()
    # BF16 deployed forward must also agree, not only FP32 synthetic unit tests.
    sample = next((r for r in train_records if len(r["features"])), None)
    if sample is None:
        raise RuntimeError("selected TRAIN windows contain no sources")
    base = provider.encode_record(sample)
    for model in models.values():
        initial = predict(model, sample, device, config)
        if any(not torch.equal(initial[key], base[key]) for key in initial):
            raise RuntimeError("paired initialization is not exact frozen V18")
    out.mkdir(parents=True)
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unavailable"
    epochs = 1 if args.mode == "smoke" else args.epochs
    identity = {"protocol": PROTOCOL, "base_checkpoint_sha256": provider.sha,
        "runtime_config_fingerprint": stable_json_fingerprint(cfg), "git_commit": git_commit, "mode": args.mode,
        "train_keys": train_keys, "calibration_keys": cal_keys, "dev_keys": use_dev_keys,
        "population_fingerprint": stable_json_fingerprint({"train": train_keys, "calibration": cal_keys}),
        "dev_manifest_fingerprint": manifest["manifest_fingerprint"], "scene_overlap": 0,
        "full_train_windows": full_train_count, "train_fraction": len(train_keys)/full_train_count,
        "epochs_per_arm": epochs, "seed": args.seed, "interaction_config": asdict(config),
        "packing": {"source_budget": args.source_budget, "window_budget": args.window_budget,
                    "whole_windows_no_source_truncation": True},
        "optimizer": {"name": "AdamW", "common_moments_restored": True, "fixed_lr": optimizers[ARMS[0]].param_groups[0]["lr"],
                      "new_module_same_lr": True, "clip_norm": 5.},
        "loss": "SmoothL1_beta1 + existence_BCE + 19*periodic_yaw + 0.25*soft_SE2",
        "checkpoint_selection": "TRAIN_scene_disjoint_real_renderer_mIoU_update0_control",
        "calibration_note": "continuation excludes these TRAIN scenes; original pretrained V18 has seen TRAIN, not a new unseen test"}
    write_json(out/"execution_contract.json", {**identity, "arguments": vars(args)})
    best = {arm: {"score": None, "update": 0, "state_dict": cpu_weights(m)} for arm, m in models.items()}
    timings = {"load_validate_inputs": time.perf_counter()-start}
    with (out/"progress.jsonl").open("x", encoding="utf-8") as handle:
        def progress(row):
            handle.write(json.dumps(finite_json(row), ensure_ascii=False, allow_nan=False)+"\n"); handle.flush()
        train_source = NuScenesWindowSource(args.dataroot, info_pkl=args.train_info, verbose=False)
        print(f"Train={len(train_records)}/{full_train_count} windows; TRAINcal={len(cal_records)}; "
              f"dev={len(dev_records)}; epochs={epochs} EACH; full shared encoder/decoder trainable.", flush=True)
        tick = time.perf_counter()
        initial_cal = evaluate_models(provider, train_source, cal_records, {}, config, progress=progress)
        for arm in ARMS:
            best[arm]["score"] = initial_cal["baseline"]["mIoU"]
            if best[arm]["score"] is None:
                raise RuntimeError("TRAIN calibration has no defined mIoU")
            atomic_checkpoint(out/(arm+"_best.pt"), checkpoint_payload({**identity, "arm": arm}, models[arm],
                role="selected_full_v18_candidate", update=0))
        timings["initial_train_calibration"] = time.perf_counter()-tick
        updates, history = 0, []
        for epoch in range(1, epochs+1):
            tick = time.perf_counter()
            updates, train_stats = train_epoch(models, optimizers, train_records, config, device,
                seed=args.seed, epoch=epoch, source_budget=args.source_budget, window_budget=args.window_budget,
                updates=updates, progress=progress)
            timings[f"epoch_{epoch}_paired_train"] = time.perf_counter()-tick
            for arm in ARMS:
                atomic_checkpoint(out/(arm+"_last.pt"), checkpoint_payload({**identity, "arm": arm, "epoch": epoch},
                    models[arm], role="last_full_v18_diagnostic", update=updates, optimizer=optimizers[arm]))
            tick = time.perf_counter()
            calibration = evaluate_models(provider, train_source, cal_records, models, config, progress=progress)
            timings[f"epoch_{epoch}_train_calibration"] = time.perf_counter()-tick
            for arm in ARMS:
                previous = best[arm]
                best[arm] = select_checkpoint(previous, calibration["variants"][arm], models[arm], updates)
                if best[arm] is not previous:
                    atomic_checkpoint(out/(arm+"_best.pt"), checkpoint_payload({**identity, "arm": arm}, models[arm],
                        role="selected_full_v18_candidate", update=updates))
                print(f"calibration epoch={epoch} arm={arm} dMiOU="
                      f"{calibration['variants'][arm]['delta_vs_v18_pp']['mIoU']:+.6f} selected_update={best[arm]['update']}", flush=True)
            row = {"event": "epoch_complete", "epoch": epoch, "updates": updates, "train": train_stats,
                   "calibration": calibration, "selected_updates": {arm: best[arm]["update"] for arm in ARMS}}
            history.append(row); progress(row); write_json(out/"train_history.json", history)
        del train_source, train_records, cal_records
        # Selected and last candidates are evaluated TOGETHER, with one shared
        # raw/Strong/base/support pass. Never choose checkpoints from dev.
        tick = time.perf_counter()
        dev_models, arms = {}, {}
        for arm in ARMS:
            selected = type(models[arm])(models[arm].v17_config).to(device)
            selected.load_state_dict(best[arm]["state_dict"], strict=True)
            dev_models[arm+"_selected"], dev_models[arm+"_last"] = selected, models[arm]
            arms[arm] = {"selected_update": best[arm]["update"], "calibration_mIoU": best[arm]["score"],
                         "parameters": sum(p.numel() for p in selected.parameters()),
                         "selected_checkpoint": str((out/(arm+"_best.pt")).resolve())}
        del optimizers
        dev_source = NuScenesWindowSource(args.dataroot, info_pkl=args.dev_info, verbose=False)
        dev = evaluate_models(provider, dev_source, dev_records, dev_models, config, progress=progress)
        timings["shared_final_dev"] = time.perf_counter()-tick
        for arm in ARMS:
            arms[arm]["screen_pass"] = (args.mode == "screen" and best[arm]["update"] > 0
                                         and dev["variants"][arm+"_selected"]["gate"]["pass"])
            atomic_checkpoint(out/(arm+"_best.pt"), checkpoint_payload({**identity, "arm": arm}, dev_models[arm+"_selected"],
                role="selected_full_v18_candidate", update=best[arm]["update"], screen_pass=arms[arm]["screen_pass"]))
        interaction_vs_control = delta(dev["variants"][ARMS[1]+"_selected"]["metrics"],
                                       dev["variants"][ARMS[0]+"_selected"]["metrics"])
        route = "stop_this_fixed_experiment_no_automatic_retry"
        if args.mode == "smoke":
            route = "smoke_only_not_effectiveness_evidence"
        elif arms[ARMS[1]]["screen_pass"] and interaction_vs_control["mIoU"] > 0:
            route = "interaction_candidate_passed_dev64_requires_larger_frozen_validation"
        elif arms[ARMS[0]]["screen_pass"]:
            route = "continuation_passed_interaction_not_justified"
        summary = {"protocol": PROTOCOL, "mode": args.mode, "git_commit": git_commit,
                   "train_windows": len(train_keys), "full_train_windows": full_train_count,
                   "calibration_windows": len(cal_keys), "dev_windows": len(dev_records), "epochs_per_arm": epochs,
                   "updates_per_arm": updates, "arms": arms, "dev": dev, "interaction_vs_control_pp": interaction_vs_control,
                   "route": route, "seconds_by_stage": timings, "elapsed_seconds": time.perf_counter()-start}
        write_json(out/"summary.json", summary)
        (out/"summary.txt").write_text(summary_text(summary), encoding="utf-8")
        print(summary_text(summary), flush=True)


if __name__ == "__main__":
    main()
