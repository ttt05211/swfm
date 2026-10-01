#!/usr/bin/env python3
"""One paired 20% experiment: frozen yaw vs training-only GT-yaw curriculum."""
from __future__ import annotations
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import subprocess
import time
import numpy as np
import torch

from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.v21_source_induction import stable_json_fingerprint
from real_motion.local_st_world_model_v18_se2 import LocalSpatialTemporalWorldModelV18SE2
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import load_manifest, align_records, delta
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256, finite_json, write_json, atomic_checkpoint
from tools.real_motion.v18_xy_trajectory_common import FrozenXYV18
from tools.real_motion.train_p0_f9_v18_xy_trajectory import DEV64_FP
from tools.real_motion.train_p0_f9_v18_source_interaction import cpu_weights, checkpoint_payload, select_checkpoint
from tools.real_motion.v18_source_interaction_common import select_population, evaluate_models, validate_records
from tools.real_motion.v18_xy_specialist_common import (
    PROTOCOL, ARMS, build_pair, predict, cache_baseline_outputs, epoch_batches, train_epoch, assert_frozen_heads,
)


def summary_text(summary):
    metric_keys = ("IoU", "mIoU", "MovingMacro", "MovingMicro")
    def headline(metrics):
        return {k: metrics.get(k) for k in metric_keys}
    def trajectory(errors):
        row = errors.get("all", {})
        return {key: row.get(key) for key in ("source_center_error_m", "source_center_error/3.0s_m", "yaw_error_deg")}
    lines = ["===== V18 XY SPECIALIST / STRICT FROZEN YAW =====", f"protocol: {PROTOCOL}",
             f"mode: {summary['mode']}", f"train_windows: {summary['train_windows']} / {summary['full_train_windows']}",
             f"epochs_per_arm: {summary['epochs_per_arm']}", f"updates_per_arm: {summary['updates_per_arm']}",
             f"calibration_windows: {summary['calibration_windows']}", f"dev_windows: {summary['dev_windows']}",
             f"fixed_tail_lr: {summary['fixed_tail_lr']}", f"teacher_cache_mib: {summary['teacher_cache_mib']:.3f}",
             "Both deploy: learned XY + unchanged Clean-E14 yaw/existence; NO GT yaw.",
             "scheduled_gt_yaw_xy: shape-loss yaw sampled GT, probability .5 -> 0; final >=1/3 pure baseline yaw.",
             "No source-interaction module, new sources, existence BCE or yaw-head loss.",
             "Selection: held-out TRAIN-scene real-renderer mIoU, update0 control; never dev."]
    base_err = summary["dev"]["baseline_motion_errors"]
    lines += [f"V18_BASE: {headline(summary['dev']['baseline'])}", f"V18_BASE_TRAJECTORY: {trajectory(base_err)}"]
    for arm in ARMS:
        info = summary["arms"][arm]
        lines += [f"\n===== {arm} =====", f"selected_update: {info['selected_update']}"]
        for role in ("selected", "last"):
            r = summary["dev"]["variants"][arm+"_"+role]
            errors = r["motion_errors"]
            per_horizon = {h: headline(v) for h, v in r["delta_vs_v18_pp"].get("per_horizon", {}).items()}
            scenes = {k: r["scene_delta"].get(k) for k in ("scenes", "positive", "zero", "negative", "mean", "median")}
            lines += [f"{role}_delta_vs_V18_pp: {headline(r['delta_vs_v18_pp'])}",
                      f"{role}_per_horizon_delta_pp: {per_horizon}",
                      f"{role}_trajectory: {trajectory(errors)}",
                      f"{role}_scene_delta: {scenes}", f"{role}_gate: {r['gate']}"]
        lines += [f"screen_pass: {info['screen_pass']}", f"selected_checkpoint: {info['selected_checkpoint']}"]
    lines += [f"\ncurriculum_minus_frozen_yaw_selected_pp: {headline(summary['curriculum_vs_frozen_yaw_pp'])}",
              f"route: {summary['route']}", f"seconds_by_stage: {summary['seconds_by_stage']}",
              f"elapsed_seconds: {summary['elapsed_seconds']:.2f}",
              "Update0/smoke/last are NOT a learned improvement. No automatic retry or expansion.",
              "Deployment requires the original Clean-E14 plus specialist (two neural forwards)."]
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
    if device.type == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("CUDA/BF16 unavailable; no silent CPU/FP16 fallback")
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
    models, optimizers = build_pair(checkpoint, device)
    frozen_state = {k: v.clone() for k, v in checkpoint["state_dict"].items()
                    if k.startswith(("yaw_head.", "existence_head."))}
    del checkpoint
    sample = next((r for r in train_records if len(r["features"])), None)
    if sample is None:
        raise RuntimeError("selected TRAIN windows contain no sources")
    base = provider.encode_record(sample)
    for model in models.values():
        initial = predict(model, sample, device, base=base)
        if any(not torch.equal(initial[key], base[key]) for key in ("residual_xy_m", "yaw_delta_rad", "existence_logits")):
            raise RuntimeError("XY initialization is not exact frozen V18 deployed forward")
    epochs = 1 if args.mode == "smoke" else args.epochs
    plans = [epoch_batches(train_records, seed=args.seed, epoch=e, source_budget=args.source_budget,
                           window_budget=args.window_budget) for e in range(1, epochs+1)]
    target_updates = sum(len(packed) for packed, _ in plans)
    if target_updates < 1 or any(not packed for packed, _ in plans):
        raise RuntimeError("no supervised XY updates in selected population")
    out.mkdir(parents=True)
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unavailable"
    tail_lr = optimizers[ARMS[0]].param_groups[0]["lr"]
    identity = {"protocol": PROTOCOL, "base_checkpoint_sha256": provider.sha,
        "runtime_config_fingerprint": stable_json_fingerprint(cfg), "git_commit": git_commit, "mode": args.mode,
        "deployment_contract": "specialist_XY_plus_frozen_CleanE14_yaw_existence_two_forward_v1",
        "train_keys": train_keys, "calibration_keys": cal_keys, "dev_keys": use_dev_keys,
        "population_fingerprint": stable_json_fingerprint({"train": train_keys, "calibration": cal_keys}),
        "dev_manifest_fingerprint": manifest["manifest_fingerprint"], "scene_overlap": 0,
        "full_train_windows": full_train_count, "train_fraction": len(train_keys)/full_train_count,
        "epochs_per_arm": epochs, "target_updates_per_arm": target_updates, "seed": args.seed,
        "packing": {"source_budget": args.source_budget, "window_budget": args.window_budget,
                    "whole_windows_no_source_truncation": True},
        "optimizer": {"name": "AdamW", "common_moments_restored": True, "fixed_lr": tail_lr, "clip_norm": 5.},
        "loss": "source_center_SmoothL1_beta1 + 0.25*original_soft_SE2_no_yaw_or_existence_loss",
        "teacher_curriculum": {"initial_probability": .5, "zero_by_fraction": 2/3,
                               "mask": "GT_yaw_only_valid_supervised_yaw_enabled_source_horizons_in_shape_loss",
                               "inference_gt_probability": 0.},
        "checkpoint_selection": "TRAIN_scene_disjoint_real_renderer_mIoU_update0_control",
        "calibration_note": "held out from this continuation, but seen by original pretrained V18; dev64 reused exploratory"}
    write_json(out/"execution_contract.json", {**identity, "arguments": vars(args)})
    best = {arm: {"score": None, "update": 0, "state_dict": cpu_weights(m)} for arm, m in models.items()}
    timings = {"load_validate_inputs": time.perf_counter()-start}
    with (out/"progress.jsonl").open("x", encoding="utf-8") as handle:
        def progress(row):
            handle.write(json.dumps(finite_json(row), ensure_ascii=False, allow_nan=False)+"\n"); handle.flush()
        print(f"Train={len(train_records)}/{full_train_count}; TRAINcal={len(cal_records)}; dev={len(dev_records)}; "
              f"epochs={epochs} EACH; fixed_tail_lr={tail_lr}; strictly frozen BASE yaw/existence.", flush=True)
        tick = time.perf_counter()
        teacher_cache, teacher_mib = cache_baseline_outputs(provider, train_records, progress=progress)
        timings["compact_teacher_cache"] = time.perf_counter()-tick
        train_source = NuScenesWindowSource(args.dataroot, info_pkl=args.train_info, verbose=False)
        tick = time.perf_counter()
        initial_cal = evaluate_models(provider, train_source, cal_records, {}, None, progress=progress, prediction_fn=predict)
        for arm in ARMS:
            best[arm]["score"] = initial_cal["baseline"]["mIoU"]
            if best[arm]["score"] is None or not np.isfinite(best[arm]["score"]):
                raise RuntimeError("TRAIN calibration has no finite mIoU")
            atomic_checkpoint(out/(arm+"_best.pt"), checkpoint_payload({**identity, "arm": arm}, models[arm],
                role="selected_xy_specialist_candidate", update=0))
        timings["initial_train_calibration"] = time.perf_counter()-tick
        updates, history = 0, []
        for epoch, (batches, empty) in enumerate(plans, 1):
            progress({"event": "epoch_plan", "epoch": epoch, "supervised_batches": len(batches),
                      "empty_supervision_batches": len(empty), "windows_without_valid_xy_in_empty_batches": sum(map(len, empty))})
            tick = time.perf_counter()
            updates, train_stats = train_epoch(models, optimizers, train_records, teacher_cache, device,
                batches=batches, epoch=epoch, updates=updates, total_updates=target_updates, seed=args.seed, progress=progress)
            timings[f"epoch_{epoch}_paired_train"] = time.perf_counter()-tick
            for arm in ARMS:
                assert_frozen_heads(models[arm], frozen_state)
                atomic_checkpoint(out/(arm+"_last.pt"), checkpoint_payload({**identity, "arm": arm, "epoch": epoch},
                    models[arm], role="last_xy_specialist_diagnostic", update=updates, optimizer=optimizers[arm]))
            tick = time.perf_counter()
            calibration = evaluate_models(provider, train_source, cal_records, models, None, progress=progress, prediction_fn=predict)
            timings[f"epoch_{epoch}_train_calibration"] = time.perf_counter()-tick
            for arm in ARMS:
                previous = best[arm]
                best[arm] = select_checkpoint(previous, calibration["variants"][arm], models[arm], updates)
                if best[arm] is not previous:
                    atomic_checkpoint(out/(arm+"_best.pt"), checkpoint_payload({**identity, "arm": arm}, models[arm],
                        role="selected_xy_specialist_candidate", update=updates))
                print(f"calibration epoch={epoch} arm={arm} dMiOU="
                      f"{calibration['variants'][arm]['delta_vs_v18_pp']['mIoU']:+.6f} selected_update={best[arm]['update']}", flush=True)
            row = {"event": "epoch_complete", "epoch": epoch, "updates": updates, "train": train_stats,
                   "calibration": calibration, "selected_updates": {arm: best[arm]["update"] for arm in ARMS}}
            history.append(row); progress(row); write_json(out/"train_history.json", history)
        if updates != target_updates:
            raise RuntimeError("paired update budget not completed")
        del train_source, train_records, cal_records, teacher_cache, optimizers
        tick = time.perf_counter()
        dev_models, arms = {}, {}
        for arm in ARMS:
            selected = LocalSpatialTemporalWorldModelV18SE2(models[arm].v17_config).to(device)
            selected.load_state_dict(best[arm]["state_dict"], strict=True)
            selected.requires_grad_(False)
            dev_models[arm+"_selected"], dev_models[arm+"_last"] = selected, models[arm]
            arms[arm] = {"selected_update": best[arm]["update"], "calibration_mIoU": best[arm]["score"],
                         "selected_checkpoint": str((out/(arm+"_best.pt")).resolve())}
        dev_source = NuScenesWindowSource(args.dataroot, info_pkl=args.dev_info, verbose=False)
        dev = evaluate_models(provider, dev_source, dev_records, dev_models, None, progress=progress, prediction_fn=predict)
        timings["shared_final_dev"] = time.perf_counter()-tick
        for arm in ARMS:
            arms[arm]["screen_pass"] = (args.mode == "screen" and best[arm]["update"] > 0
                                        and dev["variants"][arm+"_selected"]["gate"]["pass"])
            atomic_checkpoint(out/(arm+"_best.pt"), checkpoint_payload({**identity, "arm": arm}, dev_models[arm+"_selected"],
                role="selected_xy_specialist_candidate", update=best[arm]["update"], screen_pass=arms[arm]["screen_pass"]))
        comparison = delta(dev["variants"][ARMS[1]+"_selected"]["metrics"], dev["variants"][ARMS[0]+"_selected"]["metrics"])
        route = ("smoke_only_not_effectiveness_evidence" if args.mode == "smoke" else
                 "xy_specialist_candidate_passed_requires_larger_frozen_validation" if any(a["screen_pass"] for a in arms.values()) else
                 "stop_this_fixed_experiment_no_automatic_retry")
        summary = {"protocol": PROTOCOL, "mode": args.mode, "git_commit": git_commit,
                   "train_windows": len(train_keys), "full_train_windows": full_train_count,
                   "calibration_windows": len(cal_keys), "dev_windows": len(dev_records), "epochs_per_arm": epochs,
                   "updates_per_arm": updates, "fixed_tail_lr": tail_lr, "teacher_cache_mib": teacher_mib,
                   "arms": arms, "dev": dev, "curriculum_vs_frozen_yaw_pp": comparison, "route": route,
                   "seconds_by_stage": timings, "elapsed_seconds": time.perf_counter()-start}
        write_json(out/"summary.json", summary)
        (out/"summary.txt").write_text(summary_text(summary), encoding="utf-8")
        print(summary_text(summary), flush=True)


if __name__ == "__main__":
    main()
