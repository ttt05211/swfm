#!/usr/bin/env python3
"""Bounded learned XY screen. Frozen V18; no oracle or generation module."""
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
from real_motion.v18_xy_trajectory import PROTOCOL, INPUT_PROTOCOL, INPUT_KEYS, TrajectoryConfig, make_xy_model, xy_objective
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import load_manifest, align_records
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256, finite_json, write_json, atomic_checkpoint, bank_fingerprint
from tools.real_motion.v18_xy_trajectory_common import (
    FrozenXYV18, prepare_bank, predict_bank, error_summary, evaluate_dev, select_train_calibration,
)

DEV64_FP = "0cb9d69ee11d3ba2afd88a7a2436eb7453670b82b25000a5047d3ed91061101b"
KINDS = ("linear_xy_refit", "joint_xy_position", "integrated_xy_trajectory")


def cpu_weights(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def label_fit(model, bank, device):
    pred = predict_bank(model, bank, device)
    return error_summary(pred, bank["target_xy"], bank["valid"] & bank["supervised"][:, None])


def train_variants(bank, calibration, config, device, *, updates, batch_size, seed, progress=None):
    """Same sources, optimizer and active loss; selection uses TRAIN scenes only."""
    if updates < 1 or batch_size < 1: raise ValueError("invalid training budget")
    torch.manual_seed(seed)
    models = {k: make_xy_model(k, config).to(device) for k in KINDS}
    # Parameter-matched control: identical initial weights and features.
    models["integrated_xy_trajectory"].load_state_dict(models["joint_xy_position"].state_dict())
    optimizers = {k: torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=.01) for k, m in models.items()}
    best, history = {}, []
    for k, m in models.items():
        fit = label_fit(m, calibration, device)
        if fit["selection_score_m"] is None: raise RuntimeError("calibration has no valid labels")
        best[k] = {"update": 0, "calibration": fit, "state_dict": cpu_weights(m)}
    size = sum(v.nbytes for v in bank.values())
    if size > 256*2**20: raise RuntimeError("resident XY bank exceeds fixed memory budget")
    resident = {k: torch.as_tensor(v, device=device) for k, v in bank.items()}
    rng = np.random.default_rng(seed+1)
    for update in range(1, updates+1):
        # Uniform source sampling, no class/age/error weighting or mining.
        ids = torch.as_tensor(rng.choice(len(bank["query"]), batch_size,
            replace=len(bank["query"]) < batch_size), device=device)
        batch = {k: v.index_select(0, ids) for k, v in resident.items()}
        inputs = {k: batch[k] for k in INPUT_KEYS}
        for k, m in models.items():
            tick = time.perf_counter(); m.train(); optimizer = optimizers[k]
            optimizer.zero_grad(set_to_none=True)
            prediction = m(**inputs)
            loss, parts = xy_objective(prediction, inputs, batch)
            if not torch.isfinite(loss): raise RuntimeError(f"nonfinite XY loss variant={k} update={update}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            row = {"event": "train", "variant": k, "update": update, "loss": float(loss.detach()),
                   **{key: float(v) for key, v in parts.items()}, "grad_norm": float(norm),
                   "seconds": time.perf_counter()-tick}
            if progress: progress(row)
            if update == 1 or update % 64 == 0 or update == updates:
                print(f"variant={k} update={update}/{updates} loss={row['loss']:.6f} "
                    f"xy={row['xy_smooth_l1']:.6f} shape={row['shape']:.6f} seconds={row['seconds']:.3f}", flush=True)
        if update % 256 == 0 or update == updates:
            for k, m in models.items():
                fit = label_fit(m, calibration, device)
                score = fit["selection_score_m"]
                if score is None or not np.isfinite(score): raise RuntimeError("nonfinite calibration score")
                if score < best[k]["calibration"]["selection_score_m"]-1e-9:
                    best[k] = {"update": update, "calibration": fit, "state_dict": cpu_weights(m)}
                row = {"event": "calibration", "variant": k, "update": update,
                       "fit": fit, "selected_update": best[k]["update"]}
                history.append(row)
                if progress: progress(row)
                print(f"calibration variant={k} update={update} ADE={fit['all_six_ADE']['mean_m']:.6f} "
                    f"FDE3s={fit['FDE_3s']['mean_m']} selected_update={best[k]['update']}", flush=True)
    return models, best, history


def checkpoint_payload(identity, model, *, role, update, screen_pass=False):
    return {**identity, "checkpoint_role": role, "kind": model.kind,
            "model_config": asdict(model.config), "state_dict": cpu_weights(model),
            "selected_update": update, "screen_pass": bool(screen_pass)}


def summary_text(summary):
    def num(v): return "undefined" if v is None else f"{v:+.6f}"
    def motion(errors):
        row = errors.get("all",{})
        return {k:row.get(k,{}) for k in ("source_center_error_m","source_center_error/3.0s_m")}
    lines = ["===== LEARNED V18 XY TRAJECTORY =====", f"protocol: {PROTOCOL}", f"mode: {summary['mode']}",
             f"train/calibration/dev windows: {summary['populations']}", f"updates per arm: {summary['updates']}",
             f"batch_size_sources: {summary['batch_size_sources']}",
             "V18 encoder/source/yaw/existence/renderer frozen; ONLY XY changed",
             "Checkpoint selected on TRAIN-only held-out scenes, NOT on dev or oracle",
             f"V18_mIoU: {summary['dev']['baseline']['mIoU']}",
             f"V18 dev XY errors: {motion(summary['dev']['baseline_motion_errors'])}"]
    for kind, arm in summary["arms"].items():
        lines += [f"===== {kind} =====", f"parameters: {arm['parameters']}; selected_update: {arm['selected_update']}"]
        for suffix in ("selected", "last"):
            r = summary["dev"]["variants"][kind+"_"+suffix]; d = r["delta_vs_v18_pp"]
            fit = arm["label_fit"][suffix]
            lines += [f"{suffix}: dMiOU={num(d['mIoU'])} dIoU={num(d['IoU'])} "
                      f"dMovingMicro={num(d['MovingMicro'])} dMovingMacro={num(d['MovingMacro'])}",
                      f"{suffix} dev XY errors: {motion(r['motion_errors'])}",
                      f"{suffix} train ADE/FDE3s: {fit['train']['all_six_ADE']['mean_m']} / {fit['train']['FDE_3s']['mean_m']}",
                      f"{suffix} calibration ADE/FDE3s: {fit['calibration']['all_six_ADE']['mean_m']} / {fit['calibration']['FDE_3s']['mean_m']}"]
            if suffix == "selected":
                for h, row in d["per_horizon"].items():
                    lines.append(f"{h}s dMiOU={num(row['mIoU'])} dMovingMicro={num(row['MovingMicro'])}")
                scenes = r["scene_delta"]
                strata = {k:r["motion_errors"].get(k,{}).get("source_center_error_m",{})
                          for k in ("history_valid/1","history_valid/6")}
                lines += [f"dev XY history strata: {strata}",
                          f"scene_delta: { {k:v for k,v in scenes.items() if k != 'by_scene'} }",
                          f"gate: {r['gate']}", f"deployable: {arm['screen_pass']}"]
        lines.append(f"selected_checkpoint: {arm['selected_checkpoint']}")
    lines += [f"baseline TRAIN label fit: {summary['baseline_label_fit']['train']}",
              f"baseline calibration label fit: {summary['baseline_label_fit']['calibration']}",
              f"route: {summary['route']}", f"seconds_by_stage: {summary['seconds_by_stage']}",
              f"elapsed_seconds: {summary['elapsed_seconds']:.2f}",
              "Update=0, smoke, last diagnostic or failed gate is NOT an effective/deployable improvement.",
              "No automatic retry, expansion, generation experiment or dev-based checkpoint choice."]
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__); add_config_args(parser)
    for name in ("train-cache", "dev-cache", "population-manifest", "base-checkpoint", "dataroot", "dev-info", "out-dir"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--expected-base-sha256", default=CLEAN_SHA256)
    parser.add_argument("--mode", choices=("smoke", "screen"), default="screen")
    parser.add_argument("--batch-size", type=int, default=128, help="sources per arm update, NOT windows")
    parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); start = time.perf_counter()
    if min(args.batch_size, args.cpu_workers) < 1: parser.error("invalid resource budget")
    for name in ("config", "train_cache", "dev_cache", "population_manifest", "base_checkpoint", "dev_info"):
        if not str(getattr(args, name) or "").strip() or not Path(getattr(args, name)).is_file():
            parser.error(f"{name} must be an existing file")
    if not Path(args.dataroot).is_dir(): parser.error("dataroot must be an existing directory")
    out = Path(args.out_dir)
    if out.exists(): parser.error("out-dir exists; choose a NEW directory")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    torch.set_num_threads(1); torch.manual_seed(args.seed); np.random.seed(args.seed)
    cfg = load_runtime_config(args.config, args.override); pcfg = make_prepare_config(cfg)
    if pcfg.free_label != 17: raise RuntimeError("requires frozen 18-class metric contract")
    manifest, dev_keys, _ = load_manifest(args.population_manifest)
    if len(dev_keys) != 64 or len(manifest["parent_keys"]) != 512 or manifest["selected_key_fingerprint"] != DEV64_FP:
        raise RuntimeError("requires the frozen Stage-1 dev512-derived dev64 identity/order")
    train_n, cal_n, dev_n, updates = (4, 2, 2, 8) if args.mode == "smoke" else (512, 32, 64, 1024)
    train_meta, all_train = load_cache(args.train_cache)
    train_keys, cal_keys = select_train_calibration([(r["scene_name"], r["t0_token"]) for r in all_train],
        train_n, cal_n, {k[0] for k in manifest["parent_keys"]})
    train_records, cal_records = align_records(all_train, train_keys), align_records(all_train, cal_keys); del all_train
    dev_meta, all_dev = load_cache(args.dev_cache); dev_records = align_records(all_dev, dev_keys[:dev_n]); del all_dev
    for meta in (train_meta, dev_meta):
        if float(meta.get("patch_resolution_m", .8)) != .8: raise RuntimeError("frozen V18 patch resolution mismatch")
    provider = FrozenXYV18(args.base_checkpoint, args.expected_base_sha256, pcfg, device, args.cpu_workers)
    config = TrajectoryConfig(dim=provider.dim)
    out.mkdir(parents=True)
    try: git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError): git_commit = "unavailable"
    identity = {"protocol": PROTOCOL, "input_protocol": INPUT_PROTOCOL, "base_checkpoint_sha256": provider.sha,
        "runtime_config_fingerprint": stable_json_fingerprint(cfg), "git_commit": git_commit, "mode": args.mode,
        "populations": {"train": train_n, "calibration": cal_n, "dev": dev_n}, "train_keys": train_keys,
        "calibration_keys": cal_keys, "dev_keys": dev_keys[:dev_n], "dev_manifest_fingerprint": manifest["manifest_fingerprint"],
        "scene_overlap": 0, "updates": updates, "seed": args.seed, "batch_size_sources": args.batch_size,
        "checkpoint_selection": "TRAIN_scene_disjoint_0.5_times_ADE_plus_FDE3s_with_update0_control",
        "loss": "original_XY_SmoothL1_beta1_plus_0.25_frozen_yaw_soft_SE2_IoU",
        "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": .01, "clip_norm": 5.}}
    write_json(out/"execution_contract.json", {**identity, "arguments": vars(args), "model_config": asdict(config), "arms": KINDS})
    with (out/"progress.jsonl").open("x", encoding="utf-8") as handle:
        def progress(row):
            handle.write(json.dumps(finite_json(row), ensure_ascii=False, allow_nan=False)+"\n"); handle.flush()
        print("Preparing compact causal latents ONCE; no nuScenes training reads or dense voxel bank.", flush=True)
        timings = {"load_validate_inputs": time.perf_counter()-start}; tick = time.perf_counter()
        bank = prepare_bank(provider, train_records, progress=progress)
        calibration = prepare_bank(provider, cal_records, progress=progress)
        timings["shared_latent_prepare"] = time.perf_counter()-tick
        del train_records, cal_records
        identity["bank_fingerprints"] = {"train": bank_fingerprint(bank), "calibration": bank_fingerprint(calibration)}
        identity["bank_population"] = {role: {"sources": len(b["query"]), "valid_source_horizons": int(b["valid"].sum()),
            "mib": sum(v.nbytes for v in b.values())/2**20} for role,b in (("train",bank),("calibration",calibration))}
        write_json(out/"execution_contract.json", {**identity, "arguments": vars(args), "model_config": asdict(config), "arms": KINDS})
        baseline_label = {"train": error_summary(bank["base_xy"], bank["target_xy"], bank["valid"]),
            "calibration": error_summary(calibration["base_xy"], calibration["target_xy"], calibration["valid"])}
        print(f"Train sources={len(bank['query'])}; calibration sources={len(calibration['query'])}; three arms share latents.", flush=True)
        tick = time.perf_counter()
        models, best, selection = train_variants(bank, calibration, config, device,
            updates=updates, batch_size=args.batch_size, seed=args.seed, progress=progress)
        timings["three_arm_train_and_calibration"] = time.perf_counter()-tick; tick = time.perf_counter()
        dev_models, arms = {}, {}
        for kind, last in models.items():
            selected = make_xy_model(kind, config).to(device); selected.load_state_dict(best[kind]["state_dict"])
            dev_models[kind+"_selected"], dev_models[kind+"_last"] = selected, last
            fits = {role: {"train": label_fit(m, bank, device), "calibration": label_fit(m, calibration, device)}
                    for role, m in (("selected", selected), ("last", last))}
            selected_path, last_path = out/(kind+"_best.pt"), out/(kind+"_last.pt")
            # Persist completed training before real-data evaluation. Never
            # mark a model deployable until the actual frozen metric gate ran.
            atomic_checkpoint(selected_path, checkpoint_payload(identity, selected, role="selected_xy_candidate", update=best[kind]["update"]))
            atomic_checkpoint(last_path, checkpoint_payload(identity, last, role="last_xy_diagnostic", update=updates))
            arms[kind] = {"parameters": sum(p.numel() for p in last.parameters()), "selected_update": best[kind]["update"],
                "label_fit": fits, "selected_checkpoint": str(selected_path.resolve()), "last_checkpoint": str(last_path.resolve())}
        write_json(out/"calibration.json", {"selection_history": selection, "arms": arms, "baseline": baseline_label})
        del bank, calibration, models
        timings["label_diagnostics_and_checkpoints"] = time.perf_counter()-tick; tick = time.perf_counter()
        if device.type == "cuda": torch.cuda.empty_cache()
        print("TRAIN-calibration choices frozen. ONE dev pass evaluates selected + last XY; no oracle intervention.", flush=True)
        source = NuScenesWindowSource(args.dataroot, info_pkl=args.dev_info, verbose=False)
        dev = evaluate_dev(provider, source, dev_records, dev_models, progress=progress)
        timings["shared_dev_prepare_and_six_variant_render"] = time.perf_counter()-tick
        for kind, arm in arms.items():
            passed = args.mode == "screen" and arm["selected_update"] > 0 and dev["variants"][kind+"_selected"]["gate"]["pass"]
            arm["screen_pass"] = passed
            ck = checkpoint_payload(identity, dev_models[kind+"_selected"], role="selected_xy_candidate",
                update=arm["selected_update"], screen_pass=passed)
            ck["dev_gate"] = dev["variants"][kind+"_selected"]["gate"]
            atomic_checkpoint(arm["selected_checkpoint"], ck)
        summary = {**identity, "arms": arms, "baseline_label_fit": baseline_label, "dev": dev,
            "elapsed_seconds": time.perf_counter()-start, "seconds_by_stage": timings,
            "route": "screen_pass_review_before_expansion" if any(a["screen_pass"] for a in arms.values())
                     else "stop_xy_version_no_automatic_retry"}
        write_json(out/"summary.json", summary)
        (out/"summary.txt").write_text(summary_text(finite_json(summary)), encoding="utf-8")
        print((out/"summary.txt").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__": main()
