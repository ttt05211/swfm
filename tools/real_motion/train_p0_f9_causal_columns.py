#!/usr/bin/env python3
"""One bounded shared generation/refinement screen; baseline V18 is immutable."""
from __future__ import annotations
import sys
from pathlib import Path
if __package__ in (None, ""): sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataclasses import asdict
import argparse
import json
import subprocess
import time
import numpy as np
import torch

from real_motion.causal_column_completion import (PROTOCOL, FEATURE_PROTOCOL, ColumnConfig, GENERATE, REFINE,
    KEEP, ADD, REMOVE, action_targets, sample_queries)
from real_motion.causal_column_model import CausalColumnModel, column_loss
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion.causal_column_common import (FrozenColumns, FEATURE_KEYS, CALIBRATION_LEVELS,
    candidate_plan, sample_column_features, calibrate_columns, evaluate_columns)
from tools.real_motion.v18_source_interaction_common import select_population
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import load_manifest, align_records, sha256
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256, write_json, finite_json, atomic_checkpoint, bank_fingerprint
from tools.real_motion.train_p0_f9_v18_xy_trajectory import DEV64_FP

CONTRACT = {"v18_frozen": True, "grid_entry_classes": [11, 13], "generation_free_only": True,
    "refine": "visible_source_REMOVE_restore_lower_layer_and_free_ADD_v1", "static_refine_classes": [11, 13],
    "history": "same_as_V18_semantics_with_separate_visibility_and_source_membership",
    "source_alignment": "causal_same_class_ICP_worldZ_preserved_no_future_GT",
    "losses": ["importance_corrected_weighted_class_conditioned_generation_BCE",
               "importance_corrected_weighted_legal_refine_action_CE_KEEP_on_utility_tie"],
    "calibration": "subtract_log_TRAIN_weights_then_fixed_grid_TRAIN_only_actual_metrics",
    "calibration_levels": list(CALIBRATION_LEVELS), "fixed_diagnostic_thresholds": [.50, .50, .50],
    "learning_rate": 3e-4, "weight_decay": .01, "clip_norm": 1., "queries_per_kind_horizon": 4,
    "epochs_selected_using_dev": False, "evaluation_candidate_truncation": False}


def record_keys(records):
    keys = tuple((str(r["scene_name"]), str(r["t0_token"])) for r in records)
    if len(keys) != len(set(keys)): raise RuntimeError("duplicate cache identities")
    return keys


def prepare_bank(provider, source, records, model_config, seed, max_mib, progress=None):
    rng = np.random.default_rng(seed)
    parts = {k: [] for k in (*FEATURE_KEYS, "legal", "target", "weight")}
    counts = {"generation": np.zeros(2, np.float64), "refine": np.zeros(3, np.float64)}
    audit = {"generation_queries": 0, "static_refine_queries": 0, "dynamic_refine_queries": 0}
    size = 0
    for wi, record in enumerate(records, 1):
        tick = time.perf_counter(); print(f"prepare_columns_TRAIN={wi}/{len(records)}", flush=True)
        prep = provider.prepare_columns(source, record, include_gt=True)
        for h in range(6):
            plan = candidate_plan(prep, h, provider.pcfg.grid, model_config)
            target = action_targets(plan, prep.raw["future_gt_occ"][h])
            audit["generation_queries"] += int((plan.kind == GENERATE).sum())
            audit["static_refine_queries"] += int(((plan.kind == REFINE)&(plan.actor < 0)).sum())
            audit["dynamic_refine_queries"] += int(((plan.kind == REFINE)&(plan.actor >= 0)).sum())
            gen = (plan.kind == GENERATE)[:, None]&plan.legal[..., ADD]
            counts["generation"] += np.bincount((target[gen] == ADD).astype(int), minlength=2)
            ref = (plan.kind == REFINE)[:, None]&(plan.legal[..., ADD]|plan.legal[..., REMOVE])
            counts["refine"] += np.bincount(target[ref], minlength=3)
            ids, weight = sample_queries(plan, target, CONTRACT["queries_per_kind_horizon"], rng)
            if not len(ids): continue
            small = plan.subset(ids)
            features = sample_column_features(prep, h, small, provider.pcfg.grid, model_config)
            values = {**features, "legal": small.legal, "target": target[ids].astype(np.uint8), "weight": weight}
            size += sum(v.nbytes for v in values.values())
            if 2*size > max_mib*2**20:
                raise RuntimeError("compact TRAIN bank + concatenation exceeds RAM budget; no dense disk fallback")
            for k, v in values.items(): parts[k].append(v)
        if progress: progress({"event": "prepare_TRAIN", "window": wi, "windows": len(records), "bank_mib": size/2**20,
                               "seconds": time.perf_counter()-tick, "prepare_seconds": getattr(provider, "last_prepare_seconds", {})})
        print(f"prepare_columns_TRAIN={wi}/{len(records)} complete seconds={time.perf_counter()-tick:.3f} bank_mib={size/2**20:.2f}", flush=True)
    if not parts["kind"]: raise RuntimeError("no causal editable candidates in TRAIN population")
    bank = {k: np.concatenate(v) for k, v in parts.items()}
    gen = counts["generation"]; ref = counts["refine"]
    alpha = float(np.clip(gen[0]/max(gen[1], 1), 1, 20))
    present = ref[ref > 0]
    weights = np.clip((np.median(present) if len(present) else 1)/np.maximum(ref, 1), .2, 20).astype(np.float32)
    return bank, {"generation_pos_weight": alpha, "refine_class_weights": weights.tolist(),
        "full_unsampled_generation_counts": gen.tolist(), "full_unsampled_refine_action_counts": ref.tolist(),
        "candidate_audit": audit, "query_importance_restores_sampling": True}


def train_step(model, optimizer, bank, rng, device, batch_size):
    buckets = [np.flatnonzero(bank["kind"] == task) for task in (GENERATE, REFINE)]
    buckets = [b for b in buckets if len(b)]
    ids = np.concatenate([rng.choice(b, batch_size//len(buckets), replace=True) for b in buckets])
    batch = {k: torch.as_tensor(v[ids], device=device) for k, v in bank.items()}
    model.train(); optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        g, r = model(**{k: batch[k] for k in FEATURE_KEYS})
    loss, stats = column_loss(model, g, r, batch["kind"], batch["legal"], batch["target"], batch["weight"])
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    with torch.no_grad():
        p = model.calibrated_probabilities(g.detach(), r.detach(), batch["kind"], batch["legal"])
        stats.update(mean_corrected_add_probability=float(p[..., ADD].mean()),
                     mean_corrected_remove_probability=float(p[..., REMOVE].mean()),
                     sampled_positive_actions=int((batch["target"] != KEEP).sum()))
    return {"loss": float(loss.detach()), "grad_norm": float(grad), **stats}


def load_columns(path, device, *, base_sha, config_sha, allow_diagnostic=False):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if (ck.get("protocol") != PROTOCOL or ck.get("feature_protocol") != FEATURE_PROTOCOL or ck.get("training_contract") != CONTRACT
            or ck.get("base_checkpoint_sha256") != base_sha or ck.get("runtime_config_fingerprint") != config_sha
            or ck.get("checkpoint_role") not in ("calibrated_candidate", "resume_last")):
        raise RuntimeError("column checkpoint/base/config/deployment contract mismatch")
    if (ck.get("checkpoint_role") != "calibrated_candidate" or ck.get("mode") != "screen"
            or not ck.get("screen_pass") or ck.get("successful_updates", 0) <= 0) and not allow_diagnostic:
        raise RuntimeError("failed/smoke/last/identity column candidate cannot be deployed")
    model = CausalColumnModel(ColumnConfig(**ck["model_config"])).to(device)
    model.load_state_dict(ck["state_dict"], strict=True); model.eval()
    w = ck["TRAIN_weights"]
    if (not np.isclose(float(model.generation_pos_weight), w["generation_pos_weight"], rtol=0, atol=1e-6)
            or not np.allclose(model.refine_class_weights.cpu().numpy(), w["refine_class_weights"], rtol=0, atol=1e-6)):
        raise RuntimeError("persisted prior correction differs from TRAIN counts")
    # Threshold validation independent of GT/population.
    gates = ck.get("thresholds")
    if gates is not None and (len(gates) != 3 or any(t is not None and (not np.isfinite(t) or not .5 <= t <= 1) for t in gates)):
        raise RuntimeError("invalid persisted thresholds")
    if ck["checkpoint_role"] == "calibrated_candidate" and gates is None:
        raise RuntimeError("candidate has no frozen TRAIN thresholds")
    if not allow_diagnostic and not any(t is not None for t in gates):
        raise RuntimeError("disabled identity column candidate cannot be deployed")
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise RuntimeError("nonfinite persisted model state")
    model.requires_grad_(False)
    return ck, model


def summary_text(summary):
    lines = ["===== CAUSAL COLUMN GENERATION + REFINEMENT =====", f"protocol: {PROTOCOL}",
        f"mode: {summary['mode']}", f"successful_updates: {summary['successful_updates']}",
        f"train_windows: {summary['train_windows']}", f"TRAIN_calibration_windows: {summary['calibration_windows']}",
        f"bank_mib: {summary['bank_mib']:.2f}", f"thresholds_from_TRAIN_only: {summary['thresholds']}",
        f"TRAIN_weights: {summary['TRAIN_weights']}", "V18 XY/yaw/existence/weights frozen; no future GT features."]
    for population in ("dev64", "all"):
        if population not in summary["evaluation"]: continue
        report = summary["evaluation"][population]
        lines += [f"\n===== {population}: {report['windows']} windows / {report['scenes']} scenes =====",
                  f"V18_BASE: {report['baseline']['mIoU']:.6f} mIoU / {report['baseline']['MovingMicro']:.6f} MovingMicro"]
        for name, row in report["variants"].items():
            d, q, scenes = row["delta_vs_v18_pp"], row["quality"], row["scene_delta"]
            lines += [f"{name}: dMiOU={d['mIoU']:+.6f} dIoU={d['IoU']:+.6f} dMovingMicro={d['MovingMicro']:+.6f} "
                      f"add={q.get('added',0)} semantic_precision={q['addition_semantic_precision']} "
                      f"remove={q.get('removed',0)} removed_true_occ={q.get('removed_true_occupancy',0)} damaged={q.get('damaged',0)} "
                      f"scenes=+{scenes['positive']}/-{scenes['negative']}/0{scenes['zero']}"]
            if not name.startswith("diagnostic_"):
                for h, delta_h in d["per_horizon"].items():
                    lines += [f"  {h}s dMiOU={delta_h['mIoU']:+.6f} dIoU={delta_h['IoU']:+.6f} dMovingMicro={delta_h['MovingMicro']:+.6f}"]
        lines += [f"gate: {report['gate']}", f"confidence_audit: {report['confidence_audit']}"]
    lines += [f"\nsource_audit: {summary['evaluation']['source_audit']}", f"route: {summary['route']}",
              f"elapsed_seconds: {summary['elapsed_seconds']:.2f}", f"candidate_checkpoint: {summary['candidate_checkpoint']}",
              "diagnostic_* uses predeclared (.50,.50,.50), NOT another dev-selected model.",
              "All-reject/disabled output is V18 identity, NOT learned success; no automatic retry or expansion."]
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__); add_config_args(parser)
    for name in ("train-cache", "dev-cache", "population-manifest", "base-checkpoint", "dataroot", "train-info", "dev-info", "out-dir"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--expected-base-sha256", default=CLEAN_SHA256)
    parser.add_argument("--mode", choices=("smoke", "screen"), default="screen")
    parser.add_argument("--device", default="cuda"); parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256, help="columns, not windows")
    parser.add_argument("--bank-max-mib", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--resume", help="matching last.pt into a NEW directory; rebuild compact TRAIN bank")
    args = parser.parse_args(); started = time.perf_counter()
    for name in ("config", "train_cache", "dev_cache", "population_manifest", "base_checkpoint", "train_info", "dev_info"):
        if not str(getattr(args, name) or "").strip() or not Path(getattr(args, name)).is_file(): parser.error(f"{name} must be an existing file")
    if not Path(args.dataroot).is_dir() or min(args.batch_size, args.cpu_workers, args.bank_max_mib) < 1: parser.error("invalid paths/budgets")
    if args.batch_size < 2 or args.batch_size % 2: parser.error("batch-size must be even and >=2")
    out = Path(args.out_dir)
    if out.exists(): parser.error("choose a NEW output directory; original models/data must not be overwritten")
    device = torch.device(args.device)
    if device.type == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("CUDA/BF16 unavailable; no silent fallback")
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    cfg = load_runtime_config(args.config, args.override); pcfg = make_prepare_config(cfg)
    model_cfg = ColumnConfig(z_bins=int(pcfg.grid.shape_hwd[2]))
    manifest, dev64, _ = load_manifest(args.population_manifest)
    if (len(dev64) != 64 or len(manifest["parent_keys"]) != 512 or manifest["selected_key_fingerprint"] != DEV64_FP):
        raise RuntimeError("requires frozen scene-balanced dev64 manifest descending from dev512")
    _, all_train = load_cache(args.train_cache)
    all_keys = record_keys(all_train)
    if args.mode == "screen" and len(all_keys) != 20430: raise RuntimeError("screen requires confirmed full20430 V18 TRAIN cache")
    count = 1024 if args.mode == "screen" else min(4, len(all_keys)//2)
    dev_scenes = {str(s) for s, _ in manifest["parent_keys"]}
    held_scenes = 32 if args.mode == "screen" else 1
    train_keys, cal_keys = select_population(all_keys, dev_scenes, fraction=count/len(all_keys)+1e-12,
                                           calibration_scenes=held_scenes, seed=args.seed)
    if args.mode == "smoke": cal_keys = cal_keys[:2]
    records, cal_records = align_records(all_train, train_keys), align_records(all_train, cal_keys); del all_train
    if args.mode == "screen" and (len(records) != 1024 or len(cal_records) != 64):
        raise RuntimeError("screen requires exact TRAIN1024 / scene-disjoint calibration64 populations")
    _, all_dev = load_cache(args.dev_cache)
    record_keys(all_dev)
    dev_keys = tuple((str(s), str(t)) for s, t in manifest["parent_keys"]) if args.mode == "screen" else tuple(dev64[:2])
    dev_records = align_records(all_dev, dev_keys); del all_dev
    used_scenes = {s for s, _ in (*train_keys, *cal_keys)}
    if used_scenes & dev_scenes: raise RuntimeError("TRAIN/calibration/dev scene leakage")
    provider = FrozenColumns(args.base_checkpoint, args.expected_base_sha256, pcfg, device, args.cpu_workers)
    source = NuScenesWindowSource(args.dataroot, info_pkl=args.train_info, verbose=False)
    config_sha = stable_json_fingerprint(cfg); target_updates = 1024 if args.mode == "screen" else 2
    identity = {"protocol": PROTOCOL, "feature_protocol": FEATURE_PROTOCOL, "training_contract": CONTRACT,
        "model_config": asdict(model_cfg), "mode": args.mode, "base_checkpoint_sha256": provider.sha,
        "runtime_config_fingerprint": config_sha, "dev_manifest_fingerprint": manifest["manifest_fingerprint"],
        "train_keys": train_keys, "calibration_keys": cal_keys, "dev_keys": dev_keys,
        "population_fingerprint": stable_json_fingerprint({"train": train_keys, "calibration": cal_keys, "dev": dev_keys}),
        "seed": args.seed, "batch_size_columns": args.batch_size, "target_updates": target_updates,
        "info_fingerprints": {"train": sha256(args.train_info), "dev": sha256(args.dev_info)}}
    out.mkdir(parents=True)
    write_json(out/"execution_contract.json", {**identity, "arguments": vars(args), "calibration_no_dev_access": True})
    with (out/"progress.jsonl").open("x", encoding="utf-8") as handle:
        def progress(row):
            handle.write(json.dumps(finite_json(row), ensure_ascii=False, allow_nan=False)+"\n"); handle.flush()
        bank, weights = prepare_bank(provider, source, records, model_cfg, args.seed, args.bank_max_mib, progress)
        del records
        identity.update(bank_fingerprint=bank_fingerprint(bank), TRAIN_weights=weights)
        model = CausalColumnModel(model_cfg).to(device)
        model.generation_pos_weight.fill_(weights["generation_pos_weight"])
        model.refine_class_weights.copy_(torch.tensor(weights["refine_class_weights"], device=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
        rng = np.random.default_rng(args.seed+1); start_update = 0
        if args.resume:
            ck, model = load_columns(args.resume, device, base_sha=provider.sha, config_sha=config_sha, allow_diagnostic=True)
            if ck["checkpoint_role"] != "resume_last" or any(stable_json_fingerprint(ck.get(k)) != stable_json_fingerprint(v) for k, v in identity.items()):
                raise RuntimeError("resume population/features/sampling/budget mismatch")
            model.requires_grad_(True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01); optimizer.load_state_dict(ck["optimizer"])
            rng.bit_generator.state = ck["sampling_rng_state"]; torch.set_rng_state(ck["torch_rng_state"])
            if device.type == "cuda": torch.cuda.set_rng_state_all(ck["cuda_rng_states"])
            start_update = int(ck["successful_updates"])
            if not 0 <= start_update <= target_updates: raise RuntimeError("invalid resume update counter")
        def payload(update, role):
            return {**identity, "checkpoint_role": role, "successful_updates": update, "screen_pass": False,
                    "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        def save_last(update):
            atomic_checkpoint(out/"last.pt", {**payload(update, "resume_last"), "optimizer": optimizer.state_dict(),
                "sampling_rng_state": rng.bit_generator.state, "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else []})
        save_last(start_update)
        for update in range(start_update+1, target_updates+1):
            tick = time.perf_counter(); stats = train_step(model, optimizer, bank, rng, device, args.batch_size)
            progress({"event": "train", "update": update, **stats, "seconds": time.perf_counter()-tick})
            if update == 1 or update % 16 == 0 or update == target_updates:
                print(f"update={update}/{target_updates} loss={stats['loss']:.6f} grad={stats['grad_norm']:.4f} "
                      f"add_p={stats['mean_corrected_add_probability']:.4f} remove_p={stats['mean_corrected_remove_probability']:.4f} "
                      f"seconds={time.perf_counter()-tick:.3f}", flush=True)
            if update % 128 == 0 or update == target_updates: save_last(update)
        bank_mib = sum(v.nbytes for v in bank.values())/2**20; del bank
        gates, calibration = calibrate_columns(provider, source, cal_records, model, progress=progress, batch_size=args.batch_size)
        write_json(out/"TRAIN_calibration.json", calibration); del cal_records, source
        # Freeze thresholds and weights on disk BEFORE looking at any dev labels.
        atomic_checkpoint(out/"candidate.pt", {**payload(target_updates, "calibrated_candidate"), "thresholds": gates,
                          "calibration": calibration})
        ck, persisted = load_columns(out/"candidate.pt", device, base_sha=provider.sha, config_sha=config_sha, allow_diagnostic=True)
        if any(not torch.equal(v.cpu(), persisted.state_dict()[k].cpu()) for k, v in model.state_dict().items()):
            raise RuntimeError("checkpoint serialization changed model")
        dev_source = NuScenesWindowSource(args.dataroot, info_pkl=args.dev_info, verbose=False)
        evaluation = evaluate_columns(provider, dev_source, dev_records, persisted, tuple(ck["thresholds"]), progress=progress,
            batch_size=args.batch_size, dev64_keys=dev64 if args.mode == "screen" else None)
        passed = args.mode == "screen" and evaluation["all"]["gate"]["pass"] and evaluation["dev64"]["gate"]["pass"]
        ck["screen_pass"] = passed; atomic_checkpoint(out/"candidate.pt", ck)
        try: commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except (OSError, subprocess.CalledProcessError): commit = "unavailable"
        summary = {**identity, "git_commit": commit, "successful_updates": target_updates, "train_windows": len(train_keys),
            "calibration_windows": len(cal_keys), "bank_mib": bank_mib, "thresholds": gates, "evaluation": evaluation,
            "candidate_checkpoint": str((out/"candidate.pt").resolve()), "checkpoint_sha256": sha256(out/"candidate.pt"),
            "screen_pass": passed, "elapsed_seconds": time.perf_counter()-started,
            "route": "smoke_only_not_effectiveness_evidence" if args.mode == "smoke" else
                "candidate_passed_requires_explicit_full_validation" if passed else "stop_this_version_no_automatic_retry"}
        write_json(out/"summary.json", summary)
        (out/"summary.txt").write_text(summary_text(summary), encoding="utf-8")
        print(summary_text(summary), flush=True)


if __name__ == "__main__": main()
