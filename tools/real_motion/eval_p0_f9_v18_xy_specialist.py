#!/usr/bin/env python3
"""Read-only expanded XY-specialist validation; no retraining/reselection."""
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
from real_motion.v18_xy_trajectory import xy_screen_gate
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import load_manifest, align_records, delta, sha256
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256, finite_json, write_json
from tools.real_motion.v18_xy_trajectory_common import FrozenXYV18
from tools.real_motion.train_p0_f9_v18_xy_trajectory import DEV64_FP
from tools.real_motion.v18_source_interaction_common import evaluate_models, validate_records
from tools.real_motion.v18_xy_specialist_common import PROTOCOL as TRAIN_PROTOCOL, ARMS, load_candidate, predict

PROTOCOL = "p0_f9_v18_xy_specialist_expanded_validation_v1"
METRICS = ("IoU", "mIoU", "MovingMacro", "MovingMicro")


def ordered_keys(rows):
    return tuple((str(a), str(b)) for a, b in rows)


def plan_populations(records, manifest, dev64, *, population):
    """512 windows does NOT mean more scenes: also evaluate all 150 scenes."""
    keys = tuple((str(r["scene_name"]), str(r["t0_token"])) for r in records)
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate full validation identities")
    parent = ordered_keys(manifest["parent_keys"])
    dev64 = ordered_keys(dev64)
    if (len(parent) != 512 or len(set(parent)) != 512 or len(dev64) != 64 or len(set(dev64)) != 64
            or not set(dev64) <= set(parent) <= set(keys)):
        raise RuntimeError("missing or invalid frozen dev64/dev512 population")
    # First reproduce dev64, then evaluate every other key ONCE. Counts are
    # integer reductions, so subset aggregation is independent of cache order.
    seen = set(dev64)
    if population == "full4369":
        if len(keys) != 4369 or len({s for s, _ in keys}) != 150:
            raise RuntimeError("requires confirmed full4369 / 150-scene validation cache")
        old_scenes = {s for s, _ in parent}
        new_scene_keys = tuple(k for k in keys if k[0] not in old_scenes)
        if not new_scene_keys:
            raise RuntimeError("no new validation scenes outside frozen dev512")
        evaluation = dev64+tuple(k for k in keys if k not in seen)
        groups = {"dev64_reproduction": dev64, "frozen_dev512": parent, "new_scenes_only": new_scene_keys}
    elif population == "dev512":
        evaluation = dev64+tuple(k for k in parent if k not in seen)
        groups = {"dev64_reproduction": dev64}
    else:
        raise ValueError("unknown evaluation population")
    return align_records(records, evaluation), evaluation, groups


def read_bundle(model_dir, provider, config_sha, manifest, dev64):
    """Failed/last checkpoints are allowed ONLY for this read-only diagnostic.

    No state_dict or optimizer is written back. Update0 best aliases must be
    verified against actual E14 weights, never trusted from metadata alone.
    """
    model_dir = Path(model_dir)
    contract = json.loads((model_dir/"execution_contract.json").read_text(encoding="utf-8"))
    previous = json.loads((model_dir/"summary.json").read_text(encoding="utf-8"))
    if (contract.get("protocol") != TRAIN_PROTOCOL or previous.get("protocol") != TRAIN_PROTOCOL
            or contract.get("mode") != "screen" or previous.get("mode") != "screen"):
        raise RuntimeError("requires an existing real screen run, not smoke/another method")
    train, cal = ordered_keys(contract["train_keys"]), ordered_keys(contract["calibration_keys"])
    fingerprint = stable_json_fingerprint({"train": train, "calibration": cal})
    if (len(train) != 4086 or len(cal) != 64 or len(set(train)) != len(train) or len(set(cal)) != len(cal)
            or {s for s, _ in train} & {s for s, _ in cal}
            or ordered_keys(contract["dev_keys"]) != ordered_keys(dev64)
            or contract.get("population_fingerprint") != fingerprint
            or contract.get("dev_manifest_fingerprint") != manifest["manifest_fingerprint"]
            or contract.get("runtime_config_fingerprint") != config_sha or contract.get("base_checkpoint_sha256") != provider.sha):
        raise RuntimeError("original run population/base/config contract mismatch")
    models, aliases, audit = {}, {}, {}
    for arm in ARMS:
        for role, suffix in (("selected", "best"), ("last", "last")):
            path = model_dir/(arm+"_"+suffix+".pt")
            digest = sha256(path)
            ck, model = load_candidate(path, provider.device, base_sha=provider.sha, config_sha=config_sha,
                                       allow_failed_diagnostic=True)
            expected_role = "selected_xy_specialist_candidate" if role == "selected" else "last_xy_specialist_diagnostic"
            expected_update = previous["arms"][arm]["selected_update"] if role == "selected" else previous["updates_per_arm"]
            if (ck.get("arm") != arm or ck.get("checkpoint_role") != expected_role or ck.get("mode") != "screen"
                    or ck.get("selected_update") != expected_update or ck.get("population_fingerprint") != fingerprint
                    or ordered_keys(ck["train_keys"]) != train or ordered_keys(ck["calibration_keys"]) != cal
                    or ordered_keys(ck["dev_keys"]) != ordered_keys(dev64)):
                raise RuntimeError("checkpoint role/update/paired population mismatch")
            name = arm+"_"+role
            if role == "selected" and expected_update == 0:
                reference = provider.model.state_dict()
                state = model.state_dict()
                if set(state) != set(reference) or any(not torch.equal(v.detach().cpu(), reference[k].detach().cpu())
                                                       for k, v in state.items()):
                    raise RuntimeError("update0 checkpoint weights differ from frozen E14")
                aliases[name] = "V18_BASE"
                del model
            else:
                models[name] = model
            audit[name] = {"path": str(path.resolve()), "sha256": digest, "original_selected_update": ck["selected_update"],
                           "checkpoint_role": ck["checkpoint_role"], "original_screen_pass": ck["screen_pass"],
                           "baseline_alias_verified": name in aliases}
            del ck
    return models, aliases, audit, contract, previous


def attach_baseline_aliases(report, aliases, keys):
    for name in aliases:
        baseline = report["baseline"]
        scene_names = sorted({s for s, _ in keys})
        row = {"metrics": baseline, "delta_vs_v18_pp": delta(baseline, baseline),
               "scene_delta": {"scenes": len(scene_names), "positive": 0, "negative": 0, "zero": len(scene_names),
                               "by_scene": {s: 0. for s in scene_names}},
               "motion_errors": report["baseline_motion_errors"], "edit_quality": {},
               "baseline_alias_verified": True}
        row["gate"] = xy_screen_gate(row)
        report["variants"][name] = row
    return report


def check_dev64_reproduction(current, previous, *, tolerance=1e-8):
    """Do not blame population size if original dev64 no longer reproduces."""
    maximum, checked = 0., 0
    def compare(a, b, path):
        nonlocal maximum, checked
        a_finite = a is not None and np.isfinite(a)
        b_finite = b is not None and np.isfinite(b)
        if not a_finite and not b_finite:
            return
        if a_finite != b_finite:
            raise RuntimeError(f"dev64 reproduction mismatch: {path}")
        difference = abs(float(a)-float(b)); maximum = max(maximum, difference); checked += 1
        if difference > tolerance:
            raise RuntimeError(f"dev64 reproduction mismatch: {path} difference={difference}")
    for name in ("V18_BASE", *current["variants"]):
        now = current["baseline"] if name == "V18_BASE" else current["variants"][name]["metrics"]
        old = previous["dev"]["baseline"] if name == "V18_BASE" else previous["dev"]["variants"][name]["metrics"]
        for key in METRICS:
            compare(now[key], old[key], f"{name}/{key}")
        for h in ("1.0", "2.0", "3.0"):
            for key in METRICS:
                compare(now["per_horizon"][h][key], old["per_horizon"][h][key], f"{name}/{h}/{key}")
    return {"checked": True, "tolerance_pp": tolerance, "metric_values_checked": checked, "max_abs_difference_pp": maximum}


def summary_text(summary):
    lines = ["===== EXPANDED V18 XY VALIDATION (NO TRAINING) =====", f"protocol: {PROTOCOL}",
             f"population: {summary['population']}", f"windows: {summary['windows']}", f"scenes: {summary['scenes']}",
             f"original_model_dir: {summary['original_model_dir']}",
             "Models/epochs/LR frozen. No checkpoint reselection or promotion. Last remains diagnostic.",
             "All forecasts use learned XY + original predicted yaw/existence; NO GT yaw.",
             f"dev64_reproduction: {summary['dev64_reproduction']}"]
    for group, report in summary["reports"].items():
        lines += [f"\n===== {group}: {report['windows']} windows / {report['scenes']} scenes =====",
                  f"V18_BASE: { {k: report['baseline'][k] for k in METRICS} }"]
        motion_keys = ("source_center_error_m", "source_center_error/3.0s_m", "yaw_error_deg")
        def compact_motion(errors):
            return {k: {s: errors.get("all", {}).get(k, {}).get(s) for s in ("mean", "p90")} for k in motion_keys}
        lines += [f"V18_BASE trajectory: {compact_motion(report['baseline_motion_errors'])}"]
        for arm in ARMS:
            for role in ("selected", "last"):
                name = arm+"_"+role; row = report["variants"][name]
                scenes = row["scene_delta"]
                vals = [v for v in scenes.get("by_scene", {}).values() if v is not None and np.isfinite(v)]
                scene_summary = {k: scenes[k] for k in ("scenes", "positive", "negative", "zero")}
                scene_summary.update(mean=float(np.mean(vals)) if vals else None, median=float(np.median(vals)) if vals else None)
                lines += [f"{name} dMiOU={row['delta_vs_v18_pp']['mIoU']:+.6f} "
                          f"dIoU={row['delta_vs_v18_pp']['IoU']:+.6f} "
                          f"dMovingMicro={row['delta_vs_v18_pp']['MovingMicro']:+.6f} "
                          f"dMovingMacro={row['delta_vs_v18_pp']['MovingMacro']:+.6f}",
                          f"  scenes: {scene_summary}"]
                lines += [f"  trajectory: {compact_motion(row['motion_errors'])}"]
                for h, d in row["delta_vs_v18_pp"]["per_horizon"].items():
                    lines += [f"  {h}s dMiOU={d['mIoU']:+.6f} dIoU={d['IoU']:+.6f} dMovingMicro={d['MovingMicro']:+.6f}"]
    lines += [f"\nroute: {summary['route']}", f"elapsed_seconds: {summary['elapsed_seconds']:.2f}",
              "Full4369 includes the reused dev scenes; new_scenes_only excludes ALL frozen dev512 scenes.",
              "Do not use this report to retroactively change original best/checkpoint screen_pass."]
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__); add_config_args(parser)
    for name in ("dev-cache", "population-manifest", "model-dir", "base-checkpoint", "dataroot", "dev-info", "out-dir"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--population", choices=("full4369", "dev512"), default="full4369")
    parser.add_argument("--expected-base-sha256", default=CLEAN_SHA256)
    parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); start = time.perf_counter()
    for name in ("config", "dev_cache", "population_manifest", "base_checkpoint", "dev_info"):
        if not str(getattr(args, name) or "").strip() or not Path(getattr(args, name)).is_file():
            parser.error(f"{name} must be an existing file")
    if not Path(args.dataroot).is_dir() or not Path(args.model_dir).is_dir() or args.cpu_workers < 1:
        parser.error("invalid dataroot/model-dir/cpu-workers")
    out = Path(args.out_dir)
    if out.exists():
        parser.error("choose a NEW out-dir; no experiment/checkpoint overwrite")
    device = torch.device(args.device)
    if device.type == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("CUDA/BF16 unavailable; no silent fallback")
    torch.set_num_threads(1)
    cfg = load_runtime_config(args.config, args.override); pcfg = make_prepare_config(cfg)
    config_sha = stable_json_fingerprint(cfg)
    manifest, dev64, _ = load_manifest(args.population_manifest)
    if manifest["selected_key_fingerprint"] != DEV64_FP:
        raise RuntimeError("wrong frozen dev64 identity/order")
    meta, records = load_cache(args.dev_cache)
    if float(meta.get("patch_resolution_m", .8)) != .8:
        raise RuntimeError("frozen patch-resolution mismatch")
    records, keys, groups = plan_populations(records, manifest, dev64, population=args.population)
    validate_records(records)
    provider = FrozenXYV18(args.base_checkpoint, args.expected_base_sha256, pcfg, device, args.cpu_workers)
    models, aliases, audit, contract, previous = read_bundle(args.model_dir, provider, config_sha, manifest, dev64)
    used_train_scenes = {s for s, _ in ordered_keys(contract["train_keys"])+ordered_keys(contract["calibration_keys"])}
    if {s for s, _ in keys} & used_train_scenes:
        raise RuntimeError("training/expanded-validation scene overlap")
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unavailable"
    out.mkdir(parents=True)
    frozen_groups = {"all": keys, **groups}
    write_json(out/"execution_contract.json", {"protocol": PROTOCOL, "git_commit": git_commit, "arguments": vars(args),
        "base_checkpoint_sha256": provider.sha, "runtime_config_fingerprint": config_sha,
        "checkpoint_audit": audit, "population_keys": frozen_groups,
        "population_fingerprints": {g: stable_json_fingerprint(v) for g, v in frozen_groups.items()},
        "read_only_original_models": True, "reselection": False, "two_frozen_last_candidates_registered_before_evaluation": True})
    reproduced = {}
    with (out/"progress.jsonl").open("x", encoding="utf-8") as handle:
        def progress(row):
            handle.write(json.dumps(finite_json(row), ensure_ascii=False, allow_nan=False)+"\n"); handle.flush()
        def complete(group, report):
            attach_baseline_aliases(report, aliases, frozen_groups[group])
            if group == "dev64_reproduction":
                reproduced.update(check_dev64_reproduction(report, previous))
                print(f"dev64 original result reproduction PASS: {reproduced}", flush=True)
            write_json(out/(group+"_report.json"), report)
            print(f"population_complete={group} windows={report['windows']} scenes={report['scenes']}", flush=True)
            progress({"event": "population_complete", "population": group, "windows": report["windows"], "scenes": report["scenes"]})
        source = NuScenesWindowSource(args.dataroot, info_pkl=args.dev_info, verbose=False)
        report = evaluate_models(provider, source, records, models, None, progress=progress, prediction_fn=predict,
                                 population_groups=groups, population_complete_fn=complete)
    for value in audit.values():
        if sha256(value["path"]) != value["sha256"]:
            raise RuntimeError("original checkpoint modified while evaluation was running")
    subreports = report.pop("populations")
    summary = {"protocol": PROTOCOL, "population": args.population, "windows": len(keys), "scenes": len({s for s, _ in keys}),
               "original_model_dir": str(Path(args.model_dir).resolve()), "git_commit": git_commit,
               "checkpoint_audit": audit, "dev64_reproduction": reproduced,
               "reports": {args.population: report, **subreports}, "elapsed_seconds": time.perf_counter()-start,
               "route": "expanded_validation_complete_no_retraining_or_checkpoint_reselection"}
    write_json(out/"summary.json", summary)
    (out/"summary.txt").write_text(summary_text(summary), encoding="utf-8")
    print(summary_text(summary), flush=True)


if __name__ == "__main__":
    main()
