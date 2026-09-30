#!/usr/bin/env python3
"""One-pass position/yaw/supervision gap audit; no training or shape changes."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from real_motion.nuscenes_adapter import NuScenesWindowSource, gt_moving_support_sequence
from real_motion.prepared import load_nuscenes_window_raw
from real_motion.motion_transport import world_points_to_t0
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.source_evidence_audit import metric_count_delta, edit_quality
from real_motion.v18_motion_gap import PROTOCOL, VARIANTS, MotionErrors, numpy, motion_states, interaction_decomposition
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
from tools.real_motion import eval_p0_f9_v18_full_validation as full
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.eval_p0_f9_source_evidence_audit import _render_current, _scene_delta, finite_json, DYN, REPORT
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import (
    Metrics, load_manifest, align_records, assert_forward_exact, validate_clean_e14_checkpoint, delta, sha256)
from tools.real_motion.train_p0_f9_v18_se2_clean import PROTOCOL as CLEAN_PROTOCOL

CLEAN_SHA = "ffaa54a76d07b4581e68046508d841bac798396533155162a5223ba155b3af73"


def check_reference(result, reference):
    """Optional same-population reproduction against the already run audit."""
    if reference.get("protocol") != "p0_f9_source_evidence_audit_v1": raise RuntimeError("wrong reference audit protocol")
    for key in ("checkpoint_sha256", "selected_key_fingerprint", "config_sha256"):
        if result[key] != reference[key]: raise RuntimeError(f"reference identity mismatch: {key}")
    if result["arguments"]["override"] != reference["arguments"]["override"]:
        raise RuntimeError("reference runtime overrides differ")
    rows = {}
    for name, old in (("V18_BASE", "V18_BASE"), ("GT_XY_GT_YAW", "T0_GT_MOTION")):
        for metric in ("IoU", "mIoU", "MovingMacro", "MovingMicro"):
            a, b = result["variants"][name]["metrics"][metric], reference["variants"][old]["metrics"][metric]
            if not np.isclose(a, b, rtol=0, atol=1e-8, equal_nan=True):
                raise RuntimeError(f"previous {old} {metric} not reproduced: {a} != {b}")
            rows[f"{old}/{metric}"] = float(a-b)
    return {"checked": True, "exact_within_1e_8": True, "difference_pp": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__); add_config_args(p)
    for name in ("val-cache", "population-manifest", "checkpoint", "dataroot", "info-pkl", "out-dir"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--expected-checkpoint-sha256", default=CLEAN_SHA)
    p.add_argument("--reference-audit", help="optional actual path to previous source-evidence audit.json")
    p.add_argument("--device", default="cuda"); p.add_argument("--cpu-workers", type=int, default=8)
    a = p.parse_args(); started = time.perf_counter()
    if a.cpu_workers < 1: p.error("cpu-workers must be positive")
    for name in ("config", "val_cache", "population_manifest", "checkpoint", "info_pkl"):
        if not str(getattr(a, name) or "").strip() or not Path(getattr(a, name)).is_file(): p.error(f"{name} must name an existing file")
    if a.reference_audit and not Path(a.reference_audit).is_file(): p.error("reference-audit must name an existing file")
    if not Path(a.dataroot).is_dir(): p.error("dataroot must name an existing directory")
    out = Path(a.out_dir)
    if out.exists(): p.error("out-dir exists; no overwrite")
    torch.set_num_threads(1); device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; no silent fallback")
    pcfg = make_prepare_config(load_runtime_config(a.config, a.override))
    if pcfg.free_label != 17 or pcfg.frame_dt_s != .5: raise RuntimeError("frozen free-label/frame-dt mismatch")
    manifest, keys, _ = load_manifest(a.population_manifest)
    if len(keys) != 64 or len(manifest["parent_keys"]) != 512: raise RuntimeError("requires frozen dev64 from dev512")
    print("loading V18 cache and frozen checkpoint", flush=True)
    _, records = load_cache(a.val_cache); records = align_records(records, keys)
    ck, model, _ = full._load_model(a.checkpoint, CLEAN_PROTOCOL, device)
    checkpoint_sha = validate_clean_e14_checkpoint(ck, a.checkpoint, a.expected_checkpoint_sha256)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    source = NuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)
    strong = StrongW2DetConfig(free_label=17)
    states = {k: Metrics() for k in VARIANTS}
    scenes = defaultdict(lambda: {k: Metrics() for k in VARIANTS})
    quality = {k: defaultdict(int) for k in VARIANTS}
    errors, audit, timing = MotionErrors(), defaultdict(int), defaultdict(float)
    out.mkdir(parents=True)
    with (out / "progress.jsonl").open("x", encoding="utf-8") as progress:
        for wi, record in enumerate(records, 1):
            tick = time.perf_counter(); print(f"motion_gap={wi}/{len(records)} stage=raw_load", flush=True)
            window = window_from_record(record)
            raw = load_nuscenes_window_raw(source, window, pcfg, include_gt=True, io_workers=a.cpu_workers)
            timing["raw_load"] += time.perf_counter()-tick; tick = time.perf_counter()
            state = runtime._prepare_record(record, source, pcfg, strong, device, raw_window=raw)
            actual_center = world_points_to_t0(np.asarray([c["centroid_world"] for c in state["current"]]).reshape(-1, 3), state["current_pose"])[:, :2]
            if not np.allclose(actual_center, numpy(record["source_centroid_xy_t0_m"]), rtol=0, atol=2e-4):
                raise RuntimeError("cached source centre/order differs from actual Strong sources")
            runtime._stage_gpu_inputs(state, device)
            try:
                if wi == 1:
                    assert_forward_exact(model, state, device); runtime._exactness_check(model, state, pcfg, strong, device)
                outputs = runtime._model_forward(model, state["gpu"], device)
                baseline = runtime._forecast_once(model, state, pcfg, strong, device, precomputed_out=outputs)
            finally: runtime._release_gpu_inputs(state)
            timing["v18_prepare_forecast"] += time.perf_counter()-tick; tick = time.perf_counter()
            interventions = motion_states(record, outputs); errors.update(record, outputs)
            rendered = {}
            for name, (center, yaw) in interventions.items():
                centers = {h: [runtime._target_world_from_xy_cached(xy, state["source_z_t0"][i], state["current_pose"])
                               for i, xy in enumerate(center[:, h])] for h in REPORT}
                yaws = {h: yaw[:, h] for h in REPORT}
                rendered[name], _ = _render_current(state, pcfg, centers, yaws, audit=audit, condition=name)
            if any(not np.array_equal(rendered["V18_BASE"][ri], baseline[h]) for ri, h in enumerate(REPORT)):
                raise RuntimeError("zero intervention is not voxel-exact V18")
            valid = numpy(record["se2_target_valid"]).astype(bool)
            supervised = numpy(record["supervised_source"]).astype(bool)
            audit["source_occurrences"] += len(supervised)
            audit["supervised_source_occurrences"] += int(supervised.sum())
            audit["gt_xy_valid_source_horizons"] += int(valid[:, REPORT].sum())
            audit["gt_xy_unavailable_kept_v18_source_horizons"] += int((~valid[:, REPORT]).sum())
            audit["gt_xy_valid_unsupervised_source_horizons"] += int((valid[:, REPORT] & ~supervised[:, None]).sum())
            audit["gt_yaw_valid_enabled_source_horizons"] += int((valid & numpy(record["yaw_label_valid"]).astype(bool)
                & numpy(record["yaw_enabled"]).astype(bool)[:, None])[:, REPORT].sum())
            timing["intervention_render"] += time.perf_counter()-tick; tick = time.perf_counter()
            moving = gt_moving_support_sequence(source.nusc, window.t0_token, window.future_tokens,
                tuple(.5*(i+1) for i in range(6)), grid=pcfg.grid, workers=a.cpu_workers)
            timing["moving_support"] += time.perf_counter()-tick; tick = time.perf_counter()
            for ri, h in enumerate(REPORT):
                gt = raw["future_gt_occ"][h]; base = baseline[h]
                before = Metrics.counts(base, gt, moving[h][0], 17)
                for name in VARIANTS:
                    pred = rendered[name][ri]
                    counts = metric_count_delta(before, base, pred, gt, moving[h][0], DYN)
                    if wi == 1 and any(not np.array_equal(x, y) for x, y in zip(counts, Metrics.counts(pred, gt, moving[h][0], 17))):
                        raise RuntimeError("incremental/full metric mismatch")
                    states[name].update(ri, counts=counts); scenes[window.scene_name][name].update(ri, counts=counts)
                    for k, v in edit_quality(base, pred, gt).items(): quality[name][k] += v
            timing["metrics"] += time.perf_counter()-tick
            row = {"window": wi, "windows": len(records), "key": [window.scene_name, window.t0_token],
                   "elapsed_seconds": time.perf_counter()-started}
            progress.write(json.dumps(row) + "\n"); progress.flush()
            print(json.dumps(row), flush=True)
    b = states["V18_BASE"].compute()
    reports = {k: {"metrics": st.compute(), "delta_vs_v18_pp": delta(st.compute(), b),
                  "scene_delta": _scene_delta(scenes, k), "edit_quality": dict(quality[k])} for k, st in states.items()}
    try: commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError): commit = None
    result = {"protocol": PROTOCOL, "analysis_only": True, "training_performed": False,
        "windows": len(keys), "scenes": len(scenes), "report_horizons_s": [1., 2., 3.], "arguments": vars(a),
        "git_commit": commit, "checkpoint_sha256": checkpoint_sha, "selected_key_fingerprint": manifest["selected_key_fingerprint"],
        "config_sha256": sha256(a.config), "variants": reports,
        "decomposition": {k: interaction_decomposition(reports, k) for k in ("mIoU", "MovingMicro")},
        "motion_errors": errors.compute(), "source_audit": dict(audit),
        "contracts": {"xy": "absolute observed source-centroid destination, not annotation box centre",
            "yaw": "frozen semantic enable mask; missing GT retains V18",
            "supervised_only": "GT intervention restricted by original cache supervised_source, not new inference gating",
            "geometry": "same t0 sources/order/pivot/Z/shape; original A1 CLEAR/WRITE",
            "interaction": "2x2 diagnostic; XY and yaw gains are not independently additive",
            "scope": "observed t0 sources only; no births, future shape, GT survival, learned checkpoint or selector"},
        "performance": {"seconds_total": time.perf_counter()-started, "seconds_by_stage": dict(timing)}}
    result["reference_check"] = check_reference(result, json.loads(Path(a.reference_audit).read_text(encoding="utf-8"))) if a.reference_audit else {"checked": False}
    (out / "audit.json").write_text(json.dumps(finite_json(result), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"completed: {out / 'audit.json'}; no training/checkpoint modification", flush=True)


if __name__ == "__main__": main()
