"""Paired populations, original V18 objective, and unchanged formal renderer."""
from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
import numpy as np
import torch

from real_motion.v18_source_interaction import (INPUT_KEYS, SourceInteractionV18, neighbor_graph,
                                               causal_forward, InteractionConfig, PROTOCOL, ARMS)
from real_motion.v18_motion_gap import MotionErrors, numpy
from real_motion.local_st_world_model_v18_se2 import LocalSpatialTemporalWorldModelV18SE2
from real_motion.local_st_world_model_v17 import config_from_mapping_v17
from real_motion.source_evidence_audit import metric_count_delta, edit_quality
from real_motion.nuscenes_adapter import gt_moving_support_sequence
from real_motion.v18_xy_trajectory import xy_screen_gate
from tools.real_motion.v18_xy_trajectory_common import render_outputs
from tools.real_motion.train_p0_f9_v18_se2_pair import se2_objective_loss
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import Metrics, DYN, delta, assert_forward_exact
from tools.real_motion.eval_p0_f9_source_evidence_audit import _scene_delta
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime

LABEL_KEYS = ("supervised_source", "existence", "target_source_residual_xy_m",
              "target_source_displacement_xy_m", "target_yaw_rad", "se2_target_valid",
              "yaw_label_valid", "yaw_enabled")


def validate_records(records):
    for record in records:
        n = len(record["features"])
        center = torch.as_tensor(record["source_centroid_xy_t0_m"]).float()
        kta = torch.as_tensor(record["kta_displacement_xy_m"]).float()
        anchors = torch.as_tensor(record["anchors_xy_t0_m"]).float()
        if center.shape != (n, 2) or kta.shape != (n, 6, 2) or anchors.shape != (n, 6, 2):
            raise RuntimeError("source/anchor shape mismatch")
        if not torch.allclose(anchors, center[:, None]+kta, rtol=0, atol=2e-4):
            raise RuntimeError("source-center KTA anchor contract mismatch")
        valid = torch.as_tensor(record["se2_target_valid"]).bool()
        residual = torch.as_tensor(record["target_source_residual_xy_m"]).float()
        target = torch.as_tensor(record["target_source_displacement_xy_m"]).float()
        if not torch.allclose((residual+kta)[valid], target[valid], rtol=0, atol=2e-4):
            raise RuntimeError("source-center residual target contract mismatch")
        tube = torch.as_tensor(record["target_source_mask_tube"])
        if tube.shape != (n, 6, 20, 20) or not ((tube == 0) | (tube == 1)).all():
            raise RuntimeError("requires frozen binary source footprint")


def select_population(keys, dev_scenes, *, fraction=.2, calibration_scenes=32, seed=20260930):
    """Exact floor(fraction * FULL train count), round-robin over hashed scenes.

    Within each scene sample uniformly spaced temporal ranks across its ENTIRE
    frozen cache order. Held-out TRAIN calibration uses two ranks per scene.
    Scene hashes use only scene names/seed, never targets or model error.
    """
    keys = tuple((str(a), str(b)) for a, b in keys)
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate train identities")
    grouped = defaultdict(list)
    for key in keys:
        grouped[key[0]].append(key)
    if set(grouped) & set(dev_scenes):
        raise RuntimeError("train/dev scene overlap")
    order = sorted(grouped, key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest())
    if not 0 < fraction < 1 or not 0 < calibration_scenes < len(order):
        raise ValueError("invalid train/calibration split")
    held = order[:calibration_scenes]
    cal = tuple(grouped[s][i] for s in held for i in sorted(set((len(grouped[s])//4, 3*len(grouped[s])//4))))
    order = order[calibration_scenes:]
    count = int(len(keys)*fraction)
    if count < 1 or sum(len(grouped[s]) for s in order) < count:
        raise RuntimeError("insufficient scene-disjoint optimization population")
    allocation = {s: 0 for s in order}
    remaining = count
    while remaining:
        for s in order:
            if allocation[s] < len(grouped[s]):
                allocation[s] += 1
                remaining -= 1
                if remaining == 0:
                    break
    # Midpoint stratification: avoids prefix-only sampling of each scene.
    samples = {s: [grouped[s][int((j+.5)*len(grouped[s])/allocation[s])]
                   for j in range(allocation[s])] for s in order if allocation[s]}
    train = tuple(samples[s][j] for j in range(max(allocation.values())) for s in order
                  if s in samples and j < len(samples[s]))
    if len(train) != count or len(set(train)) != count or {s for s, _ in train} & set(held):
        raise RuntimeError("population selection contract failed")
    return train, cal


def window_batches(records, *, seed, epoch, source_budget=256, window_budget=8):
    """Both arms consume exactly the same ALL sources in each entire window.

    Budgets control packing, not source truncation. A large single window is
    processed alone; empty windows are consumed too and reported in coverage.
    """
    if min(source_budget, window_budget) < 1:
        raise ValueError("invalid packing budget")
    order = np.random.default_rng(seed+epoch).permutation(len(records))
    batch, size = [], 0
    for i in order:
        n = len(records[i]["features"])
        if batch and (size+n > source_budget or len(batch) >= window_budget):
            yield batch
            batch, size = [], 0
        batch.append(int(i)); size += n
    if batch:
        yield batch


def make_batch(records, device, config=InteractionConfig(), *, labels=True):
    """Only explicitly whitelisted causal tensors can enter model forward."""
    required = INPUT_KEYS + ((LABEL_KEYS) if labels else ())
    batch = {key: torch.cat([torch.as_tensor(r[key]) for r in records], dim=0) for key in required}
    graph = neighbor_graph(torch.cat([torch.as_tensor(r["source_centroid_xy_t0_m"]) for r in records]),
                           torch.cat([torch.as_tensor(r["source_class_id"]) for r in records]),
                           batch["frame_motion_features"],
                           torch.cat([torch.full((len(r["features"]),), i) for i, r in enumerate(records)]), config)
    batch.update(graph)
    for key in ("features", "kta_displacement_xy_m", "frame_motion_features", "neighbor_edge"):
        batch[key] = batch[key].float()
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def original_objective(outputs, batch):
    """Original four loss groups/weights; GT masks apply ONLY to supervision.

    Unsup sources stay in causal memory. Missing target padding is sanitized
    before grid_sample arithmetic; finite valid labels fail closed.
    """
    b = dict(batch)
    valid = b["se2_target_valid"].bool() & b["supervised_source"].bool()[:, None]
    b["se2_target_valid"] = valid
    yaw_valid = valid & b["yaw_label_valid"].bool() & b["yaw_enabled"].bool()[:, None]
    b["yaw_label_valid"] = yaw_valid
    for key in ("target_source_residual_xy_m", "target_source_displacement_xy_m", "target_yaw_rad"):
        target = b[key].float()
        mask = yaw_valid if key == "target_yaw_rad" else valid
        if not torch.isfinite(target[mask]).all():
            raise RuntimeError(f"nonfinite valid target: {key}")
        while mask.ndim < target.ndim:
            mask = mask[..., None]
        b[key] = torch.where(mask, target, torch.zeros_like(target))
    b["existence"] = torch.where(b["supervised_source"].bool()[:, None], b["existence"].float(), 0.)
    if not torch.isfinite(b["existence"]).all():
        raise RuntimeError("nonfinite supervised existence label")
    return se2_objective_loss({k: v.float() for k, v in outputs.items()}, b, yaw_weight=19.,
                             shape_weight=.25, patch_resolution_m=.8)


def build_pair(checkpoint, device):
    """Restore common AdamW moments as well as weights; append ONLY new params.

    Clean-E14 used one optimizer group. Fail closed instead of guessing another
    checkpoint's parameter order, and never mutate its tensors/state in place.
    """
    config = config_from_mapping_v17(checkpoint.get("model_config"))
    state = checkpoint.get("optimizer")
    if state is None or len(state["param_groups"]) != 1:
        raise RuntimeError("requires Clean-E14 one-group optimizer state")
    models, optimizers = {}, {}
    for arm, cls in (("v18_continuation", LocalSpatialTemporalWorldModelV18SE2),
                     ("v18_source_interaction", SourceInteractionV18)):
        model = cls(config).to(device)
        inherited = {k: p for k, p in model.named_parameters() if not k.startswith("source_interaction.")}
        if set(inherited) != {k for k, _ in LocalSpatialTemporalWorldModelV18SE2(config).named_parameters()}:
            raise RuntimeError("inherited V18 parameter identity changed")
        missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
        if unexpected or any(not k.startswith("source_interaction.") for k in missing):
            raise RuntimeError("V18 checkpoint/model mismatch")
        if arm == "v18_continuation" and missing:
            raise RuntimeError("incomplete V18 control")
        group = state["param_groups"][0]
        optimizer = torch.optim.AdamW(list(inherited.values()), lr=group["lr"], weight_decay=group["weight_decay"])
        optimizer.load_state_dict(copy.deepcopy(state))
        if arm == "v18_source_interaction":
            optimizer.add_param_group({"params": list(model.source_interaction.parameters()),
                                       **{k: v for k, v in optimizer.param_groups[0].items() if k != "params"}})
        model.requires_grad_(True)
        models[arm], optimizers[arm] = model, optimizer
    return models, optimizers


def predict(model, record, device, config):
    device = torch.device(device)
    model.eval()
    batch = make_batch([record], device, config, labels=False)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = causal_forward(model, batch)
    if any(not torch.isfinite(v).all() for v in out.values()):
        raise RuntimeError("nonfinite full-source deployed prediction")
    return out


def evaluate_models(provider, source, records, models, config, *, progress=None, prediction_fn=None):
    """Shared raw/Strong/base/support pass for all selected AND last candidates.

    Optional prediction_fn(model, record, device, config, base) lets an XY-only
    candidate reuse the frozen per-window baseline output, without repeating
    its forward or changing the existing full-model evaluation path.
    """
    names = ("V18_BASE", *models)
    metrics = {k: Metrics() for k in names}
    scenes = defaultdict(lambda: {k: Metrics() for k in names})
    errors = {k: MotionErrors() for k in names}
    quality = {k: defaultdict(int) for k in models}
    for wi, rec in enumerate(records, 1):
        print(f"paired_eval={wi}/{len(records)} variants={len(models)}", flush=True)
        window, raw, state, base = provider.prepare(source, rec, include_gt=True)
        if wi == 1:
            runtime._stage_gpu_inputs(state, provider.device)
            try:
                assert_forward_exact(provider.model, state, provider.device)
                runtime._exactness_check(provider.model, state, provider.pcfg, provider.strong, provider.device)
            finally:
                runtime._release_gpu_inputs(state)
        baseline = runtime._forecast_once(provider.model, state, provider.pcfg, provider.strong,
                                          provider.device, precomputed_out=base)
        if wi == 1 and any(not np.array_equal(p, baseline[h]) for p, h in
                            zip(render_outputs(state, provider.pcfg, base), (1, 3, 5))):
            raise RuntimeError("original-output compositor not voxel-exact V18")
        outputs = {"V18_BASE": base, **{
            k: (predict(m, rec, provider.device, config) if prediction_fn is None
                else prediction_fn(m, rec, provider.device, config, base))
            for k, m in models.items()}}
        moving = gt_moving_support_sequence(source.nusc, window.t0_token, window.future_tokens,
                 tuple(.5*(h+1) for h in range(6)), grid=provider.pcfg.grid, workers=provider.workers)
        before = [Metrics.counts(baseline[h], raw["future_gt_occ"][h], moving[h][0], 17) for h in (1, 3, 5)]
        for name, out in outputs.items():
            errors[name].update(rec, out)
            pred = [baseline[h] for h in (1, 3, 5)] if name == "V18_BASE" else render_outputs(state, provider.pcfg, out)
            for ri, h in enumerate((1, 3, 5)):
                gt, b = raw["future_gt_occ"][h], baseline[h]
                counts = metric_count_delta(before[ri], b, pred[ri], gt, moving[h][0], DYN)
                if wi == 1 and any(not np.array_equal(a, z) for a, z in
                        zip(counts, Metrics.counts(pred[ri], gt, moving[h][0], 17))):
                    raise RuntimeError("changed-cell/full-grid metric mismatch")
                metrics[name].update(ri, counts=counts)
                scenes[window.scene_name][name].update(ri, counts=counts)
                if name in quality:
                    for key, value in edit_quality(b, pred[ri], gt).items():
                        quality[name][key] += value
        if progress:
            progress({"event": "evaluation", "window": wi, "windows": len(records)})
    baseline = metrics["V18_BASE"].compute()
    reports = {}
    for name in models:
        report = {"metrics": metrics[name].compute(), "delta_vs_v18_pp": delta(metrics[name].compute(), baseline),
                  "scene_delta": _scene_delta(scenes, name), "motion_errors": errors[name].compute(),
                  "edit_quality": dict(quality[name])}
        report["gate"] = xy_screen_gate(report)
        reports[name] = report
    return {"baseline": baseline, "baseline_motion_errors": errors["V18_BASE"].compute(), "variants": reports}


def forecast(provider, source, record, model, config=InteractionConfig()):
    """Formal six-horizon deployment uses history only, NEVER future GT."""
    window, _, state, _ = provider.prepare(source, record, include_gt=False)
    out = predict(model, record, provider.device, config)
    return window, runtime._forecast_once(provider.model, state, provider.pcfg, provider.strong,
                                          provider.device, precomputed_out=out)


def load_candidate(path, device, *, base_sha, config_sha, allow_failed_diagnostic=False):
    """New full-model format cannot masquerade as the original frozen V18."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if (ck.get("protocol") != PROTOCOL or ck.get("arm") not in ARMS
            or ck.get("base_checkpoint_sha256") != base_sha or ck.get("runtime_config_fingerprint") != config_sha
            or ck.get("checkpoint_role") not in ("selected_full_v18_candidate", "last_full_v18_diagnostic")):
        raise RuntimeError("full V18 candidate checkpoint contract mismatch")
    if (ck.get("mode") != "screen" or not ck.get("screen_pass") or ck.get("selected_update", 0) <= 0
            or ck.get("checkpoint_role") != "selected_full_v18_candidate") and not allow_failed_diagnostic:
        raise RuntimeError("failed/smoke/update0/last candidate cannot be deployed")
    cls = LocalSpatialTemporalWorldModelV18SE2 if ck["arm"] == ARMS[0] else SourceInteractionV18
    model = cls(config_from_mapping_v17(ck["model_config"])).to(device)
    model.load_state_dict(ck["state_dict"], strict=True); model.eval()
    return ck, model, InteractionConfig(**ck["interaction_config"])
