"""Original V18 XY specialization with immutable Clean-E14 yaw/existence.

GT yaw is a training-only geometry curriculum, NEVER a forward input. This is
not autoregressive teacher forcing: only the pose in the shape loss is mixed.
The deployable model is a two-forward composition, not a single frozen-head net.
"""
from __future__ import annotations

import copy
import math
import torch
import torch.nn.functional as F

from real_motion.local_st_world_model_v17 import config_from_mapping_v17
from real_motion.local_st_world_model_v18_se2 import (
    LocalSpatialTemporalWorldModelV18SE2, soft_se2_transport_overlap_loss,
)
from real_motion.v18_xy_trajectory import replace_xy_outputs
from tools.real_motion.v18_source_interaction_common import window_batches
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime

PROTOCOL = "p0_f9_v18_xy_specialist_frozen_yaw_paired20_v1"
ARMS = ("frozen_yaw_xy", "scheduled_gt_yaw_xy")
INPUT_KEYS = ("features", "local_semantic_tube", "kta_displacement_xy_m",
              "frame_motion_features", "target_source_mask_tube")
LABEL_KEYS = ("supervised_source", "se2_target_valid", "target_source_residual_xy_m",
              "target_source_displacement_xy_m", "target_yaw_rad", "yaw_enabled", "yaw_label_valid")
FROZEN_PREFIXES = ("yaw_head.", "existence_head.")


def record_key(record):
    return str(record["scene_name"]), str(record["t0_token"])


def make_batch(records, device, *, labels=True):
    # No neighbor graph, GT validity pruning, or label-dependent forward.
    keys = INPUT_KEYS + (LABEL_KEYS if labels else ())
    batch = {key: torch.cat([torch.as_tensor(r[key]) for r in records], dim=0) for key in keys}
    for key in ("features", "kta_displacement_xy_m", "frame_motion_features"):
        batch[key] = batch[key].float()
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def forward_xy(model, batch):
    latent = model(*[batch[k] for k in INPUT_KEYS], return_latents=True, decode_outputs=False)
    return model.residual_head(latent["future_transport_queries"])


def build_pair(checkpoint, device):
    """Restore Clean-E14 weights AND AdamW moments, never a fresh high-LR run.

    Retain the original parameter ordering/group to load moments exactly. The
    two frozen heads stay in the group but have grad=None, so AdamW cannot even
    weight-decay them. Their outputs are still discarded at deployment.
    """
    state = checkpoint.get("optimizer")
    if state is None or len(state["param_groups"]) != 1:
        raise RuntimeError("requires Clean-E14 one-group optimizer state")
    group = state["param_groups"][0]
    config = config_from_mapping_v17(checkpoint.get("model_config"))
    models, optimizers = {}, {}
    for arm in ARMS:
        model = LocalSpatialTemporalWorldModelV18SE2(config).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        if len(group["params"]) != len(list(model.parameters())):
            raise RuntimeError("Clean-E14 optimizer parameter order/count mismatch")
        optimizer = torch.optim.AdamW(model.parameters(), lr=group["lr"], weight_decay=group["weight_decay"])
        optimizer.load_state_dict(copy.deepcopy(state))
        model.requires_grad_(True)
        model.yaw_head.requires_grad_(False)
        model.existence_head.requires_grad_(False)
        models[arm], optimizers[arm] = model, optimizer
    return models, optimizers


def assert_frozen_heads(model, reference_state):
    for name, value in model.state_dict().items():
        if name.startswith(FROZEN_PREFIXES) and not torch.equal(value.detach().cpu(), reference_state[name].cpu()):
            raise RuntimeError(f"frozen head parameter changed: {name}")
    if any(p.requires_grad or p.grad is not None for name, p in model.named_parameters()
           if name.startswith(FROZEN_PREFIXES)):
        raise RuntimeError("frozen head received gradients")


def cache_baseline_outputs(provider, records, *, progress=None, max_mib=64):
    """One per-window deployed BF16 forward; only N*6*2 scalars kept in RAM.

    No latent bank, on-disk cache, future GT, raw-data read or mixed-window
    teacher forward. Convert inference tensors outside inference_mode before
    they participate in autograd loss geometry.
    """
    cache, size = {}, 0
    for wi, rec in enumerate(records, 1):
        key = record_key(rec)
        if key in cache:
            raise RuntimeError("duplicate teacher-cache identity")
        out = provider.encode_record(rec)
        row = {k: out[k].detach().cpu().float().clone() for k in ("yaw_delta_rad", "existence_logits")}
        if any(v.shape != (len(rec["features"]), 6) or not torch.isfinite(v).all() for v in row.values()):
            raise RuntimeError("invalid frozen teacher output")
        size += sum(v.numel()*v.element_size() for v in row.values())
        if size > max_mib*2**20:
            raise RuntimeError("compact teacher RAM budget exceeded")
        cache[key] = row
        if wi == 1 or wi % 256 == 0 or wi == len(records):
            print(f"frozen_yaw_cache={wi}/{len(records)} mib={size/2**20:.2f}", flush=True)
            if progress:
                progress({"event": "frozen_teacher_cache", "window": wi, "windows": len(records), "mib": size/2**20})
    return cache, size/2**20


def teacher_probability(completed_updates, total_updates, initial=.5):
    """Linear .5 -> 0 in first 2/3 of updates; final >=1/3 baseline yaw only."""
    if total_updates < 1 or not 0 <= completed_updates < total_updates or not 0 <= initial <= 1:
        raise ValueError("invalid teacher curriculum budget")
    decay_updates = max(1, math.floor(2*total_updates/3))
    if decay_updates <= 1:
        return 0.
    return float(initial) * max(0., 1. - completed_updates/(decay_updates-1))


def xy_objective(pred_xy, batch, baseline_yaw, *, teacher_prob=0., seed=0):
    """SmoothL1 + .25 soft SE(2), only XY receives gradients.

    Both arms share the exact source-centre XY targets. Only valid yaw-enabled
    source/horizons can select GT yaw for the predicted shape; GT target shape
    always remains the ordinary supervision. No existence/yaw-head loss.
    """
    if not 0 <= teacher_prob <= 1:
        raise ValueError("invalid GT-yaw probability")
    valid = batch["se2_target_valid"].bool() & batch["supervised_source"].bool()[:, None]
    if pred_xy.shape != (*valid.shape, 2) or baseline_yaw.shape != valid.shape:
        raise ValueError("XY/frozen-yaw shape mismatch")
    if not torch.isfinite(pred_xy).all() or not torch.isfinite(baseline_yaw).all():
        raise RuntimeError("nonfinite XY/frozen-yaw prediction")
    yaw_valid = valid & batch["yaw_enabled"].bool()[:, None] & batch["yaw_label_valid"].bool()
    targets = {}
    for key, mask in (("target_source_residual_xy_m", valid),
                      ("target_source_displacement_xy_m", valid), ("target_yaw_rad", yaw_valid)):
        value = batch[key].float()
        if not torch.isfinite(value[mask]).all():
            raise RuntimeError(f"nonfinite valid target: {key}")
        use = mask if value.ndim == mask.ndim else mask[..., None]
        targets[key] = torch.where(use, value, torch.zeros_like(value))
    pred_xy = pred_xy.float()
    xy = (F.smooth_l1_loss(pred_xy[valid], targets["target_source_residual_xy_m"][valid], beta=1.)
          if bool(valid.any()) else pred_xy.sum()*0.)
    # CPU RNG is deterministic across CPU/CUDA; the mask affects LOSS ONLY.
    rng = torch.Generator().manual_seed(seed)
    chosen = (torch.rand(valid.shape, generator=rng) < teacher_prob).to(valid.device) & yaw_valid
    pose_yaw = torch.where(chosen, targets["target_yaw_rad"], baseline_yaw.detach().float())
    shape, _ = soft_se2_transport_overlap_loss(
        batch["kta_displacement_xy_m"].float()+pred_xy, targets["target_source_displacement_xy_m"],
        pose_yaw, targets["target_yaw_rad"], batch["target_source_mask_tube"][:, -1].float(),
        valid, batch["yaw_enabled"], yaw_valid, patch_resolution_m=.8, materialize_stats=False)
    loss = xy+.25*shape
    count, yaw_count = int(valid.sum()), int(yaw_valid.sum())
    return loss, {"objective_loss": float(loss.detach()), "translation_smooth_l1": float(xy.detach()),
                  "se2_shape_loss": float(shape.detach()), "valid_xy_labels": count,
                  "teacher_eligible_yaw_labels": yaw_count, "teacher_gt_yaw_labels": int(chosen.sum()),
                  "teacher_probability": teacher_prob,
                  "teacher_fraction": int(chosen.sum())/yaw_count if yaw_count else 0.}


def epoch_batches(records, *, seed, epoch, source_budget, window_budget):
    # Count actual supervised updates before scheduling. Invalid/empty windows
    # are reported separately, not allowed to prevent the final zero-GT phase.
    batches = list(window_batches(records, seed=seed, epoch=epoch,
                                  source_budget=source_budget, window_budget=window_budget))
    usable, empty = [], []
    for ids in batches:
        (usable if any(bool((torch.as_tensor(records[i]["se2_target_valid"]).bool()
                       & torch.as_tensor(records[i]["supervised_source"]).bool()[:, None]).any())
                       for i in ids) else empty).append(ids)
    return usable, empty


def train_epoch(models, optimizers, records, teacher_cache, device, *, batches,
                epoch, updates, total_updates, seed, progress=None):
    totals = {arm: {} for arm in ARMS}
    for step, ids in enumerate(batches, 1):
        rows = [records[i] for i in ids]
        batch = make_batch(rows, device)
        yaw = torch.cat([teacher_cache[record_key(r)]["yaw_delta_rad"] for r in rows]).to(device)
        probability = teacher_probability(updates, total_updates)
        for arm in ARMS:
            model, optimizer = models[arm], optimizers[arm]
            model.train(); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                xy = forward_xy(model, batch)
            loss, parts = xy_objective(xy, batch, yaw,
                teacher_prob=probability if arm == ARMS[1] else 0., seed=seed+updates)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite XY specialist loss: {arm}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            row = {"event": "train", "arm": arm, "epoch": epoch, "step": step, "steps": len(batches),
                   "update": updates+1, "windows": len(rows), "sources": len(xy), "grad_norm": float(norm), **parts}
            for key in ("objective_loss", "translation_smooth_l1", "se2_shape_loss", "teacher_fraction"):
                totals[arm][key] = totals[arm].get(key, 0.)+row[key]
            if progress:
                progress(row)
            if step == 1 or step % 32 == 0 or step == len(batches):
                print(f"arm={arm} epoch={epoch} step={step}/{len(batches)} update={updates+1} "
                      f"loss={parts['objective_loss']:.6f} xy={parts['translation_smooth_l1']:.6f} "
                      f"shape={parts['se2_shape_loss']:.6f} gt_yaw={parts['teacher_fraction']:.3%}", flush=True)
        updates += 1
    if not batches:
        raise RuntimeError("epoch had no supervised XY updates")
    return updates, {arm: {k: v/len(batches) for k, v in row.items()} for arm, row in totals.items()}


def predict(model, record, device, config=None, base=None):
    """Inference accepts only causal record inputs + frozen BASE predictions.

    Evaluation shares its already-computed baseline forward. GT fields, even
    if present in an evaluator record, cannot reach the network/compositor.
    """
    if base is None:
        raise RuntimeError("requires frozen Clean-E14 predictions, not this model's yaw")
    device = torch.device(device)
    model.eval()
    batch = make_batch([record], device, labels=False)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = replace_xy_outputs(base, forward_xy(model, batch))
    if (out["yaw_delta_rad"] is not base["yaw_delta_rad"]
            or out["existence_logits"] is not base["existence_logits"]):
        raise RuntimeError("XY specialist changed frozen yaw/existence")
    return out


def forecast(provider, source, record, model):
    window, _, state, base = provider.prepare(source, record, include_gt=False)
    out = predict(model, record, provider.device, base=base)
    return window, runtime._forecast_once(provider.model, state, provider.pcfg, provider.strong,
                                         provider.device, precomputed_out=out)


def load_candidate(path, device, *, base_sha, config_sha, allow_failed_diagnostic=False):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if (ck.get("protocol") != PROTOCOL or ck.get("arm") not in ARMS
            or ck.get("base_checkpoint_sha256") != base_sha or ck.get("runtime_config_fingerprint") != config_sha
            or ck.get("deployment_contract") != "specialist_XY_plus_frozen_CleanE14_yaw_existence_two_forward_v1"
            or ck.get("checkpoint_role") not in ("selected_xy_specialist_candidate", "last_xy_specialist_diagnostic")):
        raise RuntimeError("XY specialist checkpoint contract mismatch")
    if (ck.get("mode") != "screen" or not ck.get("screen_pass") or ck.get("selected_update", 0) <= 0
            or ck.get("checkpoint_role") != "selected_xy_specialist_candidate") and not allow_failed_diagnostic:
        raise RuntimeError("failed/smoke/update0/last candidate cannot be deployed")
    model = LocalSpatialTemporalWorldModelV18SE2(config_from_mapping_v17(ck["model_config"])).to(device)
    model.load_state_dict(ck["state_dict"], strict=True); model.eval().requires_grad_(False)
    return ck, model
