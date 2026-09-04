#!/usr/bin/env python3
"""Deployment-controlled P0-F9 sparse-geometry and safe-fusion diagnostic.

No new training is performed. The evaluator reuses the exact 128-window P0-F9
validation protocol and tests whether dense/frozen/trained World-Model proposals
contain useful motion innovation once destructive CLEAR authority is removed.

Proposal sources:
- released dense OccFM;
- frozen Top-2 20x20 OccFM, official CFG=2;
- frozen Top-2 20x20 OccFM, P0-F9 CFG=1;
- optional trained P0-F9 checkpoint (EMA by default);
- GT oracle.

Every proposal is evaluated on the same frozen MSP support with three rules:
1. takeover: clear anchor dynamics, then write proposal dynamics;
2. write_only: preserve existing anchor dynamics bit-exact, add only missing
   proposal dynamics on non-dynamic anchor voxels;
3. dynamic_union: never clear anchor dynamic presence, but allow proposal
   dynamic semantics to add or relabel.

Legacy takeover metric names are preserved for direct comparison with previous
P0-F9 reports.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "upstream_occfm"
sys.path[:0] = [str(ROOT), str(UP)]

import numpy as np
import torch

from real_motion.checkpoint import load_shape_safe, require_checkpoint_reuse
from real_motion.context import crop_prediction_and_context
from real_motion.metrics.moving_miou_v2 import (
    DYNAMIC_CLASS_IDS,
    MovingMIoUV2MultiHorizon,
    REPORT_HORIZONS_S,
    SemanticIoUAccumulator,
)
from real_motion.models.p0_f9 import P0_F9_PROTOCOL, make_p0_f9_model
from real_motion.msp import latent_support_to_bev
from real_motion.msp_wm_cache import MSP_WM_CACHE_VERSION_V2, MSPWorldModelCacheDataset
from real_motion.native_forecast import crop_coherent_source_noise, deterministic_sample_seed
from real_motion.occfm_io import (
    OccFMVAEAdapter,
    file_sha256,
    load_occfm_config,
    load_official_vae,
    load_official_wm,
    run_frozen_occfm_forecast,
)
from real_motion.repair_target import (
    apply_dynamic_repair,
    apply_dynamic_union,
    apply_dynamic_write_only,
)
from real_motion.windows import WindowPlan, crop_windows, scatter_windows
from tools.real_motion.build_p0_f9_cache_fast import P0_F9_CACHE_PROTOCOL

REPORT = {1.0: 1, 2.0: 3, 3.0: 5}
PRED_HW = (20, 20)
CONTEXT_HW = (40, 40)
FULL_HW = (50, 50)
FREE = 17
HIST_LAST = 4
OFFICIAL_CFG = 2.0
P0_F9_CFG = 1.0
SOURCE_SPATIAL_CONTRACT = "one_global_gaussian_field_cropped_into_top2_windows"
PROTOCOL = "p0_f9_safe_fusion_ablation_v4"
FUSIONS = ("takeover", "write_only", "dynamic_union")


def _new_metrics():
    return {
        "overall": {h: SemanticIoUAccumulator() for h in REPORT_HORIZONS_S},
        "moving": MovingMIoUV2MultiHorizon(),
    }


def _update(state, horizon, pred, gt, moving_support):
    state["overall"][horizon].update(pred, gt)
    state["moving"].update(horizon, pred, gt, moving_support)


def _report(state):
    by_h = {h: state["overall"][h].compute() for h in REPORT_HORIZONS_S}
    return {
        "overall": {
            "mIoU": float(np.nanmean([by_h[h]["mIoU"] for h in REPORT_HORIZONS_S])),
            "per_horizon": by_h,
        },
        "moving": state["moving"].compute(),
    }


def _metric_pair(report):
    return float(report["overall"]["mIoU"]), float(report["moving"]["mIoU"])


def _delta(a, b):
    ao, am = _metric_pair(a)
    bo, bm = _metric_pair(b)
    return {"overall": ao - bo, "moving": am - bm}


def _apply_fusion(mode, anchor, proposal, write_bev):
    if mode == "takeover":
        return apply_dynamic_repair(
            anchor,
            proposal,
            write_bev,
            dynamic_class_ids=DYNAMIC_CLASS_IDS,
            free_label=FREE,
        )
    if mode == "write_only":
        return apply_dynamic_write_only(
            anchor,
            proposal,
            write_bev,
            dynamic_class_ids=DYNAMIC_CLASS_IDS,
        )
    if mode == "dynamic_union":
        return apply_dynamic_union(
            anchor,
            proposal,
            write_bev,
            dynamic_class_ids=DYNAMIC_CLASS_IDS,
        )
    raise ValueError(f"unknown fusion mode: {mode}")


def _source_state_names(source):
    mappings = {
        "dense_official": {
            "takeover": "dense_official_same_support_fusion",
            "write_only": "dense_official_write_only",
            "dynamic_union": "dense_official_dynamic_union",
        },
        "frozen_sparse_official_cfg": {
            "takeover": "frozen_sparse_official_cfg",
            "write_only": "frozen_sparse_official_cfg_write_only",
            "dynamic_union": "frozen_sparse_official_cfg_dynamic_union",
        },
        "frozen_sparse_p0f9_cfg": {
            "takeover": "frozen_sparse_p0f9_cfg",
            "write_only": "frozen_sparse_p0f9_cfg_write_only",
            "dynamic_union": "frozen_sparse_p0f9_cfg_dynamic_union",
        },
        "trained_p0f9": {
            "takeover": "trained_p0f9_takeover",
            "write_only": "trained_p0f9_write_only",
            "dynamic_union": "trained_p0f9_dynamic_union",
        },
        "gt": {
            "takeover": "same_support_gt_oracle",
            "write_only": "same_support_gt_write_only_oracle",
            "dynamic_union": "same_support_gt_dynamic_union_oracle",
        },
    }
    if source not in mappings:
        raise ValueError(source)
    return mappings[source]


def _new_source_states(source):
    return {name: _new_metrics() for name in _source_state_names(source).values()}


def _update_source_fusions(states, source, horizon, anchor, proposal, gt, moving, write_bev):
    names = _source_state_names(source)
    outputs = {}
    for mode in FUSIONS:
        fused = _apply_fusion(mode, anchor, proposal, write_bev)
        outputs[mode] = fused
        _update(states[names[mode]], horizon, fused, gt, moving)
    return outputs


def _official_seeded_noise_like(x: torch.Tensor, seed: int) -> torch.Tensor:
    """Match the first torch.randn_like draw used by official OccFM cfm_eval."""
    if x.device.type != "cuda":
        gen = torch.Generator(device=x.device)
        gen.manual_seed(int(seed))
        return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen)
    cuda_devices = [x.device.index or 0]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        return torch.randn_like(x)


def _validate_cache(ds: MSPWorldModelCacheDataset, vae_ckpt: str) -> None:
    if ds.version != MSP_WM_CACHE_VERSION_V2:
        raise RuntimeError("safe-fusion diagnostic requires a v2 absolute-future cache")
    meta = ds.metadata
    if meta.get("protocol") != P0_F9_CACHE_PROTOCOL:
        raise RuntimeError("cache is not audited P0-F9 v2")
    if meta.get("vae_mode") != "sample":
        raise RuntimeError("diagnostic requires posterior-sampled native latents")
    if int(meta.get("native_backbone_hist_last", -1)) != HIST_LAST:
        raise RuntimeError("diagnostic requires native HIST_LAST=4 provenance")
    if int(meta.get("topk", -1)) != 2 or list(meta.get("window_hw", [])) != [20, 20]:
        raise RuntimeError("diagnostic requires frozen Top-2 20x20 routing")
    if meta.get("anchor_contract") != "strong_w2det_occ_only_v1":
        raise RuntimeError("diagnostic requires Strong-W2Det anchor cache")
    if not bool(meta.get("include_eval_payload", False)):
        raise RuntimeError("diagnostic requires validation eval payload")
    if file_sha256(vae_ckpt) != meta.get("vae_checkpoint_sha256"):
        raise RuntimeError("VAE checkpoint differs from cache provenance")


def _assert_new_conditioning_is_noop(model) -> None:
    tr = model.transition
    checks = {
        "token_prior_proj": tr.prior_proj.weight,
        "context_proj_weight": tr.context_proj.weight,
        "context_proj_bias": tr.context_proj.bias,
        "physics_gate": tr.physics_fusion.gate,
    }
    bad = [name for name, tensor in checks.items() if tensor is not None and bool((tensor.detach() != 0).any())]
    if bad:
        raise RuntimeError(
            "frozen sparse diagnostic requires exact zero-impact new conditioning; "
            f"nonzero tensors: {bad}"
        )


def _load_dense_reference(path: str | None, ds: MSPWorldModelCacheDataset, n_eval: int):
    if not path:
        return None
    if n_eval != len(ds):
        return {
            "status": "skipped_for_partial_run",
            "reason": "baseline reproduction check is only valid for the full validation set",
        }
    data = json.loads(Path(path).read_text())
    if data.get("protocol") != "p0_f9_official_occfm_native_baseline_v2":
        raise RuntimeError("dense reference is not the audited official OccFM baseline")
    if int(data.get("num_windows", -1)) != len(ds):
        raise RuntimeError("dense reference window count differs from diagnostic cache")
    expected_sha = file_sha256(ds.root / "index.json")
    if data.get("cache_index_sha256") != expected_sha:
        raise RuntimeError("dense reference was evaluated on a different cache index")
    return data


def _assert_dense_replay_matches(reference, dense_report) -> dict | None:
    if not reference or reference.get("status") == "skipped_for_partial_run":
        return None
    ref_o = float(reference["metrics"]["overall"]["mIoU"])
    ref_m = float(reference["metrics"]["moving"]["mIoU"])
    got_o, got_m = _metric_pair(dense_report)
    diff_o, diff_m = got_o - ref_o, got_m - ref_m
    if abs(diff_o) > 1e-9 or abs(diff_m) > 1e-9:
        raise RuntimeError(
            "official dense replay does not reproduce existing baseline: "
            f"dOverall={diff_o:.12g} dMoving={diff_m:.12g}"
        )
    return {"status": "bit_metric_exact", "overall_difference": diff_o, "moving_difference": diff_m}


def _require_trained_checkpoint_match(ck: dict, ds: MSPWorldModelCacheDataset, vae_ckpt: str) -> dict:
    arch = ck.get("architecture", {})
    if arch.get("protocol") != P0_F9_PROTOCOL or int(arch.get("stage", -1)) != 1:
        raise RuntimeError("trained checkpoint is not audited P0-F9 Stage-1")
    if int(arch.get("native_backbone_hist_last", -1)) != HIST_LAST:
        raise RuntimeError("trained checkpoint HIST_LAST contract differs")
    if arch.get("flow_source_spatial_contract") != SOURCE_SPATIAL_CONTRACT:
        raise RuntimeError("trained checkpoint source-noise spatial contract differs")
    if ck.get("val_cache_index_sha256") != file_sha256(ds.root / "index.json"):
        raise RuntimeError("trained checkpoint was validated against a different cache index")
    if ck.get("vae_checkpoint_sha256") != file_sha256(vae_ckpt):
        raise RuntimeError("trained checkpoint VAE provenance differs")
    ckmeta = ck.get("val_metadata") or {}
    for key in (
        "protocol",
        "source_v3_cache_index_sha256",
        "source_msp_cache_sha256",
        "msp_checkpoint_sha256",
        "vae_checkpoint_sha256",
        "vae_mode",
        "native_backbone_hist_last",
        "anchor_contract",
        "write_budget_ratio",
    ):
        if ckmeta.get(key) != ds.metadata.get(key):
            raise RuntimeError(f"trained checkpoint/eval-cache metadata mismatch for {key}")
    return arch


def _sample_payload(s, device):
    required = (
        "eval_future_gt_occ",
        "eval_strong_anchor_occ",
        "eval_repair_target_occ",
        "eval_gt_moving_support",
    )
    missing = [k for k in required if k not in s]
    if missing:
        raise RuntimeError(f"{s['sample_id']}: eval payload missing {missing}")
    write_lat = s["msp_write_support_latent"].bool()
    write_bev = latent_support_to_bev(write_lat, (200, 200)).cpu().numpy().astype(bool)
    return {
        "gt": s["eval_future_gt_occ"].cpu().numpy(),
        "anchor": s["eval_strong_anchor_occ"].cpu().numpy(),
        "repair_target": s["eval_repair_target_occ"].cpu().numpy(),
        "moving": s["eval_gt_moving_support"].cpu().numpy().astype(bool),
        "write_bev": write_bev,
        "write_ratio": float(write_lat.float().mean().item()),
        "trajectory": s["trajectory"].to(device),
    }


def _prepare_sparse_sample(s, device, *, with_context: bool):
    origins = s["window_origins"].unsqueeze(0).long().to(device)
    valid = s["window_valid"].unsqueeze(0).bool().to(device)
    plan = WindowPlan(origins, valid, PRED_HW, FULL_HW)
    hist_full = s["full_history_latent"].unsqueeze(0).to(device)
    physics_full = s["anchor_future_latent"].unsqueeze(0).to(device)
    if with_context:
        hist_w, hist_context, _ = crop_prediction_and_context(hist_full, plan, context_hw=CONTEXT_HW)
    else:
        hist_w = crop_windows(hist_full, plan)
        hist_context = None
    physics_w = crop_windows(physics_full, plan)
    B, K = hist_w.shape[:2]
    flat_valid = plan.valid.reshape(-1)

    def flat(x):
        return x.reshape(B * K, *x.shape[2:])[flat_valid]

    out = {
        "plan": plan,
        "physics_full": physics_full,
        "flat_valid": flat_valid,
        "B": B,
        "K": K,
    }
    if bool(flat_valid.any()):
        out["history"] = flat(hist_w)
        out["physics"] = flat(physics_w)
        out["context"] = None if hist_context is None else flat(hist_context)
        out["origins"] = plan.origins.reshape(B * K, 2)[flat_valid]
        traj = s["trajectory"].to(device).unsqueeze(0)
        out["trajectory"] = traj[:, None].expand(B, K, 12, 2).reshape(B * K, 12, 2)[flat_valid]
    return out


def _decode_sparse_prediction(model, vae, s, prepared, *, seed, use_amp, guidance_scale, physics_condition, context_condition):
    flat_valid = prepared["flat_valid"]
    if not bool(flat_valid.any()):
        return s["eval_strong_anchor_occ"].cpu().numpy()
    sample_seed = deterministic_sample_seed(str(s["sample_id"]), seed, stream="forecast")
    global_noise = _official_seeded_noise_like(prepared["physics_full"], sample_seed)
    initial_noise = crop_coherent_source_noise(global_noise, prepared["plan"], flat_valid)
    physics = prepared["physics"] if physics_condition else torch.zeros_like(prepared["physics"])
    context = prepared["context"] if context_condition else None
    with torch.autocast(
        device_type=prepared["physics_full"].device.type,
        dtype=torch.bfloat16,
        enabled=use_amp,
    ):
        pred = model.sample(
            prepared["history"],
            physics,
            history_context=context,
            trajectory=prepared["trajectory"],
            window_origins=prepared["origins"],
            initial_noise=initial_noise,
            guidance_scale=guidance_scale,
        )
    B, K = prepared["B"], prepared["K"]
    padded = torch.zeros(B * K, *pred.shape[1:], device=pred.device, dtype=pred.dtype)
    padded[flat_valid] = pred
    fused_latent = scatter_windows(
        padded.reshape(B, K, *pred.shape[1:]),
        prepared["plan"],
        base=prepared["physics_full"],
    )
    return vae.decode_labels(fused_latent.float())[0].cpu().numpy()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--occfm-ckpt", required=True)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dense-baseline-json", default=None)
    p.add_argument("--trained-sparse-ckpt", default=None,
                   help="optional P0-F9 Stage-1 checkpoint; step_1200.pt is recommended")
    p.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=20260904)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-windows", type=int, default=0,
                   help="0 means full validation set; use a small positive value only for smoke")
    a = p.parse_args()

    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("safe-fusion diagnostic requires CUDA")
    ds = MSPWorldModelCacheDataset(a.cache)
    _validate_cache(ds, a.vae_ckpt)
    n_eval = len(ds) if int(a.max_windows) <= 0 else min(len(ds), int(a.max_windows))
    dense_reference = _load_dense_reference(a.dense_baseline_json, ds, n_eval)

    cfg = load_occfm_config(UP, "tools/cfgs/occfm_fut.yaml")
    if int(cfg.DATA_CONFIG.HIST_LAST) != HIST_LAST:
        raise RuntimeError("pinned official OccFM HIST_LAST changed")
    if int(cfg.LOSS.SAMPLE_STEP) != 10 or float(cfg.LOSS.ALPHA_STEP) != 3.0:
        raise RuntimeError("pinned official OccFM sampler schedule changed")
    if float(cfg.LOSS.UNCOND_SCALE) != OFFICIAL_CFG:
        raise RuntimeError("pinned official OccFM guidance scale changed")

    # Pass A: dense official proposal under all fusion contracts.
    dense_wm, dense_cfg = load_official_wm(UP, a.occfm_ckpt, device)
    if int(dense_cfg.DATA_CONFIG.HIST_LAST) != HIST_LAST:
        raise RuntimeError("loaded official OccFM config HIST_LAST mismatch")
    dense_states = {
        "strong_anchor": _new_metrics(),
        "dense_official_raw": _new_metrics(),
        **_new_source_states("dense_official"),
    }
    for i in range(n_eval):
        s = ds[i]
        payload = _sample_payload(s, device)
        sample_seed = deterministic_sample_seed(str(s["sample_id"]), a.seed, stream="forecast")
        dense_pred = run_frozen_occfm_forecast(
            dense_wm,
            s["full_history_latent"],
            s["gt_future_latent"],
            trajectory=s["trajectory"],
            seed=sample_seed,
            hist_last=HIST_LAST,
        ).numpy()
        for horizon, fi in REPORT.items():
            gt, anchor, moving = payload["gt"][fi], payload["anchor"][fi], payload["moving"][fi]
            _update(dense_states["strong_anchor"], horizon, anchor, gt, moving)
            _update(dense_states["dense_official_raw"], horizon, dense_pred[fi], gt, moving)
            _update_source_fusions(
                dense_states, "dense_official", horizon, anchor, dense_pred[fi], gt, moving, payload["write_bev"][fi]
            )
        if i % 8 == 0:
            print("dense_safe_fusion_eval", i, s["sample_id"])
    dense_reports = {name: _report(state) for name, state in dense_states.items()}
    dense_reproduction = _assert_dense_replay_matches(dense_reference, dense_reports["dense_official_raw"])
    del dense_wm
    gc.collect()
    torch.cuda.empty_cache()

    # Shared frozen VAE for sparse branches.
    vae_model, _ = load_official_vae(UP, a.vae_ckpt, device)
    vae = OccFMVAEAdapter(vae_model)
    use_amp = bool(a.amp and device.type == "cuda")

    # Pass B: frozen 20x20 sparse official weights, no new condition paths.
    frozen_model = make_p0_f9_model(
        20,
        sample_steps=int(cfg.LOSS.SAMPLE_STEP),
        unconditional_probability=0.0,
        guidance_scale=OFFICIAL_CFG,
        hist_last=HIST_LAST,
    ).to(device)
    reuse = load_shape_safe(frozen_model.transition, a.occfm_ckpt, verbose=True)
    if "traj_encoder.0.weight" not in set(reuse.get("loaded_keys", ())):
        raise RuntimeError("diagnostic requires official OccFM-Fut epoch=000196 checkpoint")
    reuse_fraction = require_checkpoint_reuse(reuse, min_fraction=0.80)
    _assert_new_conditioning_is_noop(frozen_model)
    frozen_model.eval().requires_grad_(False)

    sparse_states = {
        **_new_source_states("frozen_sparse_official_cfg"),
        **_new_source_states("frozen_sparse_p0f9_cfg"),
        **_new_source_states("gt"),
    }
    oracle_checks = 0
    valid_windows = []
    write_ratios = []
    for i in range(n_eval):
        s = ds[i]
        payload = _sample_payload(s, device)
        prepared = _prepare_sparse_sample(s, device, with_context=False)
        valid_windows.append(int(prepared["plan"].valid.sum().item()))
        write_ratios.append(payload["write_ratio"])
        pred_cfg2 = _decode_sparse_prediction(
            frozen_model, vae, s, prepared,
            seed=a.seed, use_amp=use_amp, guidance_scale=OFFICIAL_CFG,
            physics_condition=False, context_condition=False,
        )
        pred_cfg1 = _decode_sparse_prediction(
            frozen_model, vae, s, prepared,
            seed=a.seed, use_amp=use_amp, guidance_scale=P0_F9_CFG,
            physics_condition=False, context_condition=False,
        )
        for horizon, fi in REPORT.items():
            gt, anchor, moving = payload["gt"][fi], payload["anchor"][fi], payload["moving"][fi]
            write_bev = payload["write_bev"][fi]
            _update_source_fusions(
                sparse_states, "frozen_sparse_official_cfg", horizon, anchor, pred_cfg2[fi], gt, moving, write_bev
            )
            _update_source_fusions(
                sparse_states, "frozen_sparse_p0f9_cfg", horizon, anchor, pred_cfg1[fi], gt, moving, write_bev
            )
            gt_outputs = _update_source_fusions(
                sparse_states, "gt", horizon, anchor, gt, gt, moving, write_bev
            )
            if not np.array_equal(gt_outputs["takeover"], payload["repair_target"][fi]):
                raise RuntimeError(
                    f"{s['sample_id']} horizon={horizon}: GT takeover oracle differs from cached repair target"
                )
            oracle_checks += 1
        if i % 8 == 0:
            print("frozen_sparse_safe_fusion_eval", i, s["sample_id"])
    sparse_reports = {name: _report(state) for name, state in sparse_states.items()}
    del frozen_model
    gc.collect()
    torch.cuda.empty_cache()

    # Pass C (optional): trained P0-F9 proposal, replayed with its exact cache/
    # architecture provenance, then fused under the same three rules.
    trained_reports = {}
    trained_meta = None
    if a.trained_sparse_ckpt:
        ck = torch.load(a.trained_sparse_ckpt, map_location="cpu", weights_only=False)
        arch = _require_trained_checkpoint_match(ck, ds, a.vae_ckpt)
        trained_model = make_p0_f9_model(
            20,
            sample_steps=int(arch.get("sample_steps", 10)),
            unconditional_probability=float(arch.get("unconditional_probability", 0.0)),
            guidance_scale=float(arch.get("guidance_scale", 1.0)),
            hist_last=HIST_LAST,
        ).to(device)
        if a.use_ema:
            ema = ck.get("ema")
            if not ema or "state_dict" not in ema:
                raise RuntimeError("trained P0-F9 checkpoint lacks EMA state")
            trained_model.load_state_dict(ema["state_dict"], strict=True)
            weight_source = "ema"
        else:
            trained_model.load_state_dict(ck["state_dict"], strict=True)
            weight_source = "raw"
        trained_model.eval().requires_grad_(False)
        trained_states = _new_source_states("trained_p0f9")
        for i in range(n_eval):
            s = ds[i]
            payload = _sample_payload(s, device)
            prepared = _prepare_sparse_sample(s, device, with_context=True)
            trained_pred = _decode_sparse_prediction(
                trained_model,
                vae,
                s,
                prepared,
                seed=a.seed,
                use_amp=use_amp,
                guidance_scale=float(arch.get("guidance_scale", 1.0)),
                physics_condition=True,
                context_condition=True,
            )
            for horizon, fi in REPORT.items():
                _update_source_fusions(
                    trained_states,
                    "trained_p0f9",
                    horizon,
                    payload["anchor"][fi],
                    trained_pred[fi],
                    payload["gt"][fi],
                    payload["moving"][fi],
                    payload["write_bev"][fi],
                )
            if i % 8 == 0:
                print("trained_p0f9_safe_fusion_eval", i, s["sample_id"])
        trained_reports = {name: _report(state) for name, state in trained_states.items()}
        trained_meta = {
            "checkpoint": str(Path(a.trained_sparse_ckpt).resolve()),
            "checkpoint_sha256": file_sha256(a.trained_sparse_ckpt),
            "step": int(ck.get("step", -1)),
            "weight_source": weight_source,
            "guidance_scale": float(arch.get("guidance_scale", 1.0)),
            "sample_steps": int(arch.get("sample_steps", 10)),
        }
        del trained_model, ck
        gc.collect()
        torch.cuda.empty_cache()

    metrics = {**dense_reports, **sparse_reports, **trained_reports}
    controlled_deltas = {
        "fusion_effect_dense_same_support_minus_strong": _delta(
            metrics["dense_official_same_support_fusion"], metrics["strong_anchor"]
        ),
        "sparse_geometry_effect_cfg2_minus_dense_same_support": _delta(
            metrics["frozen_sparse_official_cfg"], metrics["dense_official_same_support_fusion"]
        ),
        "guidance_effect_cfg1_minus_cfg2": _delta(
            metrics["frozen_sparse_p0f9_cfg"], metrics["frozen_sparse_official_cfg"]
        ),
        "frozen_sparse_cfg1_minus_strong": _delta(
            metrics["frozen_sparse_p0f9_cfg"], metrics["strong_anchor"]
        ),
        "oracle_headroom_minus_strong": _delta(metrics["same_support_gt_oracle"], metrics["strong_anchor"]),
    }

    source_keys = {
        "dense_official": _source_state_names("dense_official"),
        "frozen_sparse_official_cfg": _source_state_names("frozen_sparse_official_cfg"),
        "frozen_sparse_p0f9_cfg": _source_state_names("frozen_sparse_p0f9_cfg"),
        "gt": _source_state_names("gt"),
    }
    if trained_reports:
        source_keys["trained_p0f9"] = _source_state_names("trained_p0f9")
    safe_fusion_deltas = {}
    for source, names in source_keys.items():
        takeover = metrics[names["takeover"]]
        safe_fusion_deltas[source] = {
            "takeover_minus_strong": _delta(takeover, metrics["strong_anchor"]),
            "write_only_minus_strong": _delta(metrics[names["write_only"]], metrics["strong_anchor"]),
            "dynamic_union_minus_strong": _delta(metrics[names["dynamic_union"]], metrics["strong_anchor"]),
            "write_only_minus_takeover": _delta(metrics[names["write_only"]], takeover),
            "dynamic_union_minus_takeover": _delta(metrics[names["dynamic_union"]], takeover),
            "union_minus_write_only": _delta(metrics[names["dynamic_union"]], metrics[names["write_only"]]),
        }

    report = {
        "protocol": PROTOCOL,
        "num_windows": n_eval,
        "cache_index_sha256": file_sha256(ds.root / "index.json"),
        "official_occfm_checkpoint": str(Path(a.occfm_ckpt).resolve()),
        "official_occfm_checkpoint_sha256": file_sha256(a.occfm_ckpt),
        "vae_checkpoint_sha256": file_sha256(a.vae_ckpt),
        "trained_p0f9": trained_meta,
        "latent_distribution": "posterior_sample",
        "hist_last": HIST_LAST,
        "sample_steps": int(cfg.LOSS.SAMPLE_STEP),
        "alpha_step": float(cfg.LOSS.ALPHA_STEP),
        "seed": int(a.seed),
        "sample_seed_contract": "sha256(base,forecast,sample_id)",
        "source_noise_contract": "official-first-global-randn-50x50_then_crop_same_top2_plan",
        "window_contract": "top2_20x20_absolute_50x50_position_coordinates",
        "frozen_conditioning_contract": "no_full_context_no_physics_condition_no_finetuning",
        "fusion_contracts": {
            "takeover": "clear_anchor_dynamic_then_write_proposal_dynamic_inside_support",
            "write_only": "keep_anchor_dynamic_bit_exact_and_only_add_proposal_dynamic_on_non_dynamic_anchor",
            "dynamic_union": "never_clear_anchor_dynamic_presence_but_proposal_dynamic_may_add_or_relabel",
        },
        "official_guidance_scale": OFFICIAL_CFG,
        "p0_f9_guidance_scale": P0_F9_CFG,
        "official_transition_reuse_fraction": float(reuse_fraction),
        "official_transition_loaded_tensors": int(reuse.get("loaded", 0)),
        "official_transition_target_tensors": int(reuse.get("target_total", 0)),
        "oracle_bit_exact_checks": int(oracle_checks),
        "mean_valid_top2_windows": float(np.mean(valid_windows)) if valid_windows else 0.0,
        "mean_write_support_ratio": float(np.mean(write_ratios)) if write_ratios else 0.0,
        "dense_baseline_reference": dense_reference,
        "dense_baseline_reproduction": dense_reproduction,
        "metrics": metrics,
        "controlled_deltas": controlled_deltas,
        "safe_fusion_deltas": safe_fusion_deltas,
        "interpretation_contract": {
            "real_safe_fusion_beats_strong": "proposal contains useful sparse innovation once destructive clear is removed",
            "safe_gt_beats_strong_but_real_does_not": "safe fusion has headroom but proposal quality is insufficient",
            "safe_gt_not_above_strong": "pure additive innovation is insufficient; selective learned clear/correction is required",
            "write_only_above_union": "proposal dynamic relabels are harmful; preserve Strong dynamic semantics",
            "union_above_write_only": "proposal contributes useful dynamic class corrections",
        },
    }

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== P0-F9 SAFE-FUSION ABLATION ===")
    print(f"{'state':48s} {'Overall':>10s} {'Moving':>10s}")
    order = [
        "strong_anchor",
        "dense_official_raw",
        "dense_official_same_support_fusion",
        "dense_official_write_only",
        "dense_official_dynamic_union",
        "frozen_sparse_official_cfg",
        "frozen_sparse_official_cfg_write_only",
        "frozen_sparse_official_cfg_dynamic_union",
        "frozen_sparse_p0f9_cfg",
        "frozen_sparse_p0f9_cfg_write_only",
        "frozen_sparse_p0f9_cfg_dynamic_union",
    ]
    if trained_reports:
        order += ["trained_p0f9_takeover", "trained_p0f9_write_only", "trained_p0f9_dynamic_union"]
    order += ["same_support_gt_oracle", "same_support_gt_write_only_oracle", "same_support_gt_dynamic_union_oracle"]
    for name in order:
        o, m = _metric_pair(metrics[name])
        print(f"{name:48s} {o:10.4f} {m:10.4f}")

    print("\n=== SAFE FUSION DELTAS ===")
    for source, rows in safe_fusion_deltas.items():
        print(f"[{source}]")
        for name, d in rows.items():
            print(f"  {name:30s} Overall={d['overall']:+.4f} Moving={d['moving']:+.4f}")

    print("\n=== ORIGINAL CONTROLLED DELTAS ===")
    for name, d in controlled_deltas.items():
        print(f"{name:52s} Overall={d['overall']:+.4f} Moving={d['moving']:+.4f}")
    print("saved", out)


if __name__ == "__main__":
    main()
