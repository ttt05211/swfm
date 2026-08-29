#!/usr/bin/env python3
"""Evaluate the trained Top-2 anchor-centered Sparse World Model on real occupancy."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "upstream_occfm"
sys.path[:0] = [str(ROOT), str(UP)]

import numpy as np
import torch

from real_motion.composition import static_protected_compose
from real_motion.metrics.moving_miou_v2 import (
    DYNAMIC_CLASS_IDS, MovingMIoUV2MultiHorizon, REPORT_HORIZONS_S,
    SemanticIoUAccumulator,
)
from real_motion.msp import latent_support_to_bev
from real_motion.msp_window import window_plan_support
from real_motion.msp_wm_cache import MSPWorldModelCacheDataset
from real_motion.nuscenes_adapter import dynamic_only_semantics
from real_motion.occfm_io import OccFMVAEAdapter, file_sha256, load_official_vae
from real_motion.prepared import PreparedShardDataset
from real_motion.windows import WindowPlan, crop_windows, scatter_windows
from tools.real_motion.train_msp_sparse_wm import make_model

REPORT = {1.0: 1, 2.0: 3, 3.0: 5}
EVAL_KEYS = (
    "eval_future_gt_occ",
    "eval_static_future_occ",
    "eval_confident_static_future_mask",
    "eval_kta_future_occ",
    "eval_gt_moving_support",
)


def _prepared_map(ds):
    out = {}
    for i, e in enumerate(ds.entries):
        sid = str(e.get("sample_id", ""))
        if sid in out:
            raise RuntimeError(f"duplicate prepared sample {sid}")
        out[sid] = i
    return out


def _new_metrics():
    return {
        "overall": {h: SemanticIoUAccumulator() for h in REPORT_HORIZONS_S},
        "moving": MovingMIoUV2MultiHorizon(),
    }


def _report(state):
    overall_h = {h: state["overall"][h].compute() for h in REPORT_HORIZONS_S}
    return {
        "overall": {
            "mIoU": float(np.nanmean([overall_h[h]["mIoU"] for h in REPORT_HORIZONS_S])),
            "per_horizon": overall_h,
        },
        "moving": state["moving"].compute(),
    }


def _compose(static, dyn, protected):
    return static_protected_compose(
        torch.from_numpy(np.asarray(static)),
        torch.from_numpy(np.asarray(dyn)),
        torch.from_numpy(np.asarray(protected)),
        DYNAMIC_CLASS_IDS,
        write_support=None,
    ).numpy()


def _update(state, horizon, pred, gt, moving_support):
    state["overall"][horizon].update(pred, gt)
    state["moving"].update(horizon, pred, gt, moving_support)


def _base_from_cache_sample(s):
    missing = [k for k in EVAL_KEYS if k not in s]
    if missing:
        return None
    return {
        "future_gt_occ": s["eval_future_gt_occ"].cpu().numpy(),
        "static_future_occ": s["eval_static_future_occ"].cpu().numpy(),
        "confident_static_future_mask": s["eval_confident_static_future_mask"].cpu().numpy(),
        "kta_future_occ": s["eval_kta_future_occ"].cpu().numpy(),
        "gt_moving_support": s["eval_gt_moving_support"].cpu().numpy(),
    }


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument(
        "--prepared", default=None,
        help="optional legacy/full prepared val cache; unnecessary when --cache contains compact eval payload",
    )
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--sparse-ckpt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args()

    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    ds = MSPWorldModelCacheDataset(a.cache)
    if int(ds.metadata.get("topk", -1)) != 2:
        raise RuntimeError("P0-F3 evaluator is frozen to Top-2")
    expected_vae = ds.metadata.get("vae_checkpoint_sha256")
    if expected_vae and file_sha256(a.vae_ckpt) != expected_vae:
        raise RuntimeError("VAE checkpoint differs from routed cache")

    ck = torch.load(a.sparse_ckpt, map_location="cpu", weights_only=False)
    arch = ck.get("architecture", {})
    if int(arch.get("topk", -1)) != 2 or list(arch.get("window_hw", [])) != [20, 20]:
        raise RuntimeError("Sparse-WM checkpoint is not the frozen Top-2/20x20 model")
    model = make_model(
        20,
        sample_steps=int(arch.get("sample_steps", 10)),
        source_noise_std=float(arch.get("source_noise_std", 0.0)),
    ).to(device)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()

    vae, _ = load_official_vae(UP, a.vae_ckpt, device)
    va = OccFMVAEAdapter(vae)

    prepared = None
    pmap = None
    if a.prepared:
        prepared = PreparedShardDataset(a.prepared)
        pmap = _prepared_map(prepared)
        missing = [str(e["sample_id"]) for e in ds.entries if str(e["sample_id"]) not in pmap]
        if missing:
            raise RuntimeError(f"prepared val dataset misses {len(missing)} cache samples, e.g. {missing[:3]}")
    elif not bool(ds.metadata.get("include_eval_payload", False)):
        raise RuntimeError(
            "cache does not advertise compact eval payload; provide --prepared or rebuild val cache "
            "with build_msp_wm_cache_direct.py --include-eval-payload"
        )

    anchor_state = _new_metrics()
    model_state = _new_metrics()
    oracle_state = _new_metrics()
    unique_ratios = []
    valid_windows = []
    use_amp = bool(a.amp and device.type == "cuda")

    for i in range(len(ds)):
        s = ds[i]
        base = _base_from_cache_sample(s)
        if base is None:
            if prepared is None:
                raise RuntimeError(f"{s['sample_id']}: compact eval payload missing")
            base = prepared[pmap[str(s["sample_id"])]]

        origins = s["window_origins"].unsqueeze(0).long()
        valid = s["window_valid"].unsqueeze(0).bool()
        plan_cpu = WindowPlan(origins, valid, (20, 20), (50, 50))
        plan = WindowPlan(origins.to(device), valid.to(device), (20, 20), (50, 50))

        hist_full = s["moving_history_latent"].unsqueeze(0).to(device)
        anchor_full = s["anchor_future_latent"].unsqueeze(0).to(device)
        hist_w = crop_windows(hist_full, plan)
        anchor_w = crop_windows(anchor_full, plan)
        B, K = hist_w.shape[:2]
        flat_valid = plan.valid.reshape(-1)
        if not bool(flat_valid.any()):
            fused = anchor_full
        else:
            def flat(x):
                return x.reshape(B * K, *x.shape[2:])[flat_valid]
            fh = flat(hist_w)
            fa = flat(anchor_w)
            orig = plan.origins.reshape(B * K, 2)[flat_valid]
            traj = s["trajectory"].to(device).unsqueeze(0)
            traj = traj[:, None].expand(B, K, 12, 2).reshape(B * K, 12, 2)[flat_valid]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                pred = model.sample(fh, fa, trajectory=traj, window_origins=orig)
            pad = torch.zeros(B * K, *pred.shape[1:], device=device, dtype=pred.dtype)
            pad[flat_valid] = pred
            fused = scatter_windows(pad.reshape(B, K, *pred.shape[1:]), plan, base=anchor_full)

        decoded = va.decode_labels(fused.float())[0].cpu().numpy()
        spatial = window_plan_support(plan_cpu)[0]
        unique_ratios.append(float(spatial.float().mean().item()))
        valid_windows.append(int(valid.sum().item()))
        lat_support = spatial.unsqueeze(0).expand(6, -1, -1)
        bev_write = latent_support_to_bev(lat_support, (200, 200)).cpu().numpy().astype(bool)

        future_gt = np.asarray(base["future_gt_occ"])
        static = np.asarray(base["static_future_occ"])
        protected = np.asarray(base["confident_static_future_mask"])
        anchor_dyn = np.asarray(base["kta_future_occ"])
        moving_support = np.asarray(base["gt_moving_support"])

        for h, fi in REPORT.items():
            gt = future_gt[fi]
            anc = _compose(static[fi], anchor_dyn[fi], protected[fi])
            _update(anchor_state, h, anc, gt, moving_support[fi])

            pred_dyn = dynamic_only_semantics(decoded[fi], 17)
            repair_dyn = np.where(bev_write[fi][..., None], pred_dyn, anchor_dyn[fi])
            final = _compose(static[fi], repair_dyn, protected[fi])
            _update(model_state, h, final, gt, moving_support[fi])

            gt_dyn = dynamic_only_semantics(gt, 17)
            oracle_dyn = np.where(bev_write[fi][..., None], gt_dyn, anchor_dyn[fi])
            oracle = _compose(static[fi], oracle_dyn, protected[fi])
            _update(oracle_state, h, oracle, gt, moving_support[fi])

        if i % 8 == 0:
            print("eval", i, s["sample_id"])

    anchor_report = _report(anchor_state)
    model_report = _report(model_state)
    oracle_report = _report(oracle_state)
    am = float(anchor_report["moving"]["mIoU"])
    mm = float(model_report["moving"]["mIoU"])
    om = float(oracle_report["moving"]["mIoU"])
    report = {
        "protocol": {
            "name": "p0_f3_top2_anchor_sparse_wm_eval_v1",
            "num_windows": len(ds),
            "topk": 2,
            "window_hw": [20, 20],
            "slot_compute_ratio": float(np.mean(valid_windows) * 400.0 / 2500.0),
            "unique_latent_ratio": float(np.mean(unique_ratios)),
            "eval_source": "compact_cache_payload" if bool(ds.metadata.get("include_eval_payload", False)) else "prepared",
            "fusion": "outside selected windows preserve KTA/zero anchor; inside use decoded WM dynamic semantics",
        },
        "causal_anchor": anchor_report,
        "trained_sparse_wm": model_report,
        "top2_gt_repair_oracle": oracle_report,
        "delta_Moving_vs_anchor": mm - am,
        "remaining_Moving_headroom_to_oracle": om - mm,
        "oracle_delta_Moving_vs_anchor": om - am,
        "checkpoint": str(Path(a.sparse_ckpt).resolve()),
        "best_val_loss": ck.get("best_val_loss"),
    }
    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
