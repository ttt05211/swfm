#!/usr/bin/env python3
"""Attribute the remaining P0-F9 deployment gap with controlled GT oracles.

The diagnostic decodes each trained checkpoint exactly once on the frozen 128-val
protocol, then applies occupancy-space interventions without any optimization:

- current: unchanged decoded proposal + audited takeover fusion;
- oracle_clear_keep: perfect departure/persistence presence on anchor-dynamic voxels;
- oracle_write: perfect WRITE/no-WRITE decisions on anchor-non-dynamic voxels;
- oracle_event_presence: joint CLEAR/KEEP + WRITE presence oracle;
- oracle_semantic: correct dynamic class only where current proposal and GT both
  already predict dynamic presence;
- same_support_gt_event: GT dynamic proposal under the frozen MSP write support;
- oracle_support_gt_event: the same GT proposal after expanding support only to
  BEV cells that truly require a dynamic appearance/disappearance/relabel edit.

The difference between the two GT-event rows isolates routing/support headroom.
All other oracle rows keep the original causal support fixed.  Per-horizon and
per-class Moving-mIoU are preserved in the JSON so E5/E10 plateaus can be traced
to a concrete class/time range before a new method is designed.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "upstream_occfm"
sys.path[:0] = [str(ROOT), str(UP)]

import numpy as np
import torch

from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS, NUSCENES_LABELS
from real_motion.motion_edit_diagnostics import MotionEditAccumulator
from real_motion.msp_wm_cache import MSPWorldModelCacheDataset
from real_motion.occfm_io import OccFMVAEAdapter, file_sha256, load_official_vae
from real_motion.oracle_gap import (
    gt_edit_support_bev,
    oracle_clear_keep_presence,
    oracle_dynamic_semantics,
    oracle_event_presence,
    oracle_write_presence,
    support_coverage_counts,
)
from real_motion.repair_target import apply_dynamic_repair
from tools.real_motion import diagnose_p0_f9_training_failure as diag
from tools.real_motion import eval_p0_f9_frozen_sparse_occfm as safe


PROTOCOL = "p0_f9_oracle_gap_attribution_v1"
VARIANTS = (
    "current",
    "oracle_clear_keep",
    "oracle_write",
    "oracle_event_presence",
    "oracle_semantic",
    "same_support_gt_event",
    "oracle_support_gt_event",
)


def _parse_checkpoint(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("--checkpoint must be LABEL=/path/to/checkpoint.pt")
    label, path = text.split("=", 1)
    label, path = label.strip(), path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError("checkpoint label/path cannot be empty")
    return label, path


def _new_state():
    return {
        "metrics": safe._new_metrics(),
        "edit_all": MotionEditAccumulator(tuple(DYNAMIC_CLASS_IDS)),
        "edit_by_frame": [MotionEditAccumulator(tuple(DYNAMIC_CLASS_IDS)) for _ in range(6)],
    }


def _update_physical(state: dict, anchor, proposal, gt, support) -> None:
    state["edit_all"].update(anchor, proposal, gt, support)
    for fi in range(6):
        state["edit_by_frame"][fi].update(anchor[fi], proposal[fi], gt[fi], support[fi])


def _finish_state(state: dict) -> dict:
    by_frame = {str(i): state["edit_by_frame"][i].compute() for i in range(6)}
    return {
        "deployment": safe._report(state["metrics"]),
        "physical_edits_all_6_frames": state["edit_all"].compute(),
        "physical_edits_by_frame": by_frame,
        "physical_edits_report_horizons": {
            str(h): by_frame[str(fi)] for h, fi in diag.REPORT_FRAMES.items()
        },
    }


def _metric_pair(row: dict) -> tuple[float, float]:
    dep = row["deployment"]
    return float(dep["overall"]["mIoU"]), float(dep["moving"]["mIoU"])


def _delta(row: dict, base: dict) -> dict:
    o, m = _metric_pair(row)
    bo, bm = _metric_pair(base)
    return {"overall": o - bo, "moving": m - bm}


def _recovery(row: dict, current: dict, strong: dict) -> dict:
    o, m = _metric_pair(row)
    co, cm = _metric_pair(current)
    so, sm = _metric_pair(strong)

    def one(value, cur, target):
        gap = target - cur
        return (value - cur) / gap if gap > 1e-12 else float("nan")

    return {"overall": one(o, co, so), "moving": one(m, cm, sm)}


def _moving_class_means(report: dict) -> dict:
    per_h = report["deployment"]["moving"]["per_horizon"]
    out = {}
    for c in DYNAMIC_CLASS_IDS:
        vals = []
        horizon = {}
        for h in safe.REPORT:
            row = per_h[float(h)] if float(h) in per_h else per_h[str(float(h))]
            pc = row["per_class"]
            value = pc[int(c)] if int(c) in pc else pc[str(int(c))]
            horizon[str(float(h))] = float(value)
            if not math.isnan(float(value)):
                vals.append(float(value))
        out[str(int(c))] = {
            "class_name": NUSCENES_LABELS[int(c)],
            "per_horizon": horizon,
            "mean_over_horizons": float(np.mean(vals)) if vals else float("nan"),
        }
    return out


def _proposal_variants(anchor, proposal, gt, support):
    common = {"dynamic_class_ids": DYNAMIC_CLASS_IDS, "free_label": safe.FREE}
    clear = oracle_clear_keep_presence(anchor, proposal, gt, support, **common)
    write = oracle_write_presence(anchor, proposal, gt, support, **common)
    event = oracle_event_presence(anchor, proposal, gt, support, **common)
    semantic = oracle_dynamic_semantics(
        anchor, proposal, gt, support, dynamic_class_ids=DYNAMIC_CLASS_IDS
    )
    expanded = gt_edit_support_bev(
        anchor, gt, support, dynamic_class_ids=DYNAMIC_CLASS_IDS
    )
    return {
        "current": (proposal, support),
        "oracle_clear_keep": (clear, support),
        "oracle_write": (write, support),
        "oracle_event_presence": (event, support),
        "oracle_semantic": (semantic, support),
        "same_support_gt_event": (gt, support),
        "oracle_support_gt_event": (gt, expanded),
    }


class SupportCoverageAccumulator:
    def __init__(self):
        self.rows = {k: {"total": 0, "covered": 0} for k in ("clear", "write", "relabel", "event")}
        self.support_bev_cells = 0
        self.total_bev_cells = 0
        self.expanded_support_bev_cells = 0

    def update(self, anchor, gt, support):
        row = support_coverage_counts(
            anchor, gt, support, dynamic_class_ids=DYNAMIC_CLASS_IDS
        )
        for key in self.rows:
            self.rows[key]["total"] += int(row[key]["total"])
            self.rows[key]["covered"] += int(row[key]["covered"])
        self.support_bev_cells += int(row["support_bev_cells"])
        self.total_bev_cells += int(row["total_bev_cells"])
        expanded = gt_edit_support_bev(
            anchor, gt, support, dynamic_class_ids=DYNAMIC_CLASS_IDS
        )
        self.expanded_support_bev_cells += int(expanded.sum())

    def compute(self):
        out = {}
        for key, row in self.rows.items():
            total, covered = int(row["total"]), int(row["covered"])
            out[key] = {
                "total": total,
                "covered": covered,
                "coverage": covered / total if total else float("nan"),
            }
        out["current_support_bev_fraction"] = (
            self.support_bev_cells / self.total_bev_cells if self.total_bev_cells else float("nan")
        )
        out["gt_edit_expanded_over_current_support_area"] = (
            self.expanded_support_bev_cells / self.support_bev_cells
            if self.support_bev_cells else float("nan")
        )
        return out


def _evaluate_strong(ds, n_eval: int, device) -> dict:
    state = _new_state()
    for i in range(n_eval):
        s = ds[i]
        payload = safe._sample_payload(s, device)
        _update_physical(
            state, payload["anchor"], payload["anchor"], payload["gt"], payload["write_bev"]
        )
        for h, fi in safe.REPORT.items():
            safe._update(
                state["metrics"], h, payload["anchor"][fi], payload["gt"][fi], payload["moving"][fi]
            )
    return _finish_state(state)


@torch.no_grad()
def _evaluate_checkpoint(label, path, ds, n_eval, vae, device, *, use_ema, seed, use_amp):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    arch = safe._require_trained_checkpoint_match(ck, ds, str(vae[1])) if False else None
    # The public failure diagnostic already validates the same checkpoint/cache
    # contract; repeat its exact checks here with the actual VAE path in main.
    del arch
    state_dict, weight_source = diag._checkpoint_state(ck, use_ema)
    arch = ck.get("architecture", {})
    model = diag._build_model_from_state(state_dict, arch, device)
    states = {name: _new_state() for name in VARIANTS}
    support_acc = SupportCoverageAccumulator()

    for i in range(n_eval):
        s = ds[i]
        payload = safe._sample_payload(s, device)
        prepared = diag._prepare_probe_sample(s, device)
        proposal = safe._decode_sparse_prediction(
            model,
            vae[0],
            s,
            prepared,
            seed=seed,
            use_amp=use_amp,
            guidance_scale=float(arch.get("guidance_scale", 1.0)),
            physics_condition=True,
            context_condition=True,
        )
        variants = _proposal_variants(
            payload["anchor"], proposal, payload["gt"], payload["write_bev"]
        )
        support_acc.update(payload["anchor"], payload["gt"], payload["write_bev"])

        for name, (variant_proposal, variant_support) in variants.items():
            _update_physical(
                states[name], payload["anchor"], variant_proposal, payload["gt"], variant_support
            )
            for h, fi in safe.REPORT.items():
                final = apply_dynamic_repair(
                    payload["anchor"][fi],
                    variant_proposal[fi],
                    variant_support[fi],
                    dynamic_class_ids=DYNAMIC_CLASS_IDS,
                    free_label=safe.FREE,
                )
                safe._update(
                    states[name]["metrics"], h, final, payload["gt"][fi], payload["moving"][fi]
                )
        if i % 8 == 0:
            print("oracle_gap_rollout", label, i, s["sample_id"])

    reports = {name: _finish_state(state) for name, state in states.items()}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "checkpoint": str(Path(path).resolve()),
        "checkpoint_sha256": file_sha256(path),
        "weight_source": weight_source,
        "architecture": arch,
        "variants": reports,
        "support_coverage": support_acc.compute(),
    }


def _print_summary(label: str, row: dict, strong: dict):
    current = row["variants"]["current"]
    print(f"\n=== ORACLE GAP DECOMPOSITION: {label} ===")
    print(f"{'variant':28s} {'Overall':>9s} {'Moving':>9s} {'dOverall':>10s} {'dMoving':>9s} {'recStrongM':>11s}")
    so, sm = _metric_pair(strong)
    print(f"{'strong_anchor':28s} {so:9.4f} {sm:9.4f} {'-':>10s} {'-':>9s} {'-':>11s}")
    for name in VARIANTS:
        report = row["variants"][name]
        o, m = _metric_pair(report)
        d = _delta(report, current)
        rec = _recovery(report, current, strong)["moving"]
        print(f"{name:28s} {o:9.4f} {m:9.4f} {d['overall']:+10.4f} {d['moving']:+9.4f} {100*rec:+10.1f}%")

    print("\n=== MOVING BY HORIZON ===")
    print(f"{'variant':28s} {'1s':>9s} {'2s':>9s} {'3s':>9s}")
    for name in VARIANTS:
        per_h = row["variants"][name]["deployment"]["moving"]["per_horizon"]
        vals = []
        for h in safe.REPORT:
            r = per_h[float(h)] if float(h) in per_h else per_h[str(float(h))]
            vals.append(float(r["mIoU"]))
        print(f"{name:28s} {vals[0]:9.4f} {vals[1]:9.4f} {vals[2]:9.4f}")

    cov = row["support_coverage"]
    print("\n=== CURRENT SUPPORT COVERAGE OF GT-REQUIRED DYNAMIC EDITS ===")
    for key in ("clear", "write", "relabel", "event"):
        v = cov[key]
        print(f"{key:8s} covered={v['covered']:10d}/{v['total']:10d} coverage={100*v['coverage']:.2f}%")
    print(f"gt_edit_expanded_over_current_support_area={cov['gt_edit_expanded_over_current_support_area']:.4f}")


def _attribution(row: dict, strong: dict) -> dict:
    v = row["variants"]
    current = v["current"]
    gains = {name: _delta(v[name], current) for name in VARIANTS if name != "current"}
    clear_m = gains["oracle_clear_keep"]["moving"]
    write_m = gains["oracle_write"]["moving"]
    event_m = gains["oracle_event_presence"]["moving"]
    return {
        "gain_vs_current": gains,
        "strong_gap": {
            "overall": _metric_pair(strong)[0] - _metric_pair(current)[0],
            "moving": _metric_pair(strong)[1] - _metric_pair(current)[1],
        },
        "strong_gap_recovery": {
            name: _recovery(v[name], current, strong) for name in VARIANTS if name != "current"
        },
        "clear_write_interaction_moving": event_m - clear_m - write_m,
        "support_increment_over_same_support_gt": _delta(
            v["oracle_support_gt_event"], v["same_support_gt_event"]
        ),
        "residual_between_event_presence_and_same_support_gt": _delta(
            v["same_support_gt_event"], v["oracle_event_presence"]
        ),
    }


def _checkpoint_comparison(a_label, a, b_label, b):
    ac = _moving_class_means(a["variants"]["current"])
    bc = _moving_class_means(b["variants"]["current"])
    rows = {}
    for c in map(str, DYNAMIC_CLASS_IDS):
        per_h = {}
        for h in map(str, map(float, safe.REPORT.keys())):
            per_h[h] = bc[c]["per_horizon"][h] - ac[c]["per_horizon"][h]
        rows[c] = {
            "class_name": ac[c]["class_name"],
            "delta_per_horizon": per_h,
            "delta_mean_over_horizons": bc[c]["mean_over_horizons"] - ac[c]["mean_over_horizons"],
        }
    return {
        "from": a_label,
        "to": b_label,
        "current_delta": _delta(b["variants"]["current"], a["variants"]["current"]),
        "moving_per_class_delta": rows,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--occfm-ckpt", required=True)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--checkpoint", action="append", type=_parse_checkpoint, required=True,
                   help="repeatable LABEL=/path/to/checkpoint.pt, e.g. E5=... E10=...")
    p.add_argument("--output", required=True)
    p.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=20260904)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-windows", type=int, default=0)
    a = p.parse_args()

    labels = [x[0] for x in a.checkpoint]
    if len(labels) != len(set(labels)):
        raise ValueError("checkpoint labels must be unique")
    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("oracle-gap diagnostic requires CUDA")

    ds = MSPWorldModelCacheDataset(a.cache)
    safe._validate_cache(ds, a.vae_ckpt)
    n_eval = len(ds) if int(a.max_windows) <= 0 else min(len(ds), int(a.max_windows))
    use_amp = bool(a.amp and device.type == "cuda")

    # Validate every checkpoint against the frozen evaluation cache before any rollout.
    checkpoints = []
    for label, path in a.checkpoint:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        safe._require_trained_checkpoint_match(ck, ds, a.vae_ckpt)
        checkpoints.append((label, path))
        del ck

    vae_model, _ = load_official_vae(UP, a.vae_ckpt, device)
    vae = OccFMVAEAdapter(vae_model)
    vae_pair = (vae, a.vae_ckpt)

    strong = _evaluate_strong(ds, n_eval, device)
    results = {}
    for label, path in checkpoints:
        row = _evaluate_checkpoint(
            label,
            path,
            ds,
            n_eval,
            vae_pair,
            device,
            use_ema=bool(a.use_ema),
            seed=int(a.seed),
            use_amp=use_amp,
        )
        row["moving_per_class_current"] = _moving_class_means(row["variants"]["current"])
        row["attribution"] = _attribution(row, strong)
        results[label] = row
        _print_summary(label, row, strong)

    comparisons = []
    ordered = [x[0] for x in checkpoints]
    for i in range(len(ordered) - 1):
        comp = _checkpoint_comparison(
            ordered[i], results[ordered[i]], ordered[i + 1], results[ordered[i + 1]]
        )
        comparisons.append(comp)
        print(f"\n=== CURRENT MOVING PER-CLASS DELTA: {ordered[i+1]} - {ordered[i]} ===")
        print(f"{'class':22s} {'1s':>9s} {'2s':>9s} {'3s':>9s} {'mean':>9s}")
        for c in map(str, DYNAMIC_CLASS_IDS):
            r = comp["moving_per_class_delta"][c]
            h = r["delta_per_horizon"]
            print(
                f"{r['class_name']:22s} {h['1.0']:+9.4f} {h['2.0']:+9.4f} "
                f"{h['3.0']:+9.4f} {r['delta_mean_over_horizons']:+9.4f}"
            )

    report = {
        "protocol": PROTOCOL,
        "cache": str(Path(a.cache).resolve()),
        "cache_index_sha256": file_sha256(Path(a.cache) / "index.json"),
        "occfm_checkpoint": str(Path(a.occfm_ckpt).resolve()),
        "occfm_checkpoint_sha256": file_sha256(a.occfm_ckpt),
        "vae_checkpoint": str(Path(a.vae_ckpt).resolve()),
        "vae_checkpoint_sha256": file_sha256(a.vae_ckpt),
        "num_windows": n_eval,
        "seed": int(a.seed),
        "use_ema": bool(a.use_ema),
        "sample_steps": 10,
        "strong_anchor": strong,
        "checkpoints": results,
        "checkpoint_comparisons": comparisons,
        "oracle_contract": {
            "all_learned_oracles_keep_frozen_msp_support": True,
            "support_oracle_comparison": "GT proposal under current support vs same GT proposal under current union exact GT-required dynamic-edit BEV support",
            "oracle_clear_keep": "fix dynamic presence on anchor-dynamic source voxels only; keep uses anchor class, not GT class",
            "oracle_write": "fix required writes and remove false writes on anchor-non-dynamic voxels",
            "oracle_semantic": "GT class only where proposal and GT are already both dynamic",
            "no_training": True,
        },
    }
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("saved", a.output)


if __name__ == "__main__":
    main()
