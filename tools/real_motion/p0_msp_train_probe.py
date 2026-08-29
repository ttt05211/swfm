#!/usr/bin/env python3
"""P0-F1: train a tiny MSP and evaluate support-oracle vs sparse budget.

No occupancy world model is trained here.  The gate is deliberately upstream:
if the learned support cannot improve the GT-filled Moving oracle at comparable
10--15% latent budget, the MSP idea should be stopped before expensive WM work.
"""
import argparse
import copy
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from real_motion.composition import static_protected_compose
from real_motion.metrics.moving_miou_v2 import (
    DYNAMIC_CLASS_IDS,
    MovingMIoUV2MultiHorizon,
    REPORT_HORIZONS_S,
    SemanticIoUAccumulator,
)
from real_motion.msp import (
    FEATURE_DIM,
    MSP_CACHE_VERSION,
    MSPProbeHead,
    collate_probe_records,
    latent_support_to_bev,
    msp_probe_loss,
    rasterize_msp_scores,
    top_budget_support,
    validate_probe_record,
)
from real_motion.nuscenes_adapter import (
    NuScenesWindowSource,
    WindowTokens,
    causal_dynamic_target_semantics,
    dynamic_only_semantics,
)
from real_motion.prepared import prepare_nuscenes_window
from real_motion.runtime_config import (
    add_config_args,
    get_cfg,
    load_runtime_config,
    make_prepare_config,
    save_resolved_config,
)
from real_motion.support import downsample_support
from real_motion.windows import WindowPlanner, window_coverage

REPORT = {1.0: 1, 2.0: 3, 3.0: 5}
DEFAULT_SEED = 20260829


class RecordDataset(Dataset):
    def __init__(self, records):
        self.records = list(records)
    def __len__(self):
        return len(self.records)
    def __getitem__(self, i):
        return self.records[i]


def _load_cache(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("version") != MSP_CACHE_VERSION:
        raise RuntimeError(
            f"MSP cache {path} version {payload.get('version')} != {MSP_CACHE_VERSION}; rebuild"
        )
    meta = payload.get("metadata", {})
    if int(meta.get("feature_dim", -1)) != FEATURE_DIM:
        raise RuntimeError("MSP cache feature contract differs from current code")
    records = payload.get("records", [])
    if not records:
        raise RuntimeError(f"MSP cache {path} contains no records")
    for r in records:
        validate_probe_record(r)
    return meta, records


def _move_batch(batch, device):
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def _mean_h(acc):
    per = {h: acc[h].compute() for h in REPORT_HORIZONS_S}
    return {
        "mIoU": float(np.nanmean([per[h]["mIoU"] for h in REPORT_HORIZONS_S])),
        "per_horizon": per,
    }


@torch.no_grad()
def _val_loss(model, loader, device, positive_weight):
    model.eval()
    total = 0.0
    bce = 0.0
    nll = 0.0
    count = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        out = model(batch["features"], batch["candidate_mask"])
        loss, info = msp_probe_loss(out, batch, positive_weight=positive_weight)
        bs = len(batch["sample_id"])
        total += float(loss.item()) * bs
        bce += float(info["activation_bce"]) * bs
        nll += float(info["location_nll"]) * bs
        count += bs
    model.train()
    d = max(count, 1)
    return {"loss": total/d, "activation_bce": bce/d, "location_nll": nll/d}


def _record_window(r):
    return WindowTokens(
        scene_name=str(r["scene_name"]),
        history_tokens=tuple(r["history_tokens"]),
        t0_token=str(r["t0_token"]),
        future_tokens=tuple(r["future_tokens"]),
    )


def _new_oracle_state():
    return {
        "overall": {h: SemanticIoUAccumulator() for h in REPORT_HORIZONS_S},
        "moving": MovingMIoUV2MultiHorizon(),
        "arrival_hit": np.zeros(6, dtype=np.float64),
        "arrival_total": np.zeros(6, dtype=np.float64),
        "active_latent": np.zeros(6, dtype=np.float64),
        "num_windows": [],
        "slot_compute_ratio": [],
        "window_coverage": [],
    }


def _accumulate_arrival(state, arrival_vox, write_bev):
    a = torch.as_tensor(arrival_vox, dtype=torch.bool)
    w = torch.as_tensor(write_bev, dtype=torch.bool)
    if a.ndim != 4 or w.ndim != 3 or tuple(a.shape[:3]) != tuple(w.shape):
        raise ValueError("arrival/write support shape mismatch")
    state["arrival_total"] += a.sum(dim=(1,2,3), dtype=torch.float64).numpy()
    state["arrival_hit"] += (a & w.unsqueeze(-1)).sum(
        dim=(1,2,3), dtype=torch.float64
    ).numpy()


def _state_report(state, processed):
    denom = np.maximum(state["arrival_total"], 1.0)
    return {
        "oracle_overall": _mean_h(state["overall"]),
        "oracle_Moving-mIoU_v2": state["moving"].compute(),
        "future_arrival_recall_per_horizon": (state["arrival_hit"] / denom).tolist(),
        "active_latent_per_horizon": (state["active_latent"] / max(processed, 1)).tolist(),
        "window_backend": {
            "mean_num_windows": float(np.mean(state["num_windows"])) if state["num_windows"] else 0.0,
            "mean_slot_compute_ratio": float(np.mean(state["slot_compute_ratio"])) if state["slot_compute_ratio"] else 0.0,
            "mean_window_coverage": float(np.mean(state["window_coverage"])) if state["window_coverage"] else 1.0,
        },
    }


def _plan_cost(state, latent_support, history_lat, planner, window, latent_hw):
    req = latent_support.unsqueeze(0)
    ctx = torch.cat([history_lat, latent_support], dim=0).unsqueeze(0)
    plan = planner.plan(req, context_support=ctx)
    nw = int(plan.valid.sum())
    state["num_windows"].append(nw)
    state["slot_compute_ratio"].append(
        nw * window * window / float(latent_hw[0] * latent_hw[1])
    )
    state["window_coverage"].append(float(window_coverage(req, plan)[0]))


@torch.no_grad()
def evaluate_oracle_curve(model, records, source, pcfg, cfg, budgets, device):
    model.eval()
    latent_hw = tuple(int(v) for v in get_cfg(cfg, "UPSTREAM.LATENT_HW", [50, 50]))
    window_hw = tuple(int(v) for v in get_cfg(cfg, "MODEL.WINDOW_HW", [20, 20]))
    if window_hw[0] != window_hw[1]:
        raise ValueError("MSP probe currently expects square sparse windows")
    window = window_hw[0]
    planner = WindowPlanner(window_hw, int(get_cfg(cfg, "MODEL.MAX_WINDOWS", 8)))

    decomposition = {
        "overall": {h: SemanticIoUAccumulator() for h in REPORT_HORIZONS_S},
        "moving": MovingMIoUV2MultiHorizon(),
    }
    rule = _new_oracle_state()
    learned = {float(b): _new_oracle_state() for b in budgets}
    processed = 0

    for i, r in enumerate(records):
        batch = collate_probe_records([r])
        batch_dev = _move_batch(batch, device)
        out = model(batch_dev["features"], batch_dev["candidate_mask"])
        score = rasterize_msp_scores(
            out, batch_dev, latent_hw=latent_hw, grid=pcfg.grid
        ).cpu()
        learned_latent = {
            float(b): top_budget_support(score, float(b))[0]
            for b in budgets
        }

        w = _record_window(r)
        base = prepare_nuscenes_window(source, w, pcfg, include_gt=True)
        rule_bev = torch.from_numpy(np.asarray(base["generation_support_occ"])).bool()
        rule_lat = downsample_support(rule_bev, latent_hw, extra_radius=0)
        history_lat = downsample_support(
            torch.from_numpy(np.asarray(base["history_candidate_support"])).bool(),
            latent_hw,
            extra_radius=0,
        )
        arrival = np.asarray(base["future_moving_occ"]) != pcfg.free_label

        rule["active_latent"] += rule_lat.float().mean(dim=(1,2)).numpy()
        _accumulate_arrival(rule, arrival, rule_bev)
        _plan_cost(rule, rule_lat, history_lat, planner, window, latent_hw)

        learned_bev = {}
        for b, lat in learned_latent.items():
            state = learned[b]
            state["active_latent"] += lat.float().mean(dim=(1,2)).numpy()
            bev = latent_support_to_bev(lat, pcfg.grid.shape_hwd[:2])
            learned_bev[b] = bev
            _accumulate_arrival(state, arrival, bev)
            _plan_cost(state, lat, history_lat, planner, window, latent_hw)

        for h, fi in REPORT.items():
            gt = np.asarray(base["future_gt_occ"])[fi]
            static = torch.from_numpy(np.asarray(base["static_future_occ"])[fi])
            prot = torch.from_numpy(np.asarray(base["confident_static_future_mask"])[fi])
            all_dyn = torch.from_numpy(dynamic_only_semantics(gt, pcfg.free_label))
            dec = static_protected_compose(
                static, all_dyn, prot, DYNAMIC_CLASS_IDS, write_support=None
            ).numpy()
            decomposition["overall"][h].update(dec, gt)
            decomposition["moving"].update(
                h, dec, gt, np.asarray(base["gt_moving_support"])[fi]
            )

            rw = rule_bev[fi]
            rdyn = torch.from_numpy(
                causal_dynamic_target_semantics(gt, rw.numpy(), pcfg.free_label)
            )
            rpred = static_protected_compose(
                static, rdyn, prot, DYNAMIC_CLASS_IDS, write_support=rw
            ).numpy()
            rule["overall"][h].update(rpred, gt)
            rule["moving"].update(
                h, rpred, gt, np.asarray(base["gt_moving_support"])[fi]
            )

            for b, bev in learned_bev.items():
                lw = bev[fi]
                ldyn = torch.from_numpy(
                    causal_dynamic_target_semantics(gt, lw.numpy(), pcfg.free_label)
                )
                lpred = static_protected_compose(
                    static, ldyn, prot, DYNAMIC_CLASS_IDS, write_support=lw
                ).numpy()
                learned[b]["overall"][h].update(lpred, gt)
                learned[b]["moving"].update(
                    h, lpred, gt, np.asarray(base["gt_moving_support"])[fi]
                )

        processed += 1
        if i % 8 == 0:
            print("oracle eval", i, r["sample_id"])

    decomp_report = {
        "oracle_overall": _mean_h(decomposition["overall"]),
        "oracle_Moving-mIoU_v2": decomposition["moving"].compute(),
    }
    rule_report = _state_report(rule, processed)
    learned_report = {
        str(b): _state_report(learned[float(b)], processed)
        for b in budgets
    }
    rule_moving = float(rule_report["oracle_Moving-mIoU_v2"]["mIoU"])
    for b in budgets:
        row = learned_report[str(b)]
        row["delta_Moving_vs_rule"] = float(
            row["oracle_Moving-mIoU_v2"]["mIoU"] - rule_moving
        )
    return {
        "num_windows": processed,
        "decomposition": decomp_report,
        "frozen_hybrid_v6": rule_report,
        "learned_msp_by_budget": learned_report,
    }


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--val-info-pkl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=96)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-modes", type=int, default=4)
    p.add_argument("--positive-weight", type=float, default=2.0)
    p.add_argument("--budgets", default="0.10,0.125,0.15")
    p.add_argument("--val-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=0)
    a = p.parse_args()
    if a.steps <= 0 or a.batch_size <= 0 or a.val_every <= 0:
        raise ValueError("steps/batch-size/val-every must be positive")

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")

    cfg = load_runtime_config(a.config, a.override)
    pcfg = make_prepare_config(cfg)
    train_meta, train_records = _load_cache(a.train_cache)
    val_meta, val_records = _load_cache(a.val_cache)
    train_scenes = set(train_meta.get("scene_names", []))
    val_scenes = set(val_meta.get("scene_names", []))
    overlap = sorted(train_scenes & val_scenes)
    if overlap:
        raise RuntimeError(
            f"MSP train/val scene leakage detected ({len(overlap)} scenes), e.g. {overlap[:5]}"
        )
    if val_meta.get("selection") != "scene_disjoint_midpoint_one_window_per_scene_v1":
        raise RuntimeError("formal MSP probe requires scene-disjoint val cache")

    budgets = tuple(float(x) for x in a.budgets.split(",") if x.strip())
    if not budgets or any(not 0 < b <= 1 for b in budgets):
        raise ValueError("budgets must be comma-separated ratios in (0,1]")
    budgets = tuple(sorted(set(budgets)))

    train_loader = DataLoader(
        RecordDataset(train_records), batch_size=a.batch_size, shuffle=True,
        num_workers=a.num_workers, collate_fn=collate_probe_records, drop_last=False,
    )
    val_loader = DataLoader(
        RecordDataset(val_records), batch_size=a.batch_size, shuffle=False,
        num_workers=a.num_workers, collate_fn=collate_probe_records, drop_last=False,
    )
    model = MSPProbeHead(
        feature_dim=FEATURE_DIM,
        hidden_dim=a.hidden_dim,
        num_heads=a.num_heads,
        num_modes=a.num_modes,
        future_frames=pcfg.future_frames,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=a.lr, weight_decay=a.weight_decay
    )

    model.train()
    best_val = float("inf")
    best_state = None
    history = []
    step = 0
    iterator = iter(train_loader)
    while step < a.steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch["features"], batch["candidate_mask"])
        loss, info = msp_probe_loss(
            out, batch, positive_weight=float(a.positive_weight)
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite MSP loss at step {step}: {loss}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        step += 1

        if step == 1 or step % 20 == 0:
            print(
                f"step={step} loss={info['loss']:.5f} "
                f"bce={info['activation_bce']:.5f} nll={info['location_nll']:.5f}"
            )
        if step % a.val_every == 0 or step == a.steps:
            val = _val_loss(model, val_loader, device, float(a.positive_weight))
            row = {"step": step, "train": info, "val": val}
            history.append(row)
            print("validation", json.dumps(row))
            if val["loss"] < best_val:
                best_val = float(val["loss"])
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

    if best_state is None:
        raise RuntimeError("MSP probe never produced a validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.to(device)

    source = NuScenesWindowSource(a.dataroot, info_pkl=a.val_info_pkl, verbose=False)
    oracle = evaluate_oracle_curve(
        model, val_records, source, pcfg, cfg, budgets, device
    )
    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": best_state,
        "feature_dim": FEATURE_DIM,
        "hidden_dim": int(a.hidden_dim),
        "num_heads": int(a.num_heads),
        "num_modes": int(a.num_modes),
        "future_frames": int(pcfg.future_frames),
        "best_val_loss": best_val,
        "train_metadata": train_meta,
        "val_metadata": val_meta,
        "args": vars(a),
        "resolved_config": cfg,
    }
    torch.save(ckpt, out_dir / "msp_probe_best.pt")
    report = {
        "protocol": {
            "name": "p0_f1_budgeted_msp_probe_v1",
            "probe_only": True,
            "world_model_trained": False,
            "feature_contract": "causal_occ_components_only_no_gt_instance_features",
            "model": {
                "hidden_dim": int(a.hidden_dim),
                "num_heads": int(a.num_heads),
                "transformer_layers": 1,
                "num_modes": int(a.num_modes),
            },
            "budgets": list(budgets),
            "train_windows": len(train_records),
            "val_windows": len(val_records),
            "train_unique_scenes": len(train_scenes),
            "val_unique_scenes": len(val_scenes),
            "scene_overlap": 0,
        },
        "best_val_loss": best_val,
        "training_history": history,
        "oracle_curve": oracle,
        "decision_note": (
            "Use learned_msp_by_budget Moving oracle and window cost as the gate. "
            "Do not start full sparse-WM training solely because center NLL decreases."
        ),
    }
    (out_dir / "msp_probe_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    save_resolved_config(cfg, out_dir / "resolved_config.yaml")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
