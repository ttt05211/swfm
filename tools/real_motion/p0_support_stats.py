#!/usr/bin/env python3
"""P0-B: support sparsity, expansion stages, window connectivity, and optional latent-energy audit.

Two different quantities are reported and must not be conflated:
1) causal binary support: where the sparse WM is allowed/requested to compute;
2) latent residual energy: where a GT-moving-only semantic correction actually
   carries meaningful VAE latent innovation.

The optional energy audit uses the deterministic VAE posterior mean to avoid
sampling noise. It is diagnostic-only and may use GT Moving-v2 support.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "upstream_occfm"
sys.path[:0] = [str(ROOT), str(UP)]

import numpy as np
import torch

from real_motion.prepared import PreparedShardDataset
from real_motion.support import build_motion_tube, MotionTubeConfig, downsample_support
from real_motion.windows import WindowPlanner, window_coverage
from real_motion.runtime_config import (
    add_config_args,
    load_runtime_config,
    get_cfg,
    save_resolved_config,
)


def ratio(n, d):
    return float(n / d) if d else 1.0


def _stage_template(future_frames):
    return [
        {
            "gt": 0,
            "kta_r0": 0,
            "tube": 0,
            "kta_lat0": 0,
            "tube_lat0": 0,
            "tube_lat_extra": 0,
            "dense_bev": 0,
            "dense_latent": 0,
            "gt_lat0": 0,
        }
        for _ in range(future_frames)
    ]


def _state_template():
    keys = ("occupied", "confident_static", "moving", "uncertain", "wm_candidate")
    return {
        k: {"voxels": 0, "bev_cells": 0}
        for k in keys
    } | {"dense_voxels": 0, "dense_bev": 0}


def _accumulate_t0_state(state, sample, free_label=17):
    cur = np.asarray(sample["full_history_occ"])[-1]
    occupied = cur != free_label
    sta = np.asarray(sample["t0_confident_static_mask"], dtype=bool)
    mov = np.asarray(sample["t0_moving_mask"], dtype=bool)
    unc = np.asarray(sample["t0_uncertain_mask"], dtype=bool)
    cand = mov | unc
    masks = {
        "occupied": occupied,
        "confident_static": sta,
        "moving": mov,
        "uncertain": unc,
        "wm_candidate": cand,
    }
    state["dense_voxels"] += int(cur.size)
    state["dense_bev"] += int(cur.shape[0] * cur.shape[1])
    for name, mask in masks.items():
        state[name]["voxels"] += int(mask.sum())
        state[name]["bev_cells"] += int(mask.any(axis=-1).sum())


def _finalize_t0_state(state):
    occupied_vox = state["occupied"]["voxels"]
    out = {}
    for name in ("occupied", "confident_static", "moving", "uncertain", "wm_candidate"):
        d = state[name]
        out[name] = {
            "voxel_over_dense": ratio(d["voxels"], state["dense_voxels"]),
            "voxel_over_occupied": ratio(d["voxels"], occupied_vox),
            "bev_over_dense": ratio(d["bev_cells"], state["dense_bev"]),
        }
    return out


def summarize(samples, radii, extra, schedule, window, maxw):
    F = 6

    def mk():
        return [
            {
                "inter": 0,
                "gt": 0,
                "active": 0,
                "dense": 0,
                "l_inter": 0,
                "l_gt": 0,
                "l_active": 0,
                "l_dense": 0,
            }
            for _ in range(F)
        ]

    stats = {r: mk() for r in radii}
    scheduled = mk()
    moving = [
        {
            "moving_vox": 0,
            "occupied_vox": 0,
            "moving_bev": 0,
            "dense_bev": 0,
            "moving_latent": 0,
            "dense_latent": 0,
        }
        for _ in range(F)
    ]
    stages = _stage_template(F)
    t0_state = _state_template()
    planner = WindowPlanner((window, window), maxw)
    wr = []

    for s in samples:
        _accumulate_t0_state(t0_state, s)

        kta = torch.from_numpy(np.asarray(s["kta_support"])).bool()
        fg = np.asarray(s["future_gt_occ"])
        fm = np.asarray(s["future_moving_occ"])
        gt = torch.from_numpy((fm != 17).any(axis=-1))

        for h in range(F):
            moving[h]["moving_vox"] += int((fm[h] != 17).sum())
            moving[h]["occupied_vox"] += int((fg[h] != 17).sum())
            mb = gt[h]
            moving[h]["moving_bev"] += int(mb.sum())
            moving[h]["dense_bev"] += mb.numel()
            ml = downsample_support(mb.unsqueeze(0), (50, 50), extra_radius=extra)[0]
            moving[h]["moving_latent"] += int(ml.sum())
            moving[h]["dense_latent"] += ml.numel()

        tube = build_motion_tube(kta, MotionTubeConfig(radii=list(schedule), latent_extra_radius=0))
        gl0 = downsample_support(gt, (50, 50), extra_radius=0)
        gl = downsample_support(gt, (50, 50), extra_radius=extra)
        kta_l0 = downsample_support(kta, (50, 50), extra_radius=0)
        tube_l0 = downsample_support(tube, (50, 50), extra_radius=0)
        tl = downsample_support(tube, (50, 50), extra_radius=extra)
        hl = downsample_support(
            torch.from_numpy(np.asarray(s["history_candidate_support"])).bool(),
            (50, 50),
            extra_radius=extra,
        )

        for h in range(F):
            d = stages[h]
            d["gt"] += int(gt[h].sum())
            d["kta_r0"] += int(kta[h].sum())
            d["tube"] += int(tube[h].sum())
            d["kta_lat0"] += int(kta_l0[h].sum())
            d["tube_lat0"] += int(tube_l0[h].sum())
            d["tube_lat_extra"] += int(tl[h].sum())
            d["dense_bev"] += int(gt[h].numel())
            d["dense_latent"] += int(tl[h].numel())
            d["gt_lat0"] += int(gl0[h].sum())

        req = tl.unsqueeze(0)
        ctx = torch.cat([hl, tl], 0).unsqueeze(0)
        plan = planner.plan(req, context_support=ctx)
        hu = hl.any(0)
        ru = tl.any(0)
        conn = torch.zeros_like(ru)
        nw = int(plan.valid.sum())
        withh = 0
        for ki in range(plan.valid.shape[1]):
            if not bool(plan.valid[0, ki]):
                continue
            y, x = [int(v) for v in plan.origins[0, ki].tolist()]
            has = bool(hu[y:y + window, x:x + window].any())
            if has:
                withh += 1
                conn[y:y + window, x:x + window] |= ru[y:y + window, x:x + window]
        rc = int(ru.sum())
        wr.append({
            "future_window_coverage": float(window_coverage(req, plan)[0]),
            "history_plus_future_context_coverage": float(window_coverage(ctx, plan)[0]),
            "future_windows_with_any_history_ratio": withh / nw if nw else 1.0,
            "future_required_cells_in_history_connected_windows_ratio": (
                float(conn.sum()) / rc if rc else 1.0
            ),
            "num_windows": nw,
            "slot_compute_ratio": nw * window * window / 2500.0,
        })

        for h in range(F):
            d = scheduled[h]
            d["inter"] += int((gt[h] & tube[h]).sum())
            d["gt"] += int(gt[h].sum())
            d["active"] += int(tube[h].sum())
            d["dense"] += tube[h].numel()
            d["l_inter"] += int((gl[h] & tl[h]).sum())
            d["l_gt"] += int(gl[h].sum())
            d["l_active"] += int(tl[h].sum())
            d["l_dense"] += tl[h].numel()

        for r in radii:
            tr = build_motion_tube(kta, MotionTubeConfig(radii=[r] * F, latent_extra_radius=0))
            trl = downsample_support(tr, (50, 50), extra_radius=extra)
            for h in range(F):
                d = stats[r][h]
                d["inter"] += int((gt[h] & tr[h]).sum())
                d["gt"] += int(gt[h].sum())
                d["active"] += int(tr[h].sum())
                d["dense"] += tr[h].numel()
                d["l_inter"] += int((gl[h] & trl[h]).sum())
                d["l_gt"] += int(gl[h].sum())
                d["l_active"] += int(trl[h].sum())
                d["l_dense"] += trl[h].numel()

    out = {
        "constant_radius_scan": {},
        "true_moving_sparsity": [],
        "scheduled_radius": [],
        "support_expansion_diagnostic": {
            "t0_motion_state": _finalize_t0_state(t0_state),
            "per_horizon": [],
            "interpretation": (
                "These are binary support ratios, not latent residual nonzero/energy ratios. "
                "Use them to locate where a causal selector expands."
            ),
        },
        "proposed_window_backend": {},
    }

    for r in radii:
        out["constant_radius_scan"][str(r)] = [
            {
                "horizon_s": 0.5 * (h + 1),
                "coverage_bev": ratio(d["inter"], d["gt"]),
                "active_ratio_bev": ratio(d["active"], d["dense"]),
                "coverage_latent": ratio(d["l_inter"], d["l_gt"]),
                "active_ratio_latent": ratio(d["l_active"], d["l_dense"]),
            }
            for h, d in enumerate(stats[r])
        ]

    for h, d in enumerate(scheduled):
        out["scheduled_radius"].append({
            "horizon_s": 0.5 * (h + 1),
            "radius": int(schedule[h]),
            "coverage_bev": ratio(d["inter"], d["gt"]),
            "active_ratio_bev": ratio(d["active"], d["dense"]),
            "coverage_latent": ratio(d["l_inter"], d["l_gt"]),
            "active_ratio_latent": ratio(d["l_active"], d["l_dense"]),
        })

    for h, d in enumerate(stages):
        out["support_expansion_diagnostic"]["per_horizon"].append({
            "horizon_s": 0.5 * (h + 1),
            "gt_moving_bev_ratio": ratio(d["gt"], d["dense_bev"]),
            "kta_radius0_bev_ratio": ratio(d["kta_r0"], d["dense_bev"]),
            "scheduled_tube_bev_ratio": ratio(d["tube"], d["dense_bev"]),
            "gt_moving_latent_r0_ratio": ratio(d["gt_lat0"], d["dense_latent"]),
            "kta_radius0_latent_before_extra_ratio": ratio(d["kta_lat0"], d["dense_latent"]),
            "scheduled_tube_latent_before_extra_ratio": ratio(d["tube_lat0"], d["dense_latent"]),
            "scheduled_tube_latent_after_extra_ratio": ratio(d["tube_lat_extra"], d["dense_latent"]),
        })

    if wr:
        for k in wr[0]:
            v = [r[k] for r in wr]
            out["proposed_window_backend"][k] = {
                "mean": float(np.mean(v)),
                "p05": float(np.quantile(v, 0.05)),
                "min": float(np.min(v)),
                "max": float(np.max(v)),
            }
        out["proposed_window_backend"].update({
            "window_hw": [window, window],
            "max_windows": maxw,
        })

    for h, d in enumerate(moving):
        out["true_moving_sparsity"].append({
            "horizon_s": 0.5 * (h + 1),
            "moving_voxel_over_occupied": ratio(d["moving_vox"], d["occupied_vox"]),
            "moving_bev_over_dense": ratio(d["moving_bev"], d["dense_bev"]),
            "moving_latent_over_dense": ratio(d["moving_latent"], d["dense_latent"]),
        })

    return out


def _top_energy_fraction(flat_energy, fraction):
    if flat_energy.size == 0:
        return float("nan")
    total = float(flat_energy.sum())
    if total <= 0:
        return float("nan")
    k = max(1, int(np.ceil(float(fraction) * flat_energy.size)))
    if k >= flat_energy.size:
        return 1.0
    idx = np.argpartition(flat_energy, -k)[-k:]
    return float(flat_energy[idx].sum() / total)


def latent_energy_localization(dataset, n, vae_ckpt, device, radii, thresholds):
    from real_motion.occfm_io import load_official_vae, OccFMVAEAdapter

    vae, _ = load_official_vae(UP, vae_ckpt, device)
    ad = OccFMVAEAdapter(vae)

    all_energy = []
    threshold_hits = {float(t): 0 for t in thresholds}
    total_cells = 0
    residual_sq_sum = 0.0
    residual_numel = 0
    changed_voxels = 0
    moving_support_voxels = 0
    dense_semantic_voxels = 0

    radius_energy = {
        int(r): {"inside": 0.0, "total": 0.0, "active": 0, "dense": 0}
        for r in radii
    }
    per_h = {
        h: {
            int(r): {"inside": 0.0, "total": 0.0, "active": 0, "dense": 0}
            for r in radii
        }
        for h in range(6)
    }

    with torch.no_grad():
        for i in range(n):
            s = dataset[i]
            kta = np.asarray(s["kta_future_occ"])
            gt = np.asarray(s["future_gt_occ"])
            support = np.asarray(s["gt_moving_support"], dtype=bool)

            hybrid = np.where(support, gt, kta)
            changed_voxels += int(((hybrid != kta) & support).sum())
            moving_support_voxels += int(support.sum())
            dense_semantic_voxels += int(support.size)

            zk = ad.encode(
                torch.from_numpy(kta).unsqueeze(0),
                mode="mean",
            )[0]
            zh = ad.encode(
                torch.from_numpy(hybrid).unsqueeze(0),
                mode="mean",
            )[0]
            residual = (zh - zk).float()
            cell_energy = residual.square().sum(dim=1)
            cell_norm = cell_energy.sqrt()

            e_np = cell_energy.detach().cpu().numpy()
            n_np = cell_norm.detach().cpu().numpy()
            all_energy.append(e_np.reshape(-1))
            total_cells += int(n_np.size)
            for t in thresholds:
                threshold_hits[float(t)] += int((n_np > float(t)).sum())
            residual_sq_sum += float(residual.square().sum().item())
            residual_numel += int(residual.numel())

            moving_bev = torch.from_numpy(support.any(axis=-1)).bool()
            for r in radii:
                mask = downsample_support(
                    moving_bev,
                    (cell_energy.shape[-2], cell_energy.shape[-1]),
                    extra_radius=int(r),
                ).to(device=cell_energy.device)
                inside = float(cell_energy[mask].sum().item())
                total = float(cell_energy.sum().item())
                radius_energy[int(r)]["inside"] += inside
                radius_energy[int(r)]["total"] += total
                radius_energy[int(r)]["active"] += int(mask.sum().item())
                radius_energy[int(r)]["dense"] += int(mask.numel())
                for h in range(6):
                    mh = mask[h]
                    eh = cell_energy[h]
                    per_h[h][int(r)]["inside"] += float(eh[mh].sum().item())
                    per_h[h][int(r)]["total"] += float(eh.sum().item())
                    per_h[h][int(r)]["active"] += int(mh.sum().item())
                    per_h[h][int(r)]["dense"] += int(mh.numel())

    flat_energy = np.concatenate(all_energy) if all_energy else np.zeros(0, dtype=np.float64)
    result = {
        "status": "ok",
        "num_windows": int(n),
        "vae_latent_mode": "mean",
        "target_definition": (
            "r_motion = E(Y_hybrid)-E(Y_KTA), where Y_hybrid uses GT only inside "
            "GT Moving-v2 support and equals KTA elsewhere"
        ),
        "semantic_change": {
            "changed_voxels_where_hybrid_differs_from_kta": int(changed_voxels),
            "moving_support_voxels": int(moving_support_voxels),
            "changed_voxel_over_dense": ratio(changed_voxels, dense_semantic_voxels),
            "moving_support_voxel_over_dense": ratio(moving_support_voxels, dense_semantic_voxels),
        },
        "latent_residual": {
            "rms": float(np.sqrt(residual_sq_sum / residual_numel)) if residual_numel else float("nan"),
            "cell_fraction_above_norm_threshold": {
                f"{float(t):.0e}" if float(t) else "0": ratio(threshold_hits[float(t)], total_cells)
                for t in thresholds
            },
            "energy_at_top_cell_fraction": {
                "top_1pct": _top_energy_fraction(flat_energy, 0.01),
                "top_5pct": _top_energy_fraction(flat_energy, 0.05),
                "top_10pct": _top_energy_fraction(flat_energy, 0.10),
            },
        },
        "energy_inside_gt_moving_latent_radius": {},
        "per_horizon_energy_inside_gt_moving_latent_radius": {},
        "note": (
            "A high raw nonzero-cell fraction does not imply dense meaningful innovation. "
            "Energy concentration and energy captured near GT-moving support are the relevant diagnostics."
        ),
    }

    for r in radii:
        d = radius_energy[int(r)]
        result["energy_inside_gt_moving_latent_radius"][str(int(r))] = {
            "energy_fraction": ratio(d["inside"], d["total"]),
            "mask_active_ratio": ratio(d["active"], d["dense"]),
        }

    for h in range(6):
        rows = {}
        for r in radii:
            d = per_h[h][int(r)]
            rows[str(int(r))] = {
                "energy_fraction": ratio(d["inside"], d["total"]),
                "mask_active_ratio": ratio(d["active"], d["dense"]),
            }
        result["per_horizon_energy_inside_gt_moving_latent_radius"][str(0.5 * (h + 1))] = rows

    return result


def main():
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument("--prepared", required=True)
    p.add_argument("--radii", default="0,1,2,3,4,5,6")
    p.add_argument("--latent-extra-radius", type=int, default=None)
    p.add_argument("--schedule", default=None)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--window-size", type=int, default=None)
    p.add_argument("--window-slots", type=int, default=None)
    p.add_argument("--vae-ckpt", default=None, help="optional: enable Motion-Hybrid latent energy audit")
    p.add_argument("--device", default="cuda")
    p.add_argument("--energy-max-windows", type=int, default=None)
    p.add_argument("--energy-radii", default="0,1,2,3")
    p.add_argument("--energy-thresholds", default="0,1e-8,1e-6,1e-5,1e-4,1e-3,1e-2")
    p.add_argument("--output", required=True)
    a = p.parse_args()

    cfg = load_runtime_config(a.config, a.override)
    radii = [int(x) for x in a.radii.split(",")]
    schedule = tuple(
        int(x)
        for x in (
            a.schedule.split(",")
            if a.schedule
            else get_cfg(cfg, "MOTION.KTA_TUBE_RADII")
        )
    )
    extra = int(
        a.latent_extra_radius
        if a.latent_extra_radius is not None
        else get_cfg(cfg, "MOTION.LATENT_EXTRA_RADIUS", 1)
    )
    window = int(a.window_size or get_cfg(cfg, "MODEL.WINDOW_HW", [20, 20])[0])
    slots = int(a.window_slots or get_cfg(cfg, "MODEL.MAX_WINDOWS", 8))

    ds = PreparedShardDataset(a.prepared)
    n = len(ds) if a.max_windows is None else min(len(ds), a.max_windows)
    res = summarize((ds[i] for i in range(n)), radii, extra, schedule, window, slots)
    res["num_windows"] = n

    if a.vae_ckpt:
        energy_n = n if a.energy_max_windows is None else min(n, a.energy_max_windows)
        energy_radii = [int(x) for x in a.energy_radii.split(",")]
        thresholds = [float(x) for x in a.energy_thresholds.split(",")]
        res["motion_hybrid_latent_energy_localization"] = latent_energy_localization(
            ds,
            energy_n,
            a.vae_ckpt,
            a.device,
            energy_radii,
            thresholds,
        )
    else:
        res["motion_hybrid_latent_energy_localization"] = {
            "status": "skipped",
            "reason": "pass --vae-ckpt to run the optional VAE latent energy audit",
        }

    op = Path(a.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(res, indent=2), encoding="utf-8")
    save_resolved_config(cfg, op.with_suffix(".resolved.yaml"))

    print(json.dumps({
        "saved": str(a.output),
        "num_windows": n,
        "t0_motion_state": res["support_expansion_diagnostic"]["t0_motion_state"],
        "window_backend": res["proposed_window_backend"],
        "energy_audit": res["motion_hybrid_latent_energy_localization"].get("status"),
    }, indent=2))


if __name__ == "__main__":
    main()
