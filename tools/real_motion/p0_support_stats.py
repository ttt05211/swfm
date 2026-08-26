#!/usr/bin/env python3
"""P0-B: per-horizon sparsity and KTA motion-tube coverage trade-off."""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from real_motion.prepared import PreparedShardDataset
from real_motion.support import build_motion_tube, MotionTubeConfig, downsample_support
from real_motion.windows import WindowPlanner, window_coverage


def ratio(n, d):
    return float(n / d) if d else 1.0


def summarize(samples, radii, latent_extra, schedule=(1,2,3,4,5,6), window=20, max_windows=8):
    F = 6
    stats = {r: [{"inter":0,"gt":0,"active":0,"dense":0,
                  "l_inter":0,"l_gt":0,"l_active":0,"l_dense":0} for _ in range(F)]
             for r in radii}
    moving = [{"moving_vox":0,"occupied_vox":0,"moving_bev":0,"dense_bev":0,
               "moving_latent":0,"dense_latent":0} for _ in range(F)]
    scheduled = [{"inter":0,"gt":0,"active":0,"dense":0,
                  "l_inter":0,"l_gt":0,"l_active":0,"l_dense":0} for _ in range(F)]
    planner=WindowPlanner((window,window),max_windows)
    window_rows=[]

    for s in samples:
        kta = torch.from_numpy(np.asarray(s["kta_support"])).bool()
        # Tube coverage is a generation-reachability statistic. Use actual
        # eligible moving *arrival voxels*, not the dual-box metric support
        # (whose old-location half exists to penalize trailing ghosts).
        fut_gt = np.asarray(s["future_gt_occ"])
        fut_mov = np.asarray(s["future_moving_occ"])
        gt_bev = torch.from_numpy((fut_mov != 17).any(axis=-1))
        for h in range(F):
            moving[h]["moving_vox"] += int((fut_mov[h] != 17).sum())
            moving[h]["occupied_vox"] += int((fut_gt[h] != 17).sum())
            mov_bev = torch.from_numpy((fut_mov[h] != 17).any(axis=-1))
            moving[h]["moving_bev"] += int(mov_bev.sum())
            moving[h]["dense_bev"] += int(mov_bev.numel())
            ml = downsample_support(mov_bev.unsqueeze(0), (50,50), extra_radius=latent_extra)[0]
            moving[h]["moving_latent"] += int(ml.sum())
            moving[h]["dense_latent"] += int(ml.numel())

        tube_sched = build_motion_tube(kta, MotionTubeConfig(radii=list(schedule), latent_extra_radius=0))
        gt_lat_sched = downsample_support(gt_bev, (50,50), extra_radius=latent_extra)
        tube_lat_sched = downsample_support(tube_sched, (50,50), extra_radius=latent_extra)

        hist_lat=downsample_support(
            torch.from_numpy(np.asarray(s["history_candidate_support"])).bool(),
            (50,50),extra_radius=latent_extra)
        required=tube_lat_sched.unsqueeze(0)
        context=torch.cat([hist_lat,tube_lat_sched],dim=0).unsqueeze(0)
        plan=planner.plan(required,context_support=context)
        future_cov=float(window_coverage(required,plan)[0])
        context_cov=float(window_coverage(context,plan)[0])
        nwin=int(plan.valid.sum())

        # A global context-coverage number is not enough: different sparse
        # windows do not communicate. Quantify whether windows opened for future
        # targets actually contain any historical motion evidence themselves.
        hist_union=hist_lat.any(dim=0)
        req_union=tube_lat_sched.any(dim=0)
        future_covered_by_hist_window=torch.zeros_like(req_union)
        windows_with_history=0
        wh=ww=window
        for ki in range(plan.valid.shape[1]):
            if not bool(plan.valid[0,ki]):
                continue
            y,x=[int(v) for v in plan.origins[0,ki].tolist()]
            has_hist=bool(hist_union[y:y+wh,x:x+ww].any())
            if has_hist:
                windows_with_history += 1
                future_covered_by_hist_window[y:y+wh,x:x+ww] |= req_union[y:y+wh,x:x+ww]
        req_count=int(req_union.sum())
        same_window_future_ratio=(
            float(future_covered_by_hist_window.sum())/req_count if req_count else 1.0
        )
        window_rows.append({
            "future_window_coverage":future_cov,
            "history_plus_future_context_coverage":context_cov,
            "future_windows_with_any_history_ratio":(windows_with_history/nwin if nwin else 1.0),
            "future_required_cells_in_history_connected_windows_ratio":same_window_future_ratio,
            "num_windows":nwin,
            "slot_compute_ratio":nwin*window*window/(50*50),
        })

        for h in range(F):
            d=scheduled[h]
            d["inter"]+=int((gt_bev[h]&tube_sched[h]).sum()); d["gt"]+=int(gt_bev[h].sum())
            d["active"]+=int(tube_sched[h].sum()); d["dense"]+=int(tube_sched[h].numel())
            d["l_inter"]+=int((gt_lat_sched[h]&tube_lat_sched[h]).sum()); d["l_gt"]+=int(gt_lat_sched[h].sum())
            d["l_active"]+=int(tube_lat_sched[h].sum()); d["l_dense"]+=int(tube_lat_sched[h].numel())

        for r in radii:
            tube = build_motion_tube(kta, MotionTubeConfig(radii=[r]*F, latent_extra_radius=0))
            gt_lat = downsample_support(gt_bev, (50,50), extra_radius=latent_extra)
            tube_lat = downsample_support(tube, (50,50), extra_radius=latent_extra)
            for h in range(F):
                d = stats[r][h]
                d["inter"] += int((gt_bev[h] & tube[h]).sum())
                d["gt"] += int(gt_bev[h].sum())
                d["active"] += int(tube[h].sum())
                d["dense"] += int(tube[h].numel())
                d["l_inter"] += int((gt_lat[h] & tube_lat[h]).sum())
                d["l_gt"] += int(gt_lat[h].sum())
                d["l_active"] += int(tube_lat[h].sum())
                d["l_dense"] += int(tube_lat[h].numel())

    result = {"constant_radius_scan": {}, "true_moving_sparsity": [], "scheduled_radius": [],
              "proposed_window_backend": {}}
    for r in radii:
        result["constant_radius_scan"][str(r)] = []
        for h,d in enumerate(stats[r]):
            result["constant_radius_scan"][str(r)].append({
                "horizon_s": 0.5*(h+1),
                "coverage_bev": ratio(d["inter"], d["gt"]),
                "active_ratio_bev": ratio(d["active"], d["dense"]),
                "coverage_latent": ratio(d["l_inter"], d["l_gt"]),
                "active_ratio_latent": ratio(d["l_active"], d["l_dense"]),
            })
    for h,d in enumerate(scheduled):
        result["scheduled_radius"].append({
            "horizon_s":0.5*(h+1),"radius":int(schedule[h]),
            "coverage_bev":ratio(d["inter"],d["gt"]),"active_ratio_bev":ratio(d["active"],d["dense"]),
            "coverage_latent":ratio(d["l_inter"],d["l_gt"]),"active_ratio_latent":ratio(d["l_active"],d["l_dense"]),
        })
    if window_rows:
        for key in window_rows[0]:
            vals=[r[key] for r in window_rows]
            result["proposed_window_backend"][key]={
                "mean":float(np.mean(vals)),"p05":float(np.quantile(vals,0.05)),
                "min":float(np.min(vals)),"max":float(np.max(vals)),
            }
        result["proposed_window_backend"]["window_hw"]=[window,window]
        result["proposed_window_backend"]["max_windows"]=max_windows
    for h,d in enumerate(moving):
        result["true_moving_sparsity"].append({
            "horizon_s": 0.5*(h+1),
            "moving_voxel_over_occupied": ratio(d["moving_vox"], d["occupied_vox"]),
            "moving_bev_over_dense": ratio(d["moving_bev"], d["dense_bev"]),
            "moving_latent_over_dense": ratio(d["moving_latent"], d["dense_latent"]),
        })
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", required=True)
    p.add_argument("--radii", default="0,1,2,3,4,5,6")
    p.add_argument("--latent-extra-radius", type=int, default=1)
    p.add_argument("--schedule", default="1,2,3,4,5,6", help="time-dependent radii for the proposed tube")
    p.add_argument("--max-windows", type=int, default=None, help="limit number of dataset windows")
    p.add_argument("--window-size", type=int, default=20)
    p.add_argument("--window-slots", type=int, default=8)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    ds = PreparedShardDataset(a.prepared)
    n = len(ds) if a.max_windows is None else min(len(ds), a.max_windows)
    radii = [int(x) for x in a.radii.split(",")]
    schedule=tuple(int(x) for x in a.schedule.split(","))
    if len(schedule)!=6: raise ValueError("schedule must contain six radii")
    result = summarize((ds[i] for i in range(n)), radii, a.latent_extra_radius, schedule,
                       a.window_size, a.window_slots)
    result["num_windows"] = n
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("radius,horizon_s,coverage_bev,active_bev,coverage_latent,active_latent")
    for r in radii:
        for x in result["constant_radius_scan"][str(r)]:
            print(f"{r},{x['horizon_s']:.1f},{x['coverage_bev']:.6f},{x['active_ratio_bev']:.6f},"
                  f"{x['coverage_latent']:.6f},{x['active_ratio_latent']:.6f}")
    print("saved", a.output)


if __name__ == "__main__":
    main()
