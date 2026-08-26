#!/usr/bin/env python3
"""P0-C: audit actual hard-static blind spots and no-history innovation causes."""
import argparse, json, math, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from real_motion.prepared import PreparedShardDataset
from real_motion.nuscenes_adapter import NuScenesWindowSource, sample_ego_to_world, wrap_angle
from real_motion.geometry import quaternion_yaw
from real_motion.metrics.moving_miou_v2 import (
    SPEED_THRESHOLD_MPS, BOX_MARGIN_M, Box3D, GridSpec, rasterize_oriented_box,
)
from real_motion.runtime_config import add_config_args, load_runtime_config, save_resolved_config

REPORT = {1.0: 1, 2.0: 3, 3.0: 5}


def ann_map(nusc, token):
    sample = nusc.get("sample", token)
    out = {}
    for at in sample["anns"]:
        a = nusc.get("sample_annotation", at)
        out[a["instance_token"]] = a
    return out


def sample_time(nusc, token):
    return float(nusc.get("sample", token)["timestamp"]) / 1e6


def ann_to_t0_ego_box(ann, t0_ego_to_world, class_id):
    center_world = np.asarray(ann["translation"], dtype=np.float64)
    center_ego = (np.linalg.inv(t0_ego_to_world) @ np.r_[center_world, 1.0])[:3]
    yaw_world = quaternion_yaw(ann["rotation"])
    yaw_ego_world = math.atan2(t0_ego_to_world[1, 0], t0_ego_to_world[0, 0])
    w, l, h = ann["size"]
    return Box3D(
        token=ann["instance_token"], class_id=int(class_id),
        center_xyz=tuple(float(x) for x in center_ego),
        size_lwh=(float(l), float(w), float(h)),
        yaw=wrap_angle(yaw_world - yaw_ego_world),
    )


def main():
    p = argparse.ArgumentParser(); add_config_args(p)
    p.add_argument("--prepared", required=True); p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True); p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--output", required=True); a = p.parse_args(); cfg = load_runtime_config(a.config, a.override)
    ds = PreparedShardDataset(a.prepared); n = len(ds) if a.max_windows is None else min(len(ds), a.max_windows)
    source = NuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False); nusc = source.nusc; metric_grid = GridSpec()
    totals = {h: {"eligible_future_moving":0,"hard_static_any_instances":0,"hard_static_ge50_instances":0,"hard_static_ge80_instances":0,"future_moving_t0_voxels":0,"hard_static_future_moving_t0_voxels":0,"historically_stationary_aux":0,"no_pre_t0_observation":0,"birth_excluded":0,"death_excluded":0} for h in REPORT}
    for i in range(n):
        s=ds[i]; t0_map=ann_map(nusc,s["t0_token"]); hist_maps=[ann_map(nusc,t) for t in s["history_tokens"][:-1]]; hist_times=[sample_time(nusc,t) for t in s["history_tokens"][:-1]]; t0_time=sample_time(nusc,s["t0_token"]); t0_pose=sample_ego_to_world(nusc,s["t0_token"]); t0_sem=np.asarray(s["full_history_occ"][-1]); hard_static=np.asarray(s["t0_confident_static_mask"],dtype=bool)
        for h,fi in REPORT.items():
            records=s["moving_records"][fi]; d=totals[h]; d["eligible_future_moving"]+=len(records); excluded=s["metric_excluded"][fi]; d["birth_excluded"]+=int(excluded.get("birth_dynamic",0)); d["death_excluded"]+=int(excluded.get("death_dynamic",0))
            for rec in records:
                inst=rec["instance_token"]; cid=int(rec["class_id"]); ann0=t0_map.get(inst)
                if ann0 is None: d["no_pre_t0_observation"]+=1; continue
                box0=ann_to_t0_ego_box(ann0,t0_pose,cid); box_mask=rasterize_oriented_box(box0,metric_grid,BOX_MARGIN_M); inst_vox=box_mask&(t0_sem==cid); denom=int(inst_vox.sum()); overlap=int((inst_vox&hard_static).sum()); frac=overlap/denom if denom else 0.0
                d["future_moving_t0_voxels"]+=denom; d["hard_static_future_moving_t0_voxels"]+=overlap; d["hard_static_any_instances"]+=int(overlap>0); d["hard_static_ge50_instances"]+=int(frac>=.50); d["hard_static_ge80_instances"]+=int(frac>=.80)
                prior=None
                for hm,ht in zip(hist_maps,hist_times):
                    if inst in hm: prior=(hm[inst],ht); break
                if prior is None: d["no_pre_t0_observation"]+=1; continue
                dt=t0_time-prior[1]
                if dt>0:
                    cp=np.asarray(prior[0]["translation"],dtype=np.float64); c0=np.asarray(ann0["translation"],dtype=np.float64); hist_speed=float(np.linalg.norm(c0[:2]-cp[:2])/dt); d["historically_stationary_aux"]+=int(hist_speed<SPEED_THRESHOLD_MPS)
        if i%50==0: print("audit",i,"/",n)
    report={"num_windows":n,"horizons":{}}
    for h,d in totals.items():
        iden=d["eligible_future_moving"]; vden=d["future_moving_t0_voxels"]; out=dict(d); out.update({"hard_static_to_future_moving_any_instance_ratio":d["hard_static_any_instances"]/iden if iden else 0.0,"hard_static_to_future_moving_ge50_instance_ratio":d["hard_static_ge50_instances"]/iden if iden else 0.0,"hard_static_to_future_moving_ge80_instance_ratio":d["hard_static_ge80_instances"]/iden if iden else 0.0,"hard_static_to_future_moving_voxel_ratio":d["hard_static_future_moving_t0_voxels"]/vden if vden else 0.0,"historically_stationary_aux_ratio":d["historically_stationary_aux"]/iden if iden else 0.0,"no_pre_t0_ratio":d["no_pre_t0_observation"]/iden if iden else 0.0}); report["horizons"][str(h)]=out
    op=Path(a.output); op.parent.mkdir(parents=True,exist_ok=True); op.write_text(json.dumps(report,indent=2),encoding="utf-8"); save_resolved_config(cfg,op.with_suffix(".resolved.yaml")); print(json.dumps(report["horizons"],indent=2))
if __name__=="__main__": main()
