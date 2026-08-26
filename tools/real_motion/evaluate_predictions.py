#!/usr/bin/env python3
"""Evaluate saved SWFM predictions on prepared GT windows."""
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from real_motion.prepared import PreparedShardDataset
from real_motion.composition import static_protected_compose
from real_motion.metrics.moving_miou_v2 import (
    DYNAMIC_CLASS_IDS, REPORT_HORIZONS_S, MovingMIoUV2MultiHorizon,
    SemanticIoUAccumulator, GridSpec, BOX_MARGIN_M, rasterize_oriented_box,
)
from real_motion.nuscenes_adapter import box3d_from_dict
from real_motion.metrics.stratified import wrap_angle
from real_motion.metrics.diagnostics import harm_repair_counts

REPORT={1.0:1,2.0:3,3.0:5}


def mean_h(acc):
    per={h:acc[h].compute() for h in REPORT_HORIZONS_S}
    vals=[per[h]['mIoU'] for h in REPORT_HORIZONS_S]
    return {'mIoU':float(np.nanmean(vals)),'per_horizon':per}


def prediction_map(root):
    root=Path(root)
    idx_path=root/'index.json'
    if idx_path.exists():
        idx=json.loads(idx_path.read_text(encoding='utf-8'))
        return {e['sample_id']:root/e['file'] for e in idx['entries']}, idx
    mapping={}
    for p in sorted(root.glob('*.pt')):
        obj=torch.load(p,map_location='cpu',weights_only=False)
        mapping[str(obj['sample_id'])]=p
    return mapping, {'version':'legacy_scan'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--prepared',required=True)
    p.add_argument('--pred-dir',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--subset-records',default=None,help='optional JSONL for maneuver/KTA-hard analysis')
    p.add_argument('--max-windows',type=int,default=None)
    a=p.parse_args()

    ds=PreparedShardDataset(a.prepared)
    pred_paths,pred_index=prediction_map(a.pred_dir)
    n=len(ds) if a.max_windows is None else min(len(ds),a.max_windows)
    overall={h:SemanticIoUAccumulator() for h in REPORT_HORIZONS_S}
    dynamic={h:SemanticIoUAccumulator(DYNAMIC_CLASS_IDS) for h in REPORT_HORIZONS_S}
    moving=MovingMIoUV2MultiHorizon()
    kta_overall={h:SemanticIoUAccumulator() for h in REPORT_HORIZONS_S}
    kta_moving=MovingMIoUV2MultiHorizon()
    hr={'repair':0,'harm':0,'preserve':0,'unresolved':0,'support':0}
    used=0; missing=[]; subset_rows=[]
    metric_grid=GridSpec()

    for i in range(n):
        s=ds[i]; sid=str(s['sample_id'])
        path=pred_paths.get(sid)
        if path is None:
            missing.append(sid); continue
        obj=torch.load(path,map_location='cpu',weights_only=False)
        pred=obj['pred_occ']
        if torch.is_tensor(pred): pred=pred.numpy()
        if pred.shape != s['future_gt_occ'].shape:
            raise ValueError(f'{sid}: prediction shape {pred.shape} != GT {s["future_gt_occ"].shape}')

        kta_final=static_protected_compose(
            torch.from_numpy(s['static_future_occ']),
            torch.from_numpy(s['kta_future_occ']),
            torch.from_numpy(s['confident_static_future_mask']),
            DYNAMIC_CLASS_IDS,
            write_support=torch.from_numpy(s['generation_support_occ']),
        ).numpy()

        for h,fi in REPORT.items():
            gt=s['future_gt_occ'][fi]; sup=s['gt_moving_support'][fi]
            overall[h].update(pred[fi],gt)
            dynamic[h].update(pred[fi],gt)
            moving.update(h,pred[fi],gt,sup)
            kta_overall[h].update(kta_final[fi],gt)
            kta_moving.update(h,kta_final[fi],gt,sup)
            c=harm_repair_counts(torch.from_numpy(kta_final[fi]),torch.from_numpy(pred[fi]),
                                 torch.from_numpy(gt),torch.from_numpy(sup))
            for k in hr: hr[k]+=int(c[k])

            if a.subset_records:
                for rec in s["moving_records"][fi]:
                    dv=float(rec.get("delta_speed_mps",float("nan")))
                    ke=float(rec.get("kta_center_error_m",float("nan")))
                    if not np.isfinite(dv) or not np.isfinite(ke):
                        continue
                    b0=box3d_from_dict(rec["box0_future_ego"]); bh=box3d_from_dict(rec["boxh_future_ego"])
                    inst_sup=(rasterize_oriented_box(b0,metric_grid,BOX_MARGIN_M) |
                              rasterize_oriented_box(bh,metric_grid,BOX_MARGIN_M))
                    cid=int(rec["class_id"])
                    pp=(pred[fi]==cid)&inst_sup; gg=(gt==cid)&inst_sup
                    subset_rows.append({
                        "sample_id":sid,"horizon_s":h,"instance_token":rec["instance_token"],
                        "class_id":cid,"intersection":int((pp&gg).sum()),
                        "union":int((pp|gg).sum()),"delta_speed_mps":dv,
                        "heading_change_rad":abs(wrap_angle(float(rec["yawh_world"])-float(rec["yaw0_world"]))),
                        "turn_rate_radps":abs(wrap_angle(float(rec["yawh_world"])-float(rec["yaw0_world"])))/float(h),
                        "kta_center_error_m":ke,
                    })
        used+=1
        if i%50==0: print('eval',i,'/',n,sid)

    if used==0: raise RuntimeError('no predictions matched prepared sample_id')
    report={
        'num_prepared_considered':n,'num_predictions_used':used,'missing_count':len(missing),
        'missing_sample_ids':missing[:20],
        'SWFM':{'overall':mean_h(overall),'dynamic':mean_h(dynamic),'Moving-mIoU_v2':moving.compute()},
        'KTA_composed_baseline':{'overall':mean_h(kta_overall),'Moving-mIoU_v2':kta_moving.compute()},
        'moving_support_harm_repair_voxels':hr,
        'prediction_index':{k:v for k,v in pred_index.items() if k!='entries'},
        'subset_records_written':len(subset_rows),
    }
    if a.subset_records:
        sp=Path(a.subset_records); sp.parent.mkdir(parents=True,exist_ok=True)
        sp.write_text("\n".join(json.dumps(r) for r in subset_rows)+("\n" if subset_rows else ""),encoding='utf-8')
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report['SWFM'],indent=2)); print('saved',a.output)

if __name__=='__main__': main()
