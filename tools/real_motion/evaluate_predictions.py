#!/usr/bin/env python3
"""Evaluate SWFM predictions with frozen metrics and post-training diagnostics."""
import argparse,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from real_motion.prepared import PreparedShardDataset
from real_motion.composition import static_protected_compose
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS,REPORT_HORIZONS_S,MovingMIoUV2MultiHorizon,SemanticIoUAccumulator,GridSpec,BOX_MARGIN_M,rasterize_oriented_box
from real_motion.nuscenes_adapter import box3d_from_dict
from real_motion.metrics.stratified import record_subset_labels
from real_motion.metrics.diagnostics import harm_repair_counts,harm_repair_regions,oracle_selector
from real_motion.runtime_config import add_config_args,load_runtime_config,save_resolved_config
REPORT={1.0:1,2.0:3,3.0:5};SUBSET_NAMES=('uniform/easy','accel/decel','turning','turning+speed-change','kta-easy','kta-medium','kta-hard')
def mean_h(acc):
    per={h:acc[h].compute() for h in REPORT_HORIZONS_S};return {'mIoU':float(np.nanmean([per[h]['mIoU'] for h in REPORT_HORIZONS_S])),'per_horizon':per}
def prediction_map(root):
    root=Path(root);ip=root/'index.json'
    if ip.exists():
        idx=json.loads(ip.read_text());return {e['sample_id']:root/e['file'] for e in idx['entries']},idx
    m={}
    for p in sorted(root.glob('*.pt')):
        o=torch.load(p,map_location='cpu',weights_only=False);m[str(o['sample_id'])]=p
    return m,{'version':'legacy_scan'}
def dual(rec,grid):
    b0=box3d_from_dict(rec['box0_future_ego']);bh=box3d_from_dict(rec['boxh_future_ego']);return rasterize_oriented_box(b0,grid,BOX_MARGIN_M)|rasterize_oriented_box(bh,grid,BOX_MARGIN_M)
def main():
    p=argparse.ArgumentParser();add_config_args(p);p.add_argument('--prepared',required=True);p.add_argument('--pred-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--subset-config',default=None);p.add_argument('--max-windows',type=int,default=None);a=p.parse_args();cfg=load_runtime_config(a.config,a.override)
    sc=None
    if a.subset_config:
        sc=json.loads(Path(a.subset_config).read_text());
        if sc.get('version')!='swfm_subset_contract_v1':raise ValueError('unsupported subset config version')
    ds=PreparedShardDataset(a.prepared);paths,pidx=prediction_map(a.pred_dir);n=len(ds) if a.max_windows is None else min(len(ds),a.max_windows);overall={h:SemanticIoUAccumulator() for h in REPORT_HORIZONS_S};dynamic={h:SemanticIoUAccumulator(DYNAMIC_CLASS_IDS) for h in REPORT_HORIZONS_S};moving=MovingMIoUV2MultiHorizon();ko={h:SemanticIoUAccumulator() for h in REPORT_HORIZONS_S};km=MovingMIoUV2MultiHorizon();om=MovingMIoUV2MultiHorizon();ss={x:MovingMIoUV2MultiHorizon() for x in SUBSET_NAMES} if sc else {};sk={x:MovingMIoUV2MultiHorizon() for x in SUBSET_NAMES} if sc else {};counts={x:0 for x in SUBSET_NAMES};hr={k:0 for k in ('repair','harm','preserve','unresolved','support')};rows=[];used=0;missing=[];grid=GridSpec()
    for i in range(n):
        s=ds[i];sid=str(s['sample_id']);path=paths.get(sid)
        if path is None:missing.append(sid);continue
        obj=torch.load(path,map_location='cpu',weights_only=False);pred=obj['pred_occ'].numpy() if torch.is_tensor(obj['pred_occ']) else obj['pred_occ'];kta=static_protected_compose(torch.from_numpy(s['static_future_occ']),torch.from_numpy(s['kta_future_occ']),torch.from_numpy(s['confident_static_future_mask']),DYNAMIC_CLASS_IDS,write_support=torch.from_numpy(s['generation_support_occ'])).numpy()
        for h,fi in REPORT.items():
            gt=s['future_gt_occ'][fi];sup=s['gt_moving_support'][fi];overall[h].update(pred[fi],gt);dynamic[h].update(pred[fi],gt);moving.update(h,pred[fi],gt,sup);ko[h].update(kta[fi],gt);km.update(h,kta[fi],gt,sup);c=harm_repair_counts(torch.from_numpy(kta[fi]),torch.from_numpy(pred[fi]),torch.from_numpy(gt),torch.from_numpy(sup));
            for k in hr:hr[k]+=int(c[k])
            region=[];subs={x:np.zeros_like(sup,dtype=bool) for x in SUBSET_NAMES}
            for rec in s['moving_records'][fi]:
                r=dual(rec,grid);region.append(torch.from_numpy(r))
                if sc:
                    rr=dict(rec);rr['horizon_s']=h
                    for label in record_subset_labels(rr,float(sc['speed_change_threshold_mps']),math.radians(float(sc['turn_rate_threshold_deg_s'])),sc['kta_error_cuts_m']):
                        if label in subs:subs[label]|=r;counts[label]+=1
            reg=harm_repair_regions(torch.from_numpy(kta[fi]),torch.from_numpy(pred[fi]),torch.from_numpy(gt),region)
            for r in reg['regions']:r.update({'sample_id':sid,'horizon_s':h});rows.append(r)
            oracle=oracle_selector(torch.from_numpy(kta[fi]),torch.from_numpy(pred[fi]),torch.from_numpy(gt),torch.from_numpy(sup)).numpy();om.update(h,oracle,gt,sup)
            if sc:
                for name,sup2 in subs.items():ss[name].update(h,pred[fi],gt,sup2);sk[name].update(h,kta[fi],gt,sup2)
        used+=1
    if used==0:raise RuntimeError('no predictions matched prepared sample_id')
    wm=moving.compute();oo=om.compute();macro={'num_regions':len(rows),'repair_rate':float(np.mean([r['repair'] for r in rows])) if rows else 0.0,'harm_rate':float(np.mean([r['harm'] for r in rows])) if rows else 0.0,'mean_accuracy_delta':float(np.mean([r['delta'] for r in rows])) if rows else 0.0}
    report={'num_prepared_considered':n,'num_predictions_used':used,'missing_count':len(missing),'missing_sample_ids':missing[:20],'SWFM':{'overall':mean_h(overall),'dynamic':mean_h(dynamic),'Moving-mIoU_v2':wm},'KTA_composed_baseline':{'overall':mean_h(ko),'Moving-mIoU_v2':km.compute()},'diagnostics':{'moving_support_harm_repair_voxel_micro':hr,'moving_instance_tube_macro':macro,'oracle_selector_Moving-mIoU_v2':oo,'oracle_selector_headroom_pp':float(oo['mIoU']-wm['mIoU'])},'prediction_index':{k:v for k,v in pidx.items() if k!='entries'}}
    if sc:report['subset_contract']=sc;report['Moving-mIoU_v2_subsets']={x:{'eligible_instance_records':counts[x],'SWFM':ss[x].compute(),'KTA':sk[x].compute()} for x in SUBSET_NAMES}
    op=Path(a.output);op.parent.mkdir(parents=True,exist_ok=True);op.write_text(json.dumps(report,indent=2),encoding='utf-8');save_resolved_config(cfg,op.with_suffix('.resolved.yaml'));print(json.dumps(report['SWFM'],indent=2))
if __name__=='__main__':main()
