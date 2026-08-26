#!/usr/bin/env python3
"""P0-D: decomposition and causal-support oracle upper bounds."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from real_motion.prepared import PreparedShardDataset
from real_motion.composition import static_protected_compose
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS,MovingMIoUV2MultiHorizon,SemanticIoUAccumulator,REPORT_HORIZONS_S
from real_motion.nuscenes_adapter import gt_true_static_mask,dynamic_only_semantics
from real_motion.runtime_config import add_config_args,load_runtime_config,save_resolved_config
REPORT={1.0:1,2.0:3,3.0:5}
def mean_h(acc):
    per={h:acc[h].compute() for h in REPORT_HORIZONS_S};return {'mIoU':float(np.nanmean([per[h]['mIoU'] for h in REPORT_HORIZONS_S])),'per_horizon':per}
def main():
    p=argparse.ArgumentParser();add_config_args(p);p.add_argument('--prepared',required=True);p.add_argument('--max-windows',type=int,default=None);p.add_argument('--output',required=True);a=p.parse_args();cfg=load_runtime_config(a.config,a.override);ds=PreparedShardDataset(a.prepared);n=len(ds) if a.max_windows is None else min(len(ds),a.max_windows);oa={h:SemanticIoUAccumulator() for h in REPORT_HORIZONS_S};sa={h:SemanticIoUAccumulator() for h in REPORT_HORIZONS_S};sta={h:SemanticIoUAccumulator() for h in REPORT_HORIZONS_S};mov=MovingMIoUV2MultiHorizon();smov=MovingMIoUV2MultiHorizon()
    for i in range(n):
        s=ds[i]
        for h,fi in REPORT.items():
            static=torch.from_numpy(s['static_future_occ'][fi]);gt=s['future_gt_occ'][fi];all_dyn=torch.from_numpy(dynamic_only_semantics(gt));causal=torch.from_numpy(s['future_dynamic_target_occ'][fi]);prot=torch.from_numpy(s['confident_static_future_mask'][fi]);write=torch.from_numpy(s['generation_support_occ'][fi]);oracle=static_protected_compose(static,all_dyn,prot,DYNAMIC_CLASS_IDS,write_support=None).numpy();soracle=static_protected_compose(static,causal,prot,DYNAMIC_CLASS_IDS,write_support=write).numpy();oa[h].update(oracle,gt);sa[h].update(soracle,gt);sta[h].update(s['static_future_occ'][fi],gt,gt_true_static_mask(gt,s['gt_moving_support'][fi]));mov.update(h,oracle,gt,s['gt_moving_support'][fi]);smov.update(h,soracle,gt,s['gt_moving_support'][fi])
    report={'num_windows':n,'decomposition_oracle_overall':mean_h(oa),'decomposition_oracle_Moving-mIoU_v2':mov.compute(),'causal_support_oracle_overall':mean_h(sa),'causal_support_oracle_Moving-mIoU_v2':smov.compute(),'se3_true_static':mean_h(sta),'note':'Gap(decomposition, causal-support) isolates support/reachability loss; gap(causal-support, learned WM) isolates learning headroom.'};op=Path(a.output);op.parent.mkdir(parents=True,exist_ok=True);op.write_text(json.dumps(report,indent=2),encoding='utf-8');save_resolved_config(cfg,op.with_suffix('.resolved.yaml'));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
