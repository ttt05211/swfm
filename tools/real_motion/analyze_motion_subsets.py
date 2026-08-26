#!/usr/bin/env python3
"""Fit/freeze subset-analysis thresholds on a calibration prepared split.

This script does NOT compute IoU. Subset metrics are computed by the final
evaluator with the unchanged MovingMIoUV2MultiHorizon accumulator.
"""
import argparse, json, math, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from real_motion.prepared import PreparedShardDataset
from real_motion.metrics.stratified import quantile_difficulty
from real_motion.runtime_config import add_config_args, load_runtime_config, get_cfg, save_resolved_config
REPORT={1.0:1,2.0:3,3.0:5}

def main():
    p=argparse.ArgumentParser();add_config_args(p)
    p.add_argument('--prepared',required=True,help='train/calibration prepared split only');p.add_argument('--output',required=True);p.add_argument('--max-windows',type=int,default=None)
    a=p.parse_args();cfg=load_runtime_config(a.config,a.override);ds=PreparedShardDataset(a.prepared);n=len(ds) if a.max_windows is None else min(len(ds),a.max_windows)
    errors=[];eligible=0
    for i in range(n):
        s=ds[i]
        for _,fi in REPORT.items():
            for rec in s['moving_records'][fi]:
                eligible+=1;e=float(rec.get('kta_center_error_m',float('nan')))
                if np.isfinite(e):errors.append(e)
    if not errors:raise RuntimeError('no finite KTA center errors in calibration split')
    _,cuts=quantile_difficulty(errors)
    out={'version':'swfm_subset_contract_v1','source_prepared':str(Path(a.prepared).resolve()),'num_windows':n,'eligible_moving_instances':eligible,'finite_kta_errors':len(errors),'speed_change_threshold_mps':float(get_cfg(cfg,'ANALYSIS.SPEED_CHANGE_THRESHOLD_MPS',1.0)),'turn_rate_threshold_deg_s':float(get_cfg(cfg,'ANALYSIS.TURN_RATE_THRESHOLD_DEG_S',10.0)),'kta_error_cuts_m':[float(cuts[0]),float(cuts[1])],'note':'KTA cuts are calibration-frozen. Test evaluation must load this file and must not refit quantiles.'}
    op=Path(a.output);op.parent.mkdir(parents=True,exist_ok=True);op.write_text(json.dumps(out,indent=2),encoding='utf-8');save_resolved_config(cfg,op.with_suffix('.resolved.yaml'));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
