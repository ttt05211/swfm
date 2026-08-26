#!/usr/bin/env python3
"""Aggregate Moving-mIoU-v2-style per-instance records by maneuver/KTA difficulty.

Input JSONL is intentionally model-agnostic. Each row should contain:
  intersection, union, delta_speed_mps, turn_rate_radps, kta_center_error_m
Rows are produced by the final evaluator/diagnostic pipeline, not by inference.
"""
import argparse,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from real_motion.metrics.stratified import maneuver_bucket,quantile_difficulty


def miou(rows):
    inter=sum(float(r["intersection"]) for r in rows); union=sum(float(r["union"]) for r in rows)
    return 100.0*inter/union if union else float("nan")


def main():
    p=argparse.ArgumentParser();p.add_argument("--records",required=True);p.add_argument("--output",required=True)
    p.add_argument("--speed-change-threshold",type=float,default=1.0)
    p.add_argument("--turn-rate-threshold-deg-s",type=float,default=10.0)
    p.add_argument("--fit-kta-cuts",action="store_true",help="calibration only: fit 1/3 and 2/3 KTA-error cuts on this input")
    p.add_argument("--kta-cuts",default=None,help="frozen test cuts: easy/medium boundaries in meters, e.g. 0.8,1.7")
    a=p.parse_args()
    rows=[json.loads(x) for x in Path(a.records).read_text().splitlines() if x.strip()]
    rows=[r for r in rows if math.isfinite(float(r["delta_speed_mps"])) and math.isfinite(float(r["kta_center_error_m"]))]
    if not rows: raise RuntimeError("no finite stratification records")
    for r in rows:
        r["maneuver"]=maneuver_bucket(r["delta_speed_mps"],r["turn_rate_radps"],
            a.speed_change_threshold,math.radians(a.turn_rate_threshold_deg_s))
    if a.fit_kta_cuts == (a.kta_cuts is not None):
        raise ValueError("choose exactly one of --fit-kta-cuts (calibration) or --kta-cuts c1,c2 (frozen evaluation)")
    errors=[float(r["kta_center_error_m"]) for r in rows]
    if a.fit_kta_cuts:
        labels,cuts=quantile_difficulty(errors)
        cuts_source="fit_from_input_calibration"
    else:
        cuts=tuple(float(x) for x in a.kta_cuts.split(","))
        if len(cuts)!=2 or not cuts[0] <= cuts[1]: raise ValueError("--kta-cuts must be c1,c2 with c1<=c2")
        labels=["easy" if e<=cuts[0] else ("medium" if e<=cuts[1] else "hard") for e in errors]
        cuts_source="frozen_cli"
    for r,l in zip(rows,labels): r["kta_difficulty"]=str(l)
    report={"maneuver":{},"kta_difficulty":{},"kta_error_quantile_cuts_m":cuts,
            "kta_cuts_source":cuts_source,
            "speed_change_threshold_mps":a.speed_change_threshold,
            "turn_rate_threshold_deg_s":a.turn_rate_threshold_deg_s}
    for key in sorted(set(r["maneuver"] for r in rows)):
        rr=[r for r in rows if r["maneuver"]==key]; report["maneuver"][key]={"count":len(rr),"IoU":miou(rr)}
    for key in ("easy","medium","hard"):
        rr=[r for r in rows if r["kta_difficulty"]==key]; report["kta_difficulty"][key]={"count":len(rr),"IoU":miou(rr)}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
