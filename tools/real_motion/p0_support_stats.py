#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import torch
from real_motion.support import build_motion_tube,MotionTubeConfig,coverage_and_active_ratio

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--radii",default="0,1,2,3,4,5"); a=p.parse_args()
    obj=torch.load(a.input,map_location="cpu",weights_only=False); samples=obj["samples"] if isinstance(obj,dict) and "samples" in obj else obj; radii=[int(x) for x in a.radii.split(",")]
    print("radius,coverage,active_ratio")
    for r in radii:
        cov=[]; act=[]
        for s in samples:
            k=s["kta_support"].bool(); gt=s["gt_moving_support"].bool(); tube=build_motion_tube(k,MotionTubeConfig(radii=[r]*k.shape[-3],latent_extra_radius=0)); c,a2=coverage_and_active_ratio(gt,tube); cov.append(c); act.append(a2)
        print(f"{r},{sum(cov)/len(cov):.6f},{sum(act)/len(act):.6f}")
if __name__=="__main__": main()
