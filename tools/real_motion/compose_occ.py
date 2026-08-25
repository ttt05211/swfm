#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import torch
from real_motion.composition import static_protected_compose

def main():
    p=argparse.ArgumentParser(); p.add_argument("--static-occ",required=True); p.add_argument("--wm-occ",required=True); p.add_argument("--conf-static",required=True); p.add_argument("--dynamic-classes",required=True); p.add_argument("--output",required=True); a=p.parse_args()
    def load(x):
        o=torch.load(x,map_location="cpu",weights_only=False)
        if isinstance(o,dict) and len(o)==1: o=next(iter(o.values()))
        return o
    out=static_protected_compose(load(a.static_occ),load(a.wm_occ),load(a.conf_static),[int(x) for x in a.dynamic_classes.split(",")]); torch.save({"pred_occ":out},a.output); print("saved",a.output,out.shape)
if __name__=="__main__": main()
