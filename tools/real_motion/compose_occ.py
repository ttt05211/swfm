#!/usr/bin/env python3
"""Standalone static-protected composition helper.

For formal SWFM inference pass --write-support. Omitting it is intended only
for decomposition-oracle diagnostics.
"""
import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from real_motion.composition import static_protected_compose


def load(path):
    obj=torch.load(path,map_location='cpu',weights_only=False)
    if isinstance(obj,dict) and len(obj)==1: obj=next(iter(obj.values()))
    return obj


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--static-occ',required=True); p.add_argument('--wm-occ',required=True)
    p.add_argument('--conf-static',required=True); p.add_argument('--write-support',default=None)
    p.add_argument('--dynamic-classes',default='2,3,4,5,6,7,9,10')
    p.add_argument('--output',required=True); a=p.parse_args()
    if a.write_support is None:
        print('[WARN] no --write-support: use only for oracle/diagnostic composition')
    out=static_protected_compose(
        load(a.static_occ),load(a.wm_occ),load(a.conf_static),
        [int(x) for x in a.dynamic_classes.split(',')],
        write_support=None if a.write_support is None else load(a.write_support),
    )
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    torch.save({'pred_occ':out},a.output); print('saved',a.output,tuple(out.shape))

if __name__=='__main__': main()
