#!/usr/bin/env python3
"""Validate either a legacy single-file cache or a v2 sharded cache."""
import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from real_motion.dataset import RealMotionCacheDataset


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',required=True)
    p.add_argument('--check-samples',type=int,default=16)
    a=p.parse_args()
    ds=RealMotionCacheDataset(a.cache)
    n=min(len(ds),max(0,a.check_samples))
    for i in range(n): _=ds[i]
    print('OK samples=',len(ds),'checked=',n,'metadata=',ds.metadata)

if __name__=='__main__': main()
