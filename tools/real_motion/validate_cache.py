#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from real_motion.cache import load_cache
def main():
    p=argparse.ArgumentParser(); p.add_argument("--cache",required=True); a=p.parse_args(); d=load_cache(a.cache); print("OK",d["version"],"samples=",len(d["samples"])); print("metadata=",d.get("metadata",{}))
if __name__=="__main__": main()
