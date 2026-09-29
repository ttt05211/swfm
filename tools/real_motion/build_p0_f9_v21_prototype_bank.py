#!/usr/bin/env python3
"""Build deterministic train-only V21 source-shape prototype banks."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
import torch

from real_motion.runtime_config import add_config_args,load_runtime_config,make_prepare_config
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.v21_source_induction import (
    PROTOTYPE_PROTOCOL,annotation_map,attribute_instance_shape,
    build_prototype_bank,reliable_components_and_tokens,stable_json_fingerprint,
)
from tools.real_motion import eval_p0_f9_v18_se2 as base
from tools.real_motion.diagnose_p0_f9_v19_innovation_decomposition import CachedSource
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record

def _serialize(s):
    return {"class_id":int(s.class_id),
            "cells_ijk":torch.from_numpy(np.asarray(s.cells_ijk,dtype=np.int32)),
            "observation_key":list(s.observation_key or ("",""))}

def main():
    p=argparse.ArgumentParser(); add_config_args(p)
    p.add_argument("--train-cache",required=True)
    p.add_argument("--dataroot",required=True); p.add_argument("--info-pkl",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--k",type=int,nargs="+",default=[1,4,8,16])
    p.add_argument("--match-max-distance-m",type=float,default=4.0)
    p.add_argument("--max-windows",type=int,default=0)
    a=p.parse_args()
    pcfg=make_prepare_config(load_runtime_config(a.config,a.override))
    _,records=base.load_cache(a.train_cache)
    if a.max_windows>0: records=records[:min(len(records),a.max_windows)]
    if not records: raise RuntimeError("empty train cache")

    samples={}
    for rec in records:
        w=window_from_record(rec)
        for tok in w.history_tokens: samples.setdefault(str(tok),(str(w.scene_name),str(tok)))
    ordered=[samples[k] for k in sorted(samples)]
    source=CachedSource(a.dataroot,info_pkl=a.info_pkl,verbose=False)
    strong=StrongW2DetConfig(free_label=int(pcfg.free_label))
    by_class={}; population=[]; unresolved=ambiguous=0
    for si,(scene,tok) in enumerate(ordered,1):
        sem,obs=source.load_occ3d(scene,tok,require_lidar_mask=True); pose=np.asarray(source.pose(tok))
        _,matched,amb=reliable_components_and_tokens(
            source,scene,tok,sem,obs,pose,grid=pcfg.grid,strong_cfg=strong,
            match_max_distance_m=a.match_max_distance_m)
        anns=annotation_map(source.nusc,tok)
        masked=np.where(np.asarray(obs,bool),np.asarray(sem),int(pcfg.free_label)).astype(np.uint8)
        for inst in sorted({str(x) for x in matched if x is not None}):
            if inst in amb: ambiguous+=1; continue
            key=(tok,inst)
            attr=attribute_instance_shape(
                masked,pose,anns,inst,grid=pcfg.grid,free_label=int(pcfg.free_label),
                match_max_distance_m=a.match_max_distance_m,observation_key=key)
            if attr.ambiguous: ambiguous+=1; continue
            if attr.shape is None: unresolved+=1; continue
            by_class.setdefault(int(attr.shape.class_id),[]).append(attr.shape); population.append(key)
        if si==1 or si%500==0 or si==len(ordered):
            print(f"v21_prototypes samples {si}/{len(ordered)}",flush=True)
    if len(population)!=len(set(population)):
        raise RuntimeError("duplicate (sample_token, instance_token) observations")

    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    popfp=stable_json_fingerprint([list(x) for x in sorted(population)])
    summary={"protocol":PROTOTYPE_PROTOCOL,"unique_sample_tokens":len(ordered),
             "valid_observations":len(population),"population_fingerprint":popfp,
             "class_histogram":{str(k):len(v) for k,v in sorted(by_class.items())},
             "shape_unresolved":unresolved,"shape_ambiguous":ambiguous,
             "resolution_m":0.4,
             "normalization":"GT center+yaw only; physical scale preserved; no box-size scaling",
             "distance":"1-binary-IoU","requested_k":sorted(set(a.k))}
    for k in sorted(set(a.k)):
        if k<=0: raise ValueError("K must be positive")
        bank=build_prototype_bank(by_class,requested_k=k,population_manifest=population)
        payload={"protocol":bank.protocol,"resolution_m":bank.resolution_m,
                 "requested_k":bank.requested_k,"fingerprint":bank.fingerprint,
                 "population_fingerprint":popfp,
                 "population_manifest":[list(x) for x in bank.population_manifest],
                 "medoids_by_class":{str(cid):[_serialize(s) for s in rows]
                                     for cid,rows in bank.medoids_by_class.items()},
                 "class_available_k":{str(cid):len(rows) for cid,rows in bank.medoids_by_class.items()}}
        torch.save(payload,out/f"prototype_bank_k{k}.pt")
        (out/f"prototype_bank_k{k}.json").write_text(json.dumps({
            "protocol":payload["protocol"],"requested_k":k,"fingerprint":payload["fingerprint"],
            "population_fingerprint":popfp,"class_available_k":payload["class_available_k"],
            "medoid_observation_keys":{cid:[r["observation_key"] for r in rows]
                                       for cid,rows in payload["medoids_by_class"].items()}},indent=2),encoding="utf-8")
    (out/"prototype_population_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
