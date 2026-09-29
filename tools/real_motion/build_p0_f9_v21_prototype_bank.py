#!/usr/bin/env python3
"""Build deterministic train-only V21 source-shape prototype banks."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
import torch

from real_motion.runtime_config import add_config_args,load_runtime_config,make_prepare_config
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.v21_source_induction import (
    KMEDOIDS_CLARA_SAMPLE_SIZE,KMEDOIDS_CLARA_TRIALS,KMEDOIDS_EXACT_MAX_N,
    PROTOTYPE_PROTOCOL,annotation_map,attribute_instance_shapes,
    build_prototype_bank,reliable_components_and_tokens,
    select_scene_balanced_round_robin,stable_json_fingerprint,
)
from tools.real_motion import eval_p0_f9_v18_se2 as base
from tools.real_motion.diagnose_p0_f9_v19_innovation_decomposition import CachedSource
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record

def _serialize(s):
    return {"class_id":int(s.class_id),
            "cells_ijk":torch.from_numpy(np.asarray(s.cells_ijk,dtype=np.int32)),
            "local_xyz_m":torch.from_numpy(np.asarray(s.local_xyz_m,dtype=np.float32)),
            "observation_key":list(s.observation_key or ("",""))}

def _sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()

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
    cache_meta,records_all=base.load_cache(a.train_cache)
    total_records=len(records_all)
    all_record_keys=[(str(r["scene_name"]),str(r["t0_token"])) for r in records_all]
    if len(all_record_keys)!=len(set(all_record_keys)):
        raise RuntimeError("train cache contains duplicate (scene_name, t0_token) identities")
    by_record_key=dict(zip(all_record_keys,records_all))
    if a.max_windows>0 and a.max_windows<total_records:
        selected_record_keys=select_scene_balanced_round_robin(all_record_keys,int(a.max_windows))
        records=[by_record_key[k] for k in selected_record_keys]
        selection_rule="scene_balanced_round_robin_v1"
    else:
        records=records_all
        selected_record_keys=tuple(all_record_keys)
        selection_rule="complete_train_cache_order_v1"
    if not records: raise RuntimeError("empty train cache")

    samples={}
    for rec in records:
        w=window_from_record(rec)
        for tok in (*w.history_tokens,*w.future_tokens):
            samples.setdefault(str(tok),(str(w.scene_name),str(tok)))
    ordered=[samples[k] for k in sorted(samples)]
    source=CachedSource(a.dataroot,info_pkl=a.info_pkl,verbose=False)
    strong=StrongW2DetConfig(free_label=int(pcfg.free_label))
    by_class={}; population=[]; unresolved=ambiguous=0
    for si,(scene,tok) in enumerate(ordered,1):
        sem,obs=source.load_occ3d(scene,tok,require_lidar_mask=True); pose=np.asarray(source.pose(tok))
        components,matched,amb=reliable_components_and_tokens(
            source,scene,tok,sem,obs,pose,grid=pcfg.grid,strong_cfg=strong,
            match_max_distance_m=a.match_max_distance_m)
        anns=annotation_map(source.nusc,tok)
        masked=np.where(np.asarray(obs,bool),np.asarray(sem),int(pcfg.free_label)).astype(np.uint8)
        tokens=sorted({str(x) for x in matched if x is not None})
        keys={inst:(tok,inst) for inst in tokens}
        attrs=attribute_instance_shapes(
            masked,pose,anns,grid=pcfg.grid,free_label=int(pcfg.free_label),
            match_max_distance_m=a.match_max_distance_m,tokens=tokens,observation_keys=keys,
            components=components,component_matches=matched)
        for inst in tokens:
            if inst in amb: ambiguous+=1; continue
            key=keys[inst]; attr=attrs[inst]
            if attr.ambiguous: ambiguous+=1; continue
            if attr.shape is None: unresolved+=1; continue
            by_class.setdefault(int(attr.shape.class_id),[]).append(attr.shape); population.append(key)
        if si==1 or si%500==0 or si==len(ordered):
            print(f"v21_prototypes samples {si}/{len(ordered)}",flush=True)
    if len(population)!=len(set(population)):
        raise RuntimeError("duplicate (sample_token, instance_token) observations")

    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    popfp=stable_json_fingerprint([list(x) for x in sorted(population)])
    provenance={"train_cache":str(Path(a.train_cache).resolve()),"train_cache_sha256":_sha256(a.train_cache),
                "info_pkl":str(Path(a.info_pkl).resolve()),"info_pkl_sha256":_sha256(a.info_pkl),
                "cache_records_total":total_records,"cache_records_used":len(records),
                "complete_train_population":len(records)==total_records and int(a.max_windows)==0,
                "record_selection_rule":selection_rule,
                "selected_scene_count":len({x[0] for x in selected_record_keys}),
                "record_key_fingerprint":stable_json_fingerprint([list(x) for x in selected_record_keys]),
                "cache_version":base.SE2_CACHE_VERSION,
                "se2_target_contract":cache_meta.get("se2_target_contract")}
    summary={"protocol":PROTOTYPE_PROTOCOL,"unique_sample_tokens":len(ordered),
             "valid_observations":len(population),"population_fingerprint":popfp,
             "record_selection_rule":selection_rule,
             "selected_windows":len(records),
             "selected_scenes":len({x[0] for x in selected_record_keys}),
             "class_histogram":{str(k):len(v) for k,v in sorted(by_class.items())},
             "shape_unresolved":unresolved,"shape_ambiguous":ambiguous,
             "resolution_m":0.4,
             "normalization":"GT center+yaw only; physical scale preserved; no box-size scaling",
              "distance":"1-binary-IoU","requested_k":sorted(set(a.k)),
              "algorithm":"exact_pam_le_512_else_deterministic_clara_v1",
              "algorithm_config":{"exact_max_n":KMEDOIDS_EXACT_MAX_N,
                  "clara_sample_size":KMEDOIDS_CLARA_SAMPLE_SIZE,"clara_trials":KMEDOIDS_CLARA_TRIALS},
              "source_provenance":provenance}
    for k in sorted(set(a.k)):
        if k<=0: raise ValueError("K must be positive")
        bank=build_prototype_bank(by_class,requested_k=k,population_manifest=population,
                                  source_provenance=provenance)
        payload={"protocol":bank.protocol,"resolution_m":bank.resolution_m,
                 "requested_k":bank.requested_k,"fingerprint":bank.fingerprint,
                  "population_shape_fingerprint":bank.population_shape_fingerprint,
                  "algorithm":bank.algorithm,"algorithm_config":dict(bank.algorithm_config),
                  "source_provenance":dict(bank.source_provenance),
                 "population_fingerprint":popfp,
                 "population_manifest":[list(x) for x in bank.population_manifest],
                 "medoids_by_class":{str(cid):[_serialize(s) for s in rows]
                                     for cid,rows in bank.medoids_by_class.items()},
                 "class_available_k":{str(cid):len(rows) for cid,rows in bank.medoids_by_class.items()}}
        torch.save(payload,out/f"prototype_bank_k{k}.pt")
        (out/f"prototype_bank_k{k}.json").write_text(json.dumps({
            "protocol":payload["protocol"],"requested_k":k,"fingerprint":payload["fingerprint"],
            "population_fingerprint":popfp,"class_available_k":payload["class_available_k"],
             "population_shape_fingerprint":payload["population_shape_fingerprint"],
             "algorithm":payload["algorithm"],"source_provenance":payload["source_provenance"],
            "medoid_observation_keys":{cid:[r["observation_key"] for r in rows]
                                       for cid,rows in payload["medoids_by_class"].items()}},indent=2),encoding="utf-8")
    (out/"prototype_population_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
