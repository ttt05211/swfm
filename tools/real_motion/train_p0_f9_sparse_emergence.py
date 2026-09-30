#!/usr/bin/env python3
"""One bounded two-variant learned generation screen; no automatic next stage."""
from __future__ import annotations
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from dataclasses import asdict, replace
import json
import random
import subprocess
import time
import numpy as np
import torch

from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.sparse_emergence import PROTOCOL, FEATURE_PROTOCOL, THRESHOLDS, EmergenceConfig
from real_motion.sparse_emergence_model import SparseEmergenceDecoder, set_loss
from real_motion.v21_source_induction import select_scene_balanced_round_robin, stable_json_fingerprint
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import load_manifest, align_records
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256, finite_json, write_json, atomic_checkpoint, bank_fingerprint
from tools.real_motion.sparse_emergence_common import (
    FrozenGenerationV18, prepare_population, evaluate_points, select_calibration_threshold, candidate_scope_ceiling,
)

DEV64_FP = "0cb9d69ee11d3ba2afd88a7a2436eb7453670b82b25000a5047d3ed91061101b"


def select_train_calibration(keys, train_count, cal_count, dev_scenes):
    keys = tuple((str(a),str(b)) for a,b in keys)
    if len(keys) != len(set(keys)): raise RuntimeError("duplicate train record keys")
    if {k[0] for k in keys} & set(dev_scenes): raise RuntimeError("train/dev scene overlap")
    cal = select_scene_balanced_round_robin(keys,cal_count)
    cal_scenes = {k[0] for k in cal}
    rest = tuple(k for k in keys if k[0] not in cal_scenes)
    if len(rest) < train_count: raise RuntimeError("not enough scene-disjoint train/calibration windows")
    train = select_scene_balanced_round_robin(rest,train_count)
    return train,cal


def train_variants(bank, configs, device, *, updates, batch_size, seed, progress=None):
    """Same samples/initial weights/optimizer schedule; no dev-based selection."""
    torch.manual_seed(seed)
    initial = SparseEmergenceDecoder(next(iter(configs.values()))).state_dict()
    models, optimizers = {}, {}
    for name,cfg in configs.items():
        model = SparseEmergenceDecoder(cfg).to(device); model.load_state_dict(initial)
        models[name] = model
        optimizers[name] = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    counts = np.bincount(bank["labels"][bank["mask"]],minlength=17)
    positive_counts = counts[counts > 0]
    if not len(positive_counts): raise RuntimeError("no positive foreground supervision in causal training candidates")
    weights = np.sqrt(positive_counts.mean()/np.maximum(counts,1)).clip(.5,4)
    class_weight = torch.as_tensor(weights,dtype=torch.float32,device=device)
    rng = np.random.default_rng(seed+1)
    z = bank["history"].shape[-1]
    last = {}
    for update in range(1,updates+1):
        ids = rng.choice(len(bank["history"]),batch_size,replace=len(bank["history"]) < batch_size)
        batch = {key:torch.as_tensor(value[ids],device=device) for key,value in bank.items()}
        batch["labels"] = batch["labels"].long()
        for name,model in models.items():
            start = time.perf_counter(); model.train(); optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch["history"],batch["query"])
            loss,details = set_loss(prediction,batch["xyz"],batch["labels"],batch["mask"],batch["weight"],
                (model.config.patch_cells,model.config.patch_cells,z),class_weight)
            if not torch.isfinite(loss): raise RuntimeError(f"nonfinite loss variant={name} update={update}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            grad = {head:float(torch.sqrt(sum((p.grad.float().square().sum() for p in getattr(model,head).parameters()
                                               if p.grad is not None),torch.zeros((),device=device))).item())
                    for head in ("xyz_head","semantic_head","presence_head")}
            optimizer.step()
            last[name] = {"event":"train","variant":name,"update":update,"loss":float(loss.detach()),
                **{k:float(v) for k,v in details.items()},"grad_norm":float(norm),"head_grad":grad,
                "seconds":time.perf_counter()-start}
            if progress: progress(last[name])
            if update == 1 or update % 32 == 0 or update == updates:
                r = last[name]
                print(f"variant={name} update={update}/{updates} loss={r['loss']:.4f} geometry={r['geometry']:.4f} "
                      f"semantic={r['semantic']:.4f} presence={r['presence']:.4f} xyz_grad={grad['xyz_head']:.4f} "
                      f"seconds={r['seconds']:.3f}",flush=True)
    return models,{"class_counts":counts.tolist(),"semantic_weights":weights.tolist(),"last":last}


def summary_text(summary):
    def number(v): return "undefined" if v is None else f"{v:+.6f}"
    lines = ["===== SPARSE POINT-SET EMERGENCE =====",f"protocol: {PROTOCOL}",f"mode: {summary['mode']}",
             f"train/calibration/dev windows: {summary['populations']}",f"updates per variant: {summary['updates']}",
             "V18 frozen; no tokenizer/prototype bank; calibration thresholds frozen before dev"]
    for name,r in summary["variants"].items():
        formal,raw = r["dev_formal"],r["dev_raw_diagnostic"]
        d,q = formal["delta_vs_v18_pp"],formal["quality"]
        lines += [f"===== {name} =====",f"sampled train fit: {summary['training_diagnostics'].get('sampled_fit',{}).get(name,{})}",
            f"threshold: {r['threshold']} (None=abstain, NOT success)",
            f"formal dMiOU={number(d['mIoU'])} dIoU={number(d['IoU'])} dMovingMicro={number(d['MovingMicro'])}",
            f"added={q['added']} semantic_precision={q['semantic_precision']} "
            f"unseen_static_tp={q['unseen_static_semantic_tp']} birth_tp={q['birth_semantic_tp']} dormant_tp={q['dormant_semantic_tp']}",
            f"raw threshold=0 diagnostic: dMiOU={raw['delta_vs_v18_pp']['mIoU']:+.6f} added={raw['quality']['added']} "
            f"semantic_precision={raw['quality']['semantic_precision']}",f"gate: {formal['gate']}",f"checkpoint: {r['checkpoint']}"]
    audit = summary.get("candidate_scope_audit",{})
    if audit:
        lines += ["===== SAME-PASS CANDIDATE AUDIT (GT, NOT LEARNED) =====",
                  f"scope dMiOU={number(audit['scope_gt_delta_pp']['mIoU'])}; "
                  f"64-point/patch budget dMiOU={number(audit['budget_gt_delta_pp']['mIoU'])}",
                  f"novel quality: {audit['novel_quality']}"]
    lines += [f"route: {summary['route']}",f"elapsed_seconds: {summary['elapsed_seconds']:.2f}",
              "No automatic retry, expansion or dev threshold sweep. Screen pass is not a guarantee on full dev/test."]
    return "\n".join(lines)+"\n"


def training_fit(models, bank, device, seed):
    """Last-weight sampled TRAIN fit, not validation selection or upper bound."""
    ids = np.random.default_rng(seed+2).choice(len(bank["history"]),min(128,len(bank["history"])),replace=False)
    report = {}
    for name,model in models.items():
        model.eval(); totals = {"positive_patches":0,"hit_positive_patches":0,"positive_points":0,"correct_points":0}
        presence = []
        with torch.inference_mode():
            for start in range(0,len(ids),32):
                take = ids[start:start+32]
                b = {k:torch.as_tensor(v[take],device=device) for k,v in bank.items()}
                out = model(b["history"],b["query"])
                presence.extend(out["presence_logits"].sigmoid().cpu().tolist())
                positive = b["mask"].any(1)
                if not positive.any(): continue
                scale = torch.tensor((model.config.patch_cells,model.config.patch_cells,bank["history"].shape[-1]),device=device)
                pi = torch.floor((out["xyz"][positive]+1)*.5*scale)
                gi = torch.floor((b["xyz"][positive]+1)*.5*scale)
                same_position = (pi[:,:,None] == gi[:,None]).all(-1) & b["mask"][positive,None]
                same_class = out["semantic_logits"][positive].argmax(-1)[:,:,None] == b["labels"][positive,None]
                correct = (same_position & same_class).any(-1)
                totals["positive_patches"] += int(positive.sum())
                totals["hit_positive_patches"] += int(correct.any(1).sum())
                totals["positive_points"] += correct.numel(); totals["correct_points"] += int(correct.sum())
        report[name] = {**totals,"patch_hit_rate":totals["hit_positive_patches"]/totals["positive_patches"] if totals["positive_patches"] else None,
            "point_semantic_precision_on_positive_patches":totals["correct_points"]/totals["positive_points"] if totals["positive_points"] else None,
            "mean_presence_probability":float(np.mean(presence)),"sampled_train_patches":len(ids)}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); add_config_args(parser)
    for name in ("train-cache","dev-cache","population-manifest","base-checkpoint","dataroot","train-info","dev-info","out-dir"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--expected-base-sha256",default=CLEAN_SHA256)
    parser.add_argument("--mode",choices=("smoke","screen"),default="screen")
    parser.add_argument("--batch-size",type=int,default=64,help="patches per variant update, NOT windows")
    parser.add_argument("--cpu-workers",type=int,default=8)
    parser.add_argument("--seed",type=int,default=20260930)
    parser.add_argument("--device",default="cuda")
    args = parser.parse_args(); start = time.perf_counter()
    if min(args.batch_size,args.cpu_workers) < 1: parser.error("invalid resource budget")
    for name in ("config","train_cache","dev_cache","population_manifest","base_checkpoint","train_info","dev_info"):
        if not str(getattr(args,name) or "").strip() or not Path(getattr(args,name)).is_file(): parser.error(f"{name} must be an existing file")
    if not Path(args.dataroot).is_dir(): parser.error("dataroot must be an existing directory")
    out = Path(args.out_dir)
    if out.exists(): parser.error("out-dir exists; choose a NEW directory")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    torch.set_num_threads(1); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    cfg = load_runtime_config(args.config,args.override); pcfg = make_prepare_config(cfg)
    if pcfg.free_label != 17: raise RuntimeError("requires frozen 18-class metric contract")
    manifest,dev_keys,_ = load_manifest(args.population_manifest)
    if len(dev_keys) != 64 or len(manifest["parent_keys"]) != 512 or manifest["selected_key_fingerprint"] != DEV64_FP:
        raise RuntimeError("requires the frozen Stage-1 dev512-derived dev64 identity/order")
    train_n,cal_n,dev_n,updates = (4,2,2,8) if args.mode == "smoke" else (128,16,64,512)
    _,all_train = load_cache(args.train_cache)
    train_keys,cal_keys = select_train_calibration([(r["scene_name"],r["t0_token"]) for r in all_train],
        train_n,cal_n,{k[0] for k in manifest["parent_keys"]})
    train_records,cal_records = align_records(all_train,train_keys),align_records(all_train,cal_keys); del all_train
    _,all_dev = load_cache(args.dev_cache); dev_records = align_records(all_dev,dev_keys[:dev_n]); del all_dev
    config = EmergenceConfig()
    provider = FrozenGenerationV18(args.base_checkpoint,args.expected_base_sha256,pcfg,device,args.cpu_workers,config)
    train_source = NuScenesWindowSource(args.dataroot,info_pkl=args.train_info,verbose=False)
    out.mkdir(parents=True)
    try: git_commit = subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except (OSError,subprocess.CalledProcessError): git_commit = "unavailable"
    identity = {"protocol":PROTOCOL,"feature_protocol":FEATURE_PROTOCOL,"base_checkpoint_sha256":provider.sha,
        "runtime_config_fingerprint":stable_json_fingerprint(cfg),"git_commit":git_commit,
        "populations":{"train":train_n,"calibration":cal_n,"dev":dev_n},"train_keys":train_keys,"calibration_keys":cal_keys,
        "dev_keys":dev_keys[:dev_n],"dev_manifest_fingerprint":manifest["manifest_fingerprint"],"scene_overlap":0,
        "updates":updates,"seed":args.seed,"batch_size_patches":args.batch_size,"threshold_candidates":THRESHOLDS}
    write_json(out/"execution_contract.json",{**identity,"arguments":vars(args),"model_config":asdict(config)})
    with (out/"progress.jsonl").open("x",encoding="utf-8") as handle:
        def progress(row):
            handle.write(json.dumps(finite_json(row),ensure_ascii=False,allow_nan=False)+"\n"); handle.flush()
        rng = np.random.default_rng(args.seed)
        bank = prepare_population(provider,train_source,train_records,training=True,rng=rng,memory_limit_mib=256,progress=progress)
        identity["bank_fingerprint"] = bank_fingerprint(bank)
        print(f"bank patches={len(bank['history'])} mib={sum(a.nbytes for a in bank.values())/2**20:.1f}; shared by both variants",flush=True)
        configs = {"direct":config,"refined":replace(config,refinement=True)}
        models,diagnostics = train_variants(bank,configs,device,updates=updates,batch_size=args.batch_size,seed=args.seed,progress=progress)
        diagnostics["sampled_fit"] = training_fit(models,bank,device,args.seed)
        del bank,train_records
        print("Training complete. Calibrating on TRAIN-only held-out scenes; dev has not been evaluated.",flush=True)
        calibration = prepare_population(provider,train_source,cal_records,training=False,rng=rng,progress=progress)
        del cal_records,train_source
        cal_reports,thresholds = {},{}
        for name,model in models.items():
            cal_reports[name] = evaluate_points(model,calibration,device,THRESHOLDS,batch_size=args.batch_size)
            thresholds[name] = select_calibration_threshold(cal_reports[name])
        write_json(out/"calibration.json",{"reports":cal_reports,"frozen_thresholds":thresholds})
        del calibration
        print(f"Thresholds frozen: {thresholds}. One final frozen dev evaluation starts.",flush=True)
        dev_source = NuScenesWindowSource(args.dataroot,info_pkl=args.dev_info,verbose=False)
        dev = prepare_population(provider,dev_source,dev_records,training=False,rng=rng,progress=progress)
        del dev_records,dev_source
        scope_audit = candidate_scope_ceiling(dev,config)
        variants = {}
        for name,model in models.items():
            threshold = thresholds[name]
            reports = evaluate_points(model,dev,device,tuple(dict.fromkeys((threshold,0.))),batch_size=args.batch_size)
            path = out/(name+".pt")
            accepted = args.mode == "screen" and reports[threshold]["gate"]["pass"]
            atomic_checkpoint(path,{**identity,"checkpoint_role":"auxiliary_generation_only","model_config":asdict(model.config),
                "state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"threshold":threshold,"screen_pass":accepted})
            variants[name] = {"threshold":threshold,"calibration_formal":cal_reports[name].get(threshold),
                "dev_formal":reports[threshold],"dev_raw_diagnostic":reports[0.],"checkpoint":str(path),
                "parameters":sum(p.numel() for p in model.parameters()),"screen_pass":accepted}
        summary = {**identity,"mode":args.mode,"variants":variants,"training_diagnostics":diagnostics,"candidate_scope_audit":scope_audit,
            "elapsed_seconds":time.perf_counter()-start,"real_data_run":True,
            "route":"screen_pass_review_before_expansion" if any(r["screen_pass"] for r in variants.values()) else "stop_both_variants_no_automatic_retry"}
        write_json(out/"summary.json",summary)
        text = summary_text(finite_json(summary)); (out/"summary.txt").write_text(text,encoding="utf-8"); print(text,flush=True)


if __name__ == "__main__": main()
