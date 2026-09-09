#!/usr/bin/env python3
"""Train controlled V17 single-expert Local-STWM variants.

Variants isolate the two changes motivated by the V16 diagnosis:
  R  : V16 loss, but frame-specific motion embedding + target-source mask.
  L  : exact V16 forward path, but add differentiable transport-overlap loss.
  RL : representation and overlap loss together.

No MLP/KTA/STWM fusion or router is introduced.  KTA remains the physics anchor,
the model predicts one KTA residual trajectory, and rigid source-shape transport
remains the decoder used at evaluation.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from real_motion.local_st_world_model_v17 import (
    LOCAL_STWM_V17_CACHE_VERSION,
    MODEL_PROTOCOL_V17,
    REPRESENTATION_CONTRACT,
    LocalSTWMV17Config,
    LocalSpatialTemporalWorldModelV17,
    soft_transport_overlap_loss,
)
from real_motion.metrics.moving_miou_v2 import SPEED_THRESHOLD_MPS
from real_motion.motion_transport import FEATURE_DIM, FUTURE_FRAMES, motion_transport_loss
from real_motion.motion_transport_v2 import TARGET_CONTRACT

VARIANTS = ("R", "L", "RL")


def load_cache(path: str):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if obj.get("version") != LOCAL_STWM_V17_CACHE_VERSION:
        raise RuntimeError(f"V17 cache version mismatch: {path}")
    meta = obj.get("metadata") or {}
    if meta.get("target_contract") != TARGET_CONTRACT:
        raise RuntimeError("displacement target contract mismatch")
    if meta.get("representation_contract") != REPRESENTATION_CONTRACT:
        raise RuntimeError("V17 representation contract mismatch")
    records = obj.get("records") or []
    if not records:
        raise RuntimeError("V17 cache contains no records")
    return meta, records


def flatten_supervised(records):
    keys = (
        "features", "local_semantic_tube", "kta_displacement_xy_m",
        "target_residual_xy_m", "target_displacement_xy_m", "existence",
        "target_valid", "supervised_source", "source_class_id",
        "frame_motion_features", "target_source_mask_tube",
    )
    chunks = {k: [] for k in keys}; scene_ids = []
    for r in records:
        sup = r["supervised_source"].bool()
        if not bool(sup.any()):
            continue
        for k in keys:
            if k == "supervised_source":
                chunks[k].append(torch.ones(int(sup.sum()), dtype=torch.bool))
            elif k in {"local_semantic_tube", "target_source_mask_tube"}:
                chunks[k].append(r[k][sup].to(torch.uint8))
            elif k == "source_class_id":
                chunks[k].append(r[k][sup].long())
            elif k == "target_valid":
                chunks[k].append(r[k][sup].bool())
            else:
                chunks[k].append(r[k][sup].float())
        scene_ids.extend([str(r["scene_name"])] * int(sup.sum()))
    if not chunks["features"]:
        raise RuntimeError("cache has no supervised Strong sources")
    out = {k: torch.cat(v, dim=0) for k, v in chunks.items()}
    out["scene_ids"] = scene_ids
    return out


def make_dataset(flat):
    return TensorDataset(
        flat["features"], flat["local_semantic_tube"], flat["kta_displacement_xy_m"],
        flat["target_residual_xy_m"], flat["target_displacement_xy_m"], flat["existence"],
        flat["target_valid"], flat["supervised_source"], flat["source_class_id"],
        flat["frame_motion_features"], flat["target_source_mask_tube"],
    )


def unpack(raw, device):
    (
        f, tube, kta_disp, residual, target_disp, existence, valid, supervised,
        class_id, frame_motion, source_mask,
    ) = raw
    return {
        "features": f.to(device, non_blocking=True),
        "local_semantic_tube": tube.to(device, non_blocking=True),
        "kta_displacement_xy_m": kta_disp.to(device, non_blocking=True),
        "target_residual_xy_m": residual.to(device, non_blocking=True),
        "target_displacement_xy_m": target_disp.to(device, non_blocking=True),
        "existence": existence.to(device, non_blocking=True),
        "target_valid": valid.to(device, non_blocking=True),
        "supervised_source": supervised.to(device, non_blocking=True),
        "source_class_id": class_id.to(device, non_blocking=True),
        "frame_motion_features": frame_motion.to(device, non_blocking=True),
        "target_source_mask_tube": source_mask.to(device, non_blocking=True),
    }


def true_moving_mask(target_disp: torch.Tensor, valid: torch.Tensor, frame_dt_s: float = 0.5):
    h = target_disp.shape[1]
    dt = torch.arange(1, h + 1, device=target_disp.device, dtype=target_disp.dtype)[None] * float(frame_dt_s)
    speed = torch.linalg.vector_norm(target_disp, dim=-1) / dt
    return valid.bool() & (speed >= float(SPEED_THRESHOLD_MPS))


def _latest_errors(kd, ld, mask):
    a, b = [], []
    for i in range(mask.shape[0]):
        ids = torch.nonzero(mask[i], as_tuple=False).flatten()
        if ids.numel():
            h = int(ids[-1]); a.append(float(kd[i, h])); b.append(float(ld[i, h]))
    return a, b


def _cat_mean(xs):
    return float(torch.cat(xs).mean().item()) if xs else float("nan")


def autocast_context(device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def forward_model(model, batch, *, use_representation: bool, amp: bool, device):
    with autocast_context(device, amp):
        return model(
            batch["features"], batch["local_semantic_tube"], batch["kta_displacement_xy_m"],
            batch["frame_motion_features"] if use_representation else None,
            batch["target_source_mask_tube"] if use_representation else None,
        )


def objective_loss(outputs, batch, *, overlap_weight: float, patch_resolution_m: float):
    base, parts = motion_transport_loss(outputs, batch)
    overlap = outputs["residual_xy_m"].sum() * 0.0
    overlap_stats = {"transport_soft_iou": float("nan"), "transport_overlap_labels": 0}
    if float(overlap_weight) > 0.0:
        overlap, overlap_stats = soft_transport_overlap_loss(
            outputs["residual_xy_m"].float(),
            batch["target_residual_xy_m"].float(),
            batch["target_source_mask_tube"][:, -1].float(),
            batch["target_valid"],
            patch_resolution_m=float(patch_resolution_m),
        )
    total = base + float(overlap_weight) * overlap
    result = dict(parts)
    result.update(overlap_stats)
    result["transport_overlap_loss"] = float(overlap.detach().cpu())
    result["objective_loss"] = float(total.detach().cpu())
    return total, result


def eval_model(model, loader, device, *, amp: bool, use_representation: bool,
               overlap_weight: float, patch_resolution_m: float):
    model.eval()
    objective_sum = base_sum = traj_sum = exist_sum = overlap_sum = 0.0; n_batches = 0
    soft_iou_num = soft_iou_den = 0.0
    kta_all=[]; learned_all=[]; kta_moving=[]; learned_moving=[]
    kta_fde=[]; learned_fde=[]; kta_moving_fde=[]; learned_moving_fde=[]
    horizon = {1:[[],[]], 3:[[],[]], 5:[[],[]]}
    exist_correct=exist_total=tp=fp=fn=0
    with torch.no_grad():
        for raw in loader:
            b = unpack(raw, device)
            out = forward_model(model, b, use_representation=use_representation, amp=amp, device=device)
            loss, parts = objective_loss(
                out, b, overlap_weight=overlap_weight, patch_resolution_m=patch_resolution_m
            )
            objective_sum += float(loss); base_sum += float(parts["loss"])
            traj_sum += float(parts["trajectory_smooth_l1"]); exist_sum += float(parts["existence_bce"])
            overlap_sum += float(parts["transport_overlap_loss"]); n_batches += 1
            if int(parts["transport_overlap_labels"]) > 0 and np.isfinite(float(parts["transport_soft_iou"])):
                soft_iou_num += float(parts["transport_soft_iou"]) * int(parts["transport_overlap_labels"])
                soft_iou_den += int(parts["transport_overlap_labels"])

            valid=b["target_valid"].bool(); moving=true_moving_mask(b["target_displacement_xy_m"], valid)
            target=b["target_residual_xy_m"].to(out["residual_xy_m"].dtype)
            kd=torch.linalg.vector_norm(target.float(),dim=-1)
            ld=torch.linalg.vector_norm((out["residual_xy_m"]-target).float(),dim=-1)
            if bool(valid.any()): kta_all.append(kd[valid].cpu()); learned_all.append(ld[valid].cpu())
            if bool(moving.any()): kta_moving.append(kd[moving].cpu()); learned_moving.append(ld[moving].cpu())
            a,z=_latest_errors(kd,ld,valid); kta_fde.extend(a); learned_fde.extend(z)
            a,z=_latest_errors(kd,ld,moving); kta_moving_fde.extend(a); learned_moving_fde.extend(z)
            for hi in horizon:
                m=moving[:,hi]
                if bool(m.any()):
                    horizon[hi][0].append(kd[:,hi][m].cpu()); horizon[hi][1].append(ld[:,hi][m].cpu())

            pred_exist=out["existence_logits"]>=0.0; gt_exist=b["existence"]>0.5
            exist_correct+=int((pred_exist==gt_exist).sum()); exist_total+=int(gt_exist.numel())
            tp+=int((pred_exist&gt_exist).sum()); fp+=int((pred_exist&~gt_exist).sum()); fn+=int((~pred_exist&gt_exist).sum())

    precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1)
    report={
        "objective_loss":objective_sum/max(n_batches,1), "base_motion_loss":base_sum/max(n_batches,1),
        "trajectory_smooth_l1":traj_sum/max(n_batches,1), "existence_bce":exist_sum/max(n_batches,1),
        "transport_overlap_loss":overlap_sum/max(n_batches,1),
        "transport_soft_iou":soft_iou_num/max(soft_iou_den,1) if soft_iou_den else float("nan"),
        "kta_ade_m":_cat_mean(kta_all), "learned_ade_m":_cat_mean(learned_all),
        "kta_fde_m":float(np.mean(kta_fde)) if kta_fde else float("nan"),
        "learned_fde_m":float(np.mean(learned_fde)) if learned_fde else float("nan"),
        "true_moving_kta_ade_m":_cat_mean(kta_moving), "true_moving_learned_ade_m":_cat_mean(learned_moving),
        "true_moving_kta_fde_m":float(np.mean(kta_moving_fde)) if kta_moving_fde else float("nan"),
        "true_moving_learned_fde_m":float(np.mean(learned_moving_fde)) if learned_moving_fde else float("nan"),
        "existence_accuracy":exist_correct/max(exist_total,1), "existence_precision":precision,
        "existence_recall":recall, "existence_f1":2*precision*recall/max(precision+recall,1e-12),
    }
    for hi,seconds in ((1,1.0),(3,2.0),(5,3.0)):
        report[f"true_moving_{seconds:g}s_kta_ade_m"]=_cat_mean(horizon[hi][0])
        report[f"true_moving_{seconds:g}s_learned_ade_m"]=_cat_mean(horizon[hi][1])
    return report


def save_ckpt(path, model, optimizer, epoch, args, train_meta, val_meta, val_report,
              *, use_representation: bool, effective_overlap_weight: float):
    torch.save({
        "protocol": MODEL_PROTOCOL_V17, "epoch":int(epoch), "state_dict":model.state_dict(),
        "optimizer":optimizer.state_dict(), "feature_dim":FEATURE_DIM, "future_frames":FUTURE_FRAMES,
        "model_config":asdict(model.v17_config), "variant":str(args.variant),
        "use_representation":bool(use_representation), "overlap_weight":float(effective_overlap_weight),
        "args":vars(args), "train_cache_metadata":train_meta, "val_cache_metadata":val_meta,
        "val_report":val_report,
    }, path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train-cache",required=True); p.add_argument("--val-cache",required=True)
    p.add_argument("--output-dir",required=True); p.add_argument("--variant",choices=VARIANTS,required=True)
    p.add_argument("--epochs",type=int,default=20); p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--d-model",type=int,default=128); p.add_argument("--semantic-dim",type=int,default=32)
    p.add_argument("--heads",type=int,default=4); p.add_argument("--blocks",type=int,default=4)
    p.add_argument("--decoder-blocks",type=int,default=2); p.add_argument("--lr",type=float,default=5e-4)
    p.add_argument("--weight-decay",type=float,default=1e-4); p.add_argument("--num-workers",type=int,default=2)
    p.add_argument("--overlap-weight",type=float,default=0.25,
                   help="used only by L/RL; conservative auxiliary weight")
    p.add_argument("--seed",type=int,default=20260909); p.add_argument("--device",default="cuda")
    p.add_argument("--no-amp",action="store_true")
    a=p.parse_args()
    if a.epochs<=0 or a.batch_size<=0 or a.lr<=0 or a.overlap_weight<0:
        raise ValueError("invalid optimization arguments")
    use_representation=a.variant in {"R","RL"}
    effective_overlap=float(a.overlap_weight) if a.variant in {"L","RL"} else 0.0

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    device=torch.device(a.device if a.device!="cuda" or torch.cuda.is_available() else "cpu")
    amp=device.type=="cuda" and not bool(a.no_amp)

    train_meta,train_records=load_cache(a.train_cache); val_meta,val_records=load_cache(a.val_cache)
    train=flatten_supervised(train_records); val=flatten_supervised(val_records)
    overlap=sorted(set(train["scene_ids"]) & set(val["scene_ids"]))
    if overlap: raise RuntimeError(f"train/val scene overlap: {overlap[:5]}")
    tube_hw=int(train["local_semantic_tube"].shape[-1])
    patch_resolution=float(train_meta.get("patch_resolution_m",0.8))
    if tuple(train["local_semantic_tube"].shape[1:])!=(6,tube_hw,tube_hw):
        raise RuntimeError("unexpected train tube shape")

    gen=torch.Generator().manual_seed(int(a.seed))
    train_loader=DataLoader(make_dataset(train),batch_size=a.batch_size,shuffle=True,generator=gen,
                            num_workers=a.num_workers,pin_memory=device.type=="cuda",drop_last=False)
    val_loader=DataLoader(make_dataset(val),batch_size=a.batch_size,shuffle=False,
                          num_workers=a.num_workers,pin_memory=device.type=="cuda",drop_last=False)
    mcfg=LocalSTWMV17Config(d_model=a.d_model,semantic_dim=a.semantic_dim,heads=a.heads,blocks=a.blocks,
                            decoder_blocks=a.decoder_blocks,tube_hw=tube_hw,use_representation=use_representation)
    model=LocalSpatialTemporalWorldModelV17(mcfg).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    total_steps=max(1,a.epochs*len(train_loader)); step=0
    out_dir=Path(a.output_dir); out_dir.mkdir(parents=True,exist_ok=True)
    history=[]; best_objective=float("inf"); best_ade=float("inf")

    init_report=eval_model(model,val_loader,device,amp=amp,use_representation=use_representation,
                           overlap_weight=effective_overlap,patch_resolution_m=patch_resolution)
    if abs(float(init_report["learned_ade_m"])-float(init_report["kta_ade_m"]))>1e-5:
        raise RuntimeError("zero-init safety contract failed: fresh V17 is not KTA")
    param_count=sum(p.numel() for p in model.parameters())
    print(json.dumps({"protocol":MODEL_PROTOCOL_V17,"variant":a.variant,"use_representation":use_representation,
                      "overlap_weight":effective_overlap,"model_config":asdict(mcfg),"parameters":param_count,
                      "train_sources":len(train["features"]),"val_sources":len(val["features"]),
                      "train_scenes":len(set(train["scene_ids"])),"val_scenes":len(set(val["scene_ids"])),
                      "amp_bfloat16":amp,"initial_val":init_report},indent=2))

    for epoch in range(1,a.epochs+1):
        model.train(); running=0.0
        for raw in train_loader:
            b=unpack(raw,device)
            out=forward_model(model,b,use_representation=use_representation,amp=amp,device=device)
            loss,_=objective_loss(out,b,overlap_weight=effective_overlap,patch_resolution_m=patch_resolution)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step(); step+=1
            frac=min(step/total_steps,1.0); scale=0.1+0.9*0.5*(1.0+math.cos(math.pi*frac))
            for group in optimizer.param_groups: group["lr"]=a.lr*scale
            running+=float(loss.item())

        val_report=eval_model(model,val_loader,device,amp=amp,use_representation=use_representation,
                              overlap_weight=effective_overlap,patch_resolution_m=patch_resolution)
        row={"epoch":epoch,"train_objective_loss":running/max(len(train_loader),1),
             "lr":optimizer.param_groups[0]["lr"],**val_report}
        history.append(row); print(json.dumps(row))
        save_ckpt(out_dir/"latest.pt",model,optimizer,epoch,a,train_meta,val_meta,val_report,
                  use_representation=use_representation,effective_overlap_weight=effective_overlap)
        if epoch in {1,5,10,a.epochs}:
            save_ckpt(out_dir/f"epoch_{epoch:04d}.pt",model,optimizer,epoch,a,train_meta,val_meta,val_report,
                      use_representation=use_representation,effective_overlap_weight=effective_overlap)
        if float(val_report["objective_loss"])<best_objective:
            best_objective=float(val_report["objective_loss"])
            save_ckpt(out_dir/"best.pt",model,optimizer,epoch,a,train_meta,val_meta,val_report,
                      use_representation=use_representation,effective_overlap_weight=effective_overlap)
        if float(val_report["learned_ade_m"])<best_ade:
            best_ade=float(val_report["learned_ade_m"])
            save_ckpt(out_dir/"best_ade.pt",model,optimizer,epoch,a,train_meta,val_meta,val_report,
                      use_representation=use_representation,effective_overlap_weight=effective_overlap)

    report={"protocol":MODEL_PROTOCOL_V17,"variant":a.variant,"model_config":asdict(mcfg),
            "parameters":param_count,"best_objective_loss":best_objective,"best_learned_ade_m":best_ade,
            "initial_val":init_report,"history":history,"train_sources":len(train["features"]),
            "val_sources":len(val["features"])}
    (out_dir/"training_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
