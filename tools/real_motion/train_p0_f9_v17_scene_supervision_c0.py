#!/usr/bin/env python3
"""C0 control-fidelity experiment for V17-RL scene supervision.

The base path reproduces the historical source-level V17-RL continuation.
C0-C changes nothing in that base path. C0-S adds one independent sparse
full-scene CE gradient before the same optimizer step.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json, math, random, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from real_motion.local_st_world_model_v17 import MODEL_PROTOCOL_V17, LocalSpatialTemporalWorldModelV17, config_from_mapping_v17
from real_motion.local_stwm_scene_supervision import V17SceneCacheDataset, calibrate_scene_alpha_from_gradients, grad_vector, v17_base_loss_tensors
from real_motion.motion_transport import FUTURE_FRAMES
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from tools.real_motion.train_p0_f9_v17_local_stwm import eval_model, flatten_supervised, forward_model, load_cache, make_dataset, objective_loss, unpack
from tools.real_motion.train_p0_f9_v17_scene_supervision_pair import _pack_sources, _record_map, _scene_loss_batch, _shard_local_batches

PROTOCOL="p0_f9_v17_scene_supervision_c0_source_fidelity_v1"
ARMS=("C0-C","C0-S")
EXPECTED_START_EPOCH=5
EXPECTED_VARIANT="RL"
EXPECTED_OVERLAP=0.25

def _optimizer_step_range(optimizer):
    vals=[]
    for state in optimizer.state.values():
        if "step" in state:
            x=state["step"]; vals.append(int(x.item()) if torch.is_tensor(x) else int(x))
    return (min(vals),max(vals)) if vals else (0,0)

def _lr_scale(step,total_steps):
    frac=min(max(float(step)/max(int(total_steps),1),0.0),1.0)
    return 0.1+0.9*0.5*(1.0+math.cos(math.pi*frac))

def alpha_ramp(local_step,warmup_steps):
    if int(warmup_steps)<=1:return 1.0
    return min(max((int(local_step)-1)/float(int(warmup_steps)-1),0.0),1.0)

def enable_all_parameters(model):
    names=[]; total=0
    for name,p in model.named_parameters():
        p.requires_grad_(True); names.append(name); total+=int(p.numel())
    if not names: raise RuntimeError("V17 model unexpectedly has no parameters")
    return {"trainable_parameters":total,"trainable_names":names,"frozen_parameters":0}

def _assert_no_stateful_train_buffers(model):
    bad=[]
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm): bad.append(name or "<root>")
    if bad: raise RuntimeError(f"C0 scene path cannot isolate stateful BatchNorm modules: {bad[:5]}")

@contextmanager
def preserve_rng_state(device):
    cpu=torch.get_rng_state()
    cuda=torch.cuda.get_rng_state(device) if device.type=="cuda" else None
    try: yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None: torch.cuda.set_rng_state(cuda,device)

def _loader(dataset,*,batch_size,shuffle,generator,num_workers,prefetch_factor,pin_memory):
    kw=dict(dataset=dataset,batch_size=int(batch_size),shuffle=bool(shuffle),num_workers=int(num_workers),pin_memory=bool(pin_memory),drop_last=False)
    if generator is not None: kw["generator"]=generator
    if int(num_workers)>0:
        kw["persistent_workers"]=True; kw["prefetch_factor"]=int(prefetch_factor)
    return DataLoader(**kw)

class SourceCUDAPrefetcher:
    def __init__(self,loader,device):
        if device.type!="cuda": raise ValueError("CUDA prefetcher requires CUDA")
        self._it=iter(loader); self.device=device; self.stream=torch.cuda.Stream(device=device); self.next_batch=None; self._preload()
    def _preload(self):
        try: raw=next(self._it)
        except StopIteration:
            self.next_batch=None; return
        with torch.cuda.stream(self.stream): self.next_batch=unpack(raw,self.device)
    def __iter__(self): return self
    def __next__(self):
        if self.next_batch is None: raise StopIteration
        current=torch.cuda.current_stream(self.device); current.wait_stream(self.stream)
        batch=self.next_batch
        for value in batch.values():
            if torch.is_tensor(value) and value.is_cuda: value.record_stream(current)
        self._preload(); return batch

def _source_iterator(loader,device,cuda_prefetch):
    if cuda_prefetch and device.type=="cuda": return SourceCUDAPrefetcher(loader,device)
    return (unpack(raw,device) for raw in loader)

def _motion_params(model):
    names=[]; params=[]
    for name,p in model.named_parameters():
        if p.requires_grad and not name.startswith("existence_head"):
            names.append(name); params.append(p)
    if not params: raise RuntimeError("no residual-path parameters for C0 calibration")
    return names,params

def _norm(g): return float(torch.linalg.vector_norm(g.float()).detach().cpu())
def _cos(a,b):
    den=torch.linalg.vector_norm(a.float())*torch.linalg.vector_norm(b.float())
    if float(den.detach().cpu())<=0:return float("nan")
    return float((torch.dot(a.float(),b.float())/den).detach().cpu())

def _scene_state(scene_ds,v17_by_id,*,scene_batch_size,scene_seed):
    return {"pass_index":0,"iterator":iter(_shard_local_batches(scene_ds,v17_by_id,batch_size=int(scene_batch_size),seed=int(scene_seed),pass_index=0))}

def _next_scene_batch(state,scene_ds,v17_by_id,*,scene_batch_size,scene_seed):
    while True:
        try:return next(state["iterator"])
        except StopIteration:
            state["pass_index"]+=1
            state["iterator"]=iter(_shard_local_batches(scene_ds,v17_by_id,batch_size=int(scene_batch_size),seed=int(scene_seed),pass_index=int(state["pass_index"])))

def calibrate_c0(model,source_loader,scene_ds,v17_by_id,device,*,pcfg,amp,cuda_prefetch,scene_batch_size,scene_seed,batches,overlap_resolution_m,halo_voxels,eps,jitter,target_ratio,max_alpha):
    model.train(); _,params=_motion_params(model); rows=[]
    src_it=_source_iterator(source_loader,device,cuda_prefetch=cuda_prefetch)
    scene_state=_scene_state(scene_ds,v17_by_id,scene_batch_size=scene_batch_size,scene_seed=scene_seed)
    for bi in range(1,int(batches)+1):
        try: base_batch=next(src_it)
        except StopIteration: raise RuntimeError(f"source loader exhausted during C0 calibration at batch {bi}")
        scene_samples=_next_scene_batch(scene_state,scene_ds,v17_by_id,scene_batch_size=scene_batch_size,scene_seed=scene_seed)
        scene_batch,slices=_pack_sources(scene_samples,device)
        base_out=forward_model(model,base_batch,use_representation=True,amp=amp,device=device)
        base=v17_base_loss_tensors(base_out,base_batch,overlap_weight=EXPECTED_OVERLAP,patch_resolution_m=float(overlap_resolution_m))
        with preserve_rng_state(device):
            scene_out=forward_model(model,scene_batch,use_representation=True,amp=amp,device=device)
            scene_loss,scene_stats=_scene_loss_batch(scene_out,scene_batch,scene_samples,slices,pcfg=pcfg,halo_voxels=halo_voxels,eps=eps,jitter=jitter)
        g_pos=grad_vector(base["position"],params,retain_graph=True)
        g_ov=grad_vector(base["weighted_overlap"],params,retain_graph=True)
        g_motion=grad_vector(base["motion"],params,retain_graph=False)
        g_scene=grad_vector(scene_loss,params,retain_graph=False)
        rows.append({
            "batch":bi,"source_batch_sources":int(base_batch["features"].shape[0]),"scene_batch_scenes":len(scene_samples),
            "scene_batch_sources":int(scene_batch["features"].shape[0]),"position_grad_l2":_norm(g_pos),
            "weighted_overlap_grad_l2":_norm(g_ov),"motion_grad_l2":_norm(g_motion),"scene_unit_grad_l2":_norm(g_scene),
            "position_overlap_cosine":_cos(g_pos,g_ov),"scene_motion_cosine":_cos(g_scene,g_motion),
            "scene_query_fraction":scene_stats["query_fraction"],"position_loss":float(base["position"].detach().cpu()),
            "weighted_overlap_loss":float(base["weighted_overlap"].detach().cpu()),"scene_full_ce":float(scene_loss.detach().cpu())
        })
    cal=calibrate_scene_alpha_from_gradients([r["motion_grad_l2"] for r in rows],[r["scene_unit_grad_l2"] for r in rows],target_ratio=float(target_ratio),max_alpha=float(max_alpha))
    for key in ("position_grad_l2","weighted_overlap_grad_l2","motion_grad_l2","scene_unit_grad_l2","position_overlap_cosine","scene_motion_cosine","scene_query_fraction"):
        vals=np.asarray([r[key] for r in rows],dtype=np.float64)
        cal[f"{key}_median"]=float(np.nanmedian(vals)); cal[f"{key}_min"]=float(np.nanmin(vals)); cal[f"{key}_max"]=float(np.nanmax(vals))
    cal["calibration_contract"]="source_level_rl_motion_gradient_vs_independent_scene_gradient_all_params_v1"
    cal["batches"]=rows
    return cal

def _save_ckpt(path,model,optimizer,ck,*,arm,last_completed_epoch,local_step,global_step,scene_alpha,scene_alpha_now,args,train_meta,val_meta,calibration):
    torch.save({
        "protocol":MODEL_PROTOCOL_V17,"epoch":int(last_completed_epoch),"state_dict":model.state_dict(),"optimizer":optimizer.state_dict(),
        "feature_dim":ck.get("feature_dim"),"future_frames":FUTURE_FRAMES,"model_config":ck.get("model_config"),"variant":EXPECTED_VARIANT,
        "use_representation":True,"overlap_weight":EXPECTED_OVERLAP,"args":ck.get("args") or {},"train_cache_metadata":train_meta,"val_cache_metadata":val_meta,
        "c0_continuation":{"protocol":PROTOCOL,"arm":arm,"resume_checkpoint":str(Path(args.resume_checkpoint).resolve()),"start_epoch":EXPECTED_START_EPOCH,
        "last_completed_epoch":int(last_completed_epoch),"local_step":int(local_step),"global_optimizer_step":int(global_step),
        "source_batch_contract":"original_v17_rl_flatten_supervised_checkpoint_batch_size_v1","all_parameters_trainable":True,
        "scene_path_is_separate_from_base_batch":True,"scene_alpha_target":float(scene_alpha),"scene_alpha_current":float(scene_alpha_now),
        "calibration":calibration,"runtime_args":vars(args)}
    },path)

def _parse_save_steps(raw):
    vals=set()
    for x in str(raw).split(","):
        x=x.strip()
        if not x:continue
        v=int(x)
        if v<=0:raise ValueError("save steps must be positive")
        vals.add(v)
    return vals

def _reset_seeds(seed):
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))

def main():
    p=argparse.ArgumentParser(); add_config_args(p)
    p.add_argument("--train-cache",required=True); p.add_argument("--val-cache",required=True); p.add_argument("--scene-cache",default="")
    p.add_argument("--resume-checkpoint",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--arm",choices=ARMS,required=True)
    p.add_argument("--continuation-epochs",type=int,default=3); p.add_argument("--max-steps",type=int,default=0)
    p.add_argument("--save-steps",default="300,600"); p.add_argument("--num-workers",type=int,default=6); p.add_argument("--prefetch-factor",type=int,default=4)
    p.add_argument("--log-every",type=int,default=20); p.add_argument("--paired-shuffle-seed",type=int,default=20260910)
    p.add_argument("--scene-seed",type=int,default=20260911); p.add_argument("--scene-batch-size",type=int,default=4)
    p.add_argument("--scene-alpha",type=float,default=-1.0); p.add_argument("--scene-warmup-fraction",type=float,default=0.2)
    p.add_argument("--halo-voxels",type=int,default=2); p.add_argument("--scene-eps",type=float,default=1e-4); p.add_argument("--scene-jitter-voxels",type=float,default=0.25)
    p.add_argument("--calibration-batches",type=int,default=8); p.add_argument("--calibration-target-ratio",type=float,default=0.25)
    p.add_argument("--max-calibrated-alpha",type=float,default=1e4); p.add_argument("--calibration-output",default="")
    p.add_argument("--calibrate-only",action="store_true"); p.add_argument("--device",default="cuda"); p.add_argument("--no-amp",action="store_true")
    p.add_argument("--no-cuda-prefetch",action="store_true")
    a=p.parse_args()
    if a.continuation_epochs<=0 or a.max_steps<0:raise ValueError("invalid continuation length")
    if a.num_workers<0 or a.prefetch_factor<=0 or a.log_every<=0:raise ValueError("invalid DataLoader/runtime settings")
    if a.scene_batch_size<=0 or a.calibration_batches<=0:raise ValueError("invalid scene/calibration batch settings")
    if not 0<=a.scene_warmup_fraction<=1:raise ValueError("scene warmup fraction must be in [0,1]")
    if a.halo_voxels<0 or not 0<a.scene_eps<1/18 or a.scene_jitter_voxels<0:raise ValueError("invalid scene renderer settings")
    save_steps=_parse_save_steps(a.save_steps)
    _reset_seeds(a.paired_shuffle_seed)
    device=torch.device(a.device if a.device!="cuda" or torch.cuda.is_available() else "cpu")
    amp=device.type=="cuda" and not bool(a.no_amp); cuda_prefetch=device.type=="cuda" and not bool(a.no_cuda_prefetch)
    cfg_runtime=load_runtime_config(a.config,a.override); pcfg=make_prepare_config(cfg_runtime)

    train_meta,train_records=load_cache(a.train_cache); val_meta,val_records=load_cache(a.val_cache)
    train=flatten_supervised(train_records); val=flatten_supervised(val_records)
    overlap=sorted(set(train["scene_ids"])&set(val["scene_ids"]))
    if overlap:raise RuntimeError(f"train/val scene overlap: {overlap[:5]}")

    ck=torch.load(a.resume_checkpoint,map_location="cpu",weights_only=False)
    if ck.get("protocol")!=MODEL_PROTOCOL_V17 or str(ck.get("variant"))!=EXPECTED_VARIANT:raise RuntimeError("C0 requires a V17-RL checkpoint")
    if int(ck.get("epoch",-1))!=EXPECTED_START_EPOCH:raise RuntimeError(f"C0 requires epoch {EXPECTED_START_EPOCH}")
    if not bool(ck.get("use_representation",False)):raise RuntimeError("C0 requires the V17 representation")
    if not math.isclose(float(ck.get("overlap_weight",-1.0)),EXPECTED_OVERLAP,abs_tol=1e-12):raise RuntimeError("C0 requires historical overlap weight 0.25")

    ck_args=ck.get("args") or {}; batch_size=int(ck_args.get("batch_size",256)); base_lr=float(ck_args.get("lr",5e-4))
    weight_decay=float(ck_args.get("weight_decay",1e-4)); original_epochs=int(ck_args.get("epochs",10))
    end_epoch=EXPECTED_START_EPOCH+int(a.continuation_epochs)
    if end_epoch>original_epochs:raise RuntimeError(f"C0 continuation exceeds original schedule: end={end_epoch}, original={original_epochs}")

    def make_train_loader(seed):
        gen=torch.Generator().manual_seed(int(seed))
        return _loader(make_dataset(train),batch_size=batch_size,shuffle=True,generator=gen,num_workers=a.num_workers,prefetch_factor=a.prefetch_factor,pin_memory=device.type=="cuda")
    train_loader=make_train_loader(a.paired_shuffle_seed)
    val_loader=_loader(make_dataset(val),batch_size=batch_size,shuffle=False,generator=None,num_workers=a.num_workers,prefetch_factor=a.prefetch_factor,pin_memory=device.type=="cuda")

    model=LocalSpatialTemporalWorldModelV17(config_from_mapping_v17(ck.get("model_config"))).to(device)
    model.load_state_dict(ck["state_dict"],strict=True); trainable_report=enable_all_parameters(model); _assert_no_stateful_train_buffers(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=base_lr,weight_decay=weight_decay); optimizer.load_state_dict(ck["optimizer"])
    batches_per_epoch=len(train_loader); start_step=EXPECTED_START_EPOCH*batches_per_epoch; total_original_steps=original_epochs*batches_per_epoch
    min_opt_step,max_opt_step=_optimizer_step_range(optimizer)
    if max_opt_step!=start_step:raise RuntimeError(f"optimizer step mismatch: checkpoint max={max_opt_step}, expected={start_step}; source-count/batch contract is not historical RL")
    expected_lr=base_lr*_lr_scale(start_step,total_original_steps); actual_lr=float(optimizer.param_groups[0]["lr"])
    if not math.isclose(actual_lr,expected_lr,rel_tol=2e-6,abs_tol=1e-10):raise RuntimeError(f"resume LR mismatch: checkpoint={actual_lr}, expected={expected_lr}")
    overlap_resolution_m=float(train_meta.get("patch_resolution_m",0.8))

    print("C0 initial source-level validation ...",flush=True)
    init_report=eval_model(model,val_loader,device,amp=amp,use_representation=True,overlap_weight=EXPECTED_OVERLAP,patch_resolution_m=overlap_resolution_m)
    prior_val=ck.get("val_report") or {}
    if "learned_ade_m" in prior_val and not math.isclose(float(init_report["learned_ade_m"]),float(prior_val["learned_ade_m"]),rel_tol=0.0,abs_tol=2e-4):
        raise RuntimeError("C0 checkpoint/cache identity failed: initial learned ADE does not reproduce epoch-5")

    scene_ds=None; v17_by_id=None
    if a.calibrate_only or a.arm=="C0-S":
        if not a.scene_cache:raise ValueError("--scene-cache is required for calibration and C0-S")
        scene_ds=V17SceneCacheDataset(a.scene_cache); ids={str(e["sample_id"]) for e in scene_ds.entries}; v17_by_id=_record_map(train_records,ids)

    preflight={"protocol":PROTOCOL,"arm":a.arm,"resume_checkpoint":str(Path(a.resume_checkpoint).resolve()),"start_epoch":EXPECTED_START_EPOCH,
        "requested_end_epoch":end_epoch,"batch_size":batch_size,"batches_per_epoch":batches_per_epoch,"train_sources":int(train["features"].shape[0]),
        "val_sources":int(val["features"].shape[0]),"all_parameters_trainable":True,"trainable_parameters":int(trainable_report["trainable_parameters"]),
        "optimizer_step_range":[min_opt_step,max_opt_step],"start_lr":actual_lr,"original_total_steps":total_original_steps,
        "paired_shuffle_seed":int(a.paired_shuffle_seed),"amp_bfloat16":amp,"cuda_prefetch_one_batch":cuda_prefetch,"initial_val":init_report}
    print("=== C0 SOURCE-FIDELITY PREFLIGHT ==="); print(json.dumps(preflight,indent=2),flush=True)

    calibration=None
    if a.calibrate_only or a.arm=="C0-S":
        cal_loader=make_train_loader(a.paired_shuffle_seed)
        print("C0 source/scene alpha calibration ...",flush=True)
        calibration=calibrate_c0(model,cal_loader,scene_ds,v17_by_id,device,pcfg=pcfg,amp=amp,cuda_prefetch=cuda_prefetch,
            scene_batch_size=a.scene_batch_size,scene_seed=a.scene_seed,batches=a.calibration_batches,overlap_resolution_m=overlap_resolution_m,
            halo_voxels=a.halo_voxels,eps=a.scene_eps,jitter=a.scene_jitter_voxels,target_ratio=a.calibration_target_ratio,max_alpha=a.max_calibrated_alpha)
        payload={"preflight":preflight,"calibration":calibration}; print("=== C0 SCENE-LOSS CALIBRATION ==="); print(json.dumps(payload,indent=2),flush=True)
        print(f"recommended_scene_alpha={calibration['alpha']:.12g}",flush=True)
        if a.calibration_output:
            cp=Path(a.calibration_output); cp.parent.mkdir(parents=True,exist_ok=True); cp.write_text(json.dumps(payload,indent=2),encoding="utf-8"); print(f"saved {cp}",flush=True)
        if a.calibrate_only:return
        if a.scene_alpha<=0:raise ValueError("C0-S training requires positive --scene-alpha from calibrate-only")
        if not math.isclose(float(a.scene_alpha),float(calibration["alpha"]),rel_tol=5e-3,abs_tol=1e-12):
            raise RuntimeError(f"--scene-alpha {a.scene_alpha} differs from C0 calibration {calibration['alpha']}")

    out_dir=Path(a.output_dir); out_dir.mkdir(parents=True,exist_ok=True)
    (out_dir/"preflight.json").write_text(json.dumps({**preflight,"calibration":calibration},indent=2),encoding="utf-8")
    max_available_steps=int(a.continuation_epochs)*batches_per_epoch
    target_steps=max_available_steps if int(a.max_steps)==0 else min(int(a.max_steps),max_available_steps)
    if target_steps<=0:raise RuntimeError("C0 target step count is empty")
    warmup_steps=max(1,int(round(float(a.scene_warmup_fraction)*target_steps)))
    _reset_seeds(a.paired_shuffle_seed); train_loader=make_train_loader(a.paired_shuffle_seed)
    scene_state=_scene_state(scene_ds,v17_by_id,scene_batch_size=a.scene_batch_size,scene_seed=a.scene_seed) if a.arm=="C0-S" else None

    local_step=0; global_step=start_step; last_completed_epoch=EXPECTED_START_EPOCH; history=[]; run_started=time.perf_counter(); stop=False
    for global_epoch in range(EXPECTED_START_EPOCH+1,end_epoch+1):
        model.train(); epoch_started=time.perf_counter(); running=0.0; seen=0
        iterator=_source_iterator(train_loader,device,cuda_prefetch=cuda_prefetch); bi=0
        for bi,base_batch in enumerate(iterator,start=1):
            if local_step>=target_steps:stop=True;break
            local_step+=1; optimizer.zero_grad(set_to_none=True)
            base_out=forward_model(model,base_batch,use_representation=True,amp=amp,device=device)
            base_loss,base_parts=objective_loss(base_out,base_batch,overlap_weight=EXPECTED_OVERLAP,patch_resolution_m=overlap_resolution_m)
            base_loss.backward()
            alpha_now=0.0; scene_loss_value=float("nan"); query_fraction=float("nan"); scene_sources=0; scene_pass=-1
            if a.arm=="C0-S":
                alpha_now=float(a.scene_alpha)*alpha_ramp(local_step,warmup_steps)
                scene_samples=_next_scene_batch(scene_state,scene_ds,v17_by_id,scene_batch_size=a.scene_batch_size,scene_seed=a.scene_seed)
                scene_pass=int(scene_state["pass_index"]); scene_batch,slices=_pack_sources(scene_samples,device)
                with preserve_rng_state(device):
                    scene_out=forward_model(model,scene_batch,use_representation=True,amp=amp,device=device)
                    scene_loss,scene_stats=_scene_loss_batch(scene_out,scene_batch,scene_samples,slices,pcfg=pcfg,halo_voxels=a.halo_voxels,eps=a.scene_eps,jitter=a.scene_jitter_voxels)
                    (float(alpha_now)*scene_loss).backward()
                scene_loss_value=float(scene_loss.detach().cpu()); query_fraction=float(scene_stats["query_fraction"]); scene_sources=int(scene_batch["features"].shape[0])
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step(); global_step+=1
            next_lr=base_lr*_lr_scale(global_step,total_original_steps)
            for group in optimizer.param_groups:group["lr"]=next_lr
            running+=float(base_loss.detach().cpu()); seen+=int(base_batch["features"].shape[0])
            if local_step==1 or local_step%int(a.log_every)==0 or local_step in save_steps or local_step==target_steps:
                row={"local_step":int(local_step),"global_optimizer_step":int(global_step),"global_epoch_in_progress":int(global_epoch),
                    "source_batch_index":int(bi),"source_batch_sources":int(base_batch["features"].shape[0]),"source_seen_this_epoch":int(seen),
                    "lr":float(next_lr),"base_objective":float(base_loss.detach().cpu()),"trajectory_smooth_l1":float(base_parts["trajectory_smooth_l1"]),
                    "existence_bce":float(base_parts["existence_bce"]),"transport_overlap_loss":float(base_parts["transport_overlap_loss"]),
                    "scene_alpha":float(alpha_now),"scene_full_ce":scene_loss_value,"scene_query_fraction":query_fraction,
                    "scene_batch_sources":scene_sources,"scene_pass_index":scene_pass,"elapsed_s":time.perf_counter()-run_started}
                history.append(row); print("C0_STEP "+json.dumps(row),flush=True)
            if local_step in save_steps or local_step==target_steps:
                _save_ckpt(out_dir/f"step_{local_step:06d}.pt",model,optimizer,ck,arm=a.arm,last_completed_epoch=last_completed_epoch,
                    local_step=local_step,global_step=global_step,scene_alpha=max(float(a.scene_alpha),0.0),scene_alpha_now=alpha_now,args=a,
                    train_meta=train_meta,val_meta=val_meta,calibration=calibration)
            if local_step>=target_steps:stop=True;break

        completed_epoch=(bi==batches_per_epoch)
        if completed_epoch:
            last_completed_epoch=int(global_epoch)
            print(f"C0 source-level validation epoch {global_epoch} ...",flush=True)
            val_report=eval_model(model,val_loader,device,amp=amp,use_representation=True,overlap_weight=EXPECTED_OVERLAP,patch_resolution_m=overlap_resolution_m)
            epoch_row={"global_epoch":int(global_epoch),"local_step":int(local_step),"train_base_objective":running/max(batches_per_epoch,1),
                "lr":float(optimizer.param_groups[0]["lr"]),"epoch_seconds":time.perf_counter()-epoch_started,**val_report}
            print("=== C0 SOURCE EPOCH SUMMARY ==="); print(json.dumps(epoch_row,indent=2),flush=True)
            _save_ckpt(out_dir/f"epoch_{global_epoch:04d}.pt",model,optimizer,ck,arm=a.arm,last_completed_epoch=last_completed_epoch,
                local_step=local_step,global_step=global_step,scene_alpha=max(float(a.scene_alpha),0.0),scene_alpha_now=alpha_now,args=a,
                train_meta=train_meta,val_meta=val_meta,calibration=calibration)
        if stop:break

    report={"protocol":PROTOCOL,"arm":a.arm,"resume_checkpoint":str(Path(a.resume_checkpoint).resolve()),"start_epoch":EXPECTED_START_EPOCH,
        "last_completed_epoch":int(last_completed_epoch),"local_steps":int(local_step),"global_optimizer_step":int(global_step),"target_steps":int(target_steps),
        "source_batch_size":int(batch_size),"batches_per_epoch":int(batches_per_epoch),"all_parameters_trainable":True,
        "scene_alpha":max(float(a.scene_alpha),0.0),"calibration":calibration,"history":history,"elapsed_s":time.perf_counter()-run_started,
        "fidelity_reference":"Compare C0-C epoch_0008 to historical V17 B-C epoch_0008 under the same A1 evaluator."}
    (out_dir/"training_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("=== C0 COMPLETE ==="); print(json.dumps({"arm":a.arm,"local_steps":local_step,"last_completed_epoch":last_completed_epoch,
        "output_dir":str(out_dir),"saved_step_checkpoints":sorted(s for s in save_steps if s<=local_step)},indent=2),flush=True)

if __name__=="__main__":main()
