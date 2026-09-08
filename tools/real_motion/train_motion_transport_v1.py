#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
import torch.distributed as dist,yaml
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.motion_transport_v1.config import load_config,get
from real_motion.motion_transport_v1.data import ManifestDataset,load_manifest
from real_motion.motion_transport_v1.model import MotionTransportV1
from real_motion.motion_transport_v1.engine import init_distributed,seed_all,train,barrier
from real_motion.motion_transport_v1.routing import msp_checkpoint_provenance,sha256_file
from real_motion.motion_transport_v1.server_acceptance import formal_server_environment
def _git():
    try:return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    except Exception:return 'unknown'
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--override',action='append',default=[]);p.add_argument('--output-dir',default=None);p.add_argument('--resume',default=None);a=p.parse_args();cfg=load_config(a.config,a.override);ctx=init_distributed(require_cuda=True)
    formal_server_environment(ctx,cfg)
    if get(cfg,'loss.lambda_reference') is None or get(cfg,'training.epochs_locked') is None:raise RuntimeError('formal training requires profile-locked lambda/epochs')
    if get(cfg,'runtime.profile_formal_full_dev') is not True:raise RuntimeError('formal config must come from full-dev profile')
    if get(cfg,'runtime.profile_formal_server_gate') is not True:raise RuntimeError('formal config must come from --formal-server-gate profile')
    if int(get(cfg,'runtime.profile_world_size',-1))!=ctx.world_size:raise RuntimeError('training WORLD_SIZE differs from locked formal profile')
    seed=int(get(cfg,'training.initial_seed',3407));seed_all(seed);dr,info,manifest,msp=[get(cfg,x) for x in ('paths.dataroot','paths.info_pkl','paths.manifest','paths.msp_checkpoint')];payload=load_manifest(manifest);tr,dv=set(payload['train_scenes']),set(payload['dev_scenes']);prov=msp_checkpoint_provenance(msp);leak=sorted(set(prov['train_scene_names'])&dv)
    if tr&dv or leak:raise RuntimeError(f'data/MSP scene leakage: {leak[:8]}')
    src=NuScenesWindowSource(dr,info_pkl=info,verbose=False);train_ds=ManifestDataset(src,payload,'train');dev_ds=ManifestDataset(src,payload,'dev');pipe=MotionTransportV1(cfg,msp_checkpoint=msp,device=ctx.device);out=Path(a.output_dir or get(cfg,'paths.output_root','outputs/motion_transport_v1'))
    if ctx.is_main:
        out.mkdir(parents=True,exist_ok=True);cfg.setdefault('runtime',{})['code_commit']=_git();(out/'resolved_train_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False));(out/'data_source_manifest.json').write_text(json.dumps({'manifest':str(Path(manifest).resolve()),'manifest_sha256':sha256_file(manifest),'train_windows':len(train_ds),'dev_windows':len(dev_ds),'train_scenes':sorted(tr),'dev_scenes':sorted(dv),'msp':prov},indent=2))
    barrier(ctx);report=train(pipe,src,train_ds,dev_ds,cfg,ctx,manifest_path=manifest,msp_path=msp,output_dir=out,resume=a.resume)
    if ctx.is_main:(out/'training_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    barrier(ctx)
    if ctx.world_size>1:dist.destroy_process_group()
if __name__=='__main__':main()
