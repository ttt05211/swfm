#!/usr/bin/env python3
"""TRAIN-only ego readout screen. Existing Surface CCR / V18 remain frozen."""
import sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
from dataclasses import asdict
import json
import math
import random
import signal
import threading
import time

import numpy as np
import torch

from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from real_motion.ego_trajectory_head import EgoHeadConfig, HistoryEgoTrajectoryHead, ego_supervision, PROTOCOL
from real_motion.ego_navigation import relative_se2, navigation_commands, COMMAND_PROTOCOL,validate_window_identity
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.v21_source_induction import select_scene_balanced_round_robin
from real_motion.runtime_config import load_runtime_config, make_prepare_config
from tools.real_motion.ego_trajectory_common import (extract_history_features,stack_features,
    atomic_save,digest_file,fingerprint,save_head,restore_head,implementation_fingerprint,row_fingerprint)
from tools.real_motion.joint_surface_checkpoint_selection import load_evaluation_model, AVERAGE_NAME, AVERAGE_EPOCHS
from tools.real_motion.eval_p0_f9_joint_surface_mean_full import find_frozen_bundle
from tools.real_motion.stc_shared_execution import Predictor
from tools.real_motion.waymo_zero_shot_common import write_json
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock

ROOT=Path(__file__).resolve().parents[2]
IMPLEMENTATION=('real_motion/ego_trajectory_head.py','real_motion/ego_navigation.py',
    'tools/real_motion/ego_trajectory_common.py','tools/real_motion/train_p0_f9_surface_ego_head.py')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataroot',required=True);p.add_argument('--train-cache',required=True)
    p.add_argument('--train-info',required=True);p.add_argument('--out-dir',required=True)
    p.add_argument('--checkpoint');p.add_argument('--runs-root');p.add_argument('--run-dir');p.add_argument('--source-bundle-dir')
    p.add_argument('--config',default=str(ROOT/'configs/real_motion_occfm.yaml'))
    p.add_argument('--train-windows',type=int,default=1024);p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=64);p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--min-lr',type=float,default=3e-6);p.add_argument('--yaw-weight',type=float,default=1.)
    p.add_argument('--seed',type=int,default=21);p.add_argument('--cpu-workers',type=int,default=4)
    p.add_argument('--resume',action='store_true');p.add_argument('--checkpoint-every',type=int,default=32)
    p.add_argument('--device',default='cuda');return p


def resolve_checkpoint(a):
    if a.checkpoint:return Path(a.checkpoint).resolve()
    if not a.runs_root or not a.run_dir:raise ValueError('--checkpoint or --runs-root and --run-dir required')
    return Path(find_frozen_bundle(a.runs_root,a.run_dir,a.source_bundle_dir)['candidates'][AVERAGE_NAME]['path']).resolve()


def train_cached(head,rows,optimizer,contract,out,*,epochs,batch_size,lr,min_lr,seed,
                 checkpoint_every=32,resume=False,stop_event=None):
    """Exact cursor/order/RNG/Adam recovery; only head weights are updated."""
    out=Path(out);generator=np.random.default_rng(seed);device=next(head.parameters()).device
    epoch=cursor=updates=0;order=[];last=out/'head_last.pt'
    if resume and last.is_file():
        s=restore_head(last,head,optimizer,contract,generator)
        epoch,cursor,updates,order=s['epoch'],s['cursor'],s['updates'],s['order']
        if not (0<=epoch<=epochs and 0<=cursor<=len(rows)) or (order and sorted(order)!=list(range(len(rows)))):
            raise RuntimeError('invalid ego training cursor/permutation')
    elif resume and epoch:raise RuntimeError('missing head recovery')
    steps=math.ceil(len(rows)/batch_size)*epochs;head.train()
    def persist():save_head(last,head,optimizer,contract=contract,epoch=epoch,cursor=cursor,
        order=order,updates=updates,generator=generator)
    persist()
    while epoch<epochs:
        if not order:order=generator.permutation(len(rows)).tolist();cursor=0
        while cursor<len(rows):
            if stop_event is not None and stop_event.is_set():persist();return dict(status='stopped',updates=updates,epochs=epoch)
            ids=order[cursor:cursor+batch_size];chosen=[rows[i] for i in ids]
            rate=min_lr+(lr-min_lr)*.5*(1+math.cos(math.pi*updates/max(1,steps-1)))
            for group in optimizer.param_groups:group['lr']=rate
            bank=stack_features([r['features'] for r in chosen],device)
            commands=np.stack([r['commands'] for r in chosen]);target=np.stack([r['target'] for r in chosen])
            optimizer.zero_grad(set_to_none=True)
            loss,stats=ego_supervision(head(bank,commands),target,yaw_weight=head.config.yaw_weight)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite ego training loss; no automatic retry')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True)
            optimizer.step();cursor+=len(ids);updates+=1
            if updates%checkpoint_every==0:persist()
            if updates%16==0:
                line=dict(event='ego_train',epoch=epoch+1,update=updates,lr=rate,loss=float(loss.detach()),
                          xy=float(stats['xy']),yaw=float(stats['yaw']),grad_norm=float(grad))
                with (out/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(line)+'\n')
                print(json.dumps(line),flush=True)
        epoch+=1;cursor=0;order=[];persist()
    return dict(status='complete',updates=updates,epochs=epoch)


def main(stop_event=None,argv=None):
    a=parser().parse_args(argv);out=Path(a.out_dir).resolve()
    if (min(a.train_windows,a.epochs,a.batch_size,a.checkpoint_every)<1 or not 1<=a.cpu_workers<=8
            or not 0<a.min_lr<=a.lr or a.yaw_weight<=0):raise ValueError('invalid bounded screen config')
    for path in (a.dataroot,a.train_cache,a.train_info):
        p=Path(path).resolve()
        if out==p or out.is_relative_to(p):raise ValueError('new separate experiment output required')
    if out.exists() and not a.resume:raise FileExistsError('use a NEW output or explicit --resume SAME directory')
    if a.resume and not (out/'training.json').is_file():raise ValueError('missing SAME-directory training contract')
    if a.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(1);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    ckpt=resolve_checkpoint(a);frozen_sha=digest_file(ckpt)
    meta,joint=load_evaluation_model(ckpt,device=a.device)
    if meta.get('source_epochs')!=list(AVERAGE_EPOCHS) or not meta.get('averaging'):
        raise ValueError('frozen mean5/6/8/12/14 required; no GT model retraining')
    pcfg=make_prepare_config(load_runtime_config(a.config))
    hc=EgoHeadConfig(object_dim=joint.transport.config.d_model,surface_dim=joint.columns.width,yaw_weight=a.yaw_weight)
    header,records=load_cache(a.train_cache)
    identities={}
    for r in records:
        k=(str(r['scene_name']),str(r['t0_token']))
        if k in identities:raise ValueError('duplicate TRAIN window')
        identities[k]=dict(scene_name=k[0],t0_token=k[1],history_tokens=tuple(r['history_tokens'][-4:]),
                           future_tokens=tuple(r['future_tokens']))
    del records,header
    keys=select_scene_balanced_round_robin(sorted(identities),a.train_windows)
    if len(keys)!=a.train_windows:raise ValueError('requested TRAIN population exceeds available records')
    source=NuScenesWindowSource(a.dataroot,info_pkl=a.train_info)
    if any(scene not in source.allowed_scenes for scene,_ in keys):raise ValueError('non-TRAIN scene in bank')
    for k in keys:validate_window_identity(source.nusc,identities[k])
    # Pin metadata and historical source NPZs; do not open/decompress future labels.
    paths=sorted({source._label_path(k[0],t) for k in keys for t in identities[k]['history_tokens']})
    data_stats=[[str(p),p.stat().st_size,p.stat().st_mtime_ns] for p in paths]
    metadata_dir=Path(a.dataroot)/'v1.0-trainval'
    metadata_hashes={p.name:digest_file(p) for p in sorted(metadata_dir.glob('*.json'))}
    contract=dict(protocol=PROTOCOL,frozen_checkpoint=str(ckpt),frozen_sha256=frozen_sha,head_config=asdict(hc),
        train_cache_sha256=digest_file(a.train_cache),train_info_sha256=digest_file(a.train_info),
        runtime_config_sha256=digest_file(a.config),keys=[list(k) for k in keys],metadata_hashes=metadata_hashes,
        historical_files_fingerprint=fingerprint(data_stats),commands=COMMAND_PROTOCOL,
        schedule=dict(epochs=a.epochs,batch_size=a.batch_size,lr=a.lr,min_lr=a.min_lr,seed=a.seed),
        feature_inputs='history_only; objects+spatial_surface+ego_history; no future pose/labels/CANbus',
        command_condition='GT-derived destination-row navigation; NOT navigation-free Pred',
        implementation={p:digest_file(ROOT/p) for p in IMPLEMENTATION},
        feature_geometry_implementation=implementation_fingerprint(ROOT),
        feature_execution=dict(device=str(torch.device(a.device)),torch_version=str(torch.__version__),
            cuda_version=torch.version.cuda,WM_precision='bf16' if a.device.startswith('cuda') else 'fp32'),
        runtime_environment={k:v for k,v in sorted(__import__('os').environ.items()) if k.startswith('SWFM_')})
    if a.resume:
        if json.loads((out/'training.json').read_text(encoding='utf-8'))!=contract:raise RuntimeError('training contract changed')
    else:out.mkdir(parents=True);write_json(out/'training.json',contract)
    with evaluation_lock(out):
        return finish_training(a,out,keys,identities,source,joint,pcfg,hc,contract,ckpt,frozen_sha,stop_event)


def finish_training(a,out,keys,identities,source,joint,pcfg,hc,contract,ckpt,frozen_sha,stop_event):
    bankdir=out/'bank';bankdir.mkdir(exist_ok=True);tick=time.perf_counter()
    predictor=Predictor(joint,pcfg,a.device,workers=a.cpu_workers,geometry_mib=512,graphs=False)
    try:
        for idx,k in enumerate(keys):
            if stop_event is not None and stop_event.is_set():return 130
            path=bankdir/f'{idx:06d}.pt'
            if path.exists():
                saved=torch.load(path,map_location='cpu',weights_only=False)
                if (saved['key']!=list(k) or saved['contract_fingerprint']!=fingerprint(contract)
                        or saved.get('content_fingerprint')!=row_fingerprint(saved)):raise RuntimeError('feature shard mismatch')
                continue
            rec=identities[k];loaded=[source.load_occ3d(k[0],t) for t in rec['history_tokens']]
            poses=[source.pose(t) for t in rec['history_tokens']]
            times=[source.nusc.get('sample',t)['timestamp']/1e6 for t in rec['history_tokens']]
            raw=dict(history_occ=np.stack([x for x,_ in loaded]),history_observed=np.stack([x for _,x in loaded]),history_poses=poses)
            features=extract_history_features(predictor.provider,rec,raw,hc,timestamps_s=times)
            # Continuous future poses are supervision only, AFTER extraction.
            labels=relative_se2(poses[-1],[source.pose(t) for t in rec['future_tokens']])
            cmds=navigation_commands(source.nusc,rec['future_tokens'])
            row=dict(features=features,target=labels,commands=cmds,key=list(k),contract_fingerprint=fingerprint(contract))
            row['content_fingerprint']=row_fingerprint(row);atomic_save(path,row)
            if (idx+1)%16==0:print(f'EGO_BANK {idx+1}/{len(keys)} seconds={time.perf_counter()-tick:.1f}',flush=True)
    finally:predictor.close()
    del joint,predictor,source
    if torch.cuda.is_available():torch.cuda.empty_cache()
    rows=[torch.load(bankdir/f'{i:06d}.pt',map_location='cpu',weights_only=False) for i in range(len(keys))]
    head=HistoryEgoTrajectoryHead(hc).to(a.device);optimizer=torch.optim.AdamW(head.parameters(),lr=a.lr,weight_decay=.01)
    result=train_cached(head,rows,optimizer,contract,out,epochs=a.epochs,batch_size=a.batch_size,lr=a.lr,min_lr=a.min_lr,
        seed=a.seed,checkpoint_every=a.checkpoint_every,resume=a.resume,stop_event=stop_event)
    result.update(protocol=PROTOCOL,train_windows=len(keys),parameters=sum(p.numel() for p in head.parameters()),
        frozen_checkpoint_sha256=frozen_sha,head_checkpoint=str(out/'head_last.pt'),seconds=time.perf_counter()-tick,
        selection='final epoch only; NO dev-best selection',old_WM_CCR_updated=False)
    if digest_file(ckpt)!=frozen_sha:raise RuntimeError('frozen source checkpoint changed')
    write_json(out/'summary.json',result)
    (out/'summary.txt').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return 0 if result['status']=='complete' else 130


if __name__=='__main__':
    event=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:event.set())
    sys.exit(main(event) or 0)
