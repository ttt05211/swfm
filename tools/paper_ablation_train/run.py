"""Five new ablations: fixed one-epoch training only; no evaluation entry point."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import random
import signal
import sys
import threading
import numpy as np
import torch
import torch.distributed as dist
from real_motion.local_st_world_model_v17 import config_from_mapping_v17
from tools.joint_occ_experiments import run as base
from tools.source_scene_stage1.model import ReaderConfig
from tools.source_scene_stage2.history import DirectHistory
from tools.real_motion.joint_training_distributed import DistributedTraining
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from .model import ARMS,make_model
from .control import PROTOCOL,reference,check_inputs,plan,sha,canonical
from .data import TrainProvider,attach_train_cache,batches
from .engine import execute


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',required=True,choices=ARMS)
    p.add_argument('--out-dir',required=True)
    p.add_argument('--reference-contract',required=True,help='existing E contract JSON ONLY; no E checkpoint loaded')
    for name in ('config','train-cache','base-checkpoint','dataroot','train-info','ccr-history-cache'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--seed',type=int,default=20261011)
    p.add_argument('--motion-lr',type=float,default=5e-4)
    p.add_argument('--repair-lr',type=float,default=3e-4)
    p.add_argument('--override',action='append',default=[])
    p.add_argument('--cpu-workers',type=int,default=4)
    p.add_argument('--prefetch-workers',type=int,default=2)
    for name,value in (('frame',512),('geometry',128),('descriptor',128),('track',128)):
        p.add_argument('--'+name+'-ram-mib' if name!='frame' else '--frame-cache-mib',type=int,default=value)
    p.add_argument('--checkpoint-every',type=int,default=64,help='overwrite last.pt, keep ONE previous backup')
    p.add_argument('--max-updates',type=int,default=0,help='pause at this global cursor, unchanged schedule')
    p.add_argument('--resume',help='this SAME output directory last.pt (or its previous backup)')
    p.add_argument('--device',default='cuda:0')
    return p


def optimizer_for(model,a):
    groups=[]
    for name,part,lr,decay in (('transport',model.transport,a.motion_lr,1e-4),
                              ('refiner',model.columns,a.repair_lr,.01)):
        params=[p for p in part.parameters() if p.requires_grad]
        if params: groups.append(dict(name=name,params=params,lr=lr,initial_lr=lr,weight_decay=decay))
    return torch.optim.AdamW(groups)


def seed_all(seed):
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)


def launch(a,event,ctx):
    if a.max_updates<0 or a.checkpoint_every<1 or not 1<=a.cpu_workers<=20 or not 1<=a.prefetch_workers<=8:
        raise ValueError('invalid checkpoint/prefetch bounds')
    for name in ('frame_cache_mib','geometry_ram_mib','descriptor_ram_mib','track_ram_mib'):
        if not 0<=getattr(a,name)<=1024: raise ValueError('RAM cache must stay within 0..1024 MiB per rank')
    if ctx.device.type!='cuda' or not torch.cuda.is_bf16_supported(): raise RuntimeError('BF16 CUDA required')
    root=Path(__file__).resolve().parents[2];out=Path(a.out_dir).resolve()
    c=reference(a.reference_contract,a.arm);data=check_inputs(a,ctx,c,root)
    torch.set_num_threads(1);seed_all(a.seed)
    pcfg=base.make_prepare_config(base.load_runtime_config(a.config,a.override))
    if pcfg.frame_dt_s!=.5: raise ValueError('original .5s KTA convention required')
    _,all_train=base.load_cache(a.train_cache)
    keys,groups,budget=plan(all_train,c,a.arm)
    files=sorted(p for p in Path(__file__).parent.iterdir() if p.suffix in ('.py','.sh'))
    implementation={p.relative_to(root).as_posix():sha(p) for p in files}
    experiment=dict(reference_contract_sha256=sha(a.reference_contract),
        trained_weights_imported=False,training_only=True,implementation=implementation)
    ref=c['model']
    model=make_model(config_from_mapping_v17(ref['motion']),a.arm,expected_reference_spec=ref,
        feature_width=ref['shared_history']['width'],z_bins=ref['shared_history']['z_bins'],
        memory_width=ref['repair']['memory_width'],reader_config=ReaderConfig(**ref['reader']),
        experiment=experiment).to(ctx.device)
    if ctx.active:
        for value in model.state_dict().values(): dist.broadcast(value,src=0)
    seed_all(a.seed+ctx.rank);rng=np.random.default_rng(a.seed+ctx.rank)
    optimizer=optimizer_for(model,a)
    contract=canonical(dict(protocol=PROTOCOL,training_only=True,arm=a.arm,recipe=ARMS[a.arm],
        model=model.configs(),parameters=sum(p.numel() for p in model.parameters()),
        reference_contract_sha256=sha(a.reference_contract),world_size=ctx.world_size,
        train_keys=keys,eval_keys_metadata_only=c.get('eval_keys',[]),budget=budget,
        seed=a.seed,objective=c['objective'],known='TRAIN_dense_valid',registration='none',
        precision=c['precision'],data=data,overrides=a.override,learning_rates=c['learning_rates'],
        optimizer_groups=[{k:v for k,v in g.items() if k in ('name','initial_lr','weight_decay')} for g in optimizer.param_groups],
        implementation=implementation,execution=dict(c['execution'],frame_cache_mib=a.frame_cache_mib,
            geometry_ram_mib=a.geometry_ram_mib,descriptor_ram_mib=a.descriptor_ram_mib,track_ram_mib=a.track_ram_mib),
        inference_features=dict(age=False,FREE_bias=0.,STC_adaptation=False),evaluation='DISABLED'))
    if ctx.primary:
        if a.resume:
            rp=Path(a.resume).resolve()
            if rp.parent!=out or rp.name not in ('last.pt','last.previous.pt'):
                raise RuntimeError('resume only original output last.pt or last.previous.pt')
            if not (out/'contract.json').is_file() or json.loads((out/'contract.json').read_text())!=contract:
                raise RuntimeError('output contract differs')
        elif out.exists() and any(out.iterdir()): raise RuntimeError('NEW empty output directory required')
        out.mkdir(parents=True,exist_ok=True)
    ctx.barrier()
    with ExitStack() as stack:
        if ctx.primary:
            stack.enter_context(evaluation_lock(out))  # lease only, no evaluator is called.
            (out/'contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        ctx.barrier()
        provider=TrainProvider(a.base_checkpoint,base.CLEAN_SHA256,pcfg,ctx.device,a.cpu_workers,model,None)
        from real_motion.canonical_repair_execution import CanonicalCpuExecution
        from real_motion.surface_projection_execution import SurfaceMapExecution
        from real_motion.strong_majority_execution import ParallelNativeMajority
        provider.raw_io_workers=1
        provider.ccr_execution=SurfaceMapExecution(CanonicalCpuExecution('native_parallel',min(4,a.cpu_workers)))
        provider.strong_majority=ParallelNativeMajority(min(4,a.cpu_workers))
        stack.callback(provider.ccr_execution.close);stack.callback(provider.strong_majority.close)
        train_source=base.CachedColumnSource(base.NuScenesWindowSource(a.dataroot,info_pkl=a.train_info),
            a.frame_cache_mib,copy_on_insert=False)
        cache=attach_train_cache(provider,a,root,train_source)
        stack.callback(provider.ccr_history_cache.close);stack.callback(provider.close_tracks)
        history=DirectHistory(provider,ram_mib=a.descriptor_ram_mib)
        if ctx.primary:
            (out/'cache_policy.json').write_text(json.dumps(cache,indent=2)+'\n',encoding='utf-8')
            print(f'ABLATION_TRAIN_ONLY arm={a.arm}; RANDOM TRAIN{len(keys)}x1; '
                f'updates={budget["stop_updates"]}; schedule_denominator={budget["schedule_reference_updates"]}; '
                f'world={ctx.world_size}; EVAL=OFF; NEW={out}',flush=True)
        make_batches=lambda local:batches(provider,train_source,local,history,workers=a.prefetch_workers,depth=2)
        result=execute(model,optimizer,provider,groups,contract,ctx,rng,event,out,make_batches,
            resume=a.resume,max_updates=a.max_updates,checkpoint_every=a.checkpoint_every)
        if ctx.primary: print((out/'summary.txt').read_text(encoding='utf-8'),flush=True)
    return 0 if result['status']=='complete' else 130


def main(argv=None,event=None):
    a=parser().parse_args(argv);ctx=DistributedTraining(True)
    if not ctx.active:
        ctx.device=torch.device(a.device)
        if ctx.device.type=='cuda': torch.cuda.set_device(ctx.device)
    try: return launch(a,event or threading.Event(),ctx)
    finally: ctx.close()


if __name__=='__main__':
    event=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda *_:event.set())
    sys.exit(main(event=event))
