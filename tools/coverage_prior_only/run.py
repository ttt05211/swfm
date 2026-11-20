"""Matched RANDOM TRAIN20%x1: coverage/Prior unchanged, projection OFF."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import random
import signal
import threading
import numpy as np
import torch
import torch.distributed as dist
from real_motion.runtime_config import load_runtime_config,make_prepare_config
from real_motion.local_st_world_model_v17 import config_from_mapping_v17
from tools.source_scene_stage1.model import ReaderConfig
from tools.causal_forward_recovery.adapter import RecoveryPredictor
from tools.fpn_unet_eval.common import DenseInputs
from tools.final_semantic_evaluation.sources import window
from tools.coverage_projection.model import make_model
from tools.coverage_projection.run import ROOT,sha,write_json,make_plan,prefetch
from tools.real_motion.joint_training_distributed import DistributedTraining
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from .control import PROTOCOL,OBJECTIVE,reference,canonical
from .engine import execute


def check_groups(groups,joint):
    keys=canonical([[[r['scene_name'],r['t0_token']] for r in group] for group in groups])
    if keys != joint['group_keys']:raise ValueError('ordered global groups differ from the joint run')


def launch(a,stop,ctx):
    if ctx.world_size != 2 or ctx.device.type != 'cuda' or not torch.cuda.is_bf16_supported():
        raise ValueError('TWO BF16 training GPUs required; GLOBAL B4 stays unchanged')
    source=reference(a.joint_dir);c=source['contract'];original=source['original_contract']
    execution=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cpu_workers=a.cpu_workers)
    if execution != c['execution']:raise ValueError('torch/CUDA/CPU-worker execution differs from the joint control')
    data={name:sha(getattr(a,name)) for name in ('config','train_cache','train_info')}
    if data != c['train_data']:raise ValueError('TRAIN config/index/info hashes differ from the joint control')
    from tools.joint_occ_experiments import run as base
    _,records=base.load_cache(a.train_cache);groups=make_plan(records,original);check_groups(groups,c)
    torch.set_num_threads(1);torch.manual_seed(c['seed']);torch.cuda.manual_seed(c['seed'])
    random.seed(c['seed']);np.random.seed(c['seed'])
    spec=c['model']
    model=make_model(config_from_mapping_v17(spec['motion']),feature_width=spec['shared_history']['width'],
        z_bins=spec['shared_history']['z_bins'],memory_width=spec['repair']['memory_width'],
        reader_config=ReaderConfig(**spec['reader'])).to(ctx.device)
    if model.configs() != spec:raise ValueError('prior-only architecture differs from joint; no trained values imported')
    for value in model.state_dict().values():dist.broadcast(value,src=0)
    rank_seed=c['seed']+ctx.rank
    torch.manual_seed(rank_seed);torch.cuda.manual_seed(rank_seed);random.seed(rank_seed);np.random.seed(rank_seed)
    rng=np.random.default_rng(rank_seed)
    optimizer=torch.optim.AdamW([
        dict(params=model.transport.parameters(),lr=5e-4,initial_lr=5e-4,weight_decay=1e-4),
        dict(params=model.columns.parameters(),lr=3e-4,initial_lr=3e-4,weight_decay=.01)])
    overrides=original.get('overrides',[])
    pcfg=make_prepare_config(load_runtime_config(a.config,overrides))
    if pcfg.frame_dt_s != .5:raise ValueError('original KTA/horizon convention required')
    predictor=RecoveryPredictor(model,pcfg,ctx.device,workers=a.cpu_workers)
    try:
        inputs=DenseInputs(a.dataroot,a.train_info,cache_mib=128)
        inventory=inputs.preflight([window(r) for group in groups for r in group])
        if inventory != c['TRAIN_inventory']:raise ValueError('raw TRAIN file inventory differs from joint')
        implementation=dict(c['implementations'])
        helper=ROOT/'tools/coverage_projection/eval_inputs.py'
        implementation[helper.relative_to(ROOT).as_posix()]=sha(helper)
        implementation.update({p.relative_to(ROOT).as_posix():sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
        contract=canonical(dict(protocol=PROTOCOL,model=model.configs(),parameters=sum(p.numel() for p in model.parameters()),
            joint_dir=str(source['folder']),joint_checkpoint_sha256=source['checkpoint_sha256'],
            joint_contract_sha256=source['contract_sha256'],joint_evaluation_sha256=source['evaluation_sha256'],
            control_dir=c['control_dir'],control_sha256=c['control_sha256'],control_results_sha256=c['control_results_sha256'],
            train_keys=c['train_keys'],eval_keys=c['eval_keys'],group_keys=c['group_keys'],total_updates=1031,epochs=1,
            seed=c['seed'],objective=OBJECTIVE,learning_rates=c['learning_rates'],schedule=c['schedule'],
            known=c['known'],source_policy=c['source_policy'],registration='none',world_size=2,joint_world_size=c['world_size'],
            train_data=data,TRAIN_inventory=inventory,overrides=overrides,execution=execution,
            distributed=c['distributed'],implementations=implementation,raw_live_geometry=True,tensor_disk_cache=False,
            initialization='RANDOM same fresh factory and seed; NO joint/original trained weights or optimizer imported',
            scientific_change='projection supervision OFF, never computed; coverage/model/readout unchanged'))
        out=a.out_dir.resolve()
        if ctx.primary:
            if a.resume:
                if not (out/'contract.json').is_file() or json.loads((out/'contract.json').read_text()) != contract:
                    raise ValueError('resume needs original prior-only directory and SAME contract')
            elif out.exists():raise ValueError('NEW output directory required; previous experiments are read only')
            else:out.mkdir(parents=True)
            write_json(out/'contract.json',contract)
        ctx.barrier()
        result=execute(model,optimizer,predictor,groups,contract,ctx,rng,stop,out,
            lambda values:prefetch(values,inputs,predictor),resume=a.resume)
        if result['status'] != 'complete':raise InterruptedError('saved complete-update cursor; resume same prior-only run')
        if ctx.primary:print('PRIOR_ONLY TRAIN complete; next ONE-GPU SAME64 evaluation and fixed visual comparison.',flush=True)
    finally:predictor.close()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('joint-dir','out-dir','dataroot','train-info','train-cache'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/real_motion_occfm.yaml')
    p.add_argument('--cpu-workers',type=int,default=4);p.add_argument('--resume',action='store_true')
    return p


def main():
    a=parser().parse_args();stop=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())
    ctx=DistributedTraining(True)
    try:
        if ctx.world_size != 2:raise ValueError('use workflow.sh / torchrun with TWO GPUs')
        if ctx.device.type == 'cuda':dist.barrier(device_ids=[ctx.local_rank])
        with ExitStack() as stack:
            if ctx.primary:
                lock=a.out_dir.parent/'.prior_only_locks'/hashlib.sha256(str(a.out_dir.resolve()).encode()).hexdigest()
                lock.mkdir(parents=True,exist_ok=True);stack.enter_context(evaluation_lock(lock))
            ctx.barrier();launch(a,stop,ctx)
    finally:ctx.close()


if __name__=='__main__':main()
