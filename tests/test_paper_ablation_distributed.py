"""Actual two CPU/Gloo ranks, including an empty rank, vs global-window training."""
import copy
import os
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import threading
from tools.paper_ablation_train import engine
from tools.real_motion.joint_training_distributed import DistributedTraining
from tools.paper_ablation_train.partition import partition


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.used=torch.nn.Linear(2,18);self.unused=torch.nn.Linear(2,1)


def forward(model,provider,row):
    value=model.used(row[1]['x'])
    return dict(logits=value.T[None].expand(6,-1,-1))


def groups():
    row=lambda x:({'features':[0]*int(x)},dict(x=torch.tensor([[x,1.]]),future_gt_occ=np.full((6,1),17)))
    return [[row(1.)],[row(2.),row(3.),row(4.)],
            [row(1.),row(5.),row(3.),row(7.)]]


def rank_worker(rank,directory):
    torch.set_num_threads(1)
    os.environ.update(WORLD_SIZE='2',RANK=str(rank),LOCAL_RANK=str(rank),USE_LIBUV='0')
    ctx=DistributedTraining(True,backend='gloo',init_method=(Path(directory)/'rendezvous').resolve().as_uri())
    engine.forward_window=forward
    try:
        torch.manual_seed(21);body=Tiny();unused=copy.deepcopy(body.unused.state_dict())
        opt=torch.optim.AdamW(body.parameters(),lr=.01,weight_decay=.1)
        opt.param_groups[0]['initial_lr']=.01
        for update,group in enumerate(groups(),1):
            engine.train_step(body,opt,SimpleNamespace(device=ctx.device),partition(group,2,update-1)[rank],ctx,
                global_windows=len(group),update=update,schedule_updates=20)
        for k,v in body.unused.state_dict().items(): torch.testing.assert_close(v,unused[k],rtol=0,atol=0)
        torch.save(dict(model=body.state_dict(),optimizer=opt.state_dict()),Path(directory)/f'rank{rank}.pt')
        # Exercise the ACTUAL distributed loop, rank timing collectives and
        # per-rank RNG restore across the alternating balanced partition.
        initial=copy.deepcopy(body.state_dict())
        contract=dict(arm='m0_no_memory',parameters=sum(p.numel() for p in body.parameters()),world_size=2,
            budget=dict(run_epochs=1,train_windows=8,stop_updates=3,schedule_reference_updates=3,
                schedule='whole_1_epoch_cosine_floor0.01'))
        for mode in ('whole','resumed'):
            out=Path(directory)/mode
            if ctx.primary: out.mkdir()
            ctx.barrier();body.load_state_dict(initial)
            torch.manual_seed(33+rank);random.seed(33+rank);np.random.seed(33+rank)
            rng=np.random.default_rng(33+rank)
            optimizer=torch.optim.AdamW(body.parameters(),lr=.01,weight_decay=.1)
            optimizer.param_groups[0]['initial_lr']=.01
            def step(*a,**kw):
                rng.random();random.random();torch.rand(1);np.random.rand()
                return engine.train_step(*a,**kw)
            kwargs=dict(model=body,optimizer=optimizer,provider=SimpleNamespace(device=ctx.device),groups=groups(),
                contract=contract,ctx=ctx,rng=rng,event=threading.Event(),out=out,
                make_batches=lambda values:(v for v in values),step=step)
            result=engine.execute(**kwargs,max_updates=1 if mode=='resumed' else 0)
            if mode=='resumed':
                assert result['status']=='interrupted'
                result=engine.execute(**kwargs,resume=out/'last.pt')
            assert result['status']=='complete'
        if ctx.primary:
            whole=torch.load(Path(directory)/'whole/epoch_0001.pt',weights_only=False)
            resumed=torch.load(Path(directory)/'resumed/epoch_0001.pt',weights_only=False)
            for k,v in whole['models']['gt'].items(): torch.testing.assert_close(resumed['models']['gt'][k],v,rtol=0,atol=0)
            for key,value in whole['optimizers']['gt']['state'].items():
                for name,v in value.items(): torch.testing.assert_close(resumed['optimizers']['gt']['state'][key][name],v,rtol=0,atol=0)
            for left,right in zip(whole['rank_rng'],resumed['rank_rng']):
                torch.testing.assert_close(left['torch'],right['torch'],rtol=0,atol=0)
                assert left['sampling']==right['sampling'] and left['python']==right['python']
                np.testing.assert_array_equal(left['numpy'][1],right['numpy'][1])
            log=[__import__('json').loads(line) for line in (Path(directory)/'resumed/progress.jsonl').read_text().splitlines()]
            assert len(log)==3 and log[-1]['rank_windows']==[2,2] and len(log[-1]['rank_timing'])==2
            assert sum(log[-1]['rank_sources'])==16
    finally: ctx.close()


def test_two_ranks_empty_local_batch_and_global_denominator(tmp_path,monkeypatch):
    if not dist.is_available() or not dist.is_gloo_available(): pytest.skip('CPU Gloo unavailable')
    monkeypatch.setenv('USE_LIBUV','0')
    mp.spawn(rank_worker,args=(str(tmp_path),),nprocs=2,join=True)
    monkeypatch.setattr(engine,'forward_window',forward)
    torch.manual_seed(21);body=Tiny();ctx=DistributedTraining()
    opt=torch.optim.AdamW(body.parameters(),lr=.01,weight_decay=.1);opt.param_groups[0]['initial_lr']=.01
    for update,group in enumerate(groups(),1):
        engine.train_step(body,opt,SimpleNamespace(device=ctx.device),group,ctx,
            global_windows=len(group),update=update,schedule_updates=20)
    for rank in range(2):
        saved=torch.load(tmp_path/f'rank{rank}.pt',weights_only=False)
        for k,v in body.state_dict().items(): torch.testing.assert_close(saved['model'][k],v,rtol=1e-6,atol=1e-7)
        # Three Adam steps only for observed parameters; unused stays absent.
        assert len(saved['optimizer']['state'])==2
        assert all(int(value['step'])==3 for value in saved['optimizer']['state'].values())
