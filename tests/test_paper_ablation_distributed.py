"""Actual two CPU/Gloo ranks, including an empty rank, vs global-window training."""
import copy
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tools.paper_ablation_train import engine
from tools.real_motion.joint_training_distributed import DistributedTraining


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.used=torch.nn.Linear(2,18);self.unused=torch.nn.Linear(2,1)


def forward(model,provider,row):
    value=model.used(row[1]['x'])
    return dict(logits=value.T[None].expand(6,-1,-1))


def groups():
    row=lambda x:({},dict(x=torch.tensor([[x,1.]]),future_gt_occ=np.full((6,1),17)))
    return [[row(1.)],[row(2.),row(3.),row(4.)]]


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
            engine.train_step(body,opt,SimpleNamespace(device=ctx.device),group[rank::2],ctx,
                global_windows=len(group),update=update,schedule_updates=20)
        for k,v in body.unused.state_dict().items(): torch.testing.assert_close(v,unused[k],rtol=0,atol=0)
        torch.save(dict(model=body.state_dict(),optimizer=opt.state_dict()),Path(directory)/f'rank{rank}.pt')
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
        # Two Adam steps only for observed parameters; unused stays absent.
        assert len(saved['optimizer']['state'])==2
        assert all(int(value['step'])==2 for value in saved['optimizer']['state'].values())
