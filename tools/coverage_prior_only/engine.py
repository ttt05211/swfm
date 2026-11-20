"""Full6 CE+Lovasz ONLY; same global mean, clipping, Adam and resume semantics."""
import json
import time
import numpy as np
import torch
from tools.coverage_projection.adapter import forward_window
from tools.coverage_projection.run import local_plan,write_json
from tools.shared_history_experiments.full_gt.engine import semantic_loss
from tools.real_motion.joint_training_distributed import DistributedTraining
from tools.real_motion.joint_training_recovery import save_resume_checkpoint
from .control import PROTOCOL,OBJECTIVE


def train_step(model,optimizer,predictor,rows,*,update,total,context=None,global_windows=None,window_slots=None):
    ctx=context or DistributedTraining(False)
    if context is None: ctx.device=predictor.provider.device
    count=len(rows) if global_windows is None else global_windows
    slots=list(range(len(rows))) if window_slots is None else window_slots
    if (not 1 <= count <= 4 or len(slots) != len(rows) or len(set(slots)) != len(slots)
            or any(not 0 <= s < count for s in slots) or ctx.active and window_slots is None):
        raise ValueError('same GLOBAL B1..B4 whole-window shards required')
    model.train();optimizer.zero_grad(set_to_none=True)
    factor=.01+.99*.5*(1+np.cos(np.pi*(update-1)/total))
    for group in optimizer.param_groups: group['lr']=group['initial_lr']*factor
    sums=torch.zeros(3,device=predictor.provider.device)
    for row,poses,target in rows:
        fw=forward_window(model,predictor.provider,row,poses=poses)
        losses,ce,lovasz=semantic_loss(fw['logits'],target)
        objective=losses.mean()  # Projection is NOT constructed, rendered, or evaluated.
        if not bool(torch.isfinite(objective)): raise RuntimeError('nonfinite full6 CE+Lovasz')
        (objective/count).backward()
        sums+=torch.stack((ce.mean().detach(),lovasz.mean().detach(),objective.detach()))/count
        del fw,losses,ce,lovasz,objective
    if ctx.stop(not bool(torch.isfinite(sums).all())): raise RuntimeError('nonfinite global semantic objective')
    ctx.synchronize_gradients(model)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
    optimizer.step();optimizer.zero_grad(set_to_none=True);sums=ctx.sum(sums)
    if predictor.provider.device.type == 'cuda': torch.cuda.synchronize(predictor.provider.device)
    return dict(zip(('CE','Lovasz','loss'),sums.tolist()),grad_norm=float(norm))


def validate_resume(saved,contract):
    if saved.get('protocol') != PROTOCOL or saved.get('contract') != contract or contract.get('objective') != OBJECTIVE:
        raise ValueError('resume SAME prior-only run; joint weights/Adam must never be imported')
    n=saved.get('updates')
    if type(n) is not int or not 0 <= n <= contract['total_updates'] or len(saved['rank_rng']) != contract['world_size']:
        raise ValueError('invalid completed-update/RNG cursor')
    return n


def execute(model,optimizer,predictor,groups,contract,ctx,rng,stop,out,make_batches,*,resume=False,max_updates=0):
    out=__import__('pathlib').Path(out);total=len(groups);update=0
    if total != contract['total_updates']: raise ValueError('global update budget differs')
    if resume:
        saved=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        update=validate_resume(saved,contract);model.load_state_dict(saved['models']['gt'],strict=True)
        optimizer.load_state_dict(saved['optimizer']);ctx.restore_rng(saved['rank_rng'],rng);del saved
    local,slots=local_plan(groups,ctx.world_size,ctx.rank)
    def save():
        states=ctx.gather(ctx.rng_state(rng))
        if ctx.primary:
            payload=dict(protocol=PROTOCOL,contract=contract,updates=update,epoch=int(update==total),
                batch=0 if update==total else update,models={'gt':model.state_dict()},optimizer=optimizer.state_dict(),rank_rng=states)
            save_resume_checkpoint(out/'last.pt',payload)
            if update==total:save_resume_checkpoint(out/'epoch_0001.pt',payload)
            write_json(out/'progress.json',dict(status='complete' if update==total else 'checkpointed',updates=update,
                total_updates=total,world_size=ctx.world_size,trained_windows=sum(map(len,groups[:update])),updated_utc=time.time()))
        ctx.barrier()
    if update==total:
        missing=ctx.broadcast(not (out/'epoch_0001.pt').is_file() if ctx.primary else None)
        if missing:save()
        return dict(status='complete',updates=update)
    if not resume:save()
    if ctx.primary:print(f'PRIOR_ONLY RANDOM {update}/{total}; GPUs={ctx.world_size}; GLOBAL B4; projection=OFF; OUT={out}',flush=True)
    with make_batches(local[update:]) as iterator:
        tick=time.perf_counter()
        for rows in iterator:
            if ctx.stop(stop.is_set()) or max_updates and update>=max_updates:break
            count=len(groups[update])
            stats=train_step(model,optimizer,predictor,rows,update=update+1,total=total,context=ctx,
                global_windows=count,window_slots=slots[update])
            ranks=ctx.gather(dict(seconds=time.perf_counter()-tick,windows=len(rows),
                sources=sum(len(row[0]['source_class_id']) for row,_,_ in rows)))
            update+=1;wall=max(v['seconds'] for v in ranks)/count
            if ctx.primary:
                item=dict(update=update,windows=count,seconds_per_window=wall,rank_windows=[r['windows'] for r in ranks],
                    rank_sources=[r['sources'] for r in ranks],**stats)
                with (out/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(item)+'\n')
                write_json(out/'progress.json',dict(status='training',updates=update,total_updates=total,
                    trained_windows=sum(map(len,groups[:update])),world_size=ctx.world_size,seconds_per_window=wall,
                    rank_windows=item['rank_windows'],updated_utc=time.time()))
                if update==1 or update%16==0:
                    print(f'PRIOR_ONLY {update}/{total} CE={stats["CE"]:.4f} Lovasz={stats["Lovasz"]:.4f} '
                        f'projection=OFF s/window={wall:.4f} rank_windows={item["rank_windows"]}',flush=True)
            stopping=ctx.stop(stop.is_set())
            if update%64==0 or stopping:save()
            tick=time.perf_counter()
    save()
    return dict(status='complete' if update==total else 'stopped',updates=update)
