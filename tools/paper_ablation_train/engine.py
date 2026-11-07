"""One fixed epoch, global-window losses and complete-update recovery; NO eval."""
import json
import math
from pathlib import Path
import time
import torch
from tools.shared_history_experiments.full_gt.engine import semantic_loss
from tools.real_motion.joint_training_recovery import save_resume_checkpoint
from .control import PROTOCOL,validate_resume
from .geometry import forward_window
from .partition import partition,source_count


def train_step(model,optimizer,provider,rows,ctx,*,global_windows,update,schedule_updates):
    if not 1<=global_windows<=4 or not 1<=update<=schedule_updates: raise ValueError('invalid batch/schedule')
    model.train();optimizer.zero_grad(set_to_none=True)
    factor=.01+.99*.5*(1+math.cos(math.pi*(update-1)/schedule_updates))
    for group in optimizer.param_groups: group['lr']=group['initial_lr']*factor
    begun=time.perf_counter();totals=torch.zeros(3,device=provider.device,dtype=torch.float64)
    for row in rows:
        value=forward_window(model,provider,row)
        loss,ce,lovasz=semantic_loss(value['logits'],row[1]['future_gt_occ'])
        objective=loss.mean()/global_windows
        # C can have no t0 sources: its fixed background then has NO learnable
        # path. Do not attach a fake zero gradient that would trigger AdamW.
        if objective.requires_grad: objective.backward()
        totals+=torch.stack((ce.detach().mean(),lovasz.detach().mean(),loss.detach().mean())).double()/global_windows
        del value,loss,ce,lovasz,objective
    local_forward_backward_seconds=time.perf_counter()-begun
    if ctx.stop(not bool(torch.isfinite(totals).all())): raise RuntimeError('nonfinite full6 objective')
    used=ctx.synchronize_gradients(model)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
    if used: optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    totals=ctx.sum(totals).cpu().tolist()
    if provider.device.type=='cuda': torch.cuda.synchronize(provider.device)
    return dict(CE=totals[0],Lovasz=totals[1],loss=totals[2],grad_norm=float(norm),
        local_windows=len(rows),local_forward_backward_seconds=local_forward_backward_seconds,
        optimizer_updated=used,seconds=time.perf_counter()-begun)


def summary(result):
    c=result['contract'];b=c['budget']
    return ('===== CARRYWORLD ABLATION / TRAIN ONLY =====\n'
        f'status={result["status"]}; arm={c["arm"]}; updates={result["updates"]}/{b["stop_updates"]}; epoch={result["epoch"]}\n'
        f'RANDOM TRAIN{b["train_windows"]} x1; parameters={c["parameters"]}; GPUs={c["world_size"]}; globalB4/source128.\n'
        f'Schedule={b["schedule"]}; denominator={b["schedule_reference_updates"]}; stop after ONE epoch.\n'
        'Only full6 CE+Lovasz; GT ego Decoder/geometry; TRAIN dense-valid; no ICP/age/FREE/calibration/extra loss.\n'
        'Reference E contract supplies metadata/fresh initializer only, NO trained weights imported.\n'
        'NO VAL/STC samples, labels, quality scores or evaluation calls. No model selection.\n'
        'Read-only existing TRAIN geometry; bounded RAM history; no new dense tensor disk cache.\n')


def execute(model,optimizer,provider,groups,contract,ctx,rng,event,out,make_batches,*,
            resume=None,max_updates=0,checkpoint_every=64,step=train_step):
    """make_batches receives local rows; max_updates pauses without LR changes."""
    out=Path(out);epoch=batch=updates=0
    if resume is not None:
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        epoch,batch,updates=validate_resume(saved,contract,groups)
        if epoch!=1 and (out/'epoch_0001.pt').exists():
            raise RuntimeError('cannot rewind a run with an already published final checkpoint')
        model.load_state_dict(saved['models']['gt'],strict=True);optimizer.load_state_dict(saved['optimizers']['gt'])
        ctx.restore_rng(saved['rank_rng'],rng);del saved
    result=dict(status='running',training_only=True,contract=contract,epoch=epoch,updates=updates)

    def publish():
        if ctx.primary:
            (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            (out/'summary.txt').write_text(summary(result),encoding='utf-8')

    def save(final=False):
        states=ctx.gather(ctx.rng_state(rng))
        if ctx.primary:
            payload=dict(protocol=PROTOCOL,training_only=True,contract=contract,epoch=epoch,batch=batch,
                updates=updates,models={'gt':model.state_dict()},optimizers={'gt':optimizer.state_dict()},rank_rng=states)
            save_resume_checkpoint(out/'last.pt',payload)
            if final and not (out/'epoch_0001.pt').exists(): save_resume_checkpoint(out/'epoch_0001.pt',payload)
            publish()
        ctx.barrier()

    if epoch==1:
        result['status']='complete';save(final=True);return result
    if resume is None: save()  # recover even before the first 64-update interval.
    iterator=make_batches([partition(group,ctx.world_size,i)[ctx.rank] for i,group in enumerate(groups[batch:],batch)])
    last_saved=None;begun=time.perf_counter();session_windows=0;fetch_started=begun
    try:
        for rows in iterator:
            input_wait_seconds=time.perf_counter()-fetch_started
            if ctx.stop(event.is_set() or max_updates>0 and updates>=max_updates): break
            shards=partition(groups[batch],ctx.world_size,batch);global_windows=len(groups[batch]);tick=time.perf_counter()
            stats=step(model,optimizer,provider,rows,ctx,global_windows=global_windows,
                update=updates+1,schedule_updates=contract['budget']['schedule_reference_updates'])
            updates+=1;batch+=1;session_windows+=global_windows
            if batch==len(groups): epoch,batch=1,0
            result.update(epoch=epoch,updates=updates)
            report_step=updates==1 or updates%16==0 or epoch==1
            rank_timing=ctx.gather(dict(input_wait_seconds=input_wait_seconds,
                forward_backward_seconds=stats.get('local_forward_backward_seconds'))) if report_step else None
            if ctx.primary:
                record=dict(update=updates,epoch=epoch,batch=batch,global_windows=global_windows,
                    rank_windows=list(map(len,shards)),rank_sources=[sum(map(source_count,g)) for g in shards],
                    rank_timing=rank_timing,seconds_per_window=(time.perf_counter()-tick)/global_windows,**stats)
                with (out/'progress.jsonl').open('a',encoding='utf-8') as handle: handle.write(json.dumps(record)+'\n')
                if report_step:
                    print(f'ABLATION_TRAIN {contract["arm"]} {updates}/{len(groups)} CE={stats["CE"]:.4f} '
                        f'Lovasz={stats["Lovasz"]:.4f} seconds/window={(time.perf_counter()-begun)/session_windows:.4f} '
                        f'rank_windows={record["rank_windows"]} rank_sources={record["rank_sources"]}',flush=True)
            if epoch==1: break
            if updates%checkpoint_every==0: save();last_saved=updates
            fetch_started=time.perf_counter()
    finally:
        if hasattr(iterator,'close'): iterator.close()
    result['status']='complete' if epoch==1 else 'interrupted'
    # Every pause/final cursor denotes a fully completed global optimizer step.
    if last_saved!=updates or epoch==1: save(final=epoch==1)
    else: publish()
    return result
