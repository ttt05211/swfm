"""Latest U-Net only: small affine/head adaptation with scene-disjoint checks."""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch.nn import functional as F
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from tools.shared_history_experiments.refine.geometry import centers,inverse_grid
from tools.shared_history_experiments.full_gt.engine import semantic_loss
from tools.stc_overnight.bundle import Bundle
from .portable import load_weights
from .cache import AT,write
from .calibrate import load,scores,confusion

ARMS=('norm320_history','norm320_supervised','norm_head482_supervised',
    'decoder_fine7106_history','decoder_fine7106_supervised','decoder_fine7106_targetheavy')


@torch.no_grad()
def historical_labels(raw,poses,grid):
    """Static 3/4 agreement (free 4/4) from STC historical predictions ONLY."""
    sem=torch.as_tensor(raw['history_occ']).long();valid=torch.as_tensor(raw['history_observed']).bool()&(sem<18)
    for cls in DYNAMIC_CLASS_IDS:valid&=sem!=cls
    old=torch.as_tensor(np.asarray(raw['history_poses']),dtype=torch.float64)
    future=torch.as_tensor(np.asarray(poses),dtype=torch.float64);query=centers(grid,'cpu');result=[]
    shape=tuple(grid.shape_hwd)
    for h in range(6):
        votes=torch.zeros(18,*shape)
        for f in range(4):
            sampling=inverse_grid(query,torch.linalg.inv(old[f])@future[h],grid)[None]
            label=F.grid_sample(sem[f][None,None].float(),sampling,mode='nearest',align_corners=False,padding_mode='zeros')[0,0].long().clamp(0,17)
            weight=F.grid_sample(valid[f][None,None].float(),sampling,mode='nearest',align_corners=False,padding_mode='zeros')[0,0]
            votes.scatter_add_(0,label[None],weight[None])
        count,label=votes.max(0);good=torch.where(label==17,count>=4,count>=3)
        result.append(torch.where(good,label,255).byte().numpy())
    return np.stack(result)


def pack(cache,bundle,domains):
    c=json.loads((cache/'contract.json').read_text());sha=c['source_checkpoint_sha256'];out=cache/'training_frames';out.mkdir(exist_ok=True)
    b=Bundle(bundle,cache_mib=16);refs=[b.by_key[tuple(k)] for k in c['reference64_keys'][:32]];tic=time.monotonic()
    try:
        for domain in domains:
            for i in range(32):
                name=f'{domain}_{i:03d}'
                if all((out/f'{name}_h{h}.pt').is_file() for h in range(6)):continue
                if shutil.disk_usage(out).free<3*2**30:raise RuntimeError('keep at least 3GiB disk headroom')
                pseudo=None
                if domain=='stc':
                    r=refs[i];_,raw,_=b.history(r,'stc','dense_valid');pseudo=historical_labels(raw,r['future_poses'],b.pcfg.grid)
                with load(cache/'inputs'/(name+'.npz'),sha) as z:
                    if str(z['split'])!=('fit32' if domain=='stc' else 'source_TRAIN32'):raise ValueError('no held target scene allowed in adaptation')
                    prior=z['prior'];memory=z['memory'];labels=z['labels'];shape=z['shape'].tolist()
                    for h in range(6):
                        path=out/f'{name}_h{h}.pt'
                        if path.exists():continue
                        # Copy each view: otherwise torch.save would serialize its entire
                        # six-frame backing storage for EACH file.
                        item=dict(source_sha256=sha,key=z['key'].tolist(),split=str(z['split']),h=h,shape=shape,
                            prior=torch.from_numpy(prior[h:h+1].copy()),memory=torch.from_numpy(memory[h:h+1].copy()),
                            labels=torch.from_numpy(labels[h:h+1].copy()))
                        if pseudo is not None:item['pseudo']=torch.from_numpy(pseudo[h:h+1].copy())
                        torch.save(item,path)
                del prior,memory,labels,pseudo
                if (i+1)%8==0:print('ADAPT_PACK',domain,i+1,'/32','seconds',round(time.monotonic()-tic,1),flush=True)
    finally:b.close()


def parameters(columns,arm):
    if arm.startswith('decoder_fine'):
        return [n for n,_ in columns.named_parameters() if n.startswith(('decode_fine.','head.'))]
    selected=[]
    for name,m in columns.named_modules():
        if isinstance(m,torch.nn.GroupNorm):selected.extend((name+'.weight',name+'.bias'))
    if arm=='norm_head482_supervised':selected+=['head.weight','head.bias']
    return selected


def frame(path,sha,device):
    item=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    if item['source_sha256']!=sha or item['split'] not in ('fit32','source_TRAIN32'):raise ValueError('wrong adaptation source or held-scene contamination')
    return item,item['prior'].to(device),item['memory'].to(device)


def train(cache,out,weights,arms,device,updates=192):
    body,_,info=load_weights(weights,device);sha=info['manifest']['source_checkpoint_sha256']
    c=json.loads((cache/'contract.json').read_text());assert c['source_checkpoint_sha256']==sha
    original={k:v.detach().cpu().clone() for k,v in body.columns.state_dict().items()};out.mkdir(parents=True,exist_ok=True)
    contract=dict(protocol='latest_forward_U-Net_small_parameter_adaptation_v2',source_sha256=sha,arms=list(arms),updates=updates,
        source_TRAIN_keys=c['source_keys'],target_fit_keys=c['reference64_keys'][:32],held_keys=c['reference64_keys'][32:],
        labelled_VAL_supervision={arm:not arm.endswith('_history') for arm in arms},
        loss='source/target full-grid per-horizon CE+Lovasz; interleaved six horizons; .01 source-parameter tether',
        source_target_weights={arm:([.25,.75] if arm.endswith('_targetheavy') else [.5,.5]) for arm in arms},
        pseudo='history arm: STC static 3/4 agreement, free4/4; unknown255 only TRAIN pseudo loss',
        learning_rate={arm:(.0003 if arm.startswith('decoder_fine') else .001) for arm in arms},
        schedule=f'whole{updates} cosine floor.01; fixed FINAL; no held-best update selection',
        frozen_upstream=True,known='dense_valid',GT_future_ego=True,FPS=False)
    if (out/'contract.json').exists() and json.loads((out/'contract.json').read_text())!=contract:raise ValueError('new adaptation output required')
    write(out/'contract.json',contract);tic=time.monotonic()
    try:
        for arm in arms:
            path=out/(arm+'.pt')
            if path.exists():continue
            body.columns.load_state_dict(original,strict=True);body.requires_grad_(False);body.columns.train()
            named=dict(body.columns.named_parameters());names=parameters(body.columns,arm);params=[named[n] for n in names]
            for v in params:v.requires_grad_(True)
            peak=contract['learning_rate'][arm];opt=torch.optim.AdamW(params,lr=peak,weight_decay=0);torch.manual_seed(21);trace=[]
            anchors={n:original[n].to(device) for n in names}
            for step in range(updates):
                h=step%6;i=(step//6+h*5)%32;j=(i*13+7)%32
                for g in opt.param_groups:g['lr']=peak*(.01+.99*.5*(1+np.cos(np.pi*step/updates)))
                opt.zero_grad(set_to_none=True);losses=[]
                for branch,(domain,index) in enumerate((('source',j),('stc',i))):
                    item,prior,memory=frame(cache/'training_frames'/f'{domain}_{index:03d}_h{h}.pt',sha,device)
                    logits=body.columns(memory,prior,[h],item['shape'])
                    lab=item['pseudo'] if domain=='stc' and arm.endswith('_history') else item['labels']
                    total,ce,lovasz=semantic_loss(logits,lab.numpy());loss=total.mean()
                    (contract['source_target_weights'][arm][branch]*loss).backward();losses.append(float(loss.detach()))
                    del item,prior,memory,logits,lab,total,loss,ce,lovasz
                tether=sum((named[n]-anchors[n]).square().sum() for n in names)/sum(v.numel() for v in params)
                (.01*tether).backward();torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
                if (step+1)%24==0:
                    trace.append(dict(step=step+1,source_loss=losses[0],target_loss=losses[1]))
                    print('LATEST_ADAPT',arm,step+1,'/'+str(updates),trace[-1],'seconds',round(time.monotonic()-tic,1),flush=True)
            learned={k:v.detach().cpu().clone() for k,v in body.columns.state_dict().items()}
            for n,v in learned.items():
                if n not in names and not torch.equal(v,original[n]):raise RuntimeError('frozen parameter unexpectedly changed: '+n)
            torch.save(dict(contract=contract,arm=arm,columns=learned,trainable=names,
                trainable_parameters=sum(v.numel() for v in params),trace=trace),path)
    finally:body.columns.load_state_dict(original,strict=True);body.eval().requires_grad_(False)


@torch.no_grad()
def evaluate(cache,out,weights,arms,device,domains):
    body,_,info=load_weights(weights,device);sha=info['manifest']['source_checkpoint_sha256'];original={k:v.detach().cpu().clone() for k,v in body.columns.state_dict().items()}
    models={arm:torch.load(out/(arm+'.pt'),map_location='cpu',weights_only=False) for arm in arms}
    if any(v['contract']['source_sha256']!=sha for v in models.values()):raise ValueError('wrong latest adaptation source')
    totals={};tic=time.monotonic();(out/'eval').mkdir(exist_ok=True)
    for domain in domains:
        for i in range(32,64):
            name=f'{domain}_{i:03d}';path=out/'eval'/(name+'.json')
            if path.exists():item=json.loads(path.read_text())
            else:
                with load(cache/'inputs'/(name+'.npz'),sha) as z:
                    prior=torch.from_numpy(z['prior']).to(device);memory=torch.from_numpy(z['memory']).to(device);shape=z['shape'].tolist();pred={}
                    for arm,v in models.items():
                        body.columns.load_state_dict(v['columns'],strict=True)
                        logits=body.columns(memory,prior,range(6),shape)
                        pred[arm]=logits[list(AT)].argmax(1).byte().cpu().numpy();del logits
                    # All six forecasts for every model before accessing future targets.
                    gt=z['labels'][list(AT)].reshape(3,-1)
                    counts={n:[confusion(p[h],gt[h]).tolist() for h in range(3)] for n,p in pred.items()}
                    item=dict(key=z['key'].tolist(),confusion=counts)
                write(path,item);del prior,memory,pred
            for split in ('held32','screen16' if i<48 else 'confirmation16'):
                key=domain+'/'+split;t=totals.setdefault(key,dict(windows=0,counts={n:np.zeros((3,18,18),np.int64) for n in arms}))
                t['windows']+=1
                for n,q in item['confusion'].items():t['counts'][n]+=np.asarray(q,np.int64)
            if (i+1)%8==0:
                reports={k:dict(windows=v['windows'],reports={n:scores(c) for n,c in v['counts'].items()}) for k,v in totals.items()}
                write(out/'results.json',dict(status='running',reports=reports,contracts={n:v['contract'] for n,v in models.items()}))
                print('ADAPT_EVAL',name,'seconds',round(time.monotonic()-tic,1),{n:r['average'] for n,r in reports[domain+'/held32']['reports'].items()},flush=True)
    reports={k:dict(windows=v['windows'],reports={n:scores(c) for n,c in v['counts'].items()}) for k,v in totals.items()}
    write(out/'results.json',dict(status='complete',reports=reports,contracts={n:v['contract'] for n,v in models.items()}))
    body.columns.load_state_dict(original,strict=True)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('phase',choices=('pack','train','evaluate'))
    ap.add_argument('--cache',required=True);ap.add_argument('--out');ap.add_argument('--weights',default='../_received/forward_adaln_epoch2_local.zip')
    ap.add_argument('--bundle',default='../stc_overnight_bundle.zip');ap.add_argument('--device',default='cuda')
    ap.add_argument('--updates',type=int,choices=(192,576),default=192)
    ap.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS[:3]));ap.add_argument('--domains',nargs='+')
    a=ap.parse_args(argv);torch.set_num_threads(1);cache=Path(a.cache)
    if a.phase=='pack':return pack(cache,a.bundle,a.domains or ('stc','source'))
    if not a.out:raise ValueError('--out required')
    if a.phase=='train':return train(cache,Path(a.out),a.weights,a.arms,a.device,a.updates)
    return evaluate(cache,Path(a.out),a.weights,a.arms,a.device,a.domains or ('stc','gt'))


if __name__=='__main__':main()
