"""Two cheap alternatives: semantic-embedding anti-aliasing and Refiner flip ensemble."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from tools.causal_forward_all.common import Epoch2Predictor
from tools.stc_overnight.bundle import Bundle
from .portable import load_weights
from .cache import AT,write
from .calibrate import confusion,scores


def blur(value,amount):
    n,sx,sy,z,e=value.shape;x=value.permute(0,3,4,1,2).reshape(n,z*e,sx,sy)
    k=x.new_tensor([1.,2.,1.]);kernel=(k[:,None]*k[None,:]/16)[None,None].expand(z*e,1,3,3)
    result=F.conv2d(F.pad(x.float(),(1,1,1,1),mode='replicate'),kernel.float(),groups=z*e)
    smoothed=result.to(value.dtype).reshape(n,z,e,sx,sy).permute(0,3,4,1,2)
    return value+amount*(smoothed-value)


@contextmanager
def embedding_blur(body,amount):
    def hook(module,inputs,value):
        with torch.autocast(device_type=value.device.type,enabled=False):return blur(value,amount)
    handle=body.transport.history_encoder.semantic.register_forward_hook(hook)
    try:yield
    finally:handle.remove()


@torch.no_grad()
def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--weights',default='../_received/forward_adaln_epoch2_local.zip')
    ap.add_argument('--bundle',default='../stc_overnight_bundle.zip');ap.add_argument('--cache',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--device',default='cuda');ap.add_argument('--domains',nargs='+',default=['stc','gt']);a=ap.parse_args(argv)
    torch.set_num_threads(1);torch.manual_seed(21);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/'counts').mkdir(exist_ok=True)
    body,c,info=load_weights(a.weights,a.device);b=Bundle(a.bundle,cache_mib=16);p=Epoch2Predictor(body,info['prepare_config'],a.device,2)
    free=json.loads((Path(a.cache)/'calibration/fit.json').read_text())['chosen_free_bias']
    names=['raw','unet_y_logit_TTA','unet_y_probability_TTA','embedding_blur50','embedding_blur100'];all_names=names+[n+'_free' for n in names]
    contract=dict(protocol='latest_forward_anti_alias_and_refiner_reflection_v1',source_sha256=info['manifest']['source_checkpoint_sha256'],held_keys=c['eval_keys'][32:],
        GT_future_ego=True,known='dense_valid',full_grid=True,Moving_available=False,
        feature_methods='3x3 binomial XY semantic-embedding blur .5/1; known/geometry unchanged',
        TTA='Refiner ONLY, y-mirrored future prior+memory and restored logits; two U-Net passes, not original FPS',
        future_target_labels_used_for_features=False,free_bias=free,free_bias_fit='GT labels first32 original scenes',raw_weights_unchanged=True)
    if (out/'contract.json').exists() and json.loads((out/'contract.json').read_text())!=contract:raise ValueError('new feature output required')
    write(out/'contract.json',contract);totals={};tic=time.monotonic()
    try:
        for domain in a.domains:
            for i in range(32,64):
                name=f'{domain}_{i:03d}';path=out/'counts'/(name+'.json')
                if path.exists():item=json.loads(path.read_text())
                else:
                    r=b.by_key[tuple(c['eval_keys'][i])];rec,raw,times=b.history(r,domain,'dense_valid');row=p.prepare(rec,raw,times=times)
                    base=p.forward(row,poses=np.asarray(r['future_poses']));pred={};shape=base['shape']
                    def collect(n,x):
                        pred[n]=x[list(AT)].argmax(1).byte().cpu().numpy()
                        maximum,winner=x[:,:17].max(1);cal=torch.where(maximum>=x[:,17]+free,winner,17)
                        pred[n+'_free']=cal[list(AT)].byte().cpu().numpy()
                    collect('raw',base['logits'])
                    prior=base['prior'].reshape(6,18,*shape).flip(3).flatten(2);memory=base['memory'].flip(3)
                    reflected=body.columns(memory,prior,range(6),shape).reshape(6,18,*shape).flip(3).flatten(2)
                    collect('unet_y_logit_TTA',.5*(base['logits']+reflected))
                    value=(.5*(base['logits'].softmax(1)+reflected.softmax(1))).clamp_min(1e-12).log()
                    collect('unet_y_probability_TTA',value);del reflected,value,prior,memory,base
                    for amount,n in ((.5,'embedding_blur50'),(1.,'embedding_blur100')):
                        with embedding_blur(body,amount):fw=p.forward(row,poses=np.asarray(r['future_poses']))
                        collect(n,fw['logits']);del fw
                    # Complete six outputs for all feature variants before scoring.
                    gt=b.targets(r).reshape(3,-1);item=dict(key=[r['scene_name'],r['t0_token']],
                        confusion={n:[confusion(v[h],gt[h]).tolist() for h in range(3)] for n,v in pred.items()})
                    write(path,item);del pred,row,raw,gt
                for split in ('held32','screen16' if i<48 else 'confirmation16'):
                    key=domain+'/'+split;t=totals.setdefault(key,dict(windows=0,counts={n:np.zeros((3,18,18),np.int64) for n in all_names}));t['windows']+=1
                    for n,q in item['confusion'].items():t['counts'][n]+=np.asarray(q,np.int64)
                if (i+1)%4==0:
                    reports={k:dict(windows=v['windows'],reports={n:scores(q) for n,q in v['counts'].items()}) for k,v in totals.items()}
                    write(out/'results.json',dict(status='running',contract=contract,reports=reports));print('FEATURE_SCREEN',name,'seconds',round(time.monotonic()-tic,1),
                        {n:v['average'] for n,v in reports[domain+'/held32']['reports'].items() if not n.endswith('_free')},flush=True)
        reports={k:dict(windows=v['windows'],reports={n:scores(q) for n,q in v['counts'].items()}) for k,v in totals.items()}
        write(out/'results.json',dict(status='complete',contract=contract,reports=reports))
    finally:p.close();b.close()


if __name__=='__main__':main()
