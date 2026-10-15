"""Supervised target-domain HISTORY feature study with a scene-disjoint holdout.

This study is NOT zero-shot and its fitted scenes MUST NOT enter formal scoring.
No future occupancy, future ego or forecasting labels are used for fitting.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .bundle import Bundle


class FeatureAdapter(nn.Module):
    def __init__(self,width=64):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(width,width,3,padding=1),nn.SiLU(),nn.Conv2d(width,width,3,padding=1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,value):return value+self.net(value)


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--bundle',default='../stc_overnight_bundle.zip')
    ap.add_argument('--out',required=True);ap.add_argument('--updates',type=int,default=500);ap.add_argument('--device',default='cuda')
    a=ap.parse_args(argv);torch.set_num_threads(1);torch.manual_seed(21);np.random.seed(21)
    b=Bundle(a.bundle);body,saved,policy=b.load_model('dense_epoch2',a.device)
    rows=[r for r in b.records if r['role']=='reference64']
    rows=sorted(rows,key=lambda r:hashlib.sha256(('stc_feature_adapter_v1:'+r['scene_name']).encode()).hexdigest())
    fit,held=rows[:len(rows)//2],rows[len(rows)//2:]
    assert not {r['scene_name'] for r in fit}&{r['scene_name'] for r in held}
    cache=[]
    with torch.no_grad():
        for r in fit:
            values=[]
            for kind in ('stc','gt'):
                _,raw,_=b.history(r,kind,'dense_valid')
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                    f=body.transport.history_encoder(torch.as_tensor(raw['history_occ'],device=a.device),
                                                     torch.as_tensor(raw['history_observed'],device=a.device))
                values.append(f.cpu())
            occ=F.max_pool2d(torch.as_tensor((raw['history_occ']!=17).any(-1)).float()[:,None],2)
            cache.append((*values,occ[:,0]))
    model=FeatureAdapter().to(a.device);opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    for step in range(a.updates):
        x,y,occ=cache[step%len(cache)];x=x.to(a.device);y=y.to(a.device).float();occ=occ.to(a.device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            value=model(x);identity=model(y)
        scale=y.std((0,2,3),keepdim=True).clamp_min(.1)
        err=((value.float()-y)/scale).square().mean(1)
        loss=(err*(1+3*occ)).mean()+.5*((identity.float()-y)/scale).square().mean()
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
        for g in opt.param_groups:g['lr']=.000003+.000297*(1+np.cos(np.pi*(step+1)/a.updates))/2
        if (step+1)%50==0:print('REAL_HISTORY_FEATURE_ADAPTER',step+1,float(loss.detach()),flush=True)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    contract=dict(protocol='supervised_REAL_STC_HISTORY_feature_adapter_v1',updates=a.updates,seed=21,
        fit_scenes=[r['scene_name'] for r in fit],held_scenes=[r['scene_name'] for r in held],
        held_keys=[[r['scene_name'],r['t0_token']] for r in held],
        source='real STC historical inputs -> corresponding GT historical teacher features',
        future_labels_or_poses_read_for_training=False,
        VAL_historical_GT_USED=True,NOT_zero_shot=True,
        scope='supervised target-domain adaptation; ONLY held scenes valid for scoring; NEVER full4369 score',
        source_sha256=saved['source_sha256'],initialization='zero residual adapter; original full body frozen')
    torch.save(dict(model=model.cpu().state_dict(),contract=contract),out/'adapter.pt')
    (out/'contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8');b.close()


if __name__=='__main__':main()
