"""Small independent TRAIN-only denoising pilot. No VAL labels used for training."""
import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tools.final_semantic_evaluation import metrics
from .bundle import Bundle


class Denoiser(nn.Module):
    def __init__(self,z=16,width=48):
        super().__init__();self.z=z
        self.embedding=nn.Embedding(19,4)
        self.input=nn.Sequential(nn.Conv2d(z*4,width,3,padding=1),nn.SiLU())
        self.blocks=nn.ModuleList([nn.Sequential(nn.Conv2d(width,width,3,padding=d,dilation=d,groups=width),
            nn.SiLU(),nn.Conv2d(width,width,1),nn.SiLU()) for d in (1,2,4)])
        self.output=nn.Conv2d(width,18*z,1);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,labels):
        shape=labels.shape;known=labels<18
        feature=self.embedding(labels.long()).permute(0,3,4,1,2).flatten(1,2)
        x=self.input(feature)
        for block in self.blocks:x=x+block(x)
        delta=self.output(x).reshape(shape[0],self.z,18,shape[1],shape[2]).permute(0,2,3,4,1)
        prior=F.one_hot(labels.clamp(0,17).long(),18).permute(0,4,1,2,3).float()
        return (prior+.001).log()+delta.float()


def coarse_field(shape,device):
    value=torch.rand((1,1,20,20,4),device=device)
    return F.interpolate(value,size=shape,mode='trilinear',align_corners=False)[0,0]


@torch.no_grad()
def corrupt(gt):
    shape=gt.shape;occupied=gt!=17;pred=gt.clone();device=gt.device
    xy=torch.stack(torch.meshgrid(torch.arange(shape[0],device=device),torch.arange(shape[1],device=device),indexing='ij'),-1)
    distance=((xy+.5-100)*.4).square().sum(-1).sqrt()
    missing=(.18+.017*distance).clamp(max=.85)[...,None]
    # Uniformize a smoothed random field's marginal approximately with a logistic.
    field=coarse_field(shape,device)
    field=torch.sigmoid((field-.5)*8)
    pred[occupied & (field<missing)]=17
    neighbor=torch.full_like(gt,17)
    for axis,shift in [(0,1),(0,-1),(1,1),(1,-1),(2,1),(2,-1)]:
        moved=torch.roll(gt,shift,axis);valid=moved!=17
        border=[slice(None)]*3;border[axis]=0 if shift>0 else -1;valid[tuple(border)]=False
        use=valid & (neighbor==17);neighbor[use]=moved[use]
    false=(gt==17)&(neighbor!=17)&(coarse_field(shape,device)>.57)
    pred[false]=neighbor[false]
    # Sparse local semantic substitutions, not uniform random classes in free space.
    swap=occupied&(neighbor!=17)&(coarse_field(shape,device)>.74)
    pred[swap]=neighbor[swap]
    return pred


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--bundle',default='../stc_overnight_bundle.zip')
    ap.add_argument('--out',required=True);ap.add_argument('--updates',type=int,default=500);ap.add_argument('--device',default='cuda')
    a=ap.parse_args(argv);torch.set_num_threads(1);torch.manual_seed(21);np.random.seed(21);random.seed(21)
    b=Bundle(a.bundle);rows=[r for r in b.records if r['role']=='clean_train']
    frames=[];keys=set()
    for r in rows:
        for token in r['history_tokens']:
            key=(r['scene_name'],token)
            if key not in keys:frames.append(b.frame('gt',*key)['semantics'].copy());keys.add(key)
    model=Denoiser().to(a.device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    for step in range(a.updates):
        gt=torch.as_tensor(frames[step%len(frames)].copy(),device=a.device).long()
        if random.random()<.5:gt=gt.flip(0)
        if random.random()<.5:gt=gt.flip(1)
        pred=gt.clone() if step%5==0 else corrupt(gt)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):logits=model(pred[None])
        loss=F.cross_entropy(logits,gt[None]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
        for group in opt.param_groups:group['lr']=.00001+.00099*(1+np.cos(np.pi*(step+1)/a.updates))/2
        if (step+1)%50==0:print('DENOISE_TRAIN',step+1,'CE',float(loss.detach()),flush=True)
    contract=dict(training='original TRAIN32 historical frames only',scenes=sorted({k[0] for k in keys}),frames=len(keys),
        updates=a.updates,seed=21,objective='same-frame CE',noise='fixed spatial boundary/hole corruption',
        parameters=sum(p.numel() for p in model.parameters()),VAL_labels_used_for_training=False)
    torch.save(dict(model=model.state_dict(),contract=contract),out/'last.pt')
    counts={n:metrics.empty(1) for n in ('raw','denoised')}
    model.eval()
    with torch.no_grad():
        for r in b.records:
            if r['role']!='reference64':continue
            _,raw,_=b.history(r)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                cleaned=model(torch.as_tensor(raw['history_occ'][-1:]+0,device=a.device)).argmax(1)[0].byte().cpu().numpy()
            gt=b.frame('gt',r['scene_name'],r['t0_token'])['semantics']
            metrics.add(counts['raw'],0,raw['history_occ'][-1],gt);metrics.add(counts['denoised'],0,cleaned,gt)
    result=dict(status='complete',contract=contract,scope='same-frame current STC diagnostic; NOT future target or full VAL',
        metrics={n:metrics.report(c,(0.,),moving=False) for n,c in counts.items()})
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('DENOISE_REAL_T0',{n:v['average'] for n,v in result['metrics'].items()},flush=True);b.close()


if __name__=='__main__':main()
