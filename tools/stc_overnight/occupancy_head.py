"""TRAIN-only matched confidence calibration heads; original world model frozen.

Tiny bounded occupancy-log-odds correction, not ADD/REMOVE actions. All six
CE+Lovasz terms are used per update. Clean vs synthetic-noise source control.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tools.final_semantic_evaluation.predictor import Predictor
from tools.shared_history_experiments.full_gt.engine import semantic_loss
from .bundle import Bundle
from .denoise import corrupt
from .replay import prepare_forecast,iter_forecasts


def features(memory,prior,shape):
    coarse=F.avg_pool3d(prior.reshape(-1,18,*shape).float(),(2,2,1))
    return torch.cat((memory.float(),coarse),1)


class OccupancyCalibration(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv3d(34,16,1),nn.SiLU(),nn.Conv3d(16,16,3,padding=1,groups=16),
                               nn.SiLU(),nn.Conv3d(16,1,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x,shape):
        with torch.autocast(device_type=x.device.type,dtype=torch.bfloat16,enabled=x.device.type=='cuda'):
            beta=self.net(x)
        beta=2*beta.float().tanh()
        return F.interpolate(beta,size=shape,mode='trilinear',align_corners=False).flatten(2)


def adjust(logits,beta):
    return torch.cat((logits[:,:17]+beta,logits[:,17:]),1)


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--bundle',default='../stc_overnight_bundle.zip')
    ap.add_argument('--out',required=True);ap.add_argument('--updates',type=int,default=500);ap.add_argument('--device',default='cuda')
    a=ap.parse_args(argv);torch.set_num_threads(1);torch.manual_seed(21);np.random.seed(21)
    b=Bundle(a.bundle);body,saved,policy=b.load_model('clean1000',a.device)
    if policy!='baseline':raise ValueError('this matched screen fixes original clean1000 source policy')
    rows=[r for r in b.records if r['role']=='clean_train'];out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    contract=dict(protocol='TRAIN32_frozen_clean1000_matched_bounded_occ_heads_v1',updates=a.updates,seed=21,
        source_sha256=saved['source_sha256'],keys=[[r['scene_name'],r['t0_token']] for r in rows],
        objective='full6 CE+Lovasz, equal per-horizon; body frozen; zero-init +/-2 log-odds head',
        cache_precision='float32 exact frozen logits/memory/coarse prior; uint8 GT TRAIN labels',
        target_VAL_inputs_or_labels_used_for_training=False,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=out/'contract.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=contract:raise ValueError('new head output required')
    path.write_text(json.dumps(contract,indent=2),encoding='utf-8')
    cache=out/'cache';cache.mkdir(exist_ok=True)
    needed=len(rows)*12*(34*(100*100*16)*4+18*640000*4+640000)
    completed=list(cache.glob('*_complete.json'))
    if not completed and shutil.disk_usage(out).free<needed+5*2**30:raise RuntimeError('need ~32GiB free scratch, no source/cache deletion performed')
    print('OCC_HEAD_CACHE estimated_GiB=',needed/2**30,'TRAIN_windows=',len(rows),flush=True)
    p=Predictor(body,b.pcfg,a.device,workers=2)
    try:
        for i,r in enumerate(rows):
            rec,clean,times=b.history(r,'gt','dense_valid')
            noisy=dict(clean,history_occ=np.stack([corrupt(torch.as_tensor(s,device=a.device).long()).byte().cpu().numpy() for s in clean['history_occ']]))
            labels=b.targets(r,range(6)) # TRAIN supervision only, never supplied to forecast.
            for arm,raw in [('clean',clean),('corrupted',noisy)]:
                marker=cache/f'{arm}_{i:03d}_complete.json'
                if marker.exists():continue
                row=p.prepare(rec,raw,times=times);context=prepare_forecast(p,row,r['future_poses'])
                for hs,fw in iter_forecasts(context):
                    h=hs[0]
                    x=features(fw['memory'],fw['prior'],fw['shape']).cpu().numpy()
                    logits=fw['logits'].cpu().numpy()
                    np.savez(cache/f'{arm}_{i:03d}_{h}.npz',features=x,logits=logits,labels=labels[h:h+1])
                    del fw,x,logits
                marker.write_text(json.dumps(dict(key=contract['keys'][i],horizons=6)),encoding='utf-8')
                print('OCC_HEAD_CACHE',arm,i+1,'/',len(rows),flush=True)
    finally:p.close()
    initial=OccupancyCalibration().state_dict()
    for arm in ('clean','corrupted'):
        torch.manual_seed(21)
        head=OccupancyCalibration().to(a.device);head.load_state_dict(initial);head.train()
        opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
        for step in range(a.updates):
            i=step%len(rows);opt.zero_grad(set_to_none=True);total=0.
            for h in range(6):
                with np.load(cache/f'{arm}_{i:03d}_{h}.npz',allow_pickle=False) as z:
                    x=torch.as_tensor(z['features'],device=a.device);logits=torch.as_tensor(z['logits'],device=a.device);labels=z['labels']
                beta=head(x,b.pcfg.grid.shape_hwd);pred=adjust(logits,beta)
                loss,ce,lovasz=semantic_loss(pred,labels)
                loss=loss.mean()/6;loss.backward();total+=float(loss.detach())
                del x,logits,labels,beta,pred,loss,ce,lovasz
            torch.nn.utils.clip_grad_norm_(head.parameters(),5.);opt.step()
            for g in opt.param_groups:g['lr']=.00001+.00099*(1+np.cos(np.pi*(step+1)/a.updates))/2
            if (step+1)%50==0:
                print('OCC_CALIBRATION_TRAIN',arm,step+1,'loss',total,flush=True)
                torch.save(dict(model=head.state_dict(),contract=contract,arm=arm,update=step+1,
                    parameters=sum(v.numel() for v in head.parameters())),out/f'{arm}_last.pt')
        head.cpu()
    b.close()


if __name__=='__main__':main()
