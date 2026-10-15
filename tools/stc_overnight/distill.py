"""GT TRAIN-only paired feature denoising; frozen clean encoder teacher."""
import argparse
import copy
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.nn import functional as F
from .bundle import Bundle
from .denoise import corrupt


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--bundle',default='../stc_overnight_bundle.zip')
    ap.add_argument('--out',required=True);ap.add_argument('--updates',type=int,default=500);ap.add_argument('--device',default='cuda')
    a=ap.parse_args(argv);torch.set_num_threads(1);torch.manual_seed(21);np.random.seed(21);random.seed(21)
    b=Bundle(a.bundle);body,saved,policy=b.load_model('dense_epoch2',a.device)
    teacher=body.transport.history_encoder;student=copy.deepcopy(teacher).requires_grad_(True).train()
    records=[r for r in b.records if r['role']=='clean_train'];cache=[]
    with torch.no_grad():
        for r in records:
            _,raw,_=b.history(r,'gt','dense_valid')
            sem=torch.as_tensor(raw['history_occ'],device=a.device)
            obs=torch.as_tensor(raw['history_observed'],device=a.device)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):target=teacher(sem,obs)
            cache.append((raw['history_occ'].copy(),raw['history_observed'].copy(),target.cpu()))
    opt=torch.optim.AdamW(student.parameters(),lr=.0001,weight_decay=.0001)
    # Same fixed 32 source scenes; target STC and VAL labels are never read here.
    for step in range(a.updates):
        sem,obs,target=cache[step%len(cache)]
        sem=torch.as_tensor(sem.copy(),device=a.device).long();obs=torch.as_tensor(obs,device=a.device)
        noisy=sem.clone() if step%5==0 else torch.stack([corrupt(frame) for frame in sem])
        target=target.to(a.device).float();opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):value=student(noisy,obs)
        scale=target.std((0,2,3),keepdim=True).clamp_min(.1)
        error=((value.float()-target)/scale).square().mean(1)
        occ=F.max_pool2d((sem!=17).any(-1).float()[:,None],2)[:,0]
        loss=(error*(1+3*occ)).mean()
        loss.backward();torch.nn.utils.clip_grad_norm_(student.parameters(),5.);opt.step()
        for g in opt.param_groups:g['lr']=.000001+.000099*(1+np.cos(np.pi*(step+1)/a.updates))/2
        if (step+1)%50==0:print('FEATURE_DENOISE',step+1,'loss',float(loss.detach()),flush=True)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    contract=dict(protocol='paired_clean_GT_TRAIN_feature_distillation_v1',updates=a.updates,seed=21,
        source='32 original TRAIN windows; synthetic historical corruption; clean teacher feature targets',
        VAL_labels_or_histories_used_for_training=False,body_and_refiner_optimized=False,
        source_sha256=saved['source_sha256'],initialization='dense original epoch2 encoder')
    torch.save(dict(model=student.cpu().state_dict(),contract=contract),out/'encoder.pt')
    (out/'contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8');b.close()


if __name__=='__main__':main()
