"""Label-free diagonal feature moment alignment, source GT TRAIN -> STC history."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .bundle import Bundle


@torch.no_grad()
def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--bundle',default='../stc_overnight_bundle.zip')
    ap.add_argument('--out',required=True);ap.add_argument('--device',default='cuda')
    a=ap.parse_args(argv);torch.set_num_threads(1);b=Bundle(a.bundle)
    model,saved,policy=b.load_model('dense_epoch2',a.device);encoder=model.transport.history_encoder
    moments={}
    for role,kind in [('clean_train','gt'),('reference64','stc')]:
        total,squares,count=None,None,0
        for r in b.records:
            if r['role']!=role:continue
            _,raw,_=b.history(r,kind,'dense_valid')
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                f=encoder(torch.as_tensor(raw['history_occ'],device=a.device),torch.as_tensor(raw['history_observed'],device=a.device)).double()
            flat=f.permute(1,0,2,3).flatten(1)
            sm=flat.sum(1).cpu().numpy();sq=(flat*flat).sum(1).cpu().numpy()
            total=sm if total is None else total+sm;squares=sq if squares is None else squares+sq;count+=flat.shape[1]
        mean=total/count;std=np.sqrt(np.maximum(squares/count-mean*mean,1e-8))
        moments[role]=dict(mean=mean.tolist(),std=std.tolist(),pixels=count)
    source=moments['clean_train'];target=moments['reference64']
    scale=np.asarray(source['std'])/np.asarray(target['std']);bias=np.asarray(source['mean'])-scale*np.asarray(target['mean'])
    # One fixed halfway transform, not a grid searched coefficient.
    scale=(1+scale)/2;bias=bias/2
    out=dict(protocol='label_free_feature_half_moment_alignment_v1',scale=scale.tolist(),bias=bias.tolist(),moments=moments,
        source_GT_TRAIN_only=True,target_STC_history_only=True,future_labels_or_poses_read=False,
        declaration='unsupervised target-domain feature adaptation; NOT target-domain-information-free')
    Path(a.out).write_text(json.dumps(out,indent=2),encoding='utf-8');print('STYLE',min(scale),max(scale),flush=True);b.close()


if __name__=='__main__':main()
