"""Conditional relative posterior calibration using source/target predictions only."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .bundle import Bundle
from tools.final_semantic_evaluation import metrics


def bins(raw,poses,grid,h,device):
    shape=grid.shape_hwd
    xyz=torch.stack(torch.meshgrid(*(torch.arange(n,device=device,dtype=torch.float64) for n in shape),indexing='ij'),-1)
    xyz=(xyz+.5)*xyz.new_tensor(grid.voxel_size)+xyz.new_tensor([grid.x_min,grid.y_min,grid.z_min])
    matrix=np.linalg.inv(np.asarray(raw['history_poses'][-1]))@np.asarray(poses[h])
    at=torch.einsum('ij,...j->...i',xyz.new_tensor(matrix[:3,:3]),xyz)+xyz.new_tensor(matrix[:3,3])
    rb=torch.bucketize(at[...,:2].square().sum(-1).sqrt(),at.new_tensor([10.,20.,30.,40.]))
    z=torch.arange(shape[-1],device=device).expand(shape)
    return rb*shape[-1]+z


@torch.no_grad()
def posterior_stats(logits,ids,count):
    p=logits.softmax(0)[:17].sum(0).reshape(-1).double()
    ids=ids.reshape(-1)
    return torch.bincount(ids,weights=p,minlength=count).cpu().numpy(),torch.bincount(ids,minlength=count).cpu().numpy()


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--target',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--bundle',default='../stc_overnight_bundle.zip');ap.add_argument('--device',default='cuda')
    a=ap.parse_args(argv);torch.set_num_threads(1);source,target,out=Path(a.source),Path(a.target),Path(a.out);out.mkdir(parents=True,exist_ok=True)
    sc=json.loads((source/'contract.json').read_text(encoding='utf-8'));tc=json.loads((target/'contract.json').read_text(encoding='utf-8'))
    if sc['model_sha256']!=tc['model_sha256'] or sc.get('role')!='clean_train':raise ValueError('same model/source TRAIN required')
    sp,ns=None,None
    for file in source.glob('window_*.json'):
        data=json.loads(file.read_text(encoding='utf-8'))
        sm=np.asarray(data['spatial_occ_sum']).sum(0);nn=np.asarray(data['spatial_counts']).sum(0)
        sp=sm if sp is None else sp+sm;ns=nn if ns is None else ns+nn
    b=Bundle(a.bundle);nbins=5*b.pcfg.grid.shape_hwd[-1];tp=np.zeros(nbins);nt=np.zeros(nbins)
    for i,key in enumerate(tc['keys']):
        r=b.by_key[tuple(key)];raw={'history_poses':np.asarray(r['history_poses'])}
        with np.load(target/f'logits_{i:03d}.npz',allow_pickle=False) as z:values=z['logits']
        for j,h in enumerate((1,3,5)):
            ids=bins(raw,r['future_poses'],b.pcfg.grid,h,a.device)
            sm,nn=posterior_stats(torch.as_tensor(values[j],device=a.device),ids,nbins);tp+=sm;nt+=nn
    qs=(sp+.5)/(ns+1);qt=(tp+.5)/(nt+1)
    bias=np.log(qs/(1-qs))-np.log(qt/(1-qt))
    fit=dict(protocol='range_height_relative_binary_posterior_alignment_v1',source_contract=sc,target_contract=tc,
        range_bins_m=[0,10,20,30,40,'inf'],z_bins=b.pcfg.grid.shape_hwd[-1],occupied_logit_bias=bias.tolist(),
        source_posterior=qs.tolist(),target_posterior=qt.tolist(),source_counts=ns.tolist(),target_counts=nt.tolist(),
        target_labels_used_for_fitting=False,source_GT_future_labels_used_for_fitting=False,
        declaration='conditional posterior-marginal assumption; transductive unlabeled STC calibration; NOT pristine/target-free')
    (out/'fit.json').write_text(json.dumps(fit,indent=2),encoding='utf-8')
    print('CONDITIONAL_ALIGNMENT','bias min/max',float(bias.min()),float(bias.max()),flush=True)
    counts={n:metrics.empty(3) for n in ('original','range_height_alignment')}
    bias=torch.tensor(bias,device=a.device,dtype=torch.float32)
    for i,key in enumerate(tc['keys']):
        r=b.by_key[tuple(key)];raw={'history_poses':np.asarray(r['history_poses'])};preds={n:[] for n in counts}
        with np.load(target/f'logits_{i:03d}.npz',allow_pickle=False) as z:values=z['logits']
        for j,h in enumerate((1,3,5)):
            x=torch.as_tensor(values[j],device=a.device)
            ids=bins(raw,r['future_poses'],b.pcfg.grid,h,a.device);preds['original'].append(x.argmax(0).byte().cpu().numpy())
            adjusted=x.clone();adjusted[:17]+=bias[ids][None]
            preds['range_height_alignment'].append(adjusted.argmax(0).byte().cpu().numpy())
        gt=b.targets(r)
        for n in counts:
            for h in range(3):metrics.add(counts[n],h,preds[n][h],gt[h])
        if (i+1)%16==0:print('CONDITIONAL_SCORE',i+1,{n:metrics.report(v,(1.,2.,3.),moving=False)['average'] for n,v in counts.items()},flush=True)
    result=dict(status='complete',windows=len(tc['keys']),fit=fit,metrics={n:metrics.report(v,(1.,2.,3.),moving=False) for n,v in counts.items()},
        scope='DEV64 diagnosis only; not full-VAL verification; no optimizer/label fitting')
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');b.close()


if __name__=='__main__':main()
