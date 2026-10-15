"""Source/target posterior marginal alignment. NO target labels used for fitting.

Relative distribution alignment is inspired by AdaMatch (arXiv:2106.04732).
Here it is an inference-only occupancy study, NOT that paper's full algorithm.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tools.final_semantic_evaluation import metrics
from .bundle import Bundle


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--target',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--bundle',default='../stc_overnight_bundle.zip')
    ap.add_argument('--device',default='cuda');a=ap.parse_args(argv);torch.set_num_threads(1)
    source,target,out=Path(a.source),Path(a.target),Path(a.out);out.mkdir(parents=True,exist_ok=True)
    sc=json.loads((source/'contract.json').read_text(encoding='utf-8'));tc=json.loads((target/'contract.json').read_text(encoding='utf-8'))
    if sc['model_sha256']!=tc['model_sha256'] or sc.get('role')!='clean_train' or tc['modality']!='stc':raise ValueError('TRAIN-source / same-body STC target required')
    sums=[]
    for f in sorted(source.glob('window_*.json')):
        # Source predictive posterior ONLY; no source confusion/target labels.
        sums.append(np.asarray(json.loads(f.read_text(encoding='utf-8'))['posterior_sum']))
    sp=np.sum(sums,axis=(0,1));sp=sp/sp.sum()
    target_sum=np.zeros(18,np.float64)
    for i in range(len(tc['keys'])):
        with np.load(target/f'logits_{i:03d}.npz',allow_pickle=False) as z:values=z['logits']
        for h in range(3):
            x=torch.as_tensor(values[h],device=a.device);target_sum+=x.softmax(0).double().sum((1,2,3)).cpu().numpy()
    tp=target_sum/target_sum.sum()
    full=np.log(np.maximum(sp,1e-6)/np.maximum(tp,1e-6))
    binary=np.full(18,np.log((1-sp[17])/(1-tp[17])));binary[17]=np.log(sp[17]/tp[17])
    biases=dict(original=np.zeros(18),binary_relative_alignment=binary,semantic_relative_alignment=full)
    fit=dict(protocol='source_GT_TRAIN32_vs_UNLABELED_STC_posterior_relative_alignment_v1',
        source_contract=sc,target_contract=tc,source_posterior=sp.tolist(),target_posterior=tp.tolist(),
        biases={n:b.tolist() for n,b in biases.items()},target_GT_labels_used_for_fitting=False,
        scope='transductive posterior calibration; not target-domain-information-free; class-prior assumption may fail',
        coefficients='posterior ratios only; no threshold/coefficient sweep')
    # Fitting is complete and written BEFORE Bundle/GT scoring exists.
    (out/'fit.json').write_text(json.dumps(fit,indent=2),encoding='utf-8')
    print('POSTERIOR_ALIGNMENT','sourceOcc',1-sp[17],'targetOcc',1-tp[17],'binaryBias',float(binary[0]-binary[17]),flush=True)
    b=Bundle(a.bundle);counts={n:metrics.empty(3) for n in biases}
    for i,key in enumerate(tc['keys']):
        with np.load(target/f'logits_{i:03d}.npz',allow_pickle=False) as z:values=z['logits']
        preds={n:[] for n in counts}
        for h in range(3):
            x=torch.as_tensor(values[h],device=a.device)
            for name,bias in biases.items():
                value=x+torch.as_tensor(bias,device=a.device,dtype=x.dtype)[:,None,None,None]
                preds[name].append(value.argmax(0).byte().cpu().numpy())
        gt=b.targets(b.by_key[tuple(key)])
        for n in counts:
            for h in range(3):metrics.add(counts[n],h,preds[n][h],gt[h])
        if (i+1)%16==0:print('DISTRIBUTION',i+1,{n:metrics.report(v,(1.,2.,3.),moving=False)['average'] for n,v in counts.items()},flush=True)
    result=dict(status='complete',windows=len(tc['keys']),fit=fit,
        metrics={n:metrics.report(v,(1.,2.,3.),moving=False) for n,v in counts.items()},
        scope='same inspected DEV64, no full VAL verification; zero optimizer steps')
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');b.close()


if __name__=='__main__':main()
