"""Fresh real64 replay of the EXACT frozen server CandidatePredictor."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from tools.causal_forward_stc_candidate.common import CandidatePredictor
from tools.causal_forward_stc_candidate.recipe import SOURCE_SHA256,specification
from tools.stc_overnight.bundle import Bundle
from tools.final_semantic_evaluation import metrics
from .portable import load_weights
from .cache import AT,write


@torch.no_grad()
def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--weights',default='../_received/forward_adaln_epoch2_local.zip')
    ap.add_argument('--bundle',default='../stc_overnight_bundle.zip');ap.add_argument('--out',required=True);ap.add_argument('--device',default='cuda');a=ap.parse_args(argv)
    torch.set_num_threads(1);torch.manual_seed(21);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/'counts').mkdir(exist_ok=True)
    body,c,info=load_weights(a.weights,a.device)
    if info['manifest']['source_checkpoint_sha256']!=SOURCE_SHA256:raise ValueError('exact current body source required')
    b=Bundle(a.bundle,cache_mib=16);p=CandidatePredictor(body,info['prepare_config'],a.device,2);tic=time.monotonic();totals={}
    contract=dict(protocol='exact_frozen_server_candidate_fresh_real64_verification_v1',recipe=specification(),keys=c['eval_keys'],
        source_archive_sha256=info['archive_sha256'],torch=str(torch.__version__),GPU=torch.cuda.get_device_name() if a.device=='cuda' else 'CPU',
        full_grid=True,Moving_available=False,GT_future_ego=True,GT_STC_same_model=True,
        development='same repeatedly inspected64; first32 labelled calibration; NOT ALL4369 or independent test or FPS')
    if (out/'contract.json').exists() and json.loads((out/'contract.json').read_text())!=contract:raise ValueError('new frozen verification output required')
    write(out/'contract.json',contract)
    try:
        for domain in ('stc','gt'):
            for i,key in enumerate(c['eval_keys']):
                name=f'{domain}_{i:03d}';path=out/'counts'/(name+'.json')
                if path.exists():q=json.loads(path.read_text())
                else:
                    r=b.by_key[tuple(key)];rec,raw,times=b.history(r,domain,'dense_valid');row=p.prepare(rec,raw,times=times)
                    fw=p.forward(row,poses=np.asarray(r['future_poses']));pred=p.dense(fw);gt=b.targets(r);local={n:metrics.empty(3) for n in pred}
                    for n,values in pred.items():
                        for h,at in enumerate(AT):metrics.add(local[n],h,values[at],gt[h])
                    q=dict(key=key,counts={n:v['confusion'].tolist() for n,v in local.items()});write(path,q)
                    del fw,pred,gt,row,raw,local
                for split in ('all64','fit32' if i<32 else 'held32'):
                    group=domain+'/'+split;entry=totals.setdefault(group,dict(windows=0,counts={n:metrics.empty(3) for n in q['counts']}));entry['windows']+=1
                    for n,values in q['counts'].items():entry['counts'][n]['confusion']+=np.asarray(values,np.int64)
                if (i+1)%16==0:print('EXACT_CANDIDATE_VERIFY',domain,i+1,'/64','seconds',round(time.monotonic()-tic,1),flush=True)
        reports={k:dict(windows=v['windows'],reports={n:metrics.report(q,(1.,2.,3.),moving=False) for n,q in v['counts'].items()}) for k,v in totals.items()}
        write(out/'results.json',dict(status='complete',contract=contract,reports=reports))
        for key in ('stc/all64','stc/held32','gt/all64'):print('EXACT_CANDIDATE_FINAL',key,reports[key]['reports']['semantic_refine']['average'],flush=True)
    finally:p.close();b.close()


if __name__=='__main__':main()
