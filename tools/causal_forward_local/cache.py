"""Lossless, latest-model-only research inputs; labels enter AFTER all six predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile
import numpy as np
import torch
from tools.causal_forward_all.common import Epoch2Predictor
from tools.final_semantic_evaluation import metrics
from tools.stc_overnight.bundle import Bundle
from .portable import load_weights

AT=(1,3,5)


def write(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');temp.replace(path)


def save_arrays(path,**values):
    path=Path(path);temp=path.with_suffix('.tmp')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for key,value in values.items():
            with z.open(key+'.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,np.asarray(value),allow_pickle=False)
    temp.replace(path)


def split_name(i):return 'fit32' if i<32 else 'screen16' if i<48 else 'confirmation16'


def jobs(bundle,contract):
    ref=[bundle.by_key[tuple(k)] for k in contract['eval_keys']]
    if len(ref)!=64 or len({r['scene_name'] for r in ref})!=64 or any(r['role']!='reference64' for r in ref):
        raise ValueError('exact original reference64 scenes required')
    source=[r for r in bundle.records if r['role']=='clean_train']
    if len(source)!=32 or {r['scene_name'] for r in source}&{r['scene_name'] for r in ref}:raise ValueError('source TRAIN32 must be scene-disjoint')
    return [(f'stc_{i:03d}',r,'stc',split_name(i)) for i,r in enumerate(ref)]+[
        (f'gt_{i:03d}',r,'gt',split_name(i)) for i,r in enumerate(ref)]+[
        (f'source_{i:03d}',r,'gt','source_TRAIN32') for i,r in enumerate(source)]


def baseline_report(out,job_list):
    totals={}
    for name,_,kind,split in job_list:
        path=out/'counts'/(name+'.json')
        if not path.exists():continue
        v=json.loads(path.read_text(encoding='utf-8'))
        for part in (split,'all64' if split!='source_TRAIN32' else 'source_TRAIN32_all'):
            key=kind+'/'+part;entry=totals.setdefault(key,dict(windows=0,counts={n:metrics.empty(3) for n in v['confusion']}))
            entry['windows']+=1
            for route,counts in v['confusion'].items():entry['counts'][route]['confusion']+=np.asarray(counts,np.int64)
    return {k:dict(windows=v['windows'],reports={n:metrics.report(c,(1.,2.,3.),moving=False) for n,c in v['counts'].items()}) for k,v in totals.items()}


@torch.no_grad()
def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights',default='../_received/forward_adaln_epoch2_local.zip');ap.add_argument('--bundle',default='../stc_overnight_bundle.zip')
    ap.add_argument('--out',required=True);ap.add_argument('--device',default='cuda');ap.add_argument('--limit',type=int,default=160)
    a=ap.parse_args(argv);torch.set_num_threads(1);torch.manual_seed(21)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    body,c,info=load_weights(a.weights,a.device);b=Bundle(a.bundle,cache_mib=32)
    if b.pcfg.grid!=info['prepare_config'].grid or b.pcfg.frame_dt_s!=info['prepare_config'].frame_dt_s:raise ValueError('data grid/time differs')
    planned=jobs(b,c)
    contract=dict(protocol='latest_forward_epoch2_lossless_local_research_v1',weights_archive_sha256=info['archive_sha256'],
        source_checkpoint_sha256=info['manifest']['source_checkpoint_sha256'],source_checkpoint=info['manifest']['source_checkpoint'],
        reference64_keys=c['eval_keys'],source_keys=[[r['scene_name'],r['t0_token']] for _,r,_,s in planned if s=='source_TRAIN32'],
        split='first32 fit, next16 screen, last16 confirmation; scene-disjoint, previously inspected DEV',
        future_ego='GT; SAME Decoder and renderer',known='dense_valid',full_grid=True,
        Moving_available=False,precision='original BF16 network / FP64 geometry; lossless FP32 research caches',
        original_model_unchanged=True,torch=str(torch.__version__),GPU=torch.cuda.get_device_name() if a.device=='cuda' else 'CPU')
    path=out/'contract.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=contract:raise ValueError('research cache source/contract changed')
    write(path,contract);(out/'inputs').mkdir(exist_ok=True);(out/'counts').mkdir(exist_ok=True)
    p=Epoch2Predictor(body,info['prepare_config'],a.device,2);start=time.monotonic()
    try:
        for i,(name,r,kind,split) in enumerate(planned[:a.limit]):
            path=out/'inputs'/(name+'.npz');report=out/'counts'/(name+'.json')
            if path.exists() and report.exists():continue
            rec,raw,times=b.history(r,kind,'dense_valid');row=p.prepare(rec,raw,times=times)
            fw=p.forward(row,poses=np.asarray(r['future_poses']))
            pred=p.dense(fw)  # SIX predictions completed before any future occupancy read.
            labels=b.targets(r,tuple(range(6)));counts={n:metrics.empty(3) for n in pred}
            for route,values in pred.items():
                for h,at in enumerate(AT):metrics.add(counts[route],h,values[at],labels[at])
            # FULL six-horizon inputs allow later norm-only adaptation/readouts without
            # running the frozen upstream again. Never use these learned caches for FPS.
            save_arrays(path,prior=fw['prior'].float().cpu().numpy(),memory=fw['memory'].float().cpu().numpy(),
                logits=fw['logits'][list(AT)].float().cpu().numpy(),labels=labels,
                shape=np.asarray(fw['shape']),key=np.asarray([r['scene_name'],r['t0_token']]),
                source_sha256=np.asarray(contract['source_checkpoint_sha256']),split=np.asarray(split))
            write(report,dict(key=[r['scene_name'],r['t0_token']],confusion={n:v['confusion'].tolist() for n,v in counts.items()}))
            del fw,pred,row,raw,labels,counts
            if (i+1)%4==0 or i==0:
                reports=baseline_report(out,planned);write(out/'baseline.json',dict(status='complete' if i+1==160 else 'running',completed=i+1,reports=reports,contract=contract))
                print('LATEST_CACHE',i+1,'/160',name,'seconds',round(time.monotonic()-start,1),
                    {k:v['reports']['semantic_refine']['average'] for k,v in reports.items() if k.endswith('/all64')},flush=True)
        reports=baseline_report(out,planned);write(out/'baseline.json',dict(status='complete' if len(list((out/'counts').glob('*.json')))==160 else 'partial',
            completed=len(list((out/'counts').glob('*.json'))),reports=reports,contract=contract))
    finally:p.close();b.close()


if __name__=='__main__':main()
