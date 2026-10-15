"""Frozen SAME-model baseline vs fixed KTA+recency candidate. No training."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import threading
import time
import numpy as np
import torch
from tools.joint_occ_experiments import run as base
from tools.final_semantic_evaluation import metrics
from tools.final_semantic_evaluation.predictor import Predictor
from tools.final_semantic_evaluation.sources import NuscInputs, window
from tools.shared_source_track.adapter import SharedTrackPredictor
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from .bundle import Bundle
from .replay import prepare_forecast, iter_forecasts, multiframe_velocity


def forecast(predictor,row,poses):
    context=prepare_forecast(predictor,row,poses)
    result={name:[] for name in ('transport','semantic_refine')}
    for hs,fw in iter_forecasts(context):
        for name,key in [('transport','prior'),('semantic_refine','logits')]:
            result[name].append(fw[key].argmax(1).reshape(len(hs),*fw['shape']).byte().cpu().numpy())
        del fw
    return {name:np.concatenate(value) for name,value in result.items()}


def candidate(row,provider,times):
    value,audit=multiframe_velocity(row,provider,times)
    # Do not mutate the baseline's prepared evidence arrays.
    causal=dict(value[3][0]);reads=dict(causal['memory_reads'])
    weights=np.asarray([.125,.25,.5,1.],np.float32)
    reads['static_valid']=reads['static_valid']*weights[:,None,None,None]
    reads['coverage']=reads['coverage']*weights[reads['frame']]
    causal['memory_reads']=reads
    return (value[0],value[1],value[2],(causal,value[3][1])),audit


def summary(result):
    lines=['===== FROZEN STC / MATCHED KTA + RECENCY VERIFICATION =====',
        'status='+result['status'],f"population={result['contract']['population']}; windows={result['windows']}",
        'SAME frozen body. Original pair identities/registrations retained; no internal ego head.',
        'Candidate: half original-pair + multiframe LS velocity; pooled history weights [1/8,1/4,1/2,1].',
        'GT future ego conditioned. All SIX readouts before GT label reads. FULL GRID, no mask/alignment.',
        'NO training, threshold/coefficient sweep, GT history feature oracle, supervised VAL adapter, or automatic promotion.',
        'route                       mIoU        IoU']
    for name,value in result.get('metrics',{}).items():
        r=value['average'];fmt=lambda v:'NA' if v is None else f'{v:.6f}'
        lines.append(f"{name:26s} {fmt(r['mIoU']):>10s} {fmt(r['IoU']):>10s}")
    if 'delta' in result:lines.append('candidate_minus_baseline='+json.dumps(result['delta']))
    lines+=['Original source weights/caches unchanged. Quality evaluation, NOT FPS.',
        'DEV population is NOT full-VAL proof. Repeated development is declared.']
    return '\n'.join(lines)+'\n'


def main(argv=None,event=None):
    ap=argparse.ArgumentParser();ap.add_argument('--bundle',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--model',choices=['dense_epoch2','clean1000'],default='clean1000')
    ap.add_argument('--population',choices=['all','dev64'],default='all');ap.add_argument('--portable-subset',action='store_true')
    ap.add_argument('--dev-cache');ap.add_argument('--dev-info');ap.add_argument('--dataroot');ap.add_argument('--stc-root')
    ap.add_argument('--device',default='cuda');ap.add_argument('--resume',action='store_true')
    a=ap.parse_args(argv);event=event or threading.Event();torch.set_num_threads(1);torch.manual_seed(21)
    b=Bundle(a.bundle);body,saved,policy=b.load_model(a.model,a.device)
    if a.portable_subset:
        if a.population!='dev64':raise ValueError('portable small bundle cannot evaluate ALL VAL')
        records=[r for r in b.records if r['role']=='reference64'];inputs=None
    else:
        if not all([a.dev_cache,a.dev_info,a.dataroot,a.stc_root]):raise ValueError('server full data paths required')
        for name in ('dev_cache','dev_info'):
            expected=saved['source_contract']['data'][name]
            if base.sha256(getattr(a,name))!=expected:raise RuntimeError('original metadata differs: '+name)
        _,records=base.load_cache(a.dev_cache)
        if len(records)!=4369 or len({r['scene_name'] for r in records})!=150:raise ValueError('complete VAL4369/150 required')
        if a.population=='dev64':records=base.align_records(records,b.manifest['reference64_keys'])
        inputs=NuscInputs(a.dataroot,a.dev_info,stc_root=a.stc_root,cache_mib=256)
        inputs.preflight([window(r) for r in records],'stc')
    keys=[[r['scene_name'],r['t0_token']] for r in records]
    if len(set(map(tuple,keys)))!=len(keys) or (a.population=='dev64' and len(keys)!=64):raise ValueError('population identity')
    out=Path(a.out).resolve();contract=dict(protocol='frozen_same_body_STC_KTA_recency_v1',model=a.model,
        manifest_sha256=hashlib.sha256(b.zip.read('manifest.json')).hexdigest(),source_sha256=saved['source_sha256'],
        source_policy=policy,population=a.population,keys=keys,portable_subset=a.portable_subset,
        metadata={n:base.sha256(getattr(a,n)) for n in ('dev_cache','dev_info') } if not a.portable_subset else {},
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,device=a.device),
        known='STC dense-valid; NO GT visibility',metric_mask='none',future_GT_ego_conditioned=True,
        implementation={name:base.sha256(Path(__file__).parent/name) for name in ('verify_all.py','replay.py','bundle.py')})
    lock=out.parent/'.stc_verify_locks'/base.stable_json_fingerprint(str(out));lock.mkdir(parents=True,exist_ok=True)
    with evaluation_lock(lock):
        if not a.resume:out.mkdir(parents=True,exist_ok=False)
        elif json.loads((out/'contract.json').read_text(encoding='utf-8'))!=contract:raise RuntimeError('resume contract differs')
        base.write_json(out/'contract.json',contract)
        counts={n:metrics.empty(3) for n in ['baseline/transport','baseline/semantic_refine','candidate/transport','candidate/semantic_refine']}
        state=dict(keys=[],counts={n:{k:v.tolist() for k,v in c.items()} for n,c in counts.items()},contract=contract)
        if a.resume:
            state=json.loads((out/'state.json').read_text(encoding='utf-8'))
            if state['contract']!=contract or state['keys']!=keys[:len(state['keys'])]:raise RuntimeError('saved prefix differs')
            counts={n:{k:np.asarray(v,np.int64) for k,v in c.items()} for n,c in state['counts'].items()}
            expected=len(state['keys'])*int(np.prod(b.pcfg.grid.shape_hwd))
            if any(np.any(c['confusion'].sum((1,2))!=expected) for c in counts.values()):raise ValueError('incomplete saved windows')
        p=(SharedTrackPredictor if policy=='shared_pinned' else Predictor)(body,b.pcfg,a.device,workers=2)
        before=tensor_state_fingerprint(body.state_dict());tic=time.monotonic()
        result=dict(status='running',windows=len(state['keys']),contract=contract,scientific_updates=0)
        def publish():
            result['windows']=len(state['keys']);result['metrics']={n:metrics.report(v,(1.,2.,3.),moving=False) for n,v in counts.items()}
            left=result['metrics']['baseline/semantic_refine']['average'];right=result['metrics']['candidate/semantic_refine']['average']
            if left['mIoU'] is not None:result['delta']={k:right[k]-left[k] for k in ('mIoU','IoU')}
            base.write_json(out/'results.json',result);(out/'summary.txt').write_text(summary(result),encoding='utf-8')
            base.write_json(out/'state.json',dict(contract=contract,keys=state['keys'],counts={n:{k:v.tolist() for k,v in c.items()} for n,c in counts.items()}))
        try:
            for i,r in enumerate(records):
                if i<len(state['keys']):continue
                if event.is_set():raise InterruptedError('stopped at complete paired window')
                if inputs is None:rec,raw,times=b.history(r);poses=r['future_poses']
                else:
                    job=window(r);rec,raw,times=inputs.history(job,'stc');poses=inputs.poses(job)
                row=p.prepare(rec,raw,times=times)
                original=forecast(p,row,poses)
                modified,audit=candidate(row,p.provider,times)
                changed=forecast(p,modified,poses)
                # Both full six-frame forecasts precede label access.
                targets=b.targets(r) if inputs is None else inputs.targets(job,(1,3,5))
                for prefix,predictions in [('baseline',original),('candidate',changed)]:
                    for route,value in predictions.items():
                        for h,at in enumerate((1,3,5)):metrics.add(counts[prefix+'/'+route],h,value[at],targets[h])
                state['keys'].append(keys[i])
                if len(state['keys'])%8==0:publish()
                if len(state['keys'])%16==0:print('STC_VERIFY',len(state['keys']),'/',len(keys),'seconds',round(time.monotonic()-tic,1),flush=True)
            result['status']='complete'
            average=metrics.report(counts['candidate/semantic_refine'],(1.,2.,3.),moving=False)['average']
            result['full_target_verified']=a.population=='all' and average['IoU']>28.77 and average['mIoU']>18.97
        except InterruptedError:result['status']='stopped_at_complete_paired_window'
        except Exception:
            result['status']='failed_original_unchanged';raise
        finally:
            p.close();b.close()
            if tensor_state_fingerprint(body.state_dict())!=before:raise RuntimeError('frozen body changed')
            publish();print(summary(result),flush=True)
    return 0


if __name__=='__main__':
    event=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:event.set())
    main(event=event)
