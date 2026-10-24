"""User-supplied latest weight ZIP + existing real64 replay; no optimization."""
import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import signal
import sys
import threading
import time
import zipfile
import numpy as np
import torch
from tools.joint_occ_experiments import run as base
from tools.fpn_unet_full.model import checkpoint_model
from tools.fpn_unet_full.run import PROTOCOL as TRAIN_PROTOCOL
from tools.fpn_unet_all.run import AllPredictor,PROTOCOL as EVAL_PROTOCOL
from tools.stc_overnight.bundle import Bundle
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from . import calibration as cal
from . import motion
from . import memory
from . import evidence
from . import wide

PROTOCOL='local_full_FPN_STC_age_global_gain_free_bias_fit32_held32_v1'


def load_model(path,device):
    with zipfile.ZipFile(path) as z:
        if len(z.namelist())!=len(set(z.namelist())):raise ValueError('duplicate ZIP member')
        c=json.loads(z.read('contract.json')); evaluation=json.loads(z.read('evaluation.json'))
        payload=z.read('epoch_0002.pt')
    if c['protocol']!=EVAL_PROTOCOL or evaluation['status']!='complete' or evaluation['contract']!=c:
        raise ValueError('completed latest full4369 source artifact required')
    if hashlib.sha256(payload).hexdigest()!=c['source_sha256']:raise ValueError('source SHA differs')
    saved=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
    sc=saved['contract']
    if (saved['protocol']!=TRAIN_PROTOCOL or sc!=c['source_contract'] or saved['epoch']!=2 or saved['batch']!=0 or
        saved['updates']!=sc['total_updates'] or len(sc['train_keys'])!=20430 or len(sc['eval_keys'])!=64 or sc['parameters']!=2520442):
        raise ValueError('original FULL TRAIN20430x2 final2520442-parameter body required')
    body=checkpoint_model(saved,device)
    if sum(p.numel() for p in body.parameters())!=2520442:raise ValueError('unexpected architecture')
    return body,c,evaluation['reports']['original64']['stc_gt_age_semantic_refine']


def finish(out,contract,rows,source64):
    study=contract.get('study','logits');module=dict(logits=cal,motion=motion,memory=memory,evidence=evidence,logits_wide=wide)[study]
    fields=dict(logits=('residual_gain','free_bias'),motion=('motion_mode','motion_scale','free_bias'),
        memory=('memory_mode','memory_strength','free_bias'),evidence=('free_memory_weight','free_bias'),
        logits_wide=('residual_gain','free_bias'))[study]
    names=[module.name(c) for c in module.CANDIDATES]
    parameters=[dict(zip(fields,c)) for c in module.CANDIDATES]
    result=cal.summarize(rows,names=names,parameters=parameters);result.update(status='complete',protocol=PROTOCOL,contract=contract,
        source_same64=source64,Moving_available=False,scientific_optimizer_updates=0,
        caveat='Validation-label calibration research, NOT no-calibration inference. ALL64 includes selection scenes; NOT ALL4369.')
    measured=result['all64_descriptive_includes_selection'][names[0]]['average']
    result['baseline64_minus_server']={k:measured[k]-source64['average'][k] for k in ('mIoU','IoU')}
    base.write_json(out/'results.json',result)
    lines=['===== LOCAL FROZEN FPN / SIMPLE PROCESSING =====','study='+contract.get('study','logits'),
        'Latest FULL TRAINx2, STC age0.5, original real64. No network update / GT masks / alignment.',
        'FIRST32 future GT used ONLY to select shared scalars; LAST32 scene-disjoint check.',
        'No Moving annotations in portable bundle; dynamic semantics are NOT a Moving substitute.',
        'fit32 candidate mIoU IoU']
    for n,r in result['fit32'].items():lines.append(f'{n} {r["average"]["mIoU"]:.6f} {r["average"]["IoU"]:.6f}')
    lines.append('selected='+json.dumps(result['selected']))
    for group in ('held32','all64_descriptive_includes_selection'):
        for n,r in result[group].items():lines.append(f'{group} {n} {r["average"]["mIoU"]:.6f} {r["average"]["IoU"]:.6f}')
    lines+=['held32_delta='+json.dumps(result['held32_delta']),
        'baseline64_minus_server='+json.dumps(result['baseline64_minus_server']),result['caveat']]
    text='\n'.join(lines)+'\n';(out/'summary.txt').write_text(text,encoding='utf-8');print(text,flush=True)
    return result


def main(argv=None,event=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights-zip',required=True);ap.add_argument('--bundle',default='../stc_overnight_bundle.zip')
    ap.add_argument('--out-dir',required=True);ap.add_argument('--device',default='cuda');ap.add_argument('--resume',action='store_true')
    ap.add_argument('--study',choices=('logits','motion','memory','evidence','logits_wide'),default='logits')
    a=ap.parse_args(argv);event=event or threading.Event();torch.set_num_threads(1);torch.manual_seed(21)
    out=Path(a.out_dir).resolve();repo=Path(__file__).resolve().parents[2]
    if not a.resume:out.mkdir(parents=True,exist_ok=False)
    body,source,source64=load_model(a.weights_zip,a.device)
    # Group only execution, never the supervised future population. GN is
    # per-sample; H1 bounds memory on RTX3050 4GiB. No byte identity claim.
    body.columns.horizon_chunk=1
    bundle=Bundle(a.bundle,cache_mib=128)
    try:
        keys=source['source_contract']['eval_keys'];rows=[bundle.by_key[tuple(k)] for k in keys]
        if len({r['scene_name'] for r in rows})!=64:raise ValueError('64 distinct scenes required')
        if {r['scene_name'] for r in rows}&{k[0] for k in source['source_contract']['train_keys']}:raise ValueError('TRAIN overlap')
        pcfg=base.make_prepare_config(base.load_runtime_config(repo/'configs/real_motion_occfm.yaml',source['source_contract'].get('overrides')))
        if asdict(pcfg.grid)!=asdict(bundle.pcfg.grid):raise ValueError('portable grid differs')
        files=[*Path(__file__).parent.glob('*.py')]
        module=dict(logits=cal,motion=motion,memory=memory,evidence=evidence,logits_wide=wide)[a.study]
        candidates=module.CANDIDATES;names=[module.name(c) for c in candidates]
        contract=dict(protocol=PROTOCOL,study=a.study,source_sha256=source['source_sha256'],weights_zip_sha256=base.sha256(a.weights_zip),
            bundle_sha256=base.sha256(a.bundle),keys=keys,candidates=list(map(list,candidates)),
            horizon_chunk=1,age_half_life_s=.5,grid=asdict(pcfg.grid),
            implementation={p.name:base.sha256(p) for p in files},
            runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,device=a.device))
        contract=json.loads(json.dumps(contract))
        cp=out/'contract.json'
        if a.resume and json.loads(cp.read_text())!=contract:raise ValueError('resume same source/data/candidates only')
        base.write_json(cp,contract)
        predictor=AllPredictor(body,pcfg,a.device,workers=2);predictor.kind='stc'
        before=tensor_state_fingerprint(body.state_dict());all_counts=[];started=time.perf_counter()
        try:
            for i,row in enumerate(rows):
                if event.is_set():raise InterruptedError('complete window persisted')
                path=out/f'counts_{i:02d}.npz'
                if a.resume and path.exists():
                    with np.load(path,allow_pickle=False) as z:
                        if str(z['key'])!=json.dumps(keys[i]):raise ValueError('window identity differs')
                        value=np.array(z['counts'],copy=True)
                else:
                    rec,raw,times=bundle.history(row,'stc','dense_valid')
                    prepared=predictor.prepare(rec,raw,times=times)
                    with torch.no_grad():
                        fw=predictor.forward(prepared,poses=np.asarray(row['future_poses']))
                        if a.study!='logits':predictions=module.predict(body,predictor.provider,prepared,fw)
                        else:
                            prior=body.columns.prior_logits(fw['prior'])
                            predictions=cal.predict(fw['logits'],prior,fw['shape']);del prior
                    del fw,prepared,raw
                    # Future semantic labels are read AFTER every prediction.
                    targets=bundle.targets(row);value=cal.counts(predictions,targets,names=names)
                    np.savez_compressed(path,key=json.dumps(keys[i]),counts=value)
                    del predictions,targets
                if value.shape!=(len(candidates),3,18,18) or value.dtype.kind not in 'iu' or np.any(value<0):
                    raise ValueError('invalid window integer counts')
                if np.any(value.sum((-1,-2))!=np.prod(pcfg.grid.shape_hwd)):raise ValueError('incomplete full-grid counts')
                all_counts.append(value)
                if (i+1)%4==0 or i==0:
                    print(f'LOCAL_STC {i+1}/64 elapsed={time.perf_counter()-started:.1f}s',flush=True)
                    base.write_json(out/'progress.json',dict(status='running',windows=i+1,total=64))
            finish(out,contract,np.stack(all_counts),source64)
            base.write_json(out/'progress.json',dict(status='complete',windows=64,total=64));return 0
        except InterruptedError:
            base.write_json(out/'progress.json',dict(status='stopped',windows=len(all_counts),total=64));return 130
        finally:
            predictor.close()
            if tensor_state_fingerprint(body.state_dict())!=before:raise RuntimeError('original model changed')
    finally:bundle.close()


if __name__=='__main__':
    event=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:event.set())
    sys.exit(main(event=event) or 0)
