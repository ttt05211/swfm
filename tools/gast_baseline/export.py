"""Single-GPU frozen GAST export with per-case resume; no full-VAL scoring."""
import argparse
import io
import json
from pathlib import Path
import signal
import threading
import time
import zipfile
import numpy as np
from .common import FORMAT,SOURCE_SHA,sha,read_json,write_json,labels,array,save_npz,verify_inventory,checked_archive,file_lock


def load_prediction(path,case,shape):
    with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    n=len(case['report_indices'])
    if set(a)!={'t0','gt','g','e'}:raise ValueError('prediction fields differ')
    for name,value in a.items():
        expected=tuple(shape) if name=='t0' else (n,*shape)
        if value.shape!=expected or value.dtype!=np.uint8 or np.any(value>17):raise ValueError('invalid completed case: '+str(path))
    return a


def compute_case(model,case,archive,key,device,*,predict=None,reader=labels):
    from .model import rollout
    predict=predict or rollout;frames=case['frames'];shape=tuple(archive['grid']['shape_hwd'])
    history=np.stack([reader(f['semantics'],shape) for f in frames[:4]])
    # Complete all 6/12 futures before reading ANY target semantic file.
    prediction=predict(model,history,frames,device)
    if prediction.dtype!=np.uint8 or prediction.shape!=(len(frames)-4,*shape) or np.any(prediction>17):raise ValueError('forecast shape/labels differ')
    report=case['report_indices'];gt=np.stack([reader(frames[4+i]['semantics'],shape) for i in report])
    with zipfile.ZipFile(archive['assets']) as source:
        ours=array(source,key,'e',(len(report),*shape))
        if key.startswith('short'):
            if not np.array_equal(array(source,key,'gt',(3,*shape)),gt) or not np.array_equal(array(source,key,'t0',shape),history[-1]):
                raise ValueError('GAST/CarryWorld original GT or t0 differ: '+key)
    return dict(t0=history[-1],gt=gt,g=prediction[report],e=ours)


def packet(work,prepared,receipt):
    work=Path(work);expected=set(prepared['cases']);found={p.stem for p in (work/'predictions').glob('*.npz')}
    if expected!=found:raise ValueError('all selected cases must complete before packaging')
    manifest=dict(format=FORMAT,status='complete',grid=prepared['grid'],receipt=receipt,
        horizon_definition='2Hz: short exact 1/2/3s; long exact 1..6s from two 3s blocks',
        long_protocol='released 3s checkpoint, 2x3s rollout; predicted histories in block2; GT ego in both',
        not_official_long_term_table=True,source_sha256=SOURCE_SHA,cases={})
    for key,case in prepared['cases'].items():
        path=work/'predictions'/f'{key}.npz';load_prediction(path,case,prepared['grid']['shape_hwd'])
        manifest['cases'][key]=dict(scene=case['original']['scene'],t0_token=case['original']['t0_token'],
            history_tokens=case['original']['history_tokens'],future_tokens=case['original']['future_tokens'],
            future_ego_poses=case['original']['future_ego_poses'],report_indices=case['report_indices'],
            arrays=['t0','gt','g','e'],sha256=sha(path))
    target=work/'comparison_assets.zip';temp=target.with_name(target.name+'.tmp')
    if target.exists():
        with zipfile.ZipFile(target) as z:
            if json.loads(z.read('manifest.json'))!=manifest:raise ValueError('completed packet receipt differs')
            import hashlib
            for key,case in manifest['cases'].items():
                if hashlib.sha256(z.read('cases/'+key+'.npz')).hexdigest()!=case['sha256']:raise ValueError('completed packet predictions differ')
        print('GAST_ASSETS_REUSED',target,flush=True);return target
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for key,case in prepared['cases'].items():
            path=work/'predictions'/f'{key}.npz'
            z.write(path,'cases/'+path.name)
        z.writestr('manifest.json',json.dumps(manifest,indent=2))
    temp.replace(target);write_json(work/'comparison_manifest.json',manifest)
    print('GAST_ASSETS_READY',target,f'{target.stat().st_size/2**20:.1f} MiB; no weights or dense logits',flush=True)
    return target


def run(a,stop,*,factory=None,compute=None):
    work=a.work.resolve();prepared=read_json(work/'prepared.json')
    if prepared['format']!='GAST_CPU_prepared_v1' or sha(prepared['assets'])!=prepared['assets_sha256']:
        raise ValueError('original CarryWorld assets changed')
    checked_archive(prepared['assets']);verify_inventory(prepared['inventory'])
    if factory is None:
        from .model import load_model,cuda_check
        if a.action=='check':cuda_check(a.device)
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('GPU phase requires ONE visible GPU; CPU preparation remains usable without GPU')
        factory=load_model
    model,evidence=factory(a.gast_root,a.weight,work,a.device)
    implementation={p.name:sha(p) for p in sorted(Path(__file__).parent.iterdir()) if p.suffix in ('.py','.json')}
    receipt=dict(official=evidence,prepared_sha256=sha(work/'prepared.json'),assets_sha256=prepared['assets_sha256'],
        adaptation_implementation=implementation,GT_ego=True,scientific_updates=0,full_VAL_evaluation=False,
        histories='real4 then predicted last4 at 1.5/2/2.5/3s',extra_postprocessing=False)
    receipt_path=work/'inference_receipt.json'
    if receipt_path.exists() and read_json(receipt_path)!=receipt:raise ValueError('frozen inference source/code changed; NEW GAST_WORK required')
    write_json(receipt_path,receipt)
    keys=list(prepared['cases']);predictions=work/'predictions';predictions.mkdir(exist_ok=True)
    if a.action=='check':
        keys=[next(k for k in keys if k.startswith('short')),next(k for k in keys if k.startswith('turn'))]
    initial=read_json(work/'progress.json') if (work/'progress.json').exists() else {}
    hashes=initial.get('completed',{});shape=prepared['grid']['shape_hwd'];compute=compute or compute_case
    if set(hashes)-set(prepared['cases']):raise ValueError('unknown completed case')
    for key,digest in hashes.items():
        path=predictions/(key+'.npz')
        if sha(path)!=digest:raise ValueError('completed output changed: '+key)
        load_prediction(path,prepared['cases'][key],shape)
    for key in keys:
        if stop.is_set():raise InterruptedError('complete cases saved; rerun SAME step and directory')
        tick=time.perf_counter();path=predictions/(key+'.npz')
        if key in hashes:continue
        if path.exists():
            # Process may have stopped between the atomic NPZ and progress write.
            load_prediction(path,prepared['cases'][key],shape)
        else:
            values=compute(model,prepared['cases'][key],prepared,key,a.device)
            save_npz(path,**values);load_prediction(path,prepared['cases'][key],shape)
        hashes[key]=sha(path);complete=len(hashes)==len(prepared['cases'])
        write_json(work/'progress.json',dict(status='complete' if complete else 'running',completed=hashes,
            windows=len(hashes),total=len(prepared['cases']),phase=a.action,updated_utc=time.time(),
            last_case_seconds=time.perf_counter()-tick))
        print(f'GAST_EXPORT {len(hashes)}/{len(prepared["cases"])} {key} {prepared["cases"][key]["original"]["scene"]}',flush=True)
    if a.action=='check':
        write_json(work/'gpu_check.json',dict(status='complete',cases=keys,receipt=receipt))
        print('GAST_CHECK_OK strict weights + native short3s + predicted-history long6s; saved cases will be reused',flush=True)
    if len(hashes)==len(prepared['cases']):packet(work,prepared,receipt)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('check','infer'))
    p.add_argument('--work',type=Path,required=True);p.add_argument('--gast-root',type=Path,required=True)
    p.add_argument('--weight',type=Path);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    a.weight=a.weight or a.work/'downloads/gast_gt.pth'
    stop=threading.Event()
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.set())
    with file_lock(a.work/'export.lock'):run(a,stop)


if __name__=='__main__':main()
