"""Frozen FPN/AdaLN same64 OCC+STC association A/B. Never create an optimizer."""
import argparse
from dataclasses import asdict
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
from tools.fpn_unet_eval.common import DenseInputs
from tools.final_semantic_evaluation import metrics
from tools.final_semantic_evaluation.sources import window
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from tools.stc_overnight.bundle import Bundle
from .adapter import TrackPredictor
from .track import PROTOCOL as TRACK_PROTOCOL
from tools.causal_forward_train.calibration import CALIBRATION,predict as calibrated_readout,delta as calibration_delta

PROTOCOL = 'frozen_FPN_or_AdaLN_forward_association_same64_dense_GTego_v1'
NAMES = tuple(f'{kind}/{arm}/{route}' for kind in ('occ', 'stc') for arm in ('baseline', 'forward')
              for route in ('transport', 'semantic_refine'))


def route_names(contract):
    calibration = contract.get('output_calibration')
    if calibration is not None and calibration != CALIBRATION:raise ValueError('unknown fixed calibration')
    extra = tuple(f'{kind}/{arm}/semantic_refine_free_minus1' for kind in ('occ','stc')
        for arm in ('baseline','forward')) if calibration else ()
    return NAMES+extra


def config_identity(path, expected=None):
    import hashlib
    raw = Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    unix = hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest()
    if expected and expected not in (actual, unix): raise ValueError('original runtime configuration differs')
    return dict(file_sha256=actual,LF_sha256=unix,source_sha256=expected)


def load_source(checkpoint=None, weights_zip=None, device='cpu'):
    from tools.fpn_unet import model as fpn, run as fpn_run
    from tools.fpn_unet_full import model as full, run as full_run
    from tools.fpn_adaln_zero import model as ada, run as ada_run
    from tools.fpn_adaln_zero_full import model as ada_full, run as ada_full_run
    if bool(checkpoint) == bool(weights_zip): raise ValueError('one explicit checkpoint OR weight ZIP required')
    path = Path(checkpoint or weights_zip).resolve()
    container_sha = base.sha256(path)
    if weights_zip:
        with zipfile.ZipFile(path) as z:
            if len(z.namelist()) != len(set(z.namelist())): raise ValueError('duplicate ZIP member')
            receipt = json.loads(z.read('contract.json'))
            payload = z.read('epoch_0002.pt')
        import hashlib
        sha = hashlib.sha256(payload).hexdigest()
        if sha != receipt['source_sha256']: raise ValueError('weight ZIP source SHA differs')
        saved = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
        if json.loads(json.dumps(saved['contract'])) != receipt['source_contract']:
            raise ValueError('weight ZIP source contract differs')
    else:
        sha = container_sha
        saved = torch.load(path, map_location='cpu', weights_only=False)
    c = saved['contract']
    loaders = {fpn_run.PROTOCOL:(fpn.checkpoint_model, 1, 4086),
               full_run.PROTOCOL:(full.checkpoint_model, 2, 20430),
               ada_run.PROTOCOL:(ada.checkpoint_model, 1, 4086),
               ada_full_run.PROTOCOL:(ada_full.checkpoint_model, 2, 20430)}
    if saved.get('protocol') not in loaders: raise ValueError('only explicit current FPN/AdaLN protocols supported')
    loader, epochs, windows = loaders[saved['protocol']]
    if (c.get('protocol') != saved['protocol'] or c.get('epochs') != epochs or saved.get('epoch') != epochs or
            saved.get('batch') != 0 or saved.get('updates') != c.get('total_updates') or
            c.get('known') != 'TRAIN_OCC_STC_dense_valid' or c.get('source_policy') != 'shared_pinned' or
            c.get('registration') != 'none' or set(saved.get('models', {})) != {'gt'} or
            len(c.get('train_keys', [])) != windows or len(set(map(tuple, c['train_keys']))) != windows or
            len(c.get('eval_keys', [])) != 64 or len(set(map(tuple, c['eval_keys']))) != 64):
        raise ValueError('completed unmodified dense/no-ICP/pinned current source required')
    repo = Path(__file__).resolve().parents[2]
    training_only_drift = {}
    unused_training_files = {'tools/real_motion/joint_training_distributed.py',
                             'tools/real_motion/joint_training_recovery.py'}
    for name, digest in c.get('implementation', {}).items():
        p = (repo/name).resolve()
        if repo not in p.parents:
            raise ValueError('source path outside repository: '+name)
        current_sha = base.sha256(p)
        if current_sha != digest and name in unused_training_files:
            # Imported indirectly by existing model loaders, but no distributed
            # trainer / optimizer / training checkpoint recovery is executed.
            training_only_drift[name] = dict(recorded=digest,current=current_sha,executed=False)
        elif current_sha != digest:
            raise ValueError('original source implementation changed: '+name)
    model = loader(saved, device).eval().requires_grad_(False)
    return model, c, dict(path=str(path), file_sha256=container_sha, checkpoint_sha256=sha,
                         source_protocol=saved['protocol'], model_variant=c['model']['variant'],
                         unused_training_source_drift=training_only_drift)


def summarize(result):
    lines = ['===== FROZEN FORWARD SOURCE ASSOCIATION / SAME WEIGHTS =====', 'status='+result['status'],
        f'windows={len(result["keys"])}/{len(result["contract"]["keys"])}; scientific_updates=0',
        'source='+result['contract']['source']['model_variant'],
        'FOUR histories -> SIX readouts; dense-valid; GT future ego; full-grid 1/2/3s metrics.',
        'Baseline pinned last pair vs forward CV/greedy + unknown mutual-NN + gap reconnection.',
        'BOTH rebuild own KTA, source reads, history memory and fresh Strong. Same t0 components/order.',
        'No shapes/Kalman/ICP/class detector gates/future-label matching/model updates/threshold search.',
        'STC age0.5 in BOTH arms='+str(result['contract']['stc_age']),
        'route mIoU IoU MovingMicro']
    reports = {}
    for name in route_names(result['contract']):
        reports[name] = metrics.report({k:np.asarray(v, np.int64) for k,v in result['counts'][name].items()},
                                      (1., 2., 3.), moving=result['moving_available'])
        row = reports[name]['average']
        lines.append(name+' '+' '.join('NA' if row.get(k) is None else f'{row[k]:.6f}'
                                      for k in ('mIoU', 'IoU', 'MovingMicro')))
    result['reports'] = reports
    delta = {}
    for kind in ('occ', 'stc'):
        for route in ('transport', 'semantic_refine'):
            a, b = (reports[f'{kind}/{arm}/{route}']['average'] for arm in ('baseline', 'forward'))
            delta[kind+'/'+route] = {k:b[k]-a[k] for k in a if a[k] is not None and b[k] is not None}
    result['forward_minus_baseline'] = delta
    if result['contract'].get('output_calibration'):
        calibration = {}
        for kind in ('occ','stc'):
            for arm in ('baseline','forward'):
                a,b = (reports[f'{kind}/{arm}/{route}'] for route in
                    ('semantic_refine','semantic_refine_free_minus1'))
                calibration[kind+'/'+arm] = dict(average=calibration_delta(b['average'],a['average']),
                    per_horizon={h:calibration_delta(b['per_horizon'][h],a['per_horizon'][h]) for h in a['per_horizon']})
        result['free_minus1_minus_raw'] = calibration
        lines += ['FIXED inherited FREE-1 from previous GT-labeled FPN development; NOT refitted on AdaLN.',
            'OCC/STC share the SAME coefficient; raw and calibrated readouts share ONE forward; training unchanged.',
            'free_minus1_minus_raw='+json.dumps(calibration)]
    lines += ['forward_minus_baseline='+json.dumps(delta),
        'History links/KTA changed intentionally; this is a frozen-input-policy study, NOT numerical parity.',
        'Portable replay has NO true Moving annotations; no dynamic-class surrogate is reported.',
        'Repeated development population, NOT independent test, raw-camera FPS or automatic promotion.']
    if result.get('error'): lines.append('error='+result['error'])
    return '\n'.join(lines)+'\n'


def restore(result, contract, voxels, moving_available):
    if (result['contract'] != contract or result['keys'] != contract['keys'][:len(result['keys'])] or
            len(result['keys']) > len(contract['keys']) or set(result['counts']) != set(route_names(contract)) or
            result['moving_available'] != moving_available or len(result['audits']) != len(result['keys'])):
        raise ValueError('resume same source/data/ordered population and complete windows only')
    for fields in result['counts'].values():
        if set(fields) != {'confusion', 'moving'}: raise ValueError('invalid metric fields')
        for field, values in fields.items():
            a = np.asarray(values)
            if a.shape != (3,18,18) or a.dtype.kind not in 'iu' or np.any(a < 0): raise ValueError('invalid integer counts')
            if field == 'confusion' and np.any(a.sum((1,2)) != len(result['keys'])*voxels):
                raise ValueError('incomplete full-grid accumulated counts')
        if np.any(np.asarray(fields['moving']) > np.asarray(fields['confusion'])):
            raise ValueError('Moving support cannot exceed full grid')
    return result


def main(argv=None, event=None):
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--checkpoint'); src.add_argument('--weights-zip')
    ap.add_argument('--bundle', help='portable real64 data ONLY; bundled old weights are NEVER loaded')
    for key in ('dataroot', 'dev-info', 'dev-cache', 'stc-root'): ap.add_argument('--'+key)
    ap.add_argument('--out-dir', required=True); ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, choices=(16,64), default=64)
    ap.add_argument('--horizon-chunk', type=int, choices=(1,2,3,6), default=6)
    ap.add_argument('--cpu-workers', type=int, choices=(1,2,3,4), default=2)
    ap.add_argument('--stc-age', action='store_true', help='same existing age0.5 memory weighting in BOTH STC arms')
    ap.add_argument('--free-minus1',action='store_true',help='extra fixed labeled-calibration readout, no fitting or extra forward')
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args(argv); event = event or threading.Event()
    torch.set_num_threads(1); torch.manual_seed(21)
    out = Path(a.out_dir).resolve(); repo = Path(__file__).resolve().parents[2]
    if not a.resume: out.mkdir(parents=True, exist_ok=False)
    model, c, receipt = load_source(a.checkpoint, a.weights_zip, a.device)
    if out == Path(receipt['path']).parent or Path(receipt['path']).parent in out.parents:
        raise ValueError('output must be independent of source weight directory')
    model.columns.horizon_chunk = a.horizon_chunk
    cfg_path = repo/'configs/real_motion_occfm.yaml'
    config_receipt = config_identity(cfg_path,c.get('data', {}).get('config'))
    pcfg = base.make_prepare_config(base.load_runtime_config(cfg_path, c.get('overrides')))
    keys = [list(k) for k in c['eval_keys'][:a.limit]]
    if {k[0] for k in keys} & {k[0] for k in c['train_keys']}: raise ValueError('TRAIN/VAL scene overlap')
    if a.bundle:
        data = Bundle(a.bundle, cache_mib=128)
        if asdict(data.pcfg.grid) != asdict(pcfg.grid): raise ValueError('portable grid differs')
        jobs = [data.by_key[tuple(k)] for k in keys]
        data_contract = dict(bundle_sha256=base.sha256(a.bundle))
        moving_available = False
    else:
        if not all((a.dataroot,a.dev_info,a.dev_cache,a.stc_root)): raise ValueError('server data arguments required without bundle')
        for name in ('dev_cache', 'dev_info'):
            if base.sha256(getattr(a,name)) != c['data'][name]: raise ValueError('source metadata differs: '+name)
        _, records = base.load_cache(a.dev_cache)
        jobs = [window(r) for r in base.align_records(records, keys)]
        data = DenseInputs(a.dataroot, a.dev_info, stc_root=a.stc_root, cache_mib=256)
        data_contract = dict(dev_info_sha256=base.sha256(a.dev_info), dev_cache_sha256=base.sha256(a.dev_cache),
            inventory={k:data.preflight(jobs,k) for k in ('occ','stc')})
        moving_available = True
    implementation = {p.relative_to(repo).as_posix():base.sha256(p) for p in Path(__file__).parent.glob('*.py')}
    if a.free_minus1:
        for name in ('tools/causal_forward_train/calibration.py','tools/fpn_stc_bias448/run.py'):
            implementation[name] = base.sha256(repo/name)
    contract = json.loads(json.dumps(dict(protocol=PROTOCOL, track_protocol=TRACK_PROTOCOL, source=receipt,
        keys=keys, source_contract=c, data=data_contract, implementation=implementation,config=config_receipt,
        stc_age=a.stc_age, known='dense_valid_both', GT_future_ego=True, horizon_chunk=a.horizon_chunk,
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,device=a.device,workers=a.cpu_workers))))
    if a.free_minus1:contract['output_calibration'] = json.loads(json.dumps(CALIBRATION))
    cp = out/'contract.json'
    if a.resume and json.loads(cp.read_text(encoding='utf-8')) != contract: raise ValueError('resume contract differs')
    base.write_json(cp,contract)
    result = dict(status='running',contract=contract,keys=[],audits=[],moving_available=moving_available,
        counts={n:{k:v.tolist() for k,v in metrics.empty(3).items()} for n in route_names(contract)},scientific_updates=0)
    if a.resume:
        result = restore(json.loads((out/'results.json').read_text(encoding='utf-8')), contract,
                         int(np.prod(pcfg.grid.shape_hwd)), moving_available)
        result['status'] = 'running'
    predictor = TrackPredictor(model,pcfg,a.device,workers=a.cpu_workers)
    before = tensor_state_fingerprint(model.state_dict())
    start = time.perf_counter()

    def publish():
        text = summarize(result)
        base.write_json(out/'results.json',result)
        (out/'summary.txt').write_text(text,encoding='utf-8')
        base.write_json(out/'progress.json',dict(status=result['status'],windows=len(result['keys']),total=len(keys),
                                               launch_seconds=time.perf_counter()-start))
    try:
        publish()
        for i in range(len(result['keys']), len(keys)):
            if event.is_set(): raise InterruptedError('stopped after complete paired window')
            job = jobs[i]; predictions = {}; audits = {}
            # Alternate execution order, but keep the reported keys fixed.
            for kind in ('occ', 'stc'):
                if a.bundle: rec, raw, times = data.history(job,'gt' if kind == 'occ' else 'stc','dense_valid')
                else: rec, raw, times = data.history(job,kind)
                rows, audit = predictor.prepare_pair(rec,raw,times=times)
                poses = np.asarray(job['future_poses'] if a.bundle else data.poses(job))
                audit['inference_seconds_not_FPS'] = {}
                for arm in (('baseline','forward') if i%2 == 0 else ('forward','baseline')):
                    begin = time.perf_counter()
                    with torch.no_grad():
                        fw = predictor.infer(rows[arm],poses,stc_age=(kind == 'stc' and a.stc_age))
                        if a.free_minus1:
                            predictions[f'{kind}/{arm}/transport'] = fw['prior'].argmax(1).reshape(6,*fw['shape']).byte().cpu().numpy()
                            for route,value in calibrated_readout(fw['logits'],fw['shape']).items():
                                predictions[f'{kind}/{arm}/{route}'] = value
                        else:
                            for route,tensor in (('transport',fw['prior']),('semantic_refine',fw['logits'])):
                                predictions[f'{kind}/{arm}/{route}'] = tensor.argmax(1).reshape(6,*fw['shape']).byte().cpu().numpy()
                    audit['inference_seconds_not_FPS'][arm] = time.perf_counter()-begin
                    del fw
                audits[kind] = audit
                del rows,raw
            # ALL arms/modalities forecast ALL SIX frames BEFORE future labels.
            target = data.targets(job) if a.bundle else data.targets(job,(1,3,5))
            moving = None if a.bundle else data.moving(job,(1,3,5),(1.,2.,3.),pcfg.grid,a.cpu_workers)
            pending_counts = {}
            for name in route_names(contract):
                counts = {k:np.asarray(v,np.int64) for k,v in result['counts'][name].items()}
                for h,index in enumerate((1,3,5)):
                    metrics.add(counts,h,predictions[name][index],target[h],None if moving is None else moving[h])
                pending_counts[name] = {k:v.tolist() for k,v in counts.items()}
            result['counts'] = pending_counts
            result['keys'].append(keys[i]); result['audits'].append(audits)
            publish()
            print(f'FORWARD_TRACK {i+1}/{len(keys)} elapsed={time.perf_counter()-start:.1f}s',flush=True)
        result['status'] = 'complete'
    except InterruptedError:
        result['status'] = 'stopped'
    except Exception as error:
        result.update(status='failed',error=str(error)); raise
    finally:
        predictor.close()
        if a.bundle: data.close()
        if tensor_state_fingerprint(model.state_dict()) != before: raise RuntimeError('frozen model changed')
        if base.sha256(receipt['path']) != receipt['file_sha256']: raise RuntimeError('source file changed')
        publish()
    print(summarize(result),flush=True)
    return 0 if result['status'] == 'complete' else 130


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda *_:event.set())
    sys.exit(main(event=event) or 0)
