"""Fixed200 unlabeled STC GN320 adaptation; scene-disjoint held32 paired check."""
import argparse
import json
from pathlib import Path
import signal
import sys
import threading
import numpy as np
import torch
from tools.joint_occ_experiments import run as base
from tools.fpn_unet_stc_age.run import select_source, load_frozen
from tools.fpn_unet_eval.common import DenseInputs, FPNPredictor
from tools.final_semantic_evaluation.sources import window
from tools.final_semantic_evaluation import metrics
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.joint_training_recovery import snapshot_checkpoint, save_resume_checkpoint
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from . import core

PROTOCOL = 'full_FPN_UNet_unlabeled_STC_GN320_fit32_held32_fixed200_v1'
NAMES = ('baseline/transport', 'baseline/semantic_refine', 'norm320/semantic_refine')


def summary(result):
    lines = ['===== FROZEN FULL FPN+U-NET / UNLABELED STC GN320 =====', 'status='+result['status'],
        f'adaptation_updates={result["update"]}/200; trainable_scalars=320; base_checkpoint_updates=0',
        'FIRST32 scenes: STC historical static-consensus CE fitting only; LAST32 scenes: held-out full-grid scoring.',
        'Only 10 U-Net GroupNorm affine pairs; FPN/WM/convolutions frozen. Original checkpoint/default GT route unchanged.',
        'GT future ego CONDITIONED, no GT occupancy/masks/annotations for fitting; NOT zero-adaptation inference.',
        'No age weighting, KTA changes, posterior bias, norm192 weights or future GT fitting.',
        f'held_windows={len(result["keys"])}/32; route mIoU IoU MovingMicro']
    for name, report in result['reports'].items():
        row = report['average']; fmt = lambda v: 'NA' if v is None else f'{v:.6f}'
        lines.append(name+' '+ ' '.join(fmt(row[k]) for k in ('mIoU','IoU','MovingMicro')))
    if result['reports']:
        old = result['reports'][NAMES[1]]['average']; new = result['reports'][NAMES[2]]['average']
        delta = {k: None if old[k] is None or new[k] is None else new[k]-old[k] for k in ('mIoU','IoU','MovingMicro')}
        lines.append('norm320_minus_baseline='+json.dumps(delta))
    lines.append('Fixed final step200; no best-checkpoint/threshold search, retry, expansion or promotion. Quality time NOT FPS.')
    return '\n'.join(lines)+'\n'


def restore_counts(saved, keys, voxels):
    counts = {name: metrics.empty(3) for name in NAMES}; completed = []
    if saved:
        completed = saved['keys']
        if completed != keys[:len(completed)] or len(completed) > len(keys) or set(saved['counts']) != set(NAMES):
            raise RuntimeError('evaluation resume population/routes differ')
        for name in NAMES:
            for field in counts[name]:
                value = np.asarray(saved['counts'][name][field])
                if value.shape != (3,18,18) or value.dtype.kind not in 'iu' or np.any(value < 0):
                    raise RuntimeError('invalid saved counts')
                counts[name][field][:] = value
            if np.any(counts[name]['confusion'].sum((1,2)) != len(completed)*voxels):
                raise RuntimeError('incomplete full-grid evaluation counts')
            if np.any(counts[name]['moving'] > counts[name]['confusion']): raise RuntimeError('invalid Moving counts')
    return counts, list(completed)


def main(argv=None, event=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir'); p.add_argument('--out-dir', required=True); p.add_argument('--resume', action='store_true')
    for name in ('config','dev-cache','dev-info','dataroot','stc-root'): p.add_argument('--'+name, required=True)
    p.add_argument('--device',default='cuda'); p.add_argument('--cpu-workers',type=int,default=4)
    a = p.parse_args(argv); event = event or threading.Event()
    if not 1 <= a.cpu_workers <= 4: raise ValueError('workers1..4 required')
    if a.device.startswith('cuda') and torch.cuda.device_count() != 1: raise RuntimeError('expose ONE GPU')
    torch.set_num_threads(1); torch.manual_seed(21)
    repo = Path(__file__).resolve().parents[2]; out = Path(a.out_dir).resolve()
    if a.resume and not a.source_dir: a.source_dir = str(Path(json.loads((out/'contract.json').read_text())['source']).parent)
    source = select_source(a.source_dir,repo/'outputs/p0_f9_shared_history'); sha = base.sha256(source)
    if out == source.parent or source.parent in out.parents: raise ValueError('separate output outside source run required')
    lock = out.parent/'.fpn_stc_norm_locks'/base.stable_json_fingerprint(str(out)); lock.mkdir(parents=True,exist_ok=True)
    with evaluation_lock(lock):
        if not a.resume:
            out.mkdir(parents=True,exist_ok=False); snapshot_checkpoint(source,out/'frozen_epoch2.pt')
        frozen = out/'frozen_epoch2.pt'
        if base.sha256(frozen) != sha: raise RuntimeError('source checkpoint changed')
        c, body = load_frozen(frozen,a.device)
        for name in ('config','dev_cache','dev_info'):
            if base.sha256(getattr(a,name)) != c['data'][name]: raise RuntimeError('original metadata differs: '+name)
        for name, digest in c['implementation'].items():
            if base.sha256(repo/name) != digest: raise RuntimeError('original implementation differs: '+name)
        fit_keys, held_keys = core.split_keys(c['eval_keys'])
        _, records = base.load_cache(a.dev_cache)
        jobs = {tuple(k): window(r) for k,r in zip(c['eval_keys'],base.align_records(records,c['eval_keys']))}
        inputs = DenseInputs(a.dataroot,a.dev_info,stc_root=a.stc_root,cache_mib=256)
        inventory = inputs.preflight(list(jobs.values()),'stc')  # stat only, not future label reads
        original_inventory = {r[0]:tuple(r[1:]) for r in inventory}
        for path,size,stamp in c['STC_inventory']:
            if original_inventory.get(path) != (size,stamp): raise RuntimeError('original STC history files differ')
        files = [*Path(__file__).parent.glob('*.py'),Path(__file__).parent/'run.sh',
            repo/'tools/fpn_unet_stc_age/run.py',repo/'tools/fpn_unet_eval/common.py']
        contract = dict(protocol=PROTOCOL,source=str(source),source_sha256=sha,source_contract=c,
            fit_keys=fit_keys,held_keys=held_keys,steps=core.STEPS,lr=core.LR,parameters=core.PARAMETERS,
            objective='historical_static_consensus_CE_h0.5s_occupied3_free4_cap512',inventory=inventory,
            implementation={f.relative_to(repo).as_posix():base.sha256(f) for f in files},
            runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,device=a.device,
                cpu_workers=a.cpu_workers,dataroot=str(Path(a.dataroot).resolve()),stc_root=str(Path(a.stc_root).resolve())))
        if a.resume and json.loads((out/'contract.json').read_text()) != contract: raise RuntimeError('resume contract differs')
        base.write_json(out/'contract.json',contract)
        pcfg = base.make_prepare_config(base.load_runtime_config(a.config,c.get('overrides')))
        predictor = FPNPredictor(body,pcfg,a.device,workers=a.cpu_workers)
        original = tensor_state_fingerprint(body.state_dict()); patch_path = out/'norm_last.pt'
        fitted = torch.load(patch_path,map_location='cpu',weights_only=False) if a.resume and patch_path.exists() else None
        if fitted and fitted['contract'] != contract: raise RuntimeError('norm resume source/config differs')
        result = dict(status='running',update=fitted['update'] if fitted else 0,keys=[],reports={},contract=contract)
        counts, completed = restore_counts(None,held_keys,int(np.prod(pcfg.grid.shape_hwd)))
        def publish():
            result['keys'] = completed
            result['reports'] = {n:metrics.report(v,(1.,2.,3.)) for n,v in counts.items()} if completed else {}
            base.write_json(out/'evaluation.json',result)
            (out/'summary.txt').write_text(summary(result),encoding='utf-8')
        def save_fit(state):
            state = dict(state,contract=contract); save_resume_checkpoint(patch_path,state)
            result['update'] = state['update']; publish()
            last = state['losses'][-1] if state['losses'] else {}
            print('STC_NORM_FIT '+json.dumps(dict(update=state['update'],**{k:v for k,v in last.items() if k!='update'})),flush=True)
        try:
            if not fitted or fitted['update'] < core.STEPS:
                examples = []
                # Original frozen upstream only; NO targets(), moving() or annotation reads.
                for i,key in enumerate(fit_keys):
                    if event.is_set(): raise InterruptedError('fitting preparation interrupted; no optimizer update')
                    job = jobs[tuple(key)]; rec,raw,times = inputs.history(job,'stc')
                    row = predictor.prepare(rec,raw,times=times)
                    with torch.no_grad():
                        fw = predictor.forward(row,poses=np.asarray(inputs.poses(job)))
                        labels = core.consensus(raw,fw['poses'][0],pcfg.grid,a.device)
                        examples.append(core.cache_example(fw,labels))
                    del fw,labels,row,raw
                    if (i+1)%8 == 0: print(f'STC_NORM_PREP {i+1}/32; ONLY h0.5 prior/memory cached in RAM',flush=True)
                fitted = core.fit(body,examples,resumed=fitted,stopped=event.is_set,publish=save_fit,steps=core.STEPS)
                del examples
            result['update'] = fitted['update']
            if result['update'] != core.STEPS or event.is_set(): raise InterruptedError('complete optimizer update saved')
            patch = fitted['norm']
            eval_path = out/'counts.json'
            saved = json.loads(eval_path.read_text()) if a.resume and eval_path.exists() else None
            if saved and (saved['contract'] != contract or saved['patch_sha256'] != base.sha256(patch_path)):
                raise RuntimeError('evaluation patch/contract differs')
            counts, completed = restore_counts(saved,held_keys,int(np.prod(pcfg.grid.shape_hwd)))
            def save_counts():
                base.write_json(eval_path,dict(contract=contract,patch_sha256=base.sha256(patch_path),keys=completed,
                    counts={n:{k:v.tolist() for k,v in f.items()} for n,f in counts.items()}))
                publish()
            for i,key in enumerate(held_keys):
                if i < len(completed): continue
                if event.is_set(): raise InterruptedError('complete evaluation window saved')
                job = jobs[tuple(key)]; rec,raw,times = inputs.history(job,'stc'); row = predictor.prepare(rec,raw,times=times)
                with torch.no_grad():
                    fw = predictor.forward(row,poses=np.asarray(inputs.poses(job)))
                    with core.using_norm(body,patch):
                        adapted = body.columns(fw['memory'],fw['prior'],range(6),fw['shape'])
                    predictions = {n:v.argmax(1).reshape(6,*fw['shape']).byte().cpu().numpy() for n,v in
                        zip(NAMES,(fw['prior'],fw['logits'],adapted))}
                del fw,adapted,row,raw
                # All original/adapted predictions completed BEFORE future GT scoring.
                gt = inputs.targets(job,(1,3,5)); moving = inputs.moving(job,(1,3,5),(1.,2.,3.),pcfg.grid,predictor.provider.workers)
                delta = {n:metrics.empty(3) for n in NAMES}
                for name in NAMES:
                    for h,at in enumerate((1,3,5)): metrics.add(delta[name],h,predictions[name][at],gt[h],moving[h])
                for name in NAMES:
                    for field in counts[name]: counts[name][field] += delta[name][field]
                completed.append(key); save_counts()
                if len(completed)%8 == 0: print(f'STC_NORM_HELD {len(completed)}/32',flush=True)
            result['status'] = 'complete'; return 0
        except InterruptedError:
            result['status'] = 'stopped_at_complete_update_or_window'; return 130
        except Exception as error:
            result.update(status='failed',error=repr(error)); raise
        finally:
            predictor.close()
            if base.sha256(source) != sha or tensor_state_fingerprint(body.state_dict()) != original:
                raise RuntimeError('original checkpoint/model changed')
            publish(); print(summary(result),flush=True)


if __name__ == '__main__':
    stopped = threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda *_:stopped.set())
    sys.exit(main(event=stopped) or 0)
