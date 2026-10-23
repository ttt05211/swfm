"""One frozen STC-only same64 paired check; no training, search or promotion."""
import argparse
import json
from pathlib import Path
import signal
import sys
import threading
import numpy as np
import torch
from tools.joint_occ_experiments import run as base
from tools.fpn_unet_full.run import PROTOCOL as TRAIN_PROTOCOL
from tools.fpn_unet_full.model import checkpoint_model
from tools.fpn_unet_eval.common import DenseInputs, FPNPredictor
from tools.final_semantic_evaluation.sources import window
from tools.final_semantic_evaluation import metrics
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from .processing import candidate, HALF_LIFE_SECONDS

PROTOCOL = 'frozen_full_FPN_UNet_STC_same64_fixed_history_age_only_v1'
NAMES = ('baseline/transport', 'baseline/semantic_refine', 'age/semantic_refine')


def select_source(explicit, root):
    def valid(folder):
        value = json.loads((folder/'results.json').read_text(encoding='utf-8'))
        return (value.get('status') == 'complete' and value.get('contract', {}).get('protocol') == TRAIN_PROTOCOL
            and value.get('epoch') == 2 and (folder/'epoch_0002.pt').is_file())
    if explicit:
        folder = Path(explicit).resolve()
        if not valid(folder): raise RuntimeError('completed FULL FPN+UNet TRAINx2 directory required')
        return folder/'epoch_0002.pt'
    choices = []
    for path in Path(root).glob('fpn_unet_full_2ep_*/results.json'):
        try:
            if valid(path.parent): choices.append(path.parent.resolve())
        except (ValueError, KeyError): continue
    if len(choices) != 1: raise RuntimeError('set FPN_UNET_FULL_DIR; candidates='+str(choices))
    return choices[0]/'epoch_0002.pt'


def load_frozen(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=False); c = saved['contract']
    if (saved.get('protocol') != TRAIN_PROTOCOL or saved.get('epoch') != 2 or saved.get('batch') != 0 or
            saved.get('updates') != c.get('total_updates') or c.get('epochs') != 2 or
            len(c.get('train_keys', [])) != 20430 or len(c.get('eval_keys', [])) != 64 or
            c.get('known') != 'TRAIN_OCC_STC_dense_valid'):
        raise RuntimeError('final FULL TRAIN20430x2 FPN+UNet checkpoint required')
    return c, checkpoint_model(saved, device)


def restore(saved, contract, keys, voxels):
    counts = {n: metrics.empty(3) for n in NAMES}; completed = []
    if saved:
        completed = saved['keys']
        if saved['contract'] != contract or completed != keys[:len(completed)] or len(completed) > len(keys):
            raise RuntimeError('resume identity/population differs')
        if set(saved['counts']) != set(NAMES): raise RuntimeError('saved routes differ')
        for name in NAMES:
            for field in ('confusion', 'moving'):
                value = np.asarray(saved['counts'][name][field])
                if value.shape != (3,18,18) or value.dtype.kind not in 'iu' or np.any(value < 0):
                    raise RuntimeError('invalid integer counts')
                counts[name][field][:] = value
            if np.any(counts[name]['confusion'].sum((1,2)) != len(completed)*voxels):
                raise RuntimeError('incomplete full-grid window counts')
            if np.any(counts[name]['moving'] > counts[name]['confusion']): raise RuntimeError('invalid Moving support')
    return counts, list(completed)


def summary(result):
    lines = ['===== FROZEN FULL FPN+U-NET / STC HISTORY MEMORY AGE ONLY =====', 'status='+result['status'],
        f'windows={len(result["keys"])}/64; scientific_updates=0; half_life_s={HALF_LIFE_SECONDS}',
        'Same weights/Encoder/WM/KTA/Strong/Transport Prior. ONLY historical Memory fusion weights change.',
        'Nominal old->new weights=[1/8,1/4,1/2,1]; actual history seconds used. Numerator AND denominator weighted.',
        'No training/norm adaptation/calibration/occupancy smoothing/GT masks/GT pose alignment.',
        'STC-only candidate, NOT silently enabled for GT inputs. Same64, full-grid 1/2/3s; GT future ego conditioned.',
        'route                           mIoU       IoU MovingMicro']
    for name, report in result['reports'].items():
        row = report['average']; fmt = lambda v: 'NA' if v is None else f'{v:.6f}'
        lines.append(f'{name:29s} '+ ' '.join(f'{fmt(row[k]):>10s}' for k in ('mIoU','IoU','MovingMicro')))
    if result['reports']:
        old = result['reports']['baseline/semantic_refine']['average']; new = result['reports']['age/semantic_refine']['average']
        delta = {k:None if old[k] is None or new[k] is None else new[k]-old[k] for k in ('mIoU','IoU','MovingMicro')}
        lines.append('age_minus_baseline='+json.dumps(delta))
    lines.append('One fixed candidate; all results retained. No automatic selection, further tuning, full evaluation or deployment. NOT FPS.')
    return '\n'.join(lines)+'\n'


def main(argv=None, event=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir'); p.add_argument('--out-dir', required=True); p.add_argument('--resume', action='store_true')
    for name in ('config', 'dev-cache', 'dev-info', 'dataroot', 'stc-root'): p.add_argument('--'+name, required=True)
    p.add_argument('--device', default='cuda'); p.add_argument('--cpu-workers', type=int, default=4)
    a = p.parse_args(argv); event = event or threading.Event(); repo = Path(__file__).resolve().parents[2]; out = Path(a.out_dir).resolve()
    if not 1 <= a.cpu_workers <= 4: raise ValueError('workers1..4 required')
    if a.device.startswith('cuda') and torch.cuda.device_count() != 1: raise RuntimeError('expose ONE GPU')
    torch.set_num_threads(1)
    if a.resume and not a.source_dir: a.source_dir = str(Path(json.loads((out/'contract.json').read_text())['source']).parent)
    source = select_source(a.source_dir, repo/'outputs/p0_f9_shared_history'); source_sha = base.sha256(source)
    if out == source.parent or source.parent in out.parents: raise ValueError('independent output outside source run required')
    lock = out.parent/'.fpn_stc_age_locks'/base.stable_json_fingerprint(str(out)); lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(lock):
        if not a.resume:
            out.mkdir(parents=True, exist_ok=False); snapshot_checkpoint(source, out/'frozen_epoch2.pt')
        frozen = out/'frozen_epoch2.pt'
        if base.sha256(frozen) != source_sha: raise RuntimeError('source changed')
        c, body = load_frozen(frozen, a.device)
        for name in ('config','dev_cache','dev_info'):
            if base.sha256(getattr(a,name)) != c['data'][name]: raise RuntimeError('original metadata differs: '+name)
        for name, sha in c['implementation'].items():
            if base.sha256(repo/name) != sha: raise RuntimeError('original implementation changed: '+name)
        _, records = base.load_cache(a.dev_cache); selected = base.align_records(records, c['eval_keys'])
        keys = [list(k) for k in base.record_keys(selected)]; jobs = [window(r) for r in selected]
        if len(keys) != 64 or len(set(map(tuple, keys))) != 64: raise RuntimeError('exact same unique64 required')
        inputs = DenseInputs(a.dataroot, a.dev_info, stc_root=a.stc_root, cache_mib=256)
        inventory = inputs.preflight(jobs,'stc')
        existing = {row[0]:tuple(row[1:]) for row in inventory}
        for path,size,stamp in c['STC_inventory']:
            if existing.get(path) != (size,stamp): raise RuntimeError('original STC historical data differs')
        implementation = {f.relative_to(repo).as_posix():base.sha256(f) for f in
            [*Path(__file__).parent.glob('*.py'),Path(__file__).parent/'run.sh',repo/'tools/fpn_unet_eval/common.py']}
        contract = dict(protocol=PROTOCOL,source=str(source),source_sha256=source_sha,source_contract=c,keys=keys,
            inventory=inventory,half_life_s=HALF_LIFE_SECONDS,implementation=implementation,
            runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,device=a.device,cpu_workers=a.cpu_workers,
                dataroot=str(Path(a.dataroot).resolve()),stc_root=str(Path(a.stc_root).resolve())))
        if a.resume and json.loads((out/'contract.json').read_text()) != contract: raise RuntimeError('resume same candidate/data only')
        base.write_json(out/'contract.json',contract)
        pcfg = base.make_prepare_config(base.load_runtime_config(a.config,c.get('overrides')))
        predictor = FPNPredictor(body,pcfg,a.device,workers=a.cpu_workers)
        before = tensor_state_fingerprint(body.state_dict()); state_path = out/'state.json'
        state = json.loads(state_path.read_text()) if a.resume and state_path.exists() else None
        counts, completed = restore(state,contract,keys,int(np.prod(pcfg.grid.shape_hwd)))
        result = dict(status='running',contract=contract,keys=completed,reports={},scientific_updates=0)
        def publish():
            base.write_json(state_path,dict(contract=contract,keys=completed,
                counts={n:{k:v.tolist() for k,v in fields.items()} for n,fields in counts.items()}))
            result['reports'] = {n:metrics.report(v,(1.,2.,3.)) for n,v in counts.items()}
            base.write_json(out/'evaluation.json',result); (out/'summary.txt').write_text(summary(result),encoding='utf-8')
        try:
            with torch.no_grad():
                for i,job in enumerate(jobs):
                    if i < len(completed): continue
                    if event.is_set(): raise InterruptedError('complete window saved')
                    rec,raw,times = inputs.history(job,'stc'); row = predictor.prepare(rec,raw,times=times)
                    fw = predictor.forward(row,poses=inputs.poses(job))
                    logits, weights = candidate(body,predictor.provider,row,fw)
                    predictions = {n:v.argmax(1).reshape(6,*fw['shape']).byte().cpu().numpy() for n,v in
                        (('baseline/transport',fw['prior']),('baseline/semantic_refine',fw['logits']),('age/semantic_refine',logits))}
                    del fw,logits
                    # ONLY after both six-frame forecasts: GT occupancy/support.
                    target = inputs.targets(job,(1,3,5)); moving = inputs.moving(job,(1,3,5),(1.,2.,3.),pcfg.grid,predictor.provider.workers)
                    delta = {n:metrics.empty(3) for n in NAMES}
                    for name in NAMES:
                        for h,at in enumerate((1,3,5)): metrics.add(delta[name],h,predictions[name][at],target[h],moving[h])
                    for name in NAMES:
                        for field in counts[name]: counts[name][field] += delta[name][field]
                    completed.append(keys[i])
                    if len(completed)%8 == 0: publish(); print(f'STC_AGE {len(completed)}/64 weights={weights.tolist()}',flush=True)
            result['status']='complete'; return 0
        except InterruptedError:
            result['status']='stopped_at_complete_window'; return 130
        except Exception as error:
            result.update(status='failed',error=repr(error)); raise
        finally:
            predictor.close()
            if base.sha256(source) != source_sha or tensor_state_fingerprint(body.state_dict()) != before:
                raise RuntimeError('original checkpoint/model changed')
            publish(); print(summary(result),flush=True)


if __name__=='__main__':
    event=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:event.set())
    sys.exit(main(event=event) or 0)
