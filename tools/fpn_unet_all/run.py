"""Final FULL-TRAINx2 FPN+UNet on all VAL4369; no optimizer or FPS probe."""
import argparse
import json
import math
from pathlib import Path
import signal
import sys
import threading
import torch
from tools.joint_occ_experiments import run as base
from tools.fpn_unet_eval.common import DenseInputs, ModalInputs, FPNPredictor, merge_counts
from tools.fpn_unet_stc_age.run import select_source, load_frozen
from tools.fpn_unet_stc_age.processing import weight_history, HALF_LIFE_SECONDS
from tools.final_semantic_evaluation.engine import evaluate
from tools.final_semantic_evaluation.sources import window
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint

PROTOCOL = 'frozen_full_FPN_UNet_VAL4369_OCC_original_STC_age0.5_GTego_v1'


def select_jobs(records, original64, train_keys):
    keys = base.record_keys(records)
    if len(keys) != 4369 or len(set(keys)) != 4369 or len({k[0] for k in keys}) != 150:
        raise RuntimeError('complete unique VAL4369 / 150 scenes required; no trimmed fallback')
    original64 = [tuple(k) for k in original64]
    if len(original64) != 64 or len(set(original64)) != 64 or not set(original64) <= set(keys):
        raise RuntimeError('all original64 keys must exist, no replacements')
    if {k[0] for k in keys} & {k[0] for k in train_keys}: raise RuntimeError('TRAIN/VAL scene overlap')
    ordered = original64+sorted(set(keys)-set(original64))
    return [window(r) for r in base.align_records(records,ordered)], [list(k) for k in ordered]


class AllPredictor(FPNPredictor):
    kind = 'occ'

    @torch.no_grad()
    def forward(self, row, *, poses):
        if self.kind not in ('occ','stc'): raise ValueError('explicit OCC/STC modality required')
        if self.kind == 'stc':
            rec,raw,ego,history = row
            weighted,_ = weight_history(history,raw['_shared_source_track'].times)
            row = rec,raw,ego,weighted
        # Weight memory BEFORE the original forward, so there is only ONE
        # Encoder/WM/Prior/Memory/U-Net pass, not a baseline+candidate double run.
        return super().forward(row,poses=poses)


def summary(result):
    lines = ['===== FROZEN FULL FPN+U-NET / ALL VAL4369 =====','status='+result['status'],
        'Final FULL TRAIN20430x2 epoch2; scientific_updates=0; no norm adaptation or retraining.',
        'OCC=original model; STC=fixed history Memory age half-life0.5s ONLY; nominal weights=[1/8,1/4,1/2,1].',
        'GT future ego conditioned; FOUR histories -> SIX futures; dense-valid OCC/STC; full-grid metrics.',
        'ALL4369 / 150 VAL scenes; no planner trimming, missing-STC exclusion, visibility masks or alignment.',
        f'source={result["contract"]["source"]}; parameters={result["contract"]["source_contract"]["parameters"]}',
        'completed_windows='+json.dumps(result['progress']),
        'population route mIoU IoU MovingMicro']
    fmt = lambda v:'NA' if v is None else f'{v:.6f}'
    for group,reports in result['reports'].items():
        for route,report in reports.items():
            row = report['average']
            lines.append(group+' '+route+' '+' '.join(fmt(row[k]) for k in ('mIoU','IoU','MovingMicro')))
            if group == 'all4369':
                for horizon,m in report['per_horizon'].items():
                    lines.append(f'  {horizon}s '+' '.join(f'{k}={fmt(m[k])}' for k in ('mIoU','IoU','MovingMicro')))
    if result.get('error'): lines.append('error='+result['error'])
    lines.append('Original64 and remaining4305 integer counts SUMMED for all4369, not averages of subset scores.')
    lines.append('Quality evaluation time is NOT FPS. Source/default GT path unchanged; no automatic training or promotion.')
    return '\n'.join(lines)+'\n'


def main(argv=None,event=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir'); p.add_argument('--out-dir',required=True); p.add_argument('--resume',action='store_true')
    for n in ('config','dev-cache','dev-info','dataroot','stc-root'): p.add_argument('--'+n,required=True)
    p.add_argument('--device',default='cuda'); p.add_argument('--cpu-workers',type=int,default=4)
    a = p.parse_args(argv); event = event or threading.Event(); repo = Path(__file__).resolve().parents[2]
    if not 1 <= a.cpu_workers <= 4: raise ValueError('workers1..4 required')
    if a.device.startswith('cuda') and (torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('expose ONE BF16 GPU')
    torch.set_num_threads(1); torch.manual_seed(21); out = Path(a.out_dir).resolve()
    if a.resume and not a.source_dir:
        a.source_dir = str(Path(json.loads((out/'contract.json').read_text(encoding='utf-8'))['source']).parent)
    source = select_source(a.source_dir,repo/'outputs/p0_f9_shared_history'); sha = base.sha256(source)
    if out == source.parent or source.parent in out.parents: raise ValueError('independent output outside source run required')
    lock = out.parent/'.fpn_unet_all_locks'/base.stable_json_fingerprint(str(out)); lock.mkdir(parents=True,exist_ok=True)
    with evaluation_lock(lock):
        if not a.resume:
            out.mkdir(parents=True,exist_ok=False); snapshot_checkpoint(source,out/'frozen_epoch2.pt')
        frozen = out/'frozen_epoch2.pt'
        if base.sha256(frozen)!=sha: raise RuntimeError('source changed')
        c,body = load_frozen(frozen,a.device)
        for n in ('config','dev_cache','dev_info'):
            if base.sha256(getattr(a,n))!=c['data'][n]: raise RuntimeError('original metadata differs: '+n)
        for n,digest in c['implementation'].items():
            if base.sha256(repo/n)!=digest: raise RuntimeError('original implementation differs: '+n)
        _,records = base.load_cache(a.dev_cache); jobs,keys = select_jobs(records,c['eval_keys'],c['train_keys'])
        inputs = DenseInputs(a.dataroot,a.dev_info,stc_root=a.stc_root,cache_mib=256)
        inventory = {kind:inputs.preflight(jobs,kind) for kind in ('occ','stc')}
        current = {r[0]:tuple(r[1:]) for r in inventory['stc']}
        for path,size,stamp in c['STC_inventory']:
            if current.get(path)!=(size,stamp): raise RuntimeError('original64 STC input differs: '+path)
        files = [*Path(__file__).parent.glob('*.py'),Path(__file__).parent/'run.sh']+[
            repo/n for n in ('tools/fpn_unet_eval/common.py','tools/fpn_unet_stc_age/processing.py',
                'tools/fpn_unet_stc_age/run.py','tools/final_semantic_evaluation/engine.py',
                'tools/final_semantic_evaluation/metrics.py','tools/final_semantic_evaluation/sources.py',
                'tools/final_semantic_evaluation/predictor.py')]
        contract = dict(protocol=PROTOCOL,source=str(source),source_sha256=sha,source_contract=c,keys=keys,
            windows=4369,scenes=150,strata=dict(original64=keys[:64],remaining4305=keys[64:]),
            selection='original64_prefix_plus_ALL_sorted_remaining_no_score_filter_no_planner_trim',
            known='OCC_STC_dense_valid',metric_mask='none',STC_memory_half_life_s=HALF_LIFE_SECONDS,
            inventory=inventory,implementation={f.relative_to(repo).as_posix():base.sha256(f) for f in files},
            runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,device=a.device,cpu_workers=a.cpu_workers,
                dataroot=str(Path(a.dataroot).resolve()),stc_root=str(Path(a.stc_root).resolve())))
        if a.resume and json.loads((out/'contract.json').read_text(encoding='utf-8'))!=contract:
            raise RuntimeError('resume SAME frozen model/data/age configuration only')
        base.write_json(out/'contract.json',contract); identity = base.stable_json_fingerprint(contract)
        result = dict(status='running',contract=contract,scientific_updates=0,progress=dict(occ=0,stc=0),reports={})
        if a.resume and (out/'evaluation.json').exists():
            result = json.loads((out/'evaluation.json').read_text(encoding='utf-8'))
            if result['contract']!=contract: raise RuntimeError('evaluation identity differs')
            result['status']='running'
        pcfg = base.make_prepare_config(base.load_runtime_config(a.config,c.get('overrides')))
        predictor = AllPredictor(body,pcfg,a.device,workers=a.cpu_workers)
        before = tensor_state_fingerprint(body.state_dict())
        def publish():
            base.write_json(out/'evaluation.json',result)
            (out/'summary.txt').write_text(summary(result),encoding='utf-8')
        publish()
        try:
            with torch.no_grad():
                for kind,prefix in (('occ','occ_gt'),('stc','stc_gt_age')):
                    predictor.kind = kind; parts = []
                    for group,lo,hi in (('original64',0,64),('remaining4305',64,len(keys))):
                        path = out/f'{kind}_{group}_state.json'
                        state = json.loads(path.read_text(encoding='utf-8')) if a.resume and path.exists() else None
                        def save(value,path=path,kind=kind,lo=lo):
                            base.write_json(path,value)
                            result['progress'][kind] = lo+len(value['keys'])
                            base.write_json(out/'progress.json',dict(status='running',windows=result['progress'],total=4369))
                        print(f'FPN_ALL modality={kind} stratum={group}; all4369; STC_age={kind=="stc"}',flush=True)
                        report = evaluate(predictor,ModalInputs(inputs,kind),jobs[lo:hi],'gt',keys[lo:hi],
                            dict(protocol=PROTOCOL,identity=identity,modality=kind,stratum=group),saved=state,
                            save=save,stop_event=event)
                        result['reports'].setdefault(group,{}).update({prefix+'_'+n.split('/',1)[1]:m for n,m in report['reports'].items()})
                        parts.append(json.loads(path.read_text(encoding='utf-8'))); publish()
                    merged = merge_counts(parts,keys,math.prod(pcfg.grid.shape_hwd))
                    result['reports'].setdefault('all4369',{}).update({prefix+'_'+n.split('/',1)[1]:m for n,m in merged.items()}); publish()
            result['status']='complete'; return 0
        except InterruptedError:
            result['status']='stopped_at_complete_window'; return 130
        except Exception as error:
            result.update(status='failed_original_unchanged',error=repr(error)); raise
        finally:
            predictor.close()
            if base.sha256(source)!=sha or tensor_state_fingerprint(body.state_dict())!=before:
                raise RuntimeError('original checkpoint/model changed')
            base.write_json(out/'progress.json',dict(status=result['status'],windows=result['progress'],total=4369))
            publish(); print(summary(result),flush=True)


if __name__=='__main__':
    event = threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda *_:event.set())
    sys.exit(main(event=event) or 0)
