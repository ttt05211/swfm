#!/usr/bin/env python3
"""Read-only interim full-joint evaluation; snapshot last/epoch/final candidate."""
import sys
from pathlib import Path
if __package__ in (None, ''): sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import time
import torch

from real_motion.joint_causal_columns import FULL_PROTOCOLS
from real_motion.column_runtime_pipeline import CachedColumnSource
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint
from tools.real_motion.joint_column_full_common import EvaluationJointColumnProvider as FullJointColumnProvider
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from tools.real_motion.causal_column_common import evaluate_columns
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import load_manifest, align_records, sha256
from tools.real_motion.train_p0_f9_causal_columns import record_keys
from tools.real_motion.train_p0_f9_v18_xy_trajectory import DEV64_FP
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256,write_json


def evaluation_keys(population, dev64, dev512, full_keys, train_keys):
    """Explicit populations only; normalize JSON keys, never take a cache prefix."""
    if population not in ('dev64', 'dev512', 'full4369'): raise ValueError('invalid population')
    normalize = lambda keys: tuple((str(s), str(t)) for s, t in keys)
    frozen = normalize(dev512)
    chosen = normalize(dev64) if population == 'dev64' else frozen
    if population == 'full4369':
        chosen = normalize(full_keys)
        if len(chosen) != 4369 or len(set(chosen)) != 4369:
            raise RuntimeError('explicit full4369 requires exactly 4369 unique cache records')
        if set(frozen)-set(chosen): raise RuntimeError('frozen dev512 is not contained in full4369')
    if {s for s, _ in chosen} & {str(s) for s, _ in train_keys}: raise RuntimeError('TRAIN/dev scene overlap')
    return chosen


def summary_text(result):
    row=result['reports']['all']; ref=row['reference_metrics']['frozen_E14']
    lines=['===== INTERIM FULL LOCAL JOINT (DIAGNOSTIC ONLY) =====',
        f"population: {result['population']} / {row['windows']} windows / {row['scenes']} scenes",
        f"update: {result['attempted_updates']}  completed_epochs: {result['cursor_epoch']}  batch_cursor: {result['cursor_batch']}",
        f"history_frames: {result['history_frames']}; future_frames: 6",
        f"thresholds: {result['thresholds']}; source: {result['threshold_source']}",
        f"frozen_E14_mIoU: {ref['mIoU']:.6f} (legacy SIX histories; not a matched four-history budget)"]
    items={'transport':row['baseline'],**{k:row['variants'][k]['metrics'] for k in ('generation','refine','joint')}}
    for name,m in items.items():
        lines.append(f"{name}: mIoU={m['mIoU']:.6f} dMiOU_vs_E14={m['mIoU']-ref['mIoU']:+.6f} "
            f"dMiOU_vs_transport={m['mIoU']-row['baseline']['mIoU']:+.6f} "
            f"dMovingMicro_vs_E14={m['MovingMicro']-ref['MovingMicro']:+.6f}")
    for h in ('1.0','2.0','3.0'):
        m,r=items['joint']['per_horizon'][h],ref['per_horizon'][h]
        lines.append(f"{h}s joint_vs_E14 dMiOU={m['mIoU']-r['mIoU']:+.6f} dMovingMicro={m['MovingMicro']-r['MovingMicro']:+.6f}")
    lines += ['joint addition/removal: '+str(row['variants']['joint']['quality']),
        'joint scenes: '+str(row['variants']['joint']['scene_delta']),
        f"seconds: {result['seconds']:.2f}",f"checkpoint_snapshot: {result['snapshot']}",
        'No optimizer/RNG changes, TRAIN recalibration, dev-selected checkpoint, promotion or source checkpoint writes.']
    return '\n'.join(lines)+'\n'


def main(stop_event=None):
    p=argparse.ArgumentParser(description=__doc__);add_config_args(p)
    for k in ('checkpoint','dev-cache','population-manifest','base-checkpoint','dataroot','dev-info','out-dir'):
        p.add_argument('--'+k,required=True)
    p.add_argument('--population',choices=('dev64','dev512','full4369'),default='dev64')
    p.add_argument('--device',default='cuda');p.add_argument('--cpu-workers',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--column-feature-backend',choices=('cpu','gpu'),default='gpu')
    a=p.parse_args();out=Path(a.out_dir);started=time.perf_counter()
    if out.exists():p.error('NEW evaluation output required; never overwrite training or another evaluation')
    for k in ('config','checkpoint','dev_cache','population_manifest','base_checkpoint','dev_info'):
        if not str(getattr(a,k) or '').strip() or not Path(getattr(a,k)).is_file():p.error('missing '+k)
    if not Path(a.dataroot).is_dir() or min(a.cpu_workers,a.batch_size)<1:p.error('invalid paths/budgets')
    device=torch.device(a.device)
    if device.type == 'cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):raise RuntimeError('CUDA/BF16 required')
    torch.set_num_threads(1)
    from real_motion.native_column_cpu import backend_name,prepare_native
    if backend_name() == 'native':prepare_native()
    out.mkdir(parents=True);snapshot=out/'checkpoint_snapshot.pt'
    digest=snapshot_checkpoint(a.checkpoint,snapshot)
    cfg=load_runtime_config(a.config,a.override);pcfg=make_prepare_config(cfg)
    ck,joint=load_joint(snapshot,device,reference_sha=CLEAN_SHA256,config_sha=stable_json_fingerprint(cfg),allow_diagnostic=True)
    if ck['protocol'] not in FULL_PROTOCOLS:raise RuntimeError('interim evaluator requires FULL Local protocol')
    if not ck.get('prior_completed',True):raise RuntimeError('TRAIN prior is incomplete; no training update is ready to evaluate')
    if sha256(a.base_checkpoint) != CLEAN_SHA256:raise RuntimeError('reference E14 changed')
    if sha256(a.dev_info) != ck['info_fingerprints']['dev'] or sha256(a.dev_cache) != ck['cache_fingerprints']['dev']:
        raise RuntimeError('dev info/cache provenance differs from training')
    manifest,dev64,_=load_manifest(a.population_manifest)
    if (manifest['selected_key_fingerprint'] != DEV64_FP or manifest['manifest_fingerprint'] != ck['dev_manifest_fingerprint']
            or tuple(map(tuple,manifest['parent_keys'])) != tuple(map(tuple,ck['dev_keys']))):
        raise RuntimeError('frozen dev64/dev512 identity/order changed')
    _,all_records=load_cache(a.dev_cache);full_keys=record_keys(all_records)
    chosen=evaluation_keys(a.population,dev64,manifest['parent_keys'],full_keys,ck['train_keys'])
    records=align_records(all_records,chosen);del all_records
    gates=tuple(ck['thresholds']) if ck['checkpoint_role'] == 'calibrated_candidate' else (.5,.5,None)
    joint.eval()
    provider=FullJointColumnProvider(a.base_checkpoint,CLEAN_SHA256,pcfg,device,a.cpu_workers,joint,None)
    provider.reference_enabled=True
    source=CachedColumnSource(NuScenesWindowSource(a.dataroot,info_pkl=a.dev_info,verbose=False),256)
    # No shared persistent geometry writer during an independent evaluation.
    # The immutable checkpoint snapshot can be evaluated even if last.pt rotates.
    with (out/'progress.jsonl').open('x',encoding='utf-8') as log:
        import json
        def progress(row):log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
        try:
            report=evaluate_columns(provider,source,records,joint.columns,gates,progress=progress,
                batch_size=a.batch_size,dev64_keys=dev64 if a.population != 'dev64' else None,
                diagnostic_thresholds=None,stop_event=stop_event,feature_backend=a.column_feature_backend)
        except InterruptedError:
            write_json(out/'evaluation_status.json',{'status':'interrupted','source_checkpoint_unchanged':True,
                'snapshot':str(snapshot),'snapshot_sha256':digest,'population':a.population})
            print('Evaluation interrupted; training checkpoint unchanged; no incomplete metrics marked complete.',flush=True)
            return 130
    result={'status':'complete','source_checkpoint':str(Path(a.checkpoint).resolve()),'snapshot':str(snapshot.resolve()),
        'snapshot_sha256':digest,'population':a.population,'history_frames':ck['model_configs']['motion']['history_frames'],
        'attempted_updates':ck['attempted_updates'],'cursor_epoch':ck['cursor_epoch'],'cursor_batch':ck['cursor_batch'],
        'thresholds':gates,'threshold_source':'original_TRAIN_calibrated' if ck['checkpoint_role'] == 'calibrated_candidate' else 'fixed_monitor_0.5_0.5_REMOVE_off',
        'checkpoint_screen_pass_unchanged':ck['screen_pass'],'seconds':time.perf_counter()-started,'reports':report}
    result.update(column_feature_backend=a.column_feature_backend, batch_size=a.batch_size,
        population_key_fingerprint=stable_json_fingerprint(chosen))
    if sha256(snapshot) != digest:raise RuntimeError('immutable evaluation snapshot changed')
    write_json(out/'evaluation.json',result);(out/'summary.txt').write_text(summary_text(result),encoding='utf-8')
    print(summary_text(result),flush=True);return 0


if __name__ == '__main__':
    import signal
    from threading import Event
    event=Event()
    def stop(signum,frame):event.set()
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    sys.exit(main(event))
