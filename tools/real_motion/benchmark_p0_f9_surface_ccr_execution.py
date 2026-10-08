#!/usr/bin/env python3
"""Fixed server 20x3 paired projection/readout FPS; no quality rerun/training."""
import sys
from pathlib import Path
if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import signal
import threading
import torch
from real_motion.canonical_repair_execution import CanonicalCpuExecution
from real_motion.column_runtime_pipeline import CachedColumnSource
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.runtime_config import add_config_args,load_runtime_config,make_prepare_config
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion.run_p0_f9_shared_evidence_pilot import PilotProvider,require_cuda
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import load_manifest,align_records,sha256
from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint
from tools.real_motion.point_ccr_v18_fps_common import load_point_head
from tools.real_motion.validate_p0_f9_surface_ccr_expanded import load_surface
from tools.real_motion.surface_ccr_validation_common import paired_speed
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256,write_json
from tools.real_motion.joint_training_recovery import snapshot_checkpoint


def main(argv=None,stop_event=None):
    p=argparse.ArgumentParser(description=__doc__);add_config_args(p)
    for k in ('checkpoint','ccr-checkpoint','frozen-b-checkpoint','base-checkpoint','dev-cache',
              'population-manifest','dev-info','dataroot','out-dir'): p.add_argument('--'+k,required=True)
    p.add_argument('--device',default='cuda'); p.add_argument('--cpu-workers',type=int,default=10)
    a=p.parse_args(argv);out=Path(a.out_dir)
    if out.exists(): p.error('new output required')
    for k in ('config','checkpoint','ccr_checkpoint','frozen_b_checkpoint','base_checkpoint','dev_cache','population_manifest','dev_info'):
        if not Path(getattr(a,k)).is_file():p.error('missing '+k)
    if not Path(a.dataroot).is_dir() or not 1<=a.cpu_workers<=16:p.error('invalid dataroot/CPU budget')
    device=require_cuda(a.device);torch.set_num_threads(1);out.mkdir(parents=True)
    cfg=load_runtime_config(a.config,a.override);fp=stable_json_fingerprint(cfg)
    if sha256(a.base_checkpoint)!=CLEAN_SHA256:raise RuntimeError('Clean E14 fingerprint mismatch')
    snapshots={};digests={}
    for k in ('checkpoint','ccr_checkpoint','frozen_b_checkpoint'):
        snapshots[k]=out/(k+'.pt');digests[k]=snapshot_checkpoint(getattr(a,k),snapshots[k])
    ck,joint=load_joint(snapshots['checkpoint'],device,reference_sha=CLEAN_SHA256,config_sha=fp,allow_diagnostic=True)
    if joint.transport.config.history_frames!=4 or ck.get('cursor_epoch')!=19:
        raise RuntimeError('fixed epoch19 FOUR-history motion required')
    joint.eval().requires_grad_(False)
    manifest,keys64,_=load_manifest(a.population_manifest)
    if (manifest['manifest_fingerprint']!=ck['dev_manifest_fingerprint']
            or sha256(a.dev_cache)!=ck['cache_fingerprints']['dev']
            or sha256(a.dev_info)!=ck['info_fingerprints']['dev']): raise RuntimeError('frozen data/population mismatch')
    saved_b=torch.load(snapshots['frozen_b_checkpoint'],map_location='cpu',weights_only=False)
    baseline=load_point_head(saved_b,teacher_sha256=digests['checkpoint'],config_fingerprint=fp,
                            source_dim=joint.columns.source_dim,device=device,allow_completed_epoch_boundary=True)
    baseline.eval().requires_grad_(False);del saved_b
    saved=torch.load(snapshots['ccr_checkpoint'],map_location='cpu',weights_only=False)
    if saved['contract'].get('warm_start_head_sha256')!=digests['frozen_b_checkpoint']:
        raise RuntimeError('surface/B lineage mismatch')
    head=load_surface(saved,baseline,teacher_sha=digests['checkpoint'],config_fp=fp,
                      manifest_fp=manifest['manifest_fingerprint'],device=device);del saved
    _,records=load_cache(a.dev_cache);monitor=align_records(records,keys64);del records
    provider=PilotProvider(a.base_checkpoint,CLEAN_SHA256,make_prepare_config(cfg),device,a.cpu_workers,joint,None)
    execution=CanonicalCpuExecution('native_parallel',min(4,a.cpu_workers));provider.ccr_execution=execution
    source=CachedColumnSource(NuScenesWindowSource(a.dataroot,info_pkl=a.dev_info,verbose=False),1024)
    try:
        speed=paired_speed(provider,source,monitor,joint,head,baseline,stop_event=stop_event)
        speed['source_checkpoint_digests']=digests
        write_json(out/'speed.json',speed)
        lines=['===== SURFACE CCR EXECUTION ONLY =====','weights/support/thresholds/FPS boundary unchanged; no training or quality selection.']
        for mode,seconds in speed['six_frame_mean_seconds'].items():
            lines.append(f'{mode}: six_ms={1000*seconds:.3f} FPS={6/seconds:.3f} P90_ms={speed["p90_six_ms"][mode]:.3f}')
        lines += [f'probability + six_dense_bytes_exact: {speed["probability_and_six_dense_parity_windows"]}/20',
            'descriptor_prepare_ms reference='+str(1000*speed['surface_descriptor_prepare_seconds_per_window'])+
            ' optimized='+str(1000*speed['optimized_surface_descriptor_prepare_seconds_per_window']),
            'boundary: '+speed['boundary'],'excludes: '+speed['excludes'],
            'No automatic training/backend promotion. Local CPU microbenchmark is NOT this server FPS.']
        (out/'summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines),flush=True)
    finally: execution.close()


if __name__=='__main__':
    event=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:event.set())
    main(stop_event=event)
