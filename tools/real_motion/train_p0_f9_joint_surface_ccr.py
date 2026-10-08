#!/usr/bin/env python3
"""Clean RANDOM-init full20430 joint Surface CCR; whole-cycle cosine, strict resume."""
import sys
from pathlib import Path
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import random
import signal
import threading
import time
import numpy as np
import torch
from real_motion.joint_surface_ccr import JointSurfaceCCR, PROTOCOL
from real_motion.local_st_world_model_v17 import config_from_mapping_v17
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.column_runtime_pipeline import CachedColumnSource
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion.run_p0_f9_shared_evidence_pilot import PilotProvider, require_cuda
from tools.real_motion import ccr_screen_common as ccr
from tools.real_motion import surface_ccr_screen_common as surface
from tools.real_motion.joint_surface_ccr_common import train_batch, fit_train_prior
from tools.real_motion.joint_surface_ccr_recovery import payload, restore
from tools.real_motion.joint_training_recovery import save_resume_checkpoint, preserve_training_rng
from tools.real_motion.joint_column_full_common import prefetch_logical_superbatches
from tools.real_motion.height_field_screen_common import epoch_groups
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import load_manifest, align_records, sha256, validate_clean_e14_checkpoint
from tools.real_motion.train_p0_f9_causal_columns import record_keys
from tools.real_motion.v18_source_interaction_common import validate_records
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256, write_json, finite_json
from tools.real_motion.train_p0_f9_v18_xy_trajectory import DEV64_FP

TRAIN_WINDOWS, VAL_WINDOWS, DEV64_WINDOWS, DEV512_WINDOWS = 20430, 4369, 64, 512


def parser():
    p = argparse.ArgumentParser(description=__doc__); add_config_args(p)
    for key in ('train-cache','dev-cache','population-manifest','base-checkpoint','dataroot','train-info','dev-info','out-dir'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--resume', help='ONLY this clean-joint protocol, into a NEW directory')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--seed', type=int, default=20261008)
    p.add_argument('--window-batch-size', type=int, default=4)
    p.add_argument('--source-budget', type=int, default=128)
    p.add_argument('--motion-lr', type=float, default=5e-4)
    p.add_argument('--repair-lr', type=float, default=3e-4)
    p.add_argument('--prior-windows', type=int, default=256)
    p.add_argument('--checkpoint-every', type=int, default=128)
    p.add_argument('--stop-after-epoch', type=int, default=0)
    p.add_argument('--max-updates', type=int, default=0, help='safe stop, does NOT shorten cosine cycle')
    p.add_argument('--device', default='cuda')
    p.add_argument('--cpu-workers', type=int, default=10)
    p.add_argument('--frame-cache-mib', type=int, default=4096)
    p.add_argument('--evaluate-only', action='store_true', help='read-only snapshot scoring; needs --resume')
    p.add_argument('--eval-population', choices=('dev64','dev512','full4369'), default='dev512')
    surface.add_args(p)
    p.set_defaults(epochs=20, ccr_motion_superbatch_updates=1, ccr_batched_motion=False,
                   ccr_batched_head=True, ccr_fast_train=True, ccr_cpu_workers=4,
                   ccr_prefetch_workers=4, descriptor_disk_mib=0, descriptor_ram_mib=256)
    return p


def summary(result):
    c = result.get('contract', {}); reports = result.get('reports', {})
    lines = ['===== CLEAN JOINT V18 + SURFACE CCR =====', 'protocol: '+PROTOCOL,
             'status: '+result['status'], 'RANDOM both modules; ALL parameters live; no teacher/KD/AE.',
             '4 histories -> 6 futures; full20430 each epoch; weighted ADD raw0.5 / REMOVEoff.',
             f"completed_epochs={result.get('epoch',0)}/{c.get('epochs','?')} updates={result.get('updates',0)}",
             'schedule: whole configured cycle cosine floor0.1; no tail/automatic extension',
             'gradient_link_observed: '+str(reports.get('gradient_link_observed',False))]
    for item in reports.get('epochs', []):
        m = item['evaluation']['variants']['joint']['metrics']
        lines.append(f"epoch={item['epoch']} update={item['update']} dev64 mIoU={m['mIoU']:.6f} MovingMicro={m['MovingMicro']:.6f}")
    for key in ('final_dev512','read_only_evaluation'):
        if key in reports:
            lines.append('===== '+key+' =====')
            for name,item in reports[key]['variants'].items():
                m = item['metrics']
                lines.append(f"{name}: IoU={m['IoU']:.6f} mIoU={m['mIoU']:.6f} MovingMacro={m['MovingMacro']:.6f} MovingMicro={m['MovingMicro']:.6f}")
    lines += ['checkpoint: '+result.get('checkpoint','none'),
              'Final epoch only, no dev-best/threshold sweep/automatic promotion. Full4369 only when explicitly evaluate-only.']
    if 'error' in result: lines.append('error: '+result['error'])
    return '\n'.join(lines)+'\n'


def main(stop_event=None, argv=None):
    p = parser(); a = p.parse_args(argv); out = Path(a.out_dir)
    if out.exists(): p.error('NEW output required, including resume/evaluation; no overwrite')
    for key in ('config','train_cache','dev_cache','population_manifest','base_checkpoint','train_info','dev_info'):
        if not Path(getattr(a,key) or '').is_file(): p.error('missing '+key)
    if a.resume and not Path(a.resume).is_file(): p.error('missing resume checkpoint')
    if a.evaluate_only and not a.resume: p.error('evaluate-only requires a clean-joint --resume checkpoint')
    if (min(a.epochs,a.window_batch_size,a.source_budget,a.prior_windows,a.checkpoint_every) < 1
            or not 1 <= a.cpu_workers <= 16 or not 1 <= a.surface_query_workers <= 8
            or not 0 <= a.frame_cache_mib <= 8192 or not 0 <= a.stop_after_epoch <= a.epochs
            or a.max_updates < 0 or any(not np.isfinite(v) or v <= 0 for v in (a.motion_lr,a.repair_lr))):
        p.error('invalid training/CPU/RAM/LR budgets')
    if (not Path(a.dataroot).is_dir() or not a.ccr_history_cache or not a.ccr_val_history_cache
            or not Path(a.ccr_history_cache).is_dir() or not Path(a.ccr_val_history_cache).is_dir()
            or a.warm_start_head or a.ccr_add_only_natural_bce or a.descriptor_disk_mib
            or a.ccr_motion_superbatch_updates != 1):
        p.error('require existing TRAIN/VAL history caches; no warm start/natural loss/disk writes/frozen superbatch')
    device = require_cuda(a.device); torch.set_num_threads(1)
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    cfg = load_runtime_config(a.config,a.override); pcfg = make_prepare_config(cfg)
    base = torch.load(a.base_checkpoint,map_location='cpu',weights_only=False)
    validate_clean_e14_checkpoint(base,a.base_checkpoint,CLEAN_SHA256)
    config = replace(config_from_mapping_v17(base['model_config']),history_frames=4); del base
    meta, train = load_cache(a.train_cache); train_keys = record_keys(train); validate_records(train)
    manifest, keys64, _ = load_manifest(a.population_manifest)
    keys512 = tuple(map(tuple,manifest['parent_keys']))
    if (len(train) != TRAIN_WINDOWS or len(keys64) != DEV64_WINDOWS or len(keys512) != DEV512_WINDOWS
            or manifest['selected_key_fingerprint'] != DEV64_FP
            or {s for s,_ in train_keys} & {s for s,_ in keys512}):
        raise RuntimeError('frozen full20430/dev64/dev512 identity or scene split invalid')
    _, all_dev = load_cache(a.dev_cache); record_keys(all_dev)
    if len(all_dev) != VAL_WINDOWS: raise RuntimeError('complete VAL4369 cache required')
    dev64 = align_records(all_dev,keys64); dev512 = align_records(all_dev,keys512)
    groups = [epoch_groups(train,a.seed,i,a.window_batch_size,a.source_budget) for i in range(a.epochs)]
    sizes = [[len(g) for g in e] for e in groups]; steps = sum(map(len,groups))
    if a.max_updates > steps or a.prior_windows > len(train): p.error('stop/prior exceeds fixed population')
    # E14 is used ONLY for architecture config/renderer preflight, never weights.
    joint = JointSurfaceCCR(config,z_bins=int(pcfg.grid.shape_hwd[2])).to(device)
    provider = PilotProvider(a.base_checkpoint,CLEAN_SHA256,pcfg,device,a.cpu_workers,joint,None)
    source = CachedColumnSource(NuScenesWindowSource(a.dataroot,info_pkl=a.train_info,verbose=False),a.frame_cache_mib)
    dev_source = CachedColumnSource(NuScenesWindowSource(a.dataroot,info_pkl=a.dev_info,verbose=False),a.frame_cache_mib)
    provider.ccr_history_cache_source = source; provider.ccr_val_history_cache_source = dev_source
    optimizer = torch.optim.AdamW([
        dict(params=joint.transport.parameters(),lr=a.motion_lr,initial_lr=a.motion_lr,weight_decay=1e-4),
        dict(params=joint.columns.parameters(),lr=a.repair_lr,initial_lr=a.repair_lr,weight_decay=.01)])
    rng = np.random.default_rng(a.seed+1); root = Path(__file__).resolve().parents[2]
    surface.setup(provider,a)
    # Fast frozen-motion flags are preparation metadata only. They MUST NOT
    # dispatch the frozen screen step or precompute learnable outputs.
    provider.ccr_batched_motion = False; provider.ccr_motion_streams = 1
    contract = dict(protocol=PROTOCOL, initialization='RANDOM', history_frames=4, future_frames=6,
        model=joint.configs(), runtime_config_fingerprint=stable_json_fingerprint(cfg),
        data={k:sha256(getattr(a,k)) for k in ('train_cache','dev_cache','train_info','dev_info','base_checkpoint')},
        train_key_fingerprint=stable_json_fingerprint(train_keys),
        dev_manifest_fingerprint=manifest['manifest_fingerprint'], dataroot=str(Path(a.dataroot).resolve()),
        train_history_namespace=provider.ccr_history_cache.namespace,
        val_history_namespace=provider.ccr_val_history_cache.namespace,
        seed=a.seed, epochs=a.epochs, epoch_batches=list(map(len,groups)), epoch_batch_sizes=sizes,
        epoch_plan_fingerprint=stable_json_fingerprint([
            [[(str(r['scene_name']),str(r['t0_token'])) for r in g] for g in e] for e in groups]),
        schedule_steps=steps, schedule='whole_cycle_cosine_floor0.1_no_tail',
        window_batch=a.window_batch_size, source_budget=a.source_budget,
        motion_lr=a.motion_lr, repair_lr=a.repair_lr, prior_keys=train_keys[:a.prior_windows],
        samples_per_role=a.samples_per_role, objective='original_V18_loss_plus_equal_window_role_weighted_ADD',
        thresholds=[.5,None], clip_norm=[5.,5.], weight_decay=[1e-4,.01],
        torch_version=str(torch.__version__), device_type=device.type,
        reference_execution=bool(a.surface_reference_execution),
        implementation=stable_json_fingerprint({n:sha256(root/n) for n in (
            'real_motion/joint_surface_ccr.py','real_motion/surface_canonical_repair.py',
            'real_motion/surface_projection_execution.py','tools/real_motion/joint_surface_ccr_common.py',
            'tools/real_motion/joint_surface_ccr_recovery.py','tools/real_motion/train_p0_f9_joint_surface_ccr.py',
            'tools/real_motion/ccr_screen_common.py','tools/real_motion/surface_ccr_screen_common.py',
            'tools/real_motion/joint_column_common.py','tools/real_motion/height_field_screen_common.py')}))
    epoch = batch = updates = executed = 0; reports = {}; begun = time.perf_counter()
    if a.resume:
        saved = torch.load(a.resume,map_location='cpu',weights_only=False)
        (epoch,batch,updates,executed),reports = restore(saved,joint,optimizer,rng,contract); del saved
    out.mkdir(parents=True); result = dict(status='running',contract=contract,reports=reports)
    def persist():
        result.update(epoch=epoch,batch=batch,updates=updates,executed_windows=executed,
                      elapsed_seconds_this_invocation=time.perf_counter()-begun)
        write_json(out/'training.json',result); (out/'summary.txt').write_text(summary(result),encoding='utf-8')
    def save(name='last.pt'):
        save_resume_checkpoint(out/name,payload(joint,optimizer,rng,contract,
            epoch=epoch,batch=batch,updates=updates,executed=executed,reports=reports))
        if name=='last.pt': result['checkpoint'] = str(out/name)
    def stopped(): return stop_event is not None and stop_event.is_set()
    def evaluate(records,log):
        with preserve_training_rng(rng):
            return ccr.evaluate(provider,dev_source,records,joint,joint.columns,progress=log,stop_event=stop_event)
    try:
        with (out/'progress.jsonl').open('x',encoding='utf-8') as handle:
            def log(row):
                handle.write(json.dumps(finite_json(row),allow_nan=False)+'\n'); handle.flush()
            if a.evaluate_only:
                selected = dict(dev64=dev64,dev512=dev512,full4369=all_dev)[a.eval_population]
                reports['read_only_evaluation'] = evaluate(selected,log)
                result.update(status='complete',source_checkpoint=str(Path(a.resume).resolve()),
                              evaluation_population=a.eval_population)
                persist(); return 0
            del all_dev
            if 'train_prior' not in reports:
                with preserve_training_rng(rng):
                    reports['train_prior'] = fit_train_prior(provider,source,train[:a.prior_windows],joint,
                                                            progress=log,stop_event=stop_event)
                save(); persist()
            print(f'CLEAN_JOINT RANDOM 4->6 windows=20430 epochs={a.epochs} steps={steps} restored_updates={updates}',flush=True)
            print('LIVE JOINT: motion is NOT frozen; no frozen-output preforward/superbatch. '
                  'Hard rasterization detached ONLY; live source queries receive CCR gradients.',flush=True)
            with ThreadPoolExecutor(max_workers=provider.ccr_sample_workers) as pool:
                while epoch < a.epochs:
                    # Completing a monitor is a separate phase. Its completed
                    # result is persisted, so interruption never skips an epoch
                    # or repeats an ALREADY finished evaluation on resume.
                    if batch == len(groups[epoch]):
                        if not any(r['epoch']==epoch+1 for r in reports.get('epochs',[])):
                            report = evaluate(dev64,log)
                            reports.setdefault('epochs',[]).append(dict(epoch=epoch+1,update=updates,evaluation=report))
                            m = report['variants']['joint']['metrics']
                            print(f'JOINT_SURFACE_EPOCH {epoch+1}: mIoU={m["mIoU"]:.6f} MovingMicro={m["MovingMicro"]:.6f}',flush=True)
                            save(); persist()
                        epoch += 1; batch = 0
                        save(); save(f'epoch_{epoch:04d}.pt'); persist()
                        if stopped() or (a.stop_after_epoch and epoch>=a.stop_after_epoch):
                            result['status']='stopped'; persist(); return 0
                        continue
                    if stopped(): result['status']='stopped'; save(); persist(); return 0
                    iterator = prefetch_logical_superbatches(provider,source,groups[epoch][batch:],1,
                                                           io_workers=provider.train_io_workers)
                    previous=time.perf_counter()
                    try:
                        for bundle in iterator:
                            wait=time.perf_counter()-previous
                            if stopped(): break
                            probe=not reports.get('gradient_link_observed',False) or (updates+1)%128==0
                            stats=train_batch(joint,optimizer,provider,bundle[0],rng,updates+1,steps,pool=pool,probe=probe)
                            updates+=1; batch+=1; executed+=stats['windows']
                            norm=stats['source_query_gradient_norm']
                            if norm is not None and norm>0: reports['gradient_link_observed']=True
                            log(dict(event='train_joint_surface',epoch=epoch+1,epoch_batch=batch,update=updates,
                                     input_wait_seconds=wait,**stats))
                            if updates==1 or updates%32==0:
                                print(f'JOINT_SURFACE_TRAIN epoch={epoch+1}/{a.epochs} update={updates}/{steps} '
                                      f'seconds/window={(stats["seconds"]+wait)/stats["windows"]:.4f} '
                                      f'motion={stats["motion_loss"]:.5f} repair={stats["repair_loss"]:.5f}',flush=True)
                            if updates%a.checkpoint_every==0 or batch==len(groups[epoch]): save(); persist()
                            if stopped() or (a.max_updates and updates>=a.max_updates):
                                result['status']='stopped'; save(); persist(); return 0
                            previous=time.perf_counter()
                    finally: iterator.close()
            if 'final_dev512' not in reports:
                reports['final_dev512']=evaluate(dev512,log); save(); persist()
            result['status']='complete'; save(); save('candidate.pt'); persist()
            print(summary(result),flush=True); return 0
    except InterruptedError as exc:
        result.update(status='stopped',error=str(exc))
        if 'train_prior' in reports and not a.evaluate_only: save()
        persist(); return 0
    except BaseException as exc:
        # A failed/incomplete backward step MUST NOT publish mutated optimizer
        # state with the previous cursor. Keep the last completed snapshot.
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}'); persist(); raise
    finally:
        ccr.close(provider,result); persist()


if __name__=='__main__':
    event=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda *_:event.set())
    sys.exit(main(event) or 0)
