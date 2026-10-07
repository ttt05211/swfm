"""Build the reusable fixed causal-history geometry cache for Point CCR/joint training.

The cache contains only deterministic history/data geometry:
Strong/source extraction, causal registrations, compact canonical support/strata,
static conflict guards and fixed coordinate transforms.  It contains no future
GT, learned V18 outputs, CCR logits, optimizer state, RNG samples or gradients.

A 128-window pilot estimates final disk usage before the full TRAIN20430 build.
The builder fails closed if the projected/actual cache exceeds the requested
budget.  Training should consume it with --ccr-history-cache-mode require.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import torch

from real_motion.canonical_repair_execution import CanonicalCpuExecution
from real_motion.causal_geometry_cache import CausalGeometryCache
from real_motion.column_runtime_pipeline import CachedColumnSource, prefetch_raw_columns
from real_motion.native_column_cpu import backend_name, prepare_native
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion.ccr_screen_common import (
    CCR_HISTORY_CACHE_PROTOCOL,
    _ccr_fast_fixed_geometry,
    ccr_history_cache_namespace,
)
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import sha256
from tools.real_motion.run_p0_f9_shared_evidence_pilot import PilotProvider
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256
from tools.real_motion.train_p0_f9_causal_columns import record_keys
from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint


TRAIN_WINDOWS = 20430


def _contains_torch(value):
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_torch(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_torch(v) for v in value)
    if hasattr(value, '__dict__'):
        return _contains_torch(vars(value))
    return False


def _namespace_usage(cache):
    files=list(cache.root.glob('*.cgc'))
    total=sum(p.stat().st_size for p in files)
    return len(files),total


def _manifest(cache, args, namespace, *, records, seconds, pilot, complete):
    count,used=_namespace_usage(cache)
    payload=dict(
        protocol=CCR_HISTORY_CACHE_PROTOCOL,
        namespace=cache.namespace,
        namespace_input=namespace,
        complete=bool(complete),
        windows=int(records),
        artifacts=int(count),
        disk_bytes=int(used),
        disk_gib=float(used/2**30),
        seconds=float(seconds),
        pilot=pilot,
        train_cache_sha256=sha256(args.train_cache),
        train_info_sha256=sha256(args.train_info),
        config_fingerprint=getattr(args,'_runtime_config_fingerprint'),
        dataroot=str(Path(args.dataroot).resolve()),
        future_GT_cached=False,
        learned_outputs_cached=False,
        sampled_ids_cached=False,
        optimizer_or_rng_cached=False,
        cache_scope='fixed causal history geometry reusable by frozen-CCR diagnostics and future joint training',
    )
    path=cache.root/'manifest.json'
    path.write_text(json.dumps(payload,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
    return payload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    for key in ('checkpoint','base-checkpoint','train-cache','dataroot','train-info','cache-root'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--cpu-workers',type=int,default=10)
    p.add_argument('--prefetch-workers',type=int,default=6)
    p.add_argument('--frame-cache-mib',type=int,default=4096)
    p.add_argument('--cache-ram-mib',type=int,default=1024)
    p.add_argument('--max-cache-gib',type=float,default=32.)
    p.add_argument('--pilot-windows',type=int,default=128)
    p.add_argument('--reserve-gib',type=float,default=2.)
    a=p.parse_args()

    if (not Path(a.dataroot).is_dir() or min(a.cpu_workers,a.prefetch_workers,a.pilot_windows) < 1
            or a.prefetch_workers>a.cpu_workers or not 0<=a.frame_cache_mib<=16384
            or not 0<=a.cache_ram_mib<=16384 or not 1<=a.max_cache_gib<=128
            or not 0<=a.reserve_gib<=64):
        p.error('invalid bounded cache-build resources')
    for name in ('checkpoint','base_checkpoint','train_cache','train_info'):
        if not Path(getattr(a,name)).is_file():
            p.error('missing '+name)

    cfg=load_runtime_config(a.config,a.override)
    a._runtime_config_fingerprint=stable_json_fingerprint(cfg)
    device=torch.device(a.device)
    ck,teacher=load_joint(
        a.checkpoint,device,reference_sha=CLEAN_SHA256,
        config_sha=a._runtime_config_fingerprint,allow_diagnostic=True)
    if (teacher.transport.config.history_frames!=4 or ck.get('cursor_epoch')!=19
            or ck['model_configs'].get('adaptive_context') is not None):
        raise RuntimeError('fixed selected four-history epoch19 teacher contract required')
    for path,expected in (
        (a.train_cache,ck['cache_fingerprints']['train']),
        (a.train_info,ck['info_fingerprints']['train']),
        (a.base_checkpoint,CLEAN_SHA256),
    ):
        if sha256(path)!=expected:
            raise RuntimeError('epoch19/data provenance mismatch: '+path)

    _,records=load_cache(a.train_cache)
    keys=record_keys(records)
    if len(keys)!=TRAIN_WINDOWS or tuple(keys)!=tuple(map(tuple,ck['train_keys'])):
        raise RuntimeError('full TRAIN20430 identity/order changed')

    if backend_name()=='native':
        prepare_native()
    torch.set_num_threads(1)
    provider=PilotProvider(
        a.base_checkpoint,CLEAN_SHA256,make_prepare_config(cfg),
        device,a.cpu_workers,teacher,None)
    provider.runtime_config_fingerprint=a._runtime_config_fingerprint
    provider.ccr_execution=CanonicalCpuExecution('native_parallel',max(1,min(4,a.cpu_workers)))
    provider.fixed_geometry_builder=_ccr_fast_fixed_geometry
    provider.raw_prefetch_workers=provider.raw_prefetch_depth=min(a.prefetch_workers,a.cpu_workers)
    provider.raw_io_workers=1

    root=Path(__file__).resolve().parents[2]
    namespace=ccr_history_cache_namespace(provider,a,root)
    cache=CausalGeometryCache(
        a.cache_root,namespace,
        max_bytes=int(a.max_cache_gib*2**30),
        ram_bytes=int(a.cache_ram_mib*2**20),
        reserve_bytes=int(a.reserve_gib*2**30))
    provider.ccr_history_cache=cache
    provider.ccr_history_cache_mode='build'

    source=CachedColumnSource(
        NuScenesWindowSource(a.dataroot,info_pkl=a.train_info,verbose=False),
        a.frame_cache_mib,copy_on_insert=False)

    started=time.perf_counter()
    pilot=None
    try:
        for wi,(record,raw) in enumerate(
                prefetch_raw_columns(provider,source,records,include_gt=False),1):
            if raw.get('future_gt_occ') is not None:
                raise RuntimeError('CCR history cache builder loaded future GT')
            geometry=raw.get('_column_causal_preparation')
            if geometry is None or '_ccr_compact_support' not in geometry:
                raise RuntimeError('incomplete CCR history geometry artifact')
            if _contains_torch(geometry):
                raise RuntimeError('learned/Torch tensor leaked into fixed CCR history cache')

            if wi==a.pilot_windows:
                cache.flush()
                count,used=_namespace_usage(cache)
                if count<wi:
                    raise RuntimeError(
                        f'cache pilot incomplete: persisted={count} requested={wi}; '
                        'disk quota/free-space prevented exact persistence')
                mean=used/max(count,1)
                projected=mean*len(records)
                free=shutil.disk_usage(cache.root).free
                pilot=dict(
                    windows=wi,artifacts=count,disk_gib=used/2**30,
                    mean_kib_per_window=mean/2**10,
                    projected_full_gib=projected/2**30,
                    free_gib=free/2**30,
                    max_cache_gib=a.max_cache_gib)
                print('CCR_HISTORY_CACHE_PILOT '+json.dumps(pilot,sort_keys=True),flush=True)
                if projected>int(a.max_cache_gib*2**30):
                    raise RuntimeError(
                        f'projected CCR history cache {projected/2**30:.2f} GiB '
                        f'exceeds hard limit {a.max_cache_gib:.2f} GiB; stopped after pilot')
                if projected+int(a.reserve_gib*2**30)>free+used:
                    raise RuntimeError(
                        f'insufficient disk for projected cache plus reserve: '
                        f'projected={projected/2**30:.2f}GiB free={free/2**30:.2f}GiB')

            if wi==1 or wi%128==0 or wi==len(records):
                count,used=_namespace_usage(cache)
                elapsed=time.perf_counter()-started
                rate=wi/max(elapsed,1e-9)
                eta=(len(records)-wi)/max(rate,1e-9)/60
                print(
                    f'CCR_HISTORY_CACHE_BUILD {wi}/{len(records)} '
                    f'artifacts={count} disk={used/2**30:.2f}GiB '
                    f'rate={rate:.2f}win/s ETA_min={eta:.1f}',
                    flush=True)

        cache.flush()
        count,used=_namespace_usage(cache)
        if count!=len(records):
            raise RuntimeError(
                f'CCR history cache incomplete: artifacts={count} expected={len(records)}; '
                'do not train in require mode')
        manifest=_manifest(
            cache,a,namespace,records=len(records),
            seconds=time.perf_counter()-started,pilot=pilot,complete=True)
        print('CCR_HISTORY_CACHE_COMPLETE '+json.dumps(manifest,sort_keys=True),flush=True)
    except BaseException:
        try:
            cache.flush()
            _manifest(
                cache,a,namespace,records=len(records),
                seconds=time.perf_counter()-started,pilot=pilot,complete=False)
        finally:
            raise
    finally:
        cache.close()
        provider.ccr_execution.close()


if __name__=='__main__':
    main()
