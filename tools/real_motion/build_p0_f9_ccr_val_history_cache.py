#!/usr/bin/env python3
"""Build persistent fixed-history geometry for validation quality evaluation.

The payload is identical deterministic CCR history geometry to the TRAIN cache,
but lives in an independent VAL provenance namespace and contains no future GT,
learned outputs, sampled IDs, optimizer/RNG state, logits or gradients.

This cache is for DEV64/DEV512/full validation QUALITY evaluation only.
Official FPS measurements must not attach it because the frozen FPS boundary
includes fresh fixed-geometry construction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import torch

from real_motion.canonical_repair_execution import CanonicalCpuExecution
from real_motion.causal_geometry_cache import CausalGeometryCache
from real_motion.ccr_history_geometry import build_ccr_history_geometry
from real_motion.ccr_val_history_cache import (
    PROTOCOL as VAL_PROTOCOL, VAL_WINDOWS, namespace as val_namespace,
)
from real_motion.column_runtime_pipeline import CachedColumnSource
from real_motion.native_column_cpu import backend_name,prepare_native
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.runtime_config import add_config_args,load_runtime_config,make_prepare_config
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion.build_p0_f9_ccr_history_cache import (
    _contains_torch,_namespace_usage,_parallel_cache_rows,
)
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import sha256
from tools.real_motion.run_p0_f9_shared_evidence_pilot import PilotProvider
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256
from tools.real_motion.train_p0_f9_causal_columns import record_keys
from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint


def _manifest(cache,args,namespace_input,*,records,seconds,pilot,complete):
    count,used=_namespace_usage(cache)
    payload=dict(
        protocol=VAL_PROTOCOL,
        split='val',
        namespace=cache.namespace,
        namespace_input=namespace_input,
        complete=bool(complete),
        windows=int(records),
        artifacts=int(count),
        disk_bytes=int(used),
        disk_gib=float(used/2**30),
        seconds=float(seconds),
        pilot=pilot,
        dev_cache_sha256=sha256(args.dev_cache),
        dev_info_sha256=sha256(args.dev_info),
        config_fingerprint=args._runtime_config_fingerprint,
        dataroot=str(Path(args.dataroot).resolve()),
        future_GT_cached=False,
        learned_outputs_cached=False,
        sampled_ids_cached=False,
        optimizer_or_rng_cached=False,
        cache_scope='VAL fixed causal history geometry for quality evaluation only; excluded from official FPS',
    )
    (cache.root/'manifest.json').write_text(
        json.dumps(payload,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
    return payload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    for key in ('checkpoint','base-checkpoint','dev-cache','dataroot','dev-info','cache-root'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--cpu-workers',type=int,default=10)
    p.add_argument('--prefetch-workers',type=int,default=8)
    p.add_argument('--frame-cache-mib',type=int,default=4096)
    p.add_argument('--cache-ram-mib',type=int,default=512)
    p.add_argument('--max-cache-gib',type=float,default=8.)
    p.add_argument('--pilot-windows',type=int,default=128)
    p.add_argument('--pilot-only',action='store_true')
    p.add_argument('--reserve-gib',type=float,default=2.)
    a=p.parse_args()

    if (not Path(a.dataroot).is_dir()
            or min(a.cpu_workers,a.prefetch_workers,a.pilot_windows)<1
            or a.prefetch_workers>min(a.cpu_workers,8)
            or not 0<=a.frame_cache_mib<=16384
            or not 0<=a.cache_ram_mib<=16384
            or not 1<=a.max_cache_gib<=32
            or not 0<=a.reserve_gib<=64):
        p.error('invalid bounded VAL cache-build resources')
    for name in ('checkpoint','base_checkpoint','dev_cache','dev_info'):
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
        (a.dev_cache,ck['cache_fingerprints']['dev']),
        (a.dev_info,ck['info_fingerprints']['dev']),
        (a.base_checkpoint,CLEAN_SHA256),
    ):
        if sha256(path)!=expected:
            raise RuntimeError('epoch19/data provenance mismatch: '+path)

    _,records=load_cache(a.dev_cache)
    keys=record_keys(records)
    unique={tuple(x) for x in keys}
    if len(keys)!=VAL_WINDOWS or len(unique)!=VAL_WINDOWS:
        raise RuntimeError(f'full VAL{VAL_WINDOWS} identity/population changed')

    if backend_name()=='native':
        prepare_native()
    torch.set_num_threads(1)
    provider=PilotProvider(
        a.base_checkpoint,CLEAN_SHA256,make_prepare_config(cfg),
        device,a.cpu_workers,teacher,None)
    provider.runtime_config_fingerprint=a._runtime_config_fingerprint
    provider.ccr_execution=CanonicalCpuExecution('native',1)
    provider.fixed_geometry_builder=build_ccr_history_geometry
    provider.raw_prefetch_workers=provider.raw_prefetch_depth=min(a.prefetch_workers,a.cpu_workers)
    provider.raw_io_workers=1

    root=Path(__file__).resolve().parents[2]
    namespace_input=val_namespace(provider,a,root)
    cache=CausalGeometryCache(
        a.cache_root,namespace_input,
        max_bytes=int(a.max_cache_gib*2**30),
        ram_bytes=int(a.cache_ram_mib*2**20),
        reserve_bytes=int(a.reserve_gib*2**30),
        compression_level=6)
    provider.ccr_history_cache=cache
    provider.ccr_history_cache_mode='build'

    source=CachedColumnSource(
        NuScenesWindowSource(a.dataroot,info_pkl=a.dev_info,verbose=False),
        a.frame_cache_mib,copy_on_insert=False)

    started=time.perf_counter();pilot=None
    try:
        for wi,(record,raw) in enumerate(
                _parallel_cache_rows(provider,source,records,a.prefetch_workers),1):
            if raw.get('future_gt_occ') is not None:
                raise RuntimeError('CCR VAL history cache builder loaded future GT')
            geometry=raw.get('_column_causal_preparation')
            if geometry is None or '_ccr_compact_support' not in geometry:
                raise RuntimeError('incomplete CCR VAL history geometry artifact')
            if _contains_torch(geometry):
                raise RuntimeError('learned/Torch tensor leaked into fixed CCR VAL history cache')

            if wi==a.pilot_windows:
                cache.flush()
                count,used=_namespace_usage(cache)
                if count<wi:
                    raise RuntimeError(
                        f'VAL cache pilot incomplete: persisted={count} requested={wi}; '
                        'disk quota/free-space prevented exact persistence')
                mean=used/max(count,1);projected=mean*len(records)
                free=shutil.disk_usage(cache.root).free
                pilot=dict(
                    windows=wi,artifacts=count,disk_gib=used/2**30,
                    mean_kib_per_window=mean/2**10,
                    projected_full_gib=projected/2**30,
                    free_gib=free/2**30,max_cache_gib=a.max_cache_gib)
                print('CCR_VAL_HISTORY_CACHE_PILOT '+json.dumps(pilot,sort_keys=True),flush=True)
                if projected>int(a.max_cache_gib*2**30):
                    raise RuntimeError(
                        f'projected CCR VAL history cache {projected/2**30:.2f} GiB '
                        f'exceeds hard limit {a.max_cache_gib:.2f} GiB')
                if projected+int(a.reserve_gib*2**30)>free+used:
                    raise RuntimeError(
                        f'insufficient disk for projected VAL cache plus reserve: '
                        f'projected={projected/2**30:.2f}GiB free={free/2**30:.2f}GiB')
                if a.pilot_only:
                    manifest=_manifest(
                        cache,a,namespace_input,records=len(records),
                        seconds=time.perf_counter()-started,pilot=pilot,complete=False)
                    print('CCR_VAL_HISTORY_CACHE_PILOT_ONLY '+json.dumps(manifest,sort_keys=True),flush=True)
                    return

            if wi==1 or wi%128==0 or wi==len(records):
                count,used=_namespace_usage(cache)
                elapsed=time.perf_counter()-started
                rate=wi/max(elapsed,1e-9)
                eta=(len(records)-wi)/max(rate,1e-9)/60
                stats=cache.stats()
                print(
                    f'CCR_VAL_HISTORY_CACHE_BUILD {wi}/{len(records)} '
                    f'artifacts={count} disk={used/2**30:.2f}GiB '
                    f'rate={rate:.2f}win/s ETA_min={eta:.1f} '
                    f'parallel_windows={a.prefetch_workers} '
                    f'write_cpu_s={stats["background_write_seconds"]:.1f}',
                    flush=True)

        cache.flush()
        count,used=_namespace_usage(cache)
        if count!=len(records):
            raise RuntimeError(
                f'CCR VAL history cache incomplete: artifacts={count} expected={len(records)}')
        manifest=_manifest(
            cache,a,namespace_input,records=len(records),
            seconds=time.perf_counter()-started,pilot=pilot,complete=True)
        print('CCR_VAL_HISTORY_CACHE_COMPLETE '+json.dumps(manifest,sort_keys=True),flush=True)
    except BaseException:
        try:
            cache.flush()
            _manifest(
                cache,a,namespace_input,records=len(records),
                seconds=time.perf_counter()-started,pilot=pilot,complete=False)
        finally:
            raise
    finally:
        cache.close()
        provider.ccr_execution.close()


if __name__=='__main__':
    main()
