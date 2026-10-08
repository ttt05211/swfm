"""Stable reusable fixed-history geometry contract for Point CCR and later joint training.

This module is deliberately isolated from trainer/optimizer code so execution-only
or learning changes do not invalidate a large persistent cache.  Cached values
contain deterministic raw-history geometry only: Strong/source decomposition,
causal registrations, compact canonical support/strata, static conflict guards,
and fixed renderer geometry.  No learned outputs, future GT, sampled IDs,
optimizer/RNG state, logits or gradients are allowed.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time

from real_motion.canonical_causal_repair import build_compact_canonical_support
from real_motion.canonical_repair_context import compact_static_conflicts
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import sha256

PROTOCOL='p0_f9_ccr_history_geometry_v2'


def execution_kernels(provider):
    return getattr(getattr(provider,'ccr_execution',None),'kernels',None)


def build_ccr_history_geometry(provider, raw, record):
    """Exact fixed geometry shared by frozen-CCR diagnostics and future joint runs."""
    from types import SimpleNamespace
    from tools.real_motion.joint_column_full_common import build_ccr_training_geometry

    prefetch=max(1,getattr(provider,'raw_prefetch_workers',1))
    workers=max(1,min(2,provider.workers//prefetch))
    profile={}
    causal=build_ccr_training_geometry(
        raw,record,provider.pcfg,provider.strong,workers,profile=profile)
    causal['_ccr_fast_profile']=profile
    prep=SimpleNamespace(
        raw=raw,
        state={**causal['prepared_state'],'rec':record,'gpu':None},
        registrations=causal['registrations'])

    tick=time.perf_counter()
    compact=build_compact_canonical_support(
        prep,provider.pcfg.grid,kernels=execution_kernels(provider),executor=None)
    causal['_ccr_fast_profile']['compact_support']=time.perf_counter()-tick
    tick=time.perf_counter()
    conflicts=compact_static_conflicts(compact,prep,provider.pcfg.grid)
    causal['_ccr_fast_profile']['static_conflicts']=time.perf_counter()-tick
    causal['_ccr_fast_profile']['fast_total']=sum(
        float(v) for k,v in causal['_ccr_fast_profile'].items()
        if k not in ('total','geometry_workers') and isinstance(v,(int,float)))
    causal['_ccr_compact_support']=compact
    causal['_ccr_compact_conflicts']=conflicts

    # Keep only renderer state actually consumed after cache load. Raw history
    # stays outside the artifact and is still the source of sampled labels /
    # visibility.  This avoids persisting duplicate dense Strong tensors.
    keep=('current_pose','current','source_world_points','source_rel_xy',
          'source_z_t0','world_to_future','column_backgrounds')
    missing=[k for k in keep if k not in causal['prepared_state']]
    if missing:
        raise RuntimeError(f'incomplete compact CCR prepared state: {missing}')
    causal['prepared_state']={k:causal['prepared_state'][k] for k in keep}
    return causal


def namespace(provider,args,root):
    """Geometry-only provenance; independent of CCR optimizer/head/execution code."""
    files=(
        'real_motion/ccr_history_geometry.py',
        'real_motion/canonical_causal_repair.py',
        'real_motion/canonical_repair_context.py',
        'tools/real_motion/joint_column_full_common.py',
        'tools/real_motion/causal_column_common.py',
        'tools/real_motion/benchmark_p0_f9_v18_runtime.py',
        'real_motion/native/column_cpu.cpp',
    )
    identity=dict(
        protocol=PROTOCOL,
        prepare_config=asdict(provider.pcfg),
        strong_config=asdict(provider.strong),
        active_history_frames=int(provider.joint.transport.config.history_frames),
        train_cache_sha256=sha256(args.train_cache),
        train_info_sha256=sha256(args.train_info),
        dataroot=str(Path(args.dataroot).resolve()),
        implementation=stable_json_fingerprint({p:sha256(root/p) for p in files}),
    )
    return stable_json_fingerprint(identity)
