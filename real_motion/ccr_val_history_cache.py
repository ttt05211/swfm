"""Validation-only persistent CCR history geometry cache contract.

This is the same deterministic geometry payload as TRAIN's reusable CCR cache,
but with an independent validation provenance namespace.  It exists only to
accelerate quality evaluation (DEV64/DEV512/full-val).  Official FPS paths must
not attach this cache because their frozen timing boundary includes fresh fixed
geometry construction.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import sha256

PROTOCOL='p0_f9_ccr_val_history_geometry_v1'
VAL_WINDOWS=4369


def namespace(provider,args,root):
    """Validation provenance; intentionally independent of TRAIN cache identity."""
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
        split='val',
        prepare_config=asdict(provider.pcfg),
        strong_config=asdict(provider.strong),
        active_history_frames=int(provider.joint.transport.config.history_frames),
        dev_cache_sha256=sha256(args.dev_cache),
        dev_info_sha256=sha256(args.dev_info),
        dataroot=str(Path(args.dataroot).resolve()),
        geometry_implementation=stable_json_fingerprint({p:sha256(root/p) for p in files}),
    )
    return stable_json_fingerprint(identity)


def validate_manifest(cache,args):
    path=cache.root/'manifest.json'
    if not path.is_file():
        raise RuntimeError(f'required CCR VAL history cache manifest missing: {path}')
    m=json.loads(path.read_text(encoding='utf-8'))
    if (
        m.get('protocol')!=PROTOCOL
        or m.get('split')!='val'
        or m.get('complete') is not True
        or int(m.get('windows',-1))!=VAL_WINDOWS
        or int(m.get('artifacts',-1))!=VAL_WINDOWS
        or m.get('namespace')!=cache.namespace
        or m.get('dev_cache_sha256')!=sha256(args.dev_cache)
        or m.get('dev_info_sha256')!=sha256(args.dev_info)
        or m.get('future_GT_cached') is not False
        or m.get('learned_outputs_cached') is not False
        or m.get('sampled_ids_cached') is not False
        or m.get('optimizer_or_rng_cached') is not False
    ):
        raise RuntimeError('required CCR VAL history cache manifest is incomplete or provenance-invalid')
    return m
