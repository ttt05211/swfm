"""Immutable patch ABI, integer shard accounting, independent result namespace."""
from pathlib import Path
import numpy as np
import torch
from tools.joint_occ_experiments import run as base
from tools.final_semantic_evaluation import metrics
from tools.stc_overnight_parallel.common import read,write,parse_counts,serialized

PROTOCOL='frozen_STC_previous_plus_norm192_singleGPU_v1'
BODY_SHA='17d649046fe7eafdbd1ce08fb8350bb90c93a757faeccdd544266113070aea44'
ROUTES=('shared/transport','previous/semantic_refine','norm192/semantic_refine')
ROOT=Path(__file__).resolve().parent


def norm_state(body,patch,source_sha):
    if (patch['protocol']!='frozen_STC_refiner_norm192_patch_v1' or patch['source_sha256']!=source_sha
            or source_sha!=BODY_SHA or patch['source_role']!='clean1000' or patch['arm']!='target'
            or patch['updates']!=200 or patch['parameters']!=192
            or patch['target_future_GT_occupancy_used'] is not False):
        raise ValueError('fixed clean1000 / target-only final200 norm patch required')
    named={}
    for name,module in body.columns.named_modules():
        if isinstance(module,torch.nn.GroupNorm):
            for field in ('weight','bias'):named['columns.'+name+'.'+field]=getattr(module,field)
    if set(named)!=set(patch['state']) or sum(p.numel() for p in named.values())!=192:
        raise ValueError('norm192 named-parameter ABI differs')
    original={k:p.detach().clone() for k,p in named.items()};adapted={}
    for name,param in named.items():
        value=param.new_tensor(patch['state'][name])
        if value.shape!=param.shape or not torch.isfinite(value).all():raise ValueError('invalid norm tensor: '+name)
        adapted[name]=value
    return named,original,adapted


@torch.no_grad()
def assign(parameters,values):
    for name,param in parameters.items():param.copy_(values[name])


def merge(job,states,complete=False):
    keys=job['keys'];seen=[];total={n:metrics.empty(3) for n in ROUTES}
    if len(states)!=job['processes']:raise ValueError('shard count differs')
    for rank,state in enumerate(states):
        expected=keys[rank::job['processes']]
        if state is None:
            if complete and expected:raise ValueError('missing completed shard')
            continue
        if (state['job_sha256']!=base.stable_json_fingerprint(job) or state['rank']!=rank
                or state['keys']!=expected[:len(state['keys'])]):raise ValueError('worker contract/key prefix differs')
        if complete and (state['keys']!=expected or state['status']!='complete'):raise ValueError('incomplete shard')
        counts=parse_counts(state['counts'],len(state['keys']),job['voxel_count'],ROUTES)
        seen.extend(map(tuple,state['keys']))
        for n in ROUTES:
            for field in total[n]:total[n][field]+=counts[n][field]
    if len(seen)!=len(set(seen)) or not set(seen).issubset(set(map(tuple,keys))):raise ValueError('duplicate/foreign keys')
    if complete and set(seen)!=set(map(tuple,keys)):raise ValueError('missing full population')
    return total,[k for k in keys if tuple(k) in set(seen)]


def report(job,counts,keys,status):
    scores={n:metrics.report(v,(1.,2.,3.),moving=False) for n,v in counts.items()}
    old=scores['previous/semantic_refine']['average'];new=scores['norm192/semantic_refine']['average']
    return dict(status=status,windows=len(keys),contract=job,metrics=scores,scientific_updates=0,
        norm_minus_previous={k:new[k]-old[k] for k in ('mIoU','IoU')} if old['mIoU'] is not None else None,
        limitation='unlabeled target-input adaptation; repeatedly inspected development; no zero-adaptation/FPS claim')


def summary(r):
    j=r['contract'];lines=['===== FROZEN STC / PREVIOUS + NORM192 / SINGLE GPU =====',
        f"status={r['status']}; windows={r['windows']}/{len(j['keys'])}; processes={j['processes']}; H{j['horizon_group']}",
        'clean1000 body; previous=half-KTA + age memory + static support + frozen posterior calibration.',
        'New route ONLY replaces 192 Refiner norm affine values; no registration/new shape rules or optimizer.',
        'STC dense-valid; GT future ego conditioned; complete six readouts before target labels; FULL GRID/no masks.',
        'Fixed local final200 target-history adaptation, NOT target-domain-information-free; original source unchanged.',
        'route                         horizon       mIoU       IoU']
    for name,scores in r['metrics'].items():
        for h,v in list(scores['per_horizon'].items())+[('Avg',scores['average'])]:
            fmt=lambda x:'NA' if x is None else f'{x:.6f}'
            lines.append(f"{name:29s} {h:>5s} {fmt(v['mIoU']):>11s} {fmt(v['IoU']):>10s}")
    lines.append('norm_minus_previous='+str(r['norm_minus_previous']))
    lines.append(f"H{j['horizon_group']} numerical execution, NOT byte-identity or old FPS. Fixed method, no score-driven promotion.")
    return '\n'.join(lines)+'\n'
