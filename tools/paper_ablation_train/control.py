"""Metadata-only matching to existing E controls; never load their weights/VAL."""
import hashlib
import json
from pathlib import Path
import torch
from tools.joint_occ_experiments import run as base
from .model import ARMS

PROTOCOL = 'CarryWorld_five_ablations_train_only_v1'
FULL = 'forward_AdaLN_RANDOM_fullTRAIN20430x2_raw_every2000_v1'
SMALL = 'AdaLN_forward_RANDOM_TRAIN4086x1_singleGPU_raw_FREEreadout_v2'
TRAIN_FILES = ('config','train_cache','base_checkpoint','train_info')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def canonical(value): return json.loads(json.dumps(value,sort_keys=True))


def reference(path,arm):
    c=json.loads(Path(path).read_text(encoding='utf-8'))
    full=ARMS[arm]['fraction']==1.; count=20430 if full else 4086
    if (c.get('protocol')!=(FULL if full else SMALL) or c.get('epochs')!=(2 if full else 1)
            or c.get('world_size') not in ((1,2) if full else (1,))
            or len(c.get('train_keys',[]))!=count or len(set(map(tuple,c['train_keys'])))!=count
            or c.get('source_policy')!='causal_forward_CV_history'
            or c.get('known')!='TRAIN_OCC_STC_dense_valid' or c.get('registration')!='none'
            or c.get('parameters')!=2816122 or c.get('model',{}).get('source_policy')!='causal_forward_CV_history'
            or c.get('objective')!=dict(CE=1.,Lovasz=1.,motion=0.,existence=0.,ego=0.)
            or c.get('schedule')!=f'whole_{2 if full else 1}_epoch_cosine_floor0.01'
            or c.get('seed')!=20261011 or c.get('learning_rates')!=[5e-4,3e-4]):
        raise RuntimeError('original matched forward E contract required for '+arm)
    return c


def check_inputs(a,ctx,c,root):
    expected=dict(seed=a.seed,world_size=ctx.world_size,learning_rates=[a.motion_lr,a.repair_lr],
        overrides=a.override,precision='BF16_network_FP64_geometry_full6',
        execution=dict(torch=str(torch.__version__),cuda=torch.version.cuda,
            cpu_workers=a.cpu_workers,prefetch_workers=a.prefetch_workers))
    for name,value in expected.items():
        if c.get(name)!=value: raise RuntimeError('E control settings differ: '+name)
    data={name:sha(getattr(a,name)) for name in TRAIN_FILES}
    if any(c.get('data',{}).get(k)!=v for k,v in data.items()):
        raise RuntimeError('TRAIN file/config/architecture checksum differs from E')
    implementation={**c.get('implementation',{}),
        **(c['model'].get('experiment') or {}).get('candidate_implementation',{})}
    for name,digest in implementation.items():
        file=(root/name).resolve()
        if root not in file.parents or sha(file)!=digest:
            raise RuntimeError('E implementation differs: '+name)
    return data


def plan(all_train,c,arm):
    if len(all_train)!=20430: raise ValueError('original TRAIN20430 required')
    count=20430 if ARMS[arm]['fraction']==1. else 4086
    keys=base.select_scene_balanced_round_robin(base.record_keys(all_train),count)
    if [list(k) for k in keys]!=c['train_keys']: raise RuntimeError('ordered TRAIN keys differ from E')
    rows=base.align_records(all_train,keys)
    if {k[0] for k in c.get('eval_keys',[])} & {r['scene_name'] for r in rows}:
        raise RuntimeError('TRAIN/control-DEV scene overlap')
    groups=[base.epoch_groups(rows,c['seed'],e,4,128) for e in range(c['epochs'])]
    sizes=list(map(len,groups))
    if sum(sizes)!=c['total_updates'] or 'epochs_updates' in c and sizes!=c['epochs_updates']:
        raise RuntimeError('source128 group/update budget differs from E')
    ordered=[[[r['scene_name'],r['t0_token']] for r in group] for group in groups[0]]
    budget=dict(run_epochs=1,train_windows=count,stop_updates=sizes[0],
        schedule_reference_updates=sum(sizes),reference_epochs=c['epochs'],
        schedule='reference_cosine_prefix_floor0.01' if c['epochs']==2 else 'whole_1_epoch_cosine_floor0.01',
        group_keys_sha256=hashlib.sha256(json.dumps(ordered,separators=(',',':')).encode()).hexdigest())
    return keys,groups[0],budget


def validate_resume(saved,contract,groups):
    if (saved.get('protocol')!=PROTOCOL or not saved.get('training_only')
            or saved.get('contract')!=contract):
        raise RuntimeError('resume requires SAME arm/TRAIN/budget/implementation/world-size')
    epoch,batch,updates=(int(saved[k]) for k in ('epoch','batch','updates'))
    if ((epoch,batch,updates)!=(1,0,len(groups))
            and not (epoch==0 and 0<=batch<len(groups) and updates==batch)):
        raise RuntimeError('invalid complete-update cursor')
    if len(saved['rank_rng'])!=contract['world_size']: raise RuntimeError('resume rank count differs')
    return epoch,batch,updates
