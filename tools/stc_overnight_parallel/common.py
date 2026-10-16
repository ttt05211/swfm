"""Immutable prefix import, disjoint shards, and integer-count merging."""
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
from tools.joint_occ_experiments import run as base
from tools.final_semantic_evaluation import metrics
from tools.stc_overnight import verify_all as old

PROTOCOL='frozen_STC_two_process_H6_imported_complete_prefix_v2'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,path)


def routes(contract):
    names=['baseline/transport','baseline/semantic_refine','candidate/transport','candidate/semantic_refine']
    if contract.get('posterior_alignment'):names.append('candidate/calibrated_semantic_refine')
    if contract.get('history_support'):
        names+=['candidate/supported_'+r for r in ['transport','semantic_refine']+
            (['calibrated_semantic_refine'] if contract.get('posterior_alignment') else [])]
    return names


def parse_counts(raw,windows,volume,names):
    if set(raw)!=set(names):raise ValueError('route set differs')
    result={};marginal=None
    for name,arrays in raw.items():
        if set(arrays)!={'confusion','moving'}:raise ValueError('count fields differ')
        parsed={}
        for field,array in arrays.items():
            x=np.asarray(array)
            if x.shape!=(3,18,18) or x.dtype.kind not in 'iu' or np.any(x<0):raise ValueError('invalid integer counts')
            parsed[field]=x.astype(np.int64)
        if np.any(parsed['confusion'].sum((1,2))!=windows*volume) or np.any(parsed['moving']):
            raise ValueError('incomplete FULL-GRID windows or unexpected Moving counts')
        gt=parsed['confusion'].sum(2)
        if marginal is None:marginal=gt
        elif not np.array_equal(gt,marginal):raise ValueError('routes used different GT populations')
        result[name]=parsed
    return result


def serialized(counts):return {n:{f:v.tolist() for f,v in c.items()} for n,c in counts.items()}


def check_snapshot(job,out):
    for name,digest in job['source_snapshot_sha256'].items():
        if base.sha256(Path(out)/'source_snapshot'/name)!=digest:raise ValueError('source snapshot changed: '+name)


def inspect_source(source):
    c=read(source/'contract.json')
    if c.get('protocol')!='frozen_same_body_STC_KTA_recency_v1':raise ValueError('original serial STC output required')
    keys=c['keys']
    if len(set(map(tuple,keys)))!=len(keys):raise ValueError('duplicate original keys')
    if c['population']=='all':
        if len(keys)!=4369 or len({k[0] for k in keys})!=150:raise ValueError('ALL4369/150 required')
    elif not (c['population']=='dev64' and len(keys)==64 and c['portable_subset']):
        raise ValueError('only ALL server or portable DEV64 verification supported')
    if c['known']!='STC dense-valid; NO GT visibility' or c['metric_mask']!='none' or not c['future_GT_ego_conditioned']:
        raise ValueError('source scientific protocol differs')
    for name,digest in c['implementation'].items():
        if base.sha256(Path(old.__file__).parent/name)!=digest:raise ValueError('old execution file changed: '+name)
    return c


def snapshot(source,out,contract,volume):
    state=read(source/'state.json')
    if state['contract']!=contract or state['keys']!=contract['keys'][:len(state['keys'])]:raise ValueError('source prefix differs')
    parse_counts(state['counts'],len(state['keys']),volume,routes(contract))
    folder=out/'source_snapshot';folder.mkdir()
    names=['contract.json','state.json']
    if contract.get('posterior_alignment'):
        names.append('unlabeled_posterior_fit.json')
        fit=source/names[-1]
        if not fit.is_file() or state.get('calibration_sha256')!=base.sha256(fit):raise ValueError('source calibration incomplete/changed')
        if read(fit)['source_sha256']!=contract['source_sha256']:raise ValueError('calibration belongs to another body')
    hashes={}
    for name in names:
        payload=(source/name).read_bytes();(folder/name).write_bytes(payload)
        hashes[name]=base.sha256(folder/name)
    return state,hashes


def matching_command(argv,source,cwd):
    if '-m' not in argv or argv[argv.index('-m')+1:argv.index('-m')+2]!=['tools.stc_overnight.verify_all']:return False
    if '--out' in argv:
        at=argv.index('--out')+1
        if at>=len(argv):return False
        value=argv[at]
    else:
        found=[v.split('=',1)[1] for v in argv if v.startswith('--out=')]
        if len(found)!=1:return False
        value=found[0]
    path=Path(value)
    return (path if path.is_absolute() else Path(cwd)/path).resolve()==Path(source).resolve()


def request_source_stop(source):
    """Explicit CLI handoff: SIGINT only the EXACT old module/output, never killall."""
    if os.name!='posix' or not Path('/proc').is_dir():raise RuntimeError('--stop-source requires Linux /proc; otherwise stop old task manually')
    matches=[]
    for item in Path('/proc').iterdir():
        if not item.name.isdigit():continue
        try:
            argv=[v.decode() for v in (item/'cmdline').read_bytes().split(b'\0') if v]
            if matching_command(argv,source,(item/'cwd').resolve()):matches.append((int(item.name),argv))
        except (OSError,UnicodeError,ValueError):continue
    if len(matches)>1:raise RuntimeError('multiple old processes match exact source; no signal sent')
    for pid,argv in matches:
        # Recheck immediately before signaling to avoid targeting a changed PID.
        try:
            now=[v.decode() for v in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if v]
            if now!=argv:raise RuntimeError('source process changed; no signal sent')
            os.kill(pid,signal.SIGINT)
        except FileNotFoundError:continue
        print('HANDOFF sent SIGINT ONLY to old evaluator pid=',pid,'output=',source,flush=True)


def merge(job,imported,states,complete=False):
    c=job['legacy_contract'];names=routes(c);prefix=imported['keys'];keys=c['keys'];volume=job['voxel_count']
    if prefix!=keys[:len(prefix)]:raise ValueError('imported prefix order differs')
    total=parse_counts(imported['counts'],len(prefix),volume,names);seen=list(map(tuple,prefix))
    remaining=keys[len(prefix):]
    for rank,state in enumerate(states):
        expected=remaining[rank::2]
        if state is None:
            if complete and expected:raise ValueError('missing completed shard')
            continue
        if state['job_sha256']!=base.stable_json_fingerprint(job) or state['rank']!=rank or state['keys']!=expected[:len(state['keys'])]:
            raise ValueError('worker job/rank/order differs')
        if complete and state['keys']!=expected:raise ValueError('incomplete shard')
        part=parse_counts(state['counts'],len(state['keys']),volume,names)
        seen+=list(map(tuple,state['keys']))
        for name in total:
            for field in total[name]:total[name][field]+=part[name][field]
    seen_set=set(seen);all_set=set(map(tuple,keys))
    if len(seen)!=len(seen_set) or not seen_set.issubset(all_set):raise ValueError('duplicate/foreign windows')
    if complete and seen_set!=all_set:raise ValueError('missing full-population windows')
    return total,[key for key in keys if tuple(key) in seen_set]


def result(job,counts,keys,status):
    reports={n:metrics.report(c,(1.,2.,3.),moving=False) for n,c in counts.items()}
    out=dict(status=status,windows=len(keys),contract=job,metrics=reports,scientific_updates=0,
        imported_complete_windows=job['imported_windows'],new_complete_windows=len(keys)-job['imported_windows'])
    out['full_target_verified_by_route']={n:bool(status=='complete' and job['legacy_contract']['population']=='all'
        and len(keys)==4369 and r['average']['mIoU']>18.97 and r['average']['IoU']>28.77) for n,r in reports.items()}
    return out


def summary(r):
    group=r['contract']['horizon_group']
    gpus=r['contract']['physical_GPUs']
    lines=[f'===== FROZEN STC / {gpus} GPU / TWO PROCESSES / H{group} CONTINUATION =====',
        f"status={r['status']}; windows={r['windows']}/{len(r['contract']['legacy_contract']['keys'])}",
        f"imported complete H1 windows={r['imported_complete_windows']}; new H{group} windows={r['new_complete_windows']}",
        'SAME source weights/calibration/keys/full-grid metrics. NO training or method/threshold changes.',
        f'Execution H1 prefix + H{group} tail; numerical execution variant, NOT bitwise parity or an FPS claim.',
        'route                                            mIoU       IoU']
    for name,report in r['metrics'].items():
        avg=report['average'];fmt=lambda x:'NA' if x is None else f'{x:.6f}'
        lines.append(f"{name:46s} {fmt(avg['mIoU']):>10s} {fmt(avg['IoU']):>10s}")
    return '\n'.join(lines)+'\n'
