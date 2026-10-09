#!/usr/bin/env python3
"""Full VAL4369 evaluation of the already compared, frozen mean; no new averaging."""
import sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import copy
import json
import signal
import threading

from tools.real_motion import compare_p0_f9_joint_surface_checkpoints as shared
from tools.real_motion.joint_surface_checkpoint_selection import (
    AVERAGE_NAME,AVERAGE_EPOCHS,PROTOCOL,load_cpu_checkpoint,
)
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import sha256
from tools.real_motion.static_evidence_selector_common import write_json
from real_motion.v21_source_induction import stable_json_fingerprint


def read_frozen_bundle(directory,run_directory):
    directory=Path(directory).resolve()
    paths=[directory/'bundle.json',directory/'comparison.json']
    before=[sha256(path) for path in paths]
    bundle=json.loads(paths[0].read_text(encoding='utf-8'))
    result=json.loads(paths[1].read_text(encoding='utf-8'))
    digest=bundle.pop('fingerprint')
    if (stable_json_fingerprint(bundle)!=digest or bundle.get('protocol')!=PROTOCOL
            or bundle.get('run_directory')!=str(Path(run_directory).resolve())):
        raise RuntimeError('source bundle fingerprint/protocol/training run mismatch')
    if (result.get('protocol')!=shared.EVALUATION_PROTOCOL or result.get('status')!='complete'
            or result.get('population')!='dev512' or result.get('windows')!=shared.DEV512_WINDOWS
            or result.get('bundle_fingerprint')!=digest or AVERAGE_NAME not in result.get('reports',{})):
        raise RuntimeError('completed same-bundle DEV512 mean comparison required')
    row=bundle['candidates'][AVERAGE_NAME]
    if row.get('source_epochs')!=list(AVERAGE_EPOCHS):
        raise RuntimeError('frozen mean recipe changed')
    saved=load_cpu_checkpoint(row['path'])
    if (saved.get('protocol')!=PROTOCOL or saved.get('checkpoint_role')!='evaluation_only'
            or saved.get('resume_allowed') is not False or saved.get('source_epochs')!=list(AVERAGE_EPOCHS)
            or saved.get('weight_fingerprint')!=row['weight_fingerprint']
            or stable_json_fingerprint(saved.get('training_contract'))!=stable_json_fingerprint(bundle['audit']['contract'])):
        raise RuntimeError('frozen mean weights/recipe/contract mismatch')
    del saved
    bundle['candidates']={AVERAGE_NAME:copy.deepcopy(row)}
    bundle.update(population='full4369',selection_frozen=True,source_bundle_fingerprint=digest,
                  source_comparison_directory=str(directory),
                  source_comparison_files=[dict(path=str(p),sha256=s) for p,s in zip(paths,before)])
    if before!=[sha256(path) for path in paths]:raise RuntimeError('source comparison changed during read')
    shared.verify_sources(bundle)
    bundle['fingerprint']=stable_json_fingerprint(bundle)
    return bundle


def find_frozen_bundle(runs_root,run_directory,explicit=None):
    if explicit:return read_frozen_bundle(explicit,run_directory)
    choices=[]
    for directory in Path(runs_root).resolve().iterdir():
        if not directory.is_dir() or not all((directory/n).is_file() for n in ('bundle.json','comparison.json')):
            continue
        info=json.loads((directory/'bundle.json').read_text(encoding='utf-8'))
        if info.get('protocol')!=PROTOCOL or info.get('run_directory')!=str(Path(run_directory).resolve()):continue
        choices.append(read_frozen_bundle(directory,run_directory))
    if not choices:raise RuntimeError('no completed DEV512 mean comparison found; set --source-bundle-dir explicitly')
    fingerprints={b['candidates'][AVERAGE_NAME]['weight_fingerprint'] for b in choices}
    if len(fingerprints)!=1:
        raise RuntimeError('multiple different averaged weights found; specify --source-bundle-dir, no quality-based auto selection')
    # Equivalent completed copies only; mtime chooses provenance, never quality.
    return max(choices,key=lambda b:(Path(b['source_comparison_directory'])/'comparison.json').stat().st_mtime_ns)


def main(stop_event=None,argv=None):
    p=shared.parser();p.description=__doc__
    p.add_argument('--source-bundle-dir',help='completed DEV512 comparison; auto-discover identical frozen copies otherwise')
    a=p.parse_args(argv);a.population='full4369';out=Path(a.out_dir).resolve()
    if a.bundle_only or a.warm_start_head or a.descriptor_disk_mib or a.descriptor_cache or a.ccr_history_cache or a.ccr_add_only_natural_bce:
        p.error('evaluation only: reuse completed mean/VAL cache, no export/training/TRAIN cache/disk writes')
    if (not 1<=a.cpu_workers<=16 or not 1<=a.surface_query_workers<=8
            or not 1<=a.ccr_prefetch_workers<=4 or not 0<=a.frame_cache_mib<=8192 or a.checkpoint_every<1):
        p.error('invalid bounded CPU/RAM/checkpoint budget')
    if any((directory/'training.json').is_file() for directory in (out,*out.parents)):
        p.error('full evaluation output cannot be inside an original training run')
    if a.resume:
        if not (out/'bundle.json').is_file():p.error('resume requires the SAME full evaluation output')
        if (out/'full_validation.json').is_file():p.error('full evaluation already complete; do not rerun')
    elif out.exists():p.error('new output required; no source/results overwrite')
    if not a.resume:
        bundle=find_frozen_bundle(a.runs_root,a.run_dir,a.source_bundle_dir)
        if out.is_relative_to(Path(bundle['source_comparison_directory'])):
            p.error('output cannot be inside the source comparison')
        out.mkdir(parents=True)
    with shared.evaluation_lock(out):
        if a.resume:
            bundle=json.loads((out/'bundle.json').read_text(encoding='utf-8'));digest=bundle.pop('fingerprint')
            if stable_json_fingerprint(bundle)!=digest:raise RuntimeError('full bundle fingerprint mismatch')
            bundle['fingerprint']=digest
            if (bundle.get('protocol')!=PROTOCOL or bundle.get('selection_frozen') is not True
                    or bundle.get('population')!='full4369' or set(bundle['candidates'])!={AVERAGE_NAME}
                    or bundle.get('run_directory')!=str(Path(a.run_dir).resolve())):
                raise RuntimeError('full frozen selection changed on resume')
            if a.source_bundle_dir and str(Path(a.source_bundle_dir).resolve())!=bundle['source_comparison_directory']:
                p.error('source comparison changed on resume')
        else:write_json(out/'bundle.json',bundle)
        shared.verify_sources(bundle)
        print('FROZEN MEAN SOURCE: '+bundle['candidates'][AVERAGE_NAME]['path'],flush=True)
        print('FULL4369 only: no re-averaging, epoch20 report reuse, single-epoch reruns or weight/threshold changes.',flush=True)
        return shared.evaluate(a,out,bundle,stop_event)


if __name__=='__main__':
    stopped=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stopped.set())
    sys.exit(main(stopped) or 0)
