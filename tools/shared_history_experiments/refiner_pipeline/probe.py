"""V2 staged source execution probe + frozen five-stage drift localization."""
import importlib.util
import argparse
import json
from pathlib import Path
import signal
import sys
import threading
from tools.joint_occ_experiments import run as legacy
from tools.shared_history_experiments.refiner_execution import probe as original
from .execution import BACKENDS,execution
from .drift import audit,STAGES

BASE_MEASURE=original.measure

def measure(model,optimizer,provider,saved,batches,**kwargs):
    saved_backends,saved_execution=original.BACKENDS,original.execution
    available=list(BACKENDS);unavailable={}
    if provider.device.type=='cuda' and (sys.platform=='win32' or importlib.util.find_spec('triton') is None):
        available.remove('staged_compose')
        unavailable['staged_compose']='Linux/Triton compiler unavailable; NO compiled-speed/parity claim'
    try:
        original.BACKENDS,original.execution=tuple(available),execution
        stages=audit(model,provider,saved,batches[0][0],stop_event=kwargs.get('stop_event'),backends=tuple(available))
        for name,error in stages['failed'].items():
            unavailable[name]='compiled candidate unavailable: '+error
            available.remove(name)
        original.BACKENDS=tuple(available)
        stages['unavailable']=unavailable
        directory=kwargs.get('profile_directory')
        if directory is not None:legacy.write_json(Path(directory)/'stage_replay.json',stages)
        report=BASE_MEASURE(model,optimizer,provider,saved,batches,**kwargs)
        report.update(stage_replay=stages,unavailable=unavailable,
            fp64_arithmetic_fused=False,geometry_precision='original_float64',
            implementation_pipeline={p.name:legacy.sha256(p) for p in Path(__file__).parent.glob('*.py')})
        baseline_ok=all(all(row[s]['numerical_parity'] for s in STAGES) for row in stages['reference_replays'])
        for name,row in report['candidates'].items():
            stage_ok=all(stages['candidates'][name][s]['numerical_parity'] for s in STAGES)
            row.update(frozen_reference_stage_parity=baseline_ok,frozen_candidate_stage_parity=stage_ok)
            row['pass_gate']=bool(row['pass_gate'] and baseline_ok and stage_ok)
        return report
    finally:original.BACKENDS,original.execution=saved_backends,saved_execution


def main(event=None,argv=None):
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--out-dir',required=True)
    location,_=p.parse_known_args(argv)
    old=original.measure
    try:
        original.measure=measure
        result=original.main(event,argv)
        path=Path(location.out_dir)/'speed.json'
        if path.is_file():
            report=json.loads(path.read_text(encoding='utf-8'));stages=report['stage_replay']
            lines=['===== FROZEN F -> Q -> PRIOR -> MEMORY -> LOGITS =====']
            for i,row in enumerate(stages['reference_replays'],2):
                lines.append(f'reference replay{i}: first_exact={row["first_exact_difference"]} '
                    f'first_numeric_FAIL={row["first_numerical_failure"]} changed_voxels={row["changed_semantic_voxels"]}')
            for name,row in stages['candidates'].items():
                lines.append(f'{name}: first_exact={row["first_exact_difference"]} '
                    f'first_numeric_FAIL={row["first_numerical_failure"]} changed_voxels={row["changed_semantic_voxels"]}')
            lines.append('UNAVAILABLE '+json.dumps(report['unavailable']))
            lines.append('Original FP64 transforms/GEMM retained. No automatic promotion or scientific updates.')
            summary=path.with_name('summary.txt')
            with summary.open('a',encoding='utf-8') as handle:handle.write('\n'.join(lines)+'\n')
            print('\n'.join(lines),flush=True)
        return result
    finally:original.measure=old


if __name__=='__main__':
    event=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:event.set())
    sys.exit(main(event) or 0)
