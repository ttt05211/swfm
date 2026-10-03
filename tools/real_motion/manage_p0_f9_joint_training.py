#!/usr/bin/env python3
"""Explicit-run status, validated graceful stop and exact-recipe full resume."""
import sys
from pathlib import Path
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
from datetime import datetime
import json
import os
import signal
import subprocess
import time
import torch

from tools.real_motion.joint_training_recovery import process_start_token
from real_motion.v21_source_induction import stable_json_fingerprint

TRAINER = Path(__file__).with_name('train_p0_f9_joint_causal_columns_full.py').resolve()


def model_directory(run):
    run = Path(run).resolve()
    directory = run if (run/'execution_contract.json').is_file() else run/'model'
    if not (directory/'execution_contract.json').is_file(): raise RuntimeError('specified full run has no execution_contract.json')
    return directory


def matching_process(directory, state):
    """Never signal a guessed PID, an old PID reused by Linux, or another run."""
    if Path(state.get('out_dir', '')).resolve() != directory: raise RuntimeError('runtime status/output identity mismatch')
    pid = state.get('pid')
    if type(pid) is not int or pid <= 1: raise RuntimeError('invalid recorded trainer PID')
    token = process_start_token(pid)
    if token is None or token != state.get('process_start_token'): return False
    try:
        argv = Path(f'/proc/{pid}/cmdline').read_bytes().decode().rstrip('\0').split('\0')
        cwd = Path(f'/proc/{pid}/cwd').resolve()
        if 'train_p0_f9_joint_causal_columns_full.py' not in ' '.join(argv): return False
        scripts = [a for a in argv if Path(a).name == TRAINER.name]
        target = argv[argv.index('--out-dir')+1]
        resolved = lambda p: (cwd/Path(p)).resolve()
        return len(scripts) == 1 and resolved(scripts[0]) == TRAINER and resolved(target) == directory
    except (OSError, ValueError, IndexError): return False


def status(directory):
    path = directory/'runtime_status.json'
    if not path.is_file(): raise RuntimeError('no new runtime status; refusing to guess an old process PID')
    state = json.loads(path.read_text(encoding='utf-8'))
    return {**state, 'matching_trainer_running': matching_process(directory,state)}


def request_stop(directory, *, timeout=300.):
    row = status(directory)
    if row['matching_trainer_running']:
        # Revalidate directly before the one scoped TERM. No KILL escalation.
        if not matching_process(directory,row): raise RuntimeError('trainer identity changed before TERM')
        try: os.kill(row['pid'],signal.SIGTERM)
        except ProcessLookupError: pass  # it may have just finished normally
        print(f"STOP requested pid={row['pid']}; waiting for current update/window and atomic last.pt",flush=True)
        start = time.monotonic(); last_notice = start
        while matching_process(directory,row):
            if time.monotonic()-start >= timeout:
                raise RuntimeError('graceful stop timeout; NO kill -9 was sent; inspect the run log')
            if time.monotonic()-last_notice >= 10:
                print('still waiting for safe stop/cache drain; no forced signal',flush=True); last_notice=time.monotonic()
            time.sleep(.25)
    final = status(directory)
    checkpoint = directory/'last.pt'
    if not checkpoint.is_file(): raise RuntimeError('no published full checkpoint; initialization did not finish')
    ck = torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck.get('checkpoint_role') != 'resume_last': raise RuntimeError('unexpected checkpoint role')
    print(json.dumps({'checkpoint':str(checkpoint),'update':ck['attempted_updates'],
        'cursor_epoch':ck['cursor_epoch'],'cursor_batch':ck['cursor_batch'],
        'prior_completed':ck.get('prior_completed',True),'phase':final['phase'],
        'note':'stopped/finished is a safe boundary; a crashed process can only recover its LAST periodic checkpoint'},ensure_ascii=False),flush=True)
    return 0


def resume_command(directory, checkpoint=None, new_out=None, *, sampling_workers=None, profile_every=None, expected_update=None):
    state_path = directory/'runtime_status.json'
    if state_path.is_file() and status(directory)['matching_trainer_running']:
        raise RuntimeError('original trainer is still running; stop it before resume')
    contract = json.loads((directory/'execution_contract.json').read_text(encoding='utf-8'))
    ck_path = Path(checkpoint).resolve() if checkpoint else directory/'last.pt'
    ck = torch.load(ck_path,map_location='cpu',weights_only=False)
    if ck.get('checkpoint_role') != 'resume_last': raise RuntimeError('resume requires full last.pt (or explicit last.previous.pt), not epoch/candidate')
    if expected_update is not None and (type(expected_update) is not int or expected_update < 0
                                       or ck.get('attempted_updates') != expected_update):
        raise RuntimeError('checkpoint completed update does not match expected update')
    for key in ('protocol','training_contract','train_keys','dev_keys','seed','epochs','window_batch_size','source_budget','model_configs'):
        if stable_json_fingerprint(ck.get(key)) != stable_json_fingerprint(contract.get(key)):
            raise RuntimeError('checkpoint/original execution contract mismatch at '+key)
    args = dict(contract['arguments'])
    # Explicit performance-only overrides. No batch/source/epoch/LR/seed/RNG
    # override is accepted; original cache budgets are also preserved.
    if sampling_workers is not None:
        if type(sampling_workers) is not int or not 1 <= sampling_workers <= 8:
            raise ValueError('resume sampling workers must be 1..8 combined CPU workers')
        args['sampling_workers'] = sampling_workers
    if profile_every is not None:
        if type(profile_every) is not int or profile_every < 0:
            raise ValueError('nonnegative profiling interval required')
        args['profile_every'] = profile_every
    root = directory.parent if directory.name == 'model' else directory
    out = Path(new_out).resolve() if new_out else root.parent/(
        f"full{ck['epochs']}_history{ck['model_configs']['motion']['history_frames']}_resume_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}")
    if out.exists(): raise RuntimeError('NEW resume output required; original experiment is never overwritten')
    args.update(out_dir=str(out/'model'),resume=str(ck_path),history_frames=ck['model_configs']['motion']['history_frames'],
                prewarm_causal_cache=False)  # populated geometry is reused, never explicitly re-prefilled
    command = [sys.executable,'-u',str(TRAINER)]
    booleans = {'paired_control','persistent_sampling_pool','reference_cpu_pipeline','prewarm_causal_cache'}
    path_keys = {'config','train_cache','dev_cache','population_manifest','base_checkpoint','dataroot',
        'train_info','dev_info','out_dir','resume','causal_geometry_cache'}
    cwd = Path(contract.get('launch_cwd',Path.cwd()))
    for key,value in args.items():
        if value is None: continue
        if key in booleans:
            if value: command.append('--'+key.replace('_','-'))
        elif key == 'override':
            for entry in value: command.extend(['--override',str(entry)])
        else:
            if key in path_keys:
                if not Path(value).is_absolute() and 'launch_cwd' not in contract:
                    raise RuntimeError('old relative path has no original launch cwd; cannot safely guess '+key)
                value = str((cwd/Path(value)).resolve())
            command.extend(['--'+key.replace('_','-'),str(value)])
    env = os.environ.copy()
    env.update(SWFM_COLUMN_CPU_BACKEND='native',SWFM_COLUMN_CPU_BUNDLE='1',OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    env.setdefault('SWFM_COLUMN_CPU_HORIZONS', '1')
    repository = TRAINER.parents[2]
    env['PYTHONPATH'] = os.pathsep.join([str(repository),str(repository/'upstream_occfm'),env.get('PYTHONPATH','')])
    env.setdefault('CUDA_VISIBLE_DEVICES','0')
    return command, env, out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('status','stop','resume'))
    p.add_argument('--run-dir',required=True)
    p.add_argument('--checkpoint',help='explicit full backup checkpoint; no silent fallback')
    p.add_argument('--out-dir',help='NEW root for resume (model subdirectory created by trainer)')
    p.add_argument('--print-command',action='store_true',help='read-only recipe inspection, do not launch')
    p.add_argument('--sampling-workers',type=int,help='performance-only override, 1..8 combined CPU workers')
    p.add_argument('--profile-every',type=int,help='performance-only stage timing interval; 0 disables')
    p.add_argument('--expected-update',type=int,help='refuse a checkpoint other than this committed update')
    p.add_argument('--timeout',type=float,default=300.)
    a = p.parse_args(); directory=model_directory(a.run_dir)
    if a.action == 'status': print(json.dumps(status(directory),ensure_ascii=False,indent=2)); return 0
    if a.action == 'stop':
        if not 0 < a.timeout <= 3600: p.error('positive bounded stop timeout required')
        return request_stop(directory,timeout=a.timeout)
    command,env,out=resume_command(directory,a.checkpoint,a.out_dir,sampling_workers=a.sampling_workers,
                                  profile_every=a.profile_every,expected_update=a.expected_update)
    print(json.dumps({'resume_output':str(out),'command':command,'cpu_backend':env['SWFM_COLUMN_CPU_BACKEND'],
        'cpu_horizon_pipeline':env['SWFM_COLUMN_CPU_HORIZONS'],
        'checkpoint_role':'full optimizer/RNG/cursor, NOT weight-only'},ensure_ascii=False,indent=2),flush=True)
    if a.print_command: return 0
    out.mkdir(parents=True)
    # Real log file, not a pipe: Ctrl-C cannot close stdout before the trainer
    # serializes its last committed update. Original experiment stays untouched.
    print(f"Training log: {out/'run.log'}; progress: {out/'model/progress.jsonl'}",flush=True)
    with (out/'run.log').open('x',encoding='utf-8') as log:
        child = subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        def forward(signum,frame):
            if child.poll() is None: child.send_signal(signal.SIGTERM)
        previous = {s:signal.signal(s,forward) for s in (signal.SIGINT,signal.SIGTERM)}
        try:
            output_enabled = True
            with (out/'run.log').open('r',encoding='utf-8',errors='replace') as reader:
                while True:
                    text=reader.read(65536)
                    if text and output_enabled:
                        try: sys.stdout.write(text);sys.stdout.flush()
                        except BrokenPipeError: output_enabled=False
                    if child.poll() is not None:
                        text=reader.read()
                        if text and output_enabled:
                            try:sys.stdout.write(text);sys.stdout.flush()
                            except BrokenPipeError:pass
                        return child.returncode
                    time.sleep(.2)
        finally:
            for s,handler in previous.items(): signal.signal(s,handler)


if __name__ == '__main__': sys.exit(main())
