"""Run all FIVE arms sequentially on the same TWO GPUs. Training ONLY."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from .control import PROTOCOL,sha,canonical

ARMS=('c_transport','d_kta_refiner','m0_no_memory','m1_ego_memory','s0_pool')
SUITE_PROTOCOL='CarryWorld_five_arms_two_GPU_training_queue_v1'
ROOT=Path(__file__).resolve().parents[2]


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir',required=True,help='new suite directory, or the SAME directory to continue')
    p.add_argument('--gpus',default='0,1')
    p.add_argument('--full-reference',default=str(ROOT/'outputs/p0_f9_shared_history/adaln_forward_full_2ep_20261026_215351/contract.json'))
    p.add_argument('--small-reference',default=str(ROOT/'outputs/p0_f9_shared_history/adaln_forward_single_free20_20261025_161644/C_train/contract.json'))
    p.add_argument('--checkpoint-every',type=int,default=64)
    return p


def write_json(path,value):
    fd,temporary=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as handle:
            json.dump(value,handle,ensure_ascii=False,indent=2);handle.write('\n');handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,path)
    finally:
        if Path(temporary).exists(): Path(temporary).unlink()


@contextmanager
def lease(directory):
    with (directory/'suite.lock').open('a+b') as handle:
        if os.fstat(handle.fileno()).st_size==0: handle.write(b'0');handle.flush()
        handle.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as error: raise RuntimeError('this training suite is already running') from error
        try: yield
        finally:
            handle.seek(0)
            if os.name=='nt': msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def manifest(a):
    devices=a.gpus.split(',')
    if len(devices)!=2 or any(not d.strip() for d in devices) or len(set(devices))!=2:
        raise ValueError('exactly TWO distinct GPU IDs required')
    if a.checkpoint_every<1: raise ValueError('positive checkpoint interval required')
    files=sorted(p for p in Path(__file__).parent.iterdir() if p.suffix in ('.py','.sh'))
    return canonical(dict(protocol=SUITE_PROTOCOL,training_only=True,arms=list(ARMS),gpus=a.gpus,
        world_size=2,reference_sha256=dict(full=sha(a.full_reference),small=sha(a.small_reference)),
        implementation={p.relative_to(ROOT).as_posix():sha(p) for p in files},
        checkpoint_every=a.checkpoint_every,
        environment={n:os.environ.get(n) for n in ('TRAIN_CACHE','TRAIN_INFO','DATAROOT','E14','CCR_HISTORY_CACHE')}))


def action(directory,arm,m):
    if not directory.exists() or not any(directory.iterdir()): return 'new'
    contract=directory/'contract.json';result=directory/'results.json';final=directory/'epoch_0001.pt'
    if not contract.is_file(): raise RuntimeError('nonempty arm directory has no contract: '+str(directory))
    c=json.loads(contract.read_text(encoding='utf-8'))
    reference=m['reference_sha256']['full' if arm in ARMS[:2] else 'small']
    if (c.get('protocol')!=PROTOCOL or c.get('arm')!=arm or c.get('world_size')!=2
            or c.get('evaluation')!='DISABLED' or c.get('implementation')!=m['implementation']
            or c.get('reference_contract_sha256')!=reference):
        raise RuntimeError('arm contract differs from this TWO-GPU suite: '+str(directory))
    if final.is_file() and result.is_file():
        r=json.loads(result.read_text(encoding='utf-8'))
        if r.get('status')=='complete' and r.get('contract')==c:
            # Check cursor and rank RNG in the actual final checkpoint, not only
            # a stale/tampered complete flag. No model forward or labels read.
            import torch
            saved=torch.load(final,map_location='cpu',weights_only=False)
            if (saved.get('protocol')!=PROTOCOL or saved.get('contract')!=c or not saved.get('training_only')
                    or saved.get('epoch')!=1 or saved.get('batch')!=0
                    or saved.get('updates')!=c['budget']['stop_updates'] or len(saved.get('rank_rng',[]))!=2):
                raise RuntimeError('invalid completed arm checkpoint: '+str(final))
            return 'complete'
    if (directory/'last.pt').is_file(): return 'resume'
    if {p.name for p in directory.iterdir()}<={'contract.json','evaluation.lock','cache_policy.json'}: return 'initial'
    raise RuntimeError('arm has no recoverable last.pt; preserved directory: '+str(directory))


class ChildRunner:
    def __init__(self): self.child=None;self.stopped=False

    def stop(self,sig,*_):
        self.stopped=True
        if self.child is not None and self.child.poll() is None:
            try:
                if os.name=='posix': os.killpg(self.child.pid,sig)
                else: self.child.send_signal(sig)
            except ProcessLookupError: pass

    def __call__(self,command,env,log):
        # Stream a live combined stdout AND each arm's own log, without
        # buffering an entire epoch or starting another GPU/model process.
        if self.stopped: return 130
        with log.open('a',encoding='utf-8') as handle:
            self.child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,bufsize=1,start_new_session=os.name=='posix')
            if self.stopped: self.stop(signal.SIGTERM)
            try:
                for line in self.child.stdout:
                    handle.write(line);handle.flush();print(line,end='',flush=True)
                return self.child.wait()
            finally:
                self.child.stdout.close()
                if self.child.poll() is None:
                    self.stop(signal.SIGTERM);self.child.wait()
                self.child=None


def execute(a,runner=None):
    runner=runner or ChildRunner();m=manifest(a);out=Path(a.out_dir).resolve()
    out.mkdir(parents=True,exist_ok=True)
    with lease(out):
        path=out/'suite_contract.json'
        if path.exists():
            if json.loads(path.read_text(encoding='utf-8'))!=m: raise RuntimeError('SAME suite/code/settings/GPUs required to continue')
        elif any(p.name!='suite.lock' for p in out.iterdir()): raise RuntimeError('NEW suite directory required')
        else: write_json(path,m)
        state=dict(status='running',training_only=True,world_size=2,completed=[],current=None)
        write_json(out/'suite_state.json',state)
        for arm in ARMS:
            if runner.stopped: state['status']='interrupted';break
            directory=out/arm
            try: mode=action(directory,arm,m)
            except Exception as error:
                state.update(status='failed',current=arm,error=repr(error))
                write_json(out/'suite_state.json',state);raise
            if mode=='complete':
                print('SUITE_SKIP complete '+arm,flush=True);state['completed'].append(arm)
                write_json(out/'suite_state.json',state);continue
            state.update(current=arm,status='running');write_json(out/'suite_state.json',state)
            env=dict(os.environ,ABLATION_GPUS=a.gpus,ABLATION_DIR=str(directory),
                ABLATION_RESUME='1' if mode=='resume' else '0',
                ABLATION_REFERENCE=str(Path(a.full_reference if arm in ARMS[:2] else a.small_reference).resolve()))
            command=['bash',str(ROOT/'tools/paper_ablation_train/run.sh'),arm,
                '--checkpoint-every',str(a.checkpoint_every)]
            print(f'SUITE_START {arm}; mode={mode}; GPUs={a.gpus}; EVAL=OFF; OUT={directory}',flush=True)
            try: code=runner(command,env,out/(arm+'.log'))
            except Exception as error:
                state.update(status='failed',error=repr(error))
                write_json(out/'suite_state.json',state);raise
            if code!=0 or runner.stopped:
                state.update(status='interrupted' if runner.stopped or code==130 else 'failed',exit_code=code)
                write_json(out/'suite_state.json',state);return code or 130
            try:
                if action(directory,arm,m)!='complete': raise RuntimeError('child exited without complete fixed-epoch checkpoint: '+arm)
            except Exception as error:
                state.update(status='failed',error=repr(error))
                write_json(out/'suite_state.json',state);raise
            state['completed'].append(arm);write_json(out/'suite_state.json',state)
        if len(state['completed'])==len(ARMS): state.update(status='complete',current=None)
        write_json(out/'suite_state.json',state)
        print(f'SUITE {state["status"]}; completed={len(state["completed"])}/5; EVAL=OFF; OUT={out}',flush=True)
        return 0 if state['status']=='complete' else 130


def main(argv=None):
    a=parser().parse_args(argv);runner=ChildRunner()
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,runner.stop)
    return execute(a,runner)


if __name__=='__main__': sys.exit(main())
