"""Two-GPU sequential queue, completion/continuation, load balancing and budgets."""
import itertools
import json
import os
import sys
from pathlib import Path
import pytest
import torch
from tools.paper_ablation_train import suite
from tools.paper_ablation_train.partition import partition,source_count
from tools.paper_ablation_train.control import PROTOCOL


def args(tmp_path):
    full=tmp_path/'full.json';small=tmp_path/'small.json'
    full.write_text('full reference');small.write_text('small reference')
    return suite.parser().parse_args(['--out-dir',str(tmp_path/'queue'),
        '--full-reference',str(full),'--small-reference',str(small)])


def arm_state(directory,arm,m,*,complete):
    directory.mkdir(exist_ok=True)
    c=dict(protocol=PROTOCOL,training_only=True,arm=arm,world_size=2,evaluation='DISABLED',
        implementation=m['implementation'],reference_contract_sha256=m['reference_sha256']['full' if arm in suite.ARMS[:2] else 'small'],
        budget=dict(stop_updates=3))
    (directory/'contract.json').write_text(json.dumps(c))
    saved=dict(protocol=PROTOCOL,contract=c,training_only=True,rank_rng=[{},{}],
        epoch=1 if complete else 0,batch=0 if complete else 1,updates=3 if complete else 1)
    torch.save(saved,directory/'last.pt')
    if complete:
        torch.save(saved,directory/'epoch_0001.pt')
        (directory/'results.json').write_text(json.dumps(dict(status='complete',contract=c)))


def test_every_B4_equal_windows_and_optimal_source_balance_no_drop():
    for n in range(1,5):
        for sizes in itertools.product((0,1,7),repeat=n):
            group=[dict(id=i,features=[0]*s) for i,s in enumerate(sizes)]
            for step in (0,1):
                shards=partition(group,2,step)
                assert sorted(r['id'] for g in shards for r in g)==list(range(n))
                assert abs(len(shards[0])-len(shards[1]))<=1
                if n==4: assert list(map(len,shards))==[2,2]
                loads=[sum(map(source_count,g)) for g in shards]
                best=min(abs(sum(sizes[i] for i in ids)-sum(sizes[i] for i in range(n) if i not in ids))
                    for ids in itertools.combinations(range(n),(n+1)//2))
                assert abs(loads[0]-loads[1])==best
                assert all([r['id'] for r in g]==sorted(r['id'] for r in g) for g in shards)
            assert partition(group,2,1)==partition(group,2,0)[::-1]
    with pytest.raises(ValueError): partition([],2)


def test_suite_runs_five_sequential_two_GPU_jobs_and_skips_complete_on_resume(tmp_path,monkeypatch):
    a=args(tmp_path);calls=[];m=suite.manifest(a)
    # Stale per-arm environment cannot redirect every arm to one directory or
    # accidentally use the full reference for small arms.
    monkeypatch.setenv('ABLATION_DIR','WRONG');monkeypatch.setenv('ABLATION_REFERENCE','WRONG')
    class Runner:
        stopped=False
        def __call__(self,command,env,log):
            arm=command[2];calls.append(arm)
            assert env['ABLATION_GPUS']=='0,1' and env['ABLATION_RESUME']=='0'
            assert Path(env['ABLATION_DIR'])==Path(a.out_dir)/arm
            assert Path(env['ABLATION_REFERENCE'])==Path(a.full_reference if arm in suite.ARMS[:2] else a.small_reference)
            assert '--checkpoint-every' in command and log.name==arm+'.log'
            arm_state(Path(env['ABLATION_DIR']),arm,m,complete=True)
            return 0
    assert suite.execute(a,Runner())==0 and calls==list(suite.ARMS)
    state=json.loads((Path(a.out_dir)/'suite_state.json').read_text())
    assert state['status']=='complete' and state['completed']==list(suite.ARMS)
    class FailIfCalled:
        stopped=False
        def __call__(self,*a): pytest.fail('completed arms must not run again')
    assert suite.execute(a,FailIfCalled())==0


def test_suite_failure_stops_queue_then_resumes_only_unfinished_arm(tmp_path):
    a=args(tmp_path);m=suite.manifest(a);calls=[]
    class StopAfterFirst:
        stopped=False
        def __call__(self,command,env,log):
            arm=command[2];calls.append(arm)
            arm_state(Path(env['ABLATION_DIR']),arm,m,complete=False)
            return 130
    assert suite.execute(a,StopAfterFirst())==130 and calls==[suite.ARMS[0]]
    assert json.loads((Path(a.out_dir)/'suite_state.json').read_text())['status']=='interrupted'
    class Continue:
        stopped=False
        def __call__(self,command,env,log):
            arm=command[2];calls.append(arm)
            assert env['ABLATION_RESUME']==('1' if arm==suite.ARMS[0] else '0')
            arm_state(Path(env['ABLATION_DIR']),arm,m,complete=True)
            return 0
    assert suite.execute(a,Continue())==0
    assert calls==[suite.ARMS[0],*suite.ARMS]


def test_suite_rejects_single_GPU_changed_manifest_or_wrong_final_cursor(tmp_path):
    a=args(tmp_path)
    a.gpus='0'
    with pytest.raises(ValueError,match='TWO'): suite.manifest(a)
    a.gpus='0,0'
    with pytest.raises(ValueError,match='TWO'): suite.manifest(a)
    a.gpus='0,1';m=suite.manifest(a);out=Path(a.out_dir);out.mkdir()
    suite.write_json(out/'suite_contract.json',dict(m,world_size=1))
    with pytest.raises(RuntimeError,match='SAME'): suite.execute(a)
    arm=suite.ARMS[0];directory=out/arm;arm_state(directory,arm,m,complete=True)
    path=directory/'epoch_0001.pt';saved=torch.load(path,weights_only=False)
    saved['updates']=2;torch.save(saved,path)
    with pytest.raises(RuntimeError,match='checkpoint'): suite.action(directory,arm,m)


def test_bash_launchers_require_two_GPUs_and_queue_has_no_eval_flags():
    root=Path(suite.__file__).parent
    sh=(root/'run.sh').read_text()
    assert '--nproc_per_node=2' in sh and 'n==2' in sh
    help=suite.parser().format_help()
    assert all(x not in help for x in ('--eval','--stc','--dev','--epochs'))


def test_child_runner_streams_log_without_buffering_training_or_ignoring_stop(tmp_path,capsys):
    runner=suite.ChildRunner();log=tmp_path/'child.log'
    code=runner([sys.executable,'-u','-c','print("TRAIN_QUEUE_LOG")'],dict(os.environ),log)
    assert code==0 and log.read_text()=='TRAIN_QUEUE_LOG\n'
    assert 'TRAIN_QUEUE_LOG' in capsys.readouterr().out
    runner.stopped=True
    assert runner(['must_not_start'],dict(os.environ),log)==130
