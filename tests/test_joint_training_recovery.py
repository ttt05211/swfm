"""Crash-safe publish, immutable eval snapshots and never signal a guessed PID."""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from tools.real_motion import joint_training_recovery as recovery
from tools.real_motion import manage_p0_f9_joint_training as manage
from tools.real_motion.static_evidence_selector_common import write_json


def test_atomic_resume_retains_one_previous_and_failed_save_preserves_last(tmp_path):
    path=tmp_path/'last.pt'
    recovery.save_resume_checkpoint(path,{'update':1,'tensor':torch.tensor([1.])})
    first=path.read_bytes()
    recovery.save_resume_checkpoint(path,{'update':2,'tensor':torch.tensor([2.])})
    assert (tmp_path/'last.previous.pt').read_bytes() == first
    second=path.read_bytes()
    with patch.object(recovery.torch,'save',side_effect=OSError('simulated disk full')):
        with pytest.raises(OSError,match='disk full'):recovery.save_resume_checkpoint(path,{'update':3})
    assert path.read_bytes() == second and (tmp_path/'last.previous.pt').read_bytes() == first
    assert not list(tmp_path.glob('*.tmp'))
    assert torch.load(path,weights_only=False)['update'] == 2


def test_snapshot_tracks_one_open_file_not_subsequent_atomic_last_replacement(tmp_path):
    source=tmp_path/'last.pt';dest=tmp_path/'snapshot.pt'
    source.write_bytes(b'old checkpoint'*100000)
    original=source.read_bytes();replacement=tmp_path/'new.pt';replacement.write_bytes(b'new checkpoint')
    # Linux allows replacement of an open inode; Windows does not. Emulate the
    # Linux old-descriptor/new-path split on Windows without requiring a server.
    opened=tmp_path/'opened_version.pt';opened.write_bytes(original)
    actual=Path.open
    def open_with_replacement(path,*args,**kwargs):
        handle=actual(opened if os.name == 'nt' and path == source and args == ('rb',) else path,*args,**kwargs)
        if path == source and args == ('rb',):os.replace(replacement,source)
        return handle
    with patch.object(Path,'open',open_with_replacement):digest=recovery.snapshot_checkpoint(source,dest)
    assert source.read_bytes() == b'new checkpoint' and dest.read_bytes() == original
    assert digest == hashlib.sha256(original).hexdigest()
    with pytest.raises(RuntimeError,match='NEW'):recovery.snapshot_checkpoint(source,dest)


def test_evaluation_rng_guard_restores_sampling_global_numpy_python_torch_on_cancel():
    rng=np.random.default_rng(19);state=copy.deepcopy(rng.bit_generator.state)
    ts=torch.get_rng_state().clone();ns=np.random.get_state();rs=random.getstate()
    with pytest.raises(InterruptedError):
        with recovery.preserve_training_rng(rng):
            rng.random(200);np.random.rand(20);torch.rand(20);random.random()
            raise InterruptedError('window boundary')
    assert rng.bit_generator.state == state and torch.equal(ts,torch.get_rng_state()) and rs == random.getstate()
    restored=np.random.get_state()
    assert ns[0] == restored[0] and np.array_equal(ns[1],restored[1]) and ns[2:] == restored[2:]


def test_snapshot_does_not_delete_a_destination_created_by_someone_else(tmp_path):
    source=tmp_path/'last.pt';dest=tmp_path/'snapshot.pt';source.write_bytes(b'checkpoint')
    actual=Path.open
    def race(path,*args,**kwargs):
        if path == dest and args == ('xb',):
            with actual(dest,'wb') as f: f.write(b'other writer')
        return actual(path,*args,**kwargs)
    with patch.object(Path,'open',race),pytest.raises(FileExistsError): recovery.snapshot_checkpoint(source,dest)
    assert dest.read_bytes() == b'other writer' and source.read_bytes() == b'checkpoint'


def test_snapshot_rejects_in_place_mutation_and_removes_only_its_partial_copy(tmp_path):
    source=tmp_path/'last.pt';dest=tmp_path/'snapshot.pt';source.write_bytes(b'checkpoint')
    before=SimpleNamespace(st_size=10,st_mtime_ns=1)
    after=SimpleNamespace(st_size=10,st_mtime_ns=2)
    with patch.object(recovery.os,'fstat',side_effect=[before,after]),pytest.raises(RuntimeError,match='IN PLACE'):
        recovery.snapshot_checkpoint(source,dest)
    assert not dest.exists() and source.read_bytes() == b'checkpoint'


@pytest.mark.parametrize('bad',[{'prior_cursor':-1},{'prior_cursor':5},{'prior_completed':False,'attempted_updates':1},
    {'prior_counts':{'generation':[np.nan,0],'refine':[0,0,0]}},{'prior_completed':True,'prior_cursor':3}])
def test_prior_resume_fails_closed_on_invalid_partial_counts(bad):
    ck={'prior_completed':False,'prior_cursor':2,'attempted_updates':0,'prior_counts':{'generation':[3,1],'refine':[2,3,4]}}
    ck.update(bad)
    with pytest.raises(RuntimeError,match='prior resume'):recovery.validate_prior_resume(ck,4)
    assert recovery.validate_prior_resume({'attempted_updates':1},4)[:2] == (True,4)


def manager_fixture(tmp_path):
    directory=tmp_path/'original'/'model';directory.mkdir(parents=True)
    ck={'protocol':'full','training_contract':{'frozen':True},'train_keys':(('train','a'),),
        'dev_keys':(('dev','b'),),'seed':3,'epochs':15,'window_batch_size':4,'source_budget':128,
        'model_configs':{'motion':{'history_frames':4}},'checkpoint_role':'resume_last'}
    args={'config':str(tmp_path/'config.yaml'),'out_dir':str(directory),'epochs':15,'seed':3,
        'window_batch_size':4,'source_budget':128,'history_frames':4,'override':['A=1'],
        'paired_control':False,'persistent_sampling_pool':True,'prewarm_causal_cache':True,
        'resume':None,'sampling_workers':4,'checkpoint_every':128}
    torch.save(ck,directory/'last.pt')
    write_json(directory/'execution_contract.json',{**ck,'arguments':args,'launch_cwd':str(tmp_path)})
    return directory,ck


def test_resume_launcher_reads_original_recipe_tuple_keys_and_never_changes_epochs_or_batch(tmp_path):
    directory,ck=manager_fixture(tmp_path)
    before={p:p.read_bytes() for p in directory.iterdir()}
    new=tmp_path/'new'
    command,env,out=manage.resume_command(directory,new_out=new)
    assert out == new and not new.exists()  # command construction is read-only
    value=lambda key:command[command.index('--'+key)+1]
    assert value('epochs') == '15' and value('window-batch-size') == '4' and value('source-budget') == '128'
    assert value('history-frames') == '4' and value('seed') == '3' and value('sampling-workers') == '4'
    assert value('resume') == str(directory/'last.pt') and value('out-dir') == str(new/'model')
    assert '--persistent-sampling-pool' in command and '--prewarm-causal-cache' not in command
    assert value('override') == 'A=1' and env['SWFM_COLUMN_CPU_BACKEND'] == 'native'
    assert all(p.read_bytes() == b for p,b in before.items())
    new.mkdir()
    with pytest.raises(RuntimeError,match='NEW resume'):manage.resume_command(directory,new_out=new)
    ck['epochs']=20;torch.save(ck,directory/'last.pt')
    with pytest.raises(RuntimeError,match='mismatch'):manage.resume_command(directory)


def test_resume_performance_overrides_do_not_change_scientific_recipe_or_source_checkpoint(tmp_path):
    directory, ck = manager_fixture(tmp_path)
    ck['attempted_updates'] = 20694; torch.save(ck, directory/'last.pt')
    before = {p: p.read_bytes() for p in directory.iterdir()}
    command, env, out = manage.resume_command(directory, sampling_workers=6, profile_every=32,
                                             expected_update=20694)
    value = lambda key: command[command.index('--'+key)+1]
    assert value('sampling-workers') == '6' and value('profile-every') == '32'
    for key, expected in (('epochs', '15'), ('window-batch-size', '4'), ('source-budget', '128'), ('seed', '3')):
        assert value(key) == expected
    assert value('resume') == str(directory/'last.pt') and env['SWFM_COLUMN_CPU_HORIZONS'] == '1'
    assert not out.exists() and all(p.read_bytes() == data for p, data in before.items())
    with pytest.raises(RuntimeError, match='expected update'):
        manage.resume_command(directory, expected_update=20693)
    for count in (0, 9, -1):
        with pytest.raises(ValueError): manage.resume_command(directory, sampling_workers=count)


def test_stop_refuses_wrong_directory_and_pid_reuse_never_sends_any_signal(tmp_path):
    directory,ck=manager_fixture(tmp_path)
    state={'out_dir':str(directory),'pid':12345,'process_start_token':'old','phase':'training'}
    with patch.object(manage,'process_start_token',return_value='new'),patch.object(manage.os,'kill') as kill:
        assert not manage.matching_process(directory,state)
        kill.assert_not_called()
    with pytest.raises(RuntimeError,match='identity mismatch'):manage.matching_process(directory,{**state,'out_dir':str(tmp_path)})
    write_json(directory/'runtime_status.json',state)
    with patch.object(manage,'process_start_token',return_value=None),patch.object(manage.os,'kill') as kill:
        ck.update(attempted_updates=12,cursor_epoch=0,cursor_batch=12);torch.save(ck,directory/'last.pt')
        assert manage.request_stop(directory) == 0
        kill.assert_not_called()


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
def test_gpu_resume_override_is_readonly_and_performance_only(tmp_path, backend):
    directory, ck = manager_fixture(tmp_path)
    before = {p: p.read_bytes() for p in directory.iterdir()}
    command, _, out = manage.resume_command(directory, column_feature_backend=backend)
    value = lambda key: command[command.index('--'+key)+1]
    assert value('column-feature-backend') == backend
    assert value('window-batch-size') == '4' and value('source-budget') == '128'
    assert value('epochs') == '15' and value('resume') == str(directory/'last.pt')
    assert not out.exists() and all(p.read_bytes() == data for p, data in before.items())
    with pytest.raises(ValueError, match='must be cpu or gpu'):
        manage.resume_command(directory, column_feature_backend='float32_warp')


def test_stop_scopes_single_term_and_waits_no_force_kill(tmp_path):
    directory,ck=manager_fixture(tmp_path)
    ck.update(attempted_updates=12,cursor_epoch=0,cursor_batch=12);torch.save(ck,directory/'last.pt')
    state={'out_dir':str(directory),'pid':12345,'process_start_token':'same','phase':'training'}
    write_json(directory/'runtime_status.json',state)
    with patch.object(manage,'matching_process',side_effect=[True,True,False,False]),patch.object(manage.os,'kill') as kill:
        assert manage.request_stop(directory) == 0
        kill.assert_called_once_with(12345,manage.signal.SIGTERM)


def test_completed_extension_launcher_is_explicit_dual_and_parent_is_read_only(tmp_path):
    directory, ck = manager_fixture(tmp_path)
    ck.update(cursor_epoch=15, cursor_batch=0, attempted_updates=77172, target_updates=77172, prior_completed=True)
    torch.save(ck, directory/'last.pt')
    before = {p: p.read_bytes() for p in directory.iterdir()}
    command, env, out = manage.resume_command(directory, extend_to=20, gpus='0,1', column_feature_backend='gpu')
    value = lambda key: command[command.index('--'+key)+1]
    assert '--extend-completed-run' in command and '--distributed' in command
    assert 'torch.distributed.run' in command and '--nproc_per_node=2' in command
    assert value('epochs') == '20' and value('window-batch-size') == '4' and value('source-budget') == '128'
    assert env['CUDA_VISIBLE_DEVICES'] == '0,1' and value('resume') == str(directory/'last.pt')
    assert not out.exists() and before == {p: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(RuntimeError, match='world size'): manage.resume_command(directory, gpus='0,1')
    for ids in ('0,0', '0,1,2', 'banana', ''):
        with pytest.raises(ValueError): manage.resume_command(directory, extend_to=20, gpus=ids)
    ck['cursor_batch'] = 1; torch.save(ck, directory/'last.pt')
    with pytest.raises(RuntimeError, match='fully completed'): manage.resume_command(directory, extend_to=20, gpus='0,1')


def test_stopped_dual_extension_resume_retains_epochs_rank_count_and_recipe(tmp_path):
    directory, ck = manager_fixture(tmp_path)
    ck.update(epochs=20, distributed_training={'world_size': 2}, continuation={'parent': 'old'})
    torch.save(ck, directory/'last.pt')
    contract = json.loads((directory/'execution_contract.json').read_text())
    contract.update(epochs=20); contract['arguments'].update(epochs=20, extend_completed_run=True, distributed=True)
    write_json(directory/'execution_contract.json', contract)
    command, env, _ = manage.resume_command(directory, gpus='0,1')
    assert '--extend-completed-run' not in command and '--distributed' in command
    assert command[command.index('--epochs')+1] == '20' and env['CUDA_VISIBLE_DEVICES'] == '0,1'
    with pytest.raises(RuntimeError, match='world size'): manage.resume_command(directory, gpus='0')
    with pytest.raises(RuntimeError, match='ORIGINAL'): manage.resume_command(directory, extend_to=25, gpus='0,1')
