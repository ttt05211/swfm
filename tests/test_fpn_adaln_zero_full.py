"""Actual full planner/JSON checks, isolated loader and reused CUDA recovery."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from tools.fpn_adaln_zero_full import run, model
from tools.fpn_adaln_zero import run as pilot, model as pilot_model
from tools.fpn_unet_full import run as original


def control_fixture(tmp_path):
    train = [dict(scene_name='TRAIN',t0_token=str(i),features=[None]) for i in range(20430)]
    dev = [dict(scene_name='VAL',t0_token=str(i),history_tokens=['past']) for i in range(4369)]
    eval_keys = [['VAL',str(i)] for i in range(64)]
    path = tmp_path/'reference.json'
    path.write_text(json.dumps(dict(contract=dict(eval_keys=eval_keys))),encoding='utf-8')
    args = train,dev,path,20261011
    keys,evaluation,groups = run._plan(*args)
    stc = tmp_path/'VAL'/'past'/'labels.npz'
    stc.parent.mkdir(parents=True); stc.write_bytes(b'inventory metadata only')
    stat = stc.stat()
    c = dict(protocol=original.PROTOCOL,model=dict(variant=run.CONTROL_VARIANT),
        epochs=2,total_updates=sum(map(len,groups)),epochs_updates=list(map(len,groups)),
        train_keys=keys,eval_keys=eval_keys,seed=20261011,world_size=2,parameters=2520442,
        known='TRAIN_OCC_STC_dense_valid',registration='none',source_policy='shared_pinned',
        objective=dict(CE=1.,Lovasz=1.,ego=0.,motion=0.,existence=0.),
        learning_rates=[5e-4,3e-4],overrides=[],schedule='whole_2_epoch_cosine_floor0.01',
        precision='BF16_network_FP64_geometry_full6',initialization='RANDOM_no_pilot_weights',
        checkpoint_every=2000,evaluation='epoch_end_only_same64',data={},implementation={},
        execution=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cpu_workers=4,prefetch_workers=2),
        STC_inventory=[(str(stc.resolve()),stat.st_size,stat.st_mtime_ns)])
    report = dict(windows=64,keys=eval_keys,variants={
        n:dict(mIoU=1.,IoU=2.,MovingMicro=3.) for n in run.NAMES})
    control = dict(status='complete',epoch=2,updates=c['total_updates'],contract=c,
        evaluations={str(update):copy.deepcopy(report) for update in (len(groups[0]),c['total_updates'])})
    control = json.loads(json.dumps(control))
    (tmp_path/'results.json').write_text(json.dumps(control),encoding='utf-8')
    return control,args,(keys,evaluation,groups)


def test_actual_full_plan_JSON_control_and_strict_mismatch(tmp_path,monkeypatch):
    control,args,expected = control_fixture(tmp_path)
    a = SimpleNamespace(control_dir=tmp_path,stc_root=tmp_path,seed=20261011,
        motion_lr=5e-4,repair_lr=3e-4,override=[],checkpoint_every=2000,cpu_workers=4,prefetch_workers=2)
    ctx = SimpleNamespace(world_size=2)
    run.check_settings(a,ctx,control)
    with pytest.raises(RuntimeError,match='seed'):
        run.check_settings(SimpleNamespace(**dict(vars(a),seed=22)),ctx,control)
    monkeypatch.setattr(run,'_launch',lambda *a:run.runner.plan(*args))
    old_factory,old_plan = run.runner.make_model,run.runner.plan
    actual = run.launch(a,None,ctx)
    assert actual == expected
    assert len(actual[0]) == 20430 and len(actual[2]) == 2
    for groups in actual[2]:
        assert sum(map(len,groups)) == 20430
    for field in ('train_keys','eval_keys','epochs_updates','total_updates','STC_inventory'):
        changed = copy.deepcopy(control)
        if field.endswith('_keys'):
            changed['contract'][field][:2] = changed['contract'][field][1::-1]
            if field == 'eval_keys':
                for e in changed['evaluations'].values():e['keys'] = changed['contract'][field]
        elif field == 'epochs_updates':
            changed['contract'][field][0] += 1; changed['contract'][field][1] -= 1
            first = str(changed['contract'][field][0]-1)
            changed['evaluations'][str(int(first)+1)] = changed['evaluations'].pop(first)
        elif field == 'total_updates':
            changed['contract'][field] += 1; changed['contract']['epochs_updates'][1] += 1
            changed['updates'] += 1
            changed['evaluations'][str(changed['updates'])] = changed['evaluations'].pop(str(control['updates']))
        else:
            changed['contract'][field][0][1] += 1
        (tmp_path/'results.json').write_text(json.dumps(changed),encoding='utf-8')
        message = ('STC inventory' if field == 'STC_inventory' else
            'epochs_updates|total_updates' if field == 'total_updates' else field)
        with pytest.raises(RuntimeError,match=message):run.launch(a,None,ctx)
        assert run.runner.make_model is old_factory and run.runner.plan is old_plan
    assert original.PROTOCOL != run.PROTOCOL and original.launch is not run.runner.launch
    assert run.runner.forward_window is original.forward_window and run.runner.evaluate is original.evaluate
    assert 'full TRAIN20430x2' in run.parser().format_help()


def test_report_all_epochs_and_reject_pilot_and_changed_comparison(tmp_path):
    control,_,_ = control_fixture(tmp_path)
    candidate = copy.deepcopy(control)
    candidate['contract']['protocol'] = run.PROTOCOL
    candidate['contract']['parameters'] = 2816122
    for report in candidate['evaluations'].values():
        for m in report['variants'].values():m.update(mIoU=2.,IoU=1.,MovingMicro=2.)
    text = run.compare(candidate,control)
    assert text.count('===== update=') == 2 and '"MovingMicro": -1.0' in text
    assert '2816122' in text and 'NOT FPS' in text
    candidate['contract']['known'] = 'lidar_known'
    with pytest.raises(RuntimeError,match='known'):run.compare(candidate,control)
    changed = copy.deepcopy(control); changed['contract']['protocol'] = pilot.PROTOCOL
    (tmp_path/'results.json').write_text(json.dumps(changed),encoding='utf-8')
    with pytest.raises(RuntimeError):run.reference(tmp_path)


def test_loader_same_pilot_architecture_protocol_and_fingerprint():
    from test_fpn_adaln_zero import fixture
    body,runner,_ = fixture()
    try:
        path = Path('tools/fpn_adaln_zero/model.py')
        body.experiment = dict(candidate_implementation={path.as_posix():original.base.sha256(path)})
        saved = dict(protocol=run.PROTOCOL,contract=dict(protocol=run.PROTOCOL,model=body.configs()),
            models={'gt':body.state_dict()})
        restored = model.checkpoint_model(saved)
        assert restored.configs() == body.configs() and not any(p.requires_grad for p in restored.parameters())
        for k,v in body.state_dict().items():torch.testing.assert_close(restored.state_dict()[k],v,rtol=0,atol=0)
        with pytest.raises(RuntimeError):pilot_model.checkpoint_model(saved)
        with pytest.raises(RuntimeError):model.checkpoint_model(dict(saved,protocol=pilot.PROTOCOL))
        broken = copy.deepcopy(saved)
        broken['contract']['model']['experiment']['candidate_implementation'][path.as_posix()] = 'wrong'
        with pytest.raises(RuntimeError,match='implementation'):model.checkpoint_model(broken)
    finally:runner.close()


def test_actual_CUDA_two_epoch_Adam_RNG_resume_only_epoch_end_evaluations(tmp_path,monkeypatch):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():pytest.skip('BF16 CUDA')
    import test_fpn_unet_full as baseline_test
    # The original actual-CUDA test performs real H6 forward/backward, saves
    # epoch1, restores Adam/rankRNG at epoch2 and ensures final eval is not repeated.
    monkeypatch.setattr(baseline_test,'run',run.runner)
    monkeypatch.setattr(baseline_test,'model',model)
    monkeypatch.setattr(run.runner,'parser',run._parser)
    monkeypatch.setattr(run.runner,'launch',run._launch)
    baseline_test.test_actual_cuda_two_epoch_pause_resume_Adam_RNG_and_only_two_evals(tmp_path,monkeypatch)
