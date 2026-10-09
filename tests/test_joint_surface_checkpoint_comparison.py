import copy
import json
from pathlib import Path
from types import SimpleNamespace
from threading import Event

import numpy as np
import pytest
import torch

from real_motion.joint_surface_ccr import JointSurfaceCCR
from real_motion.surface_canonical_repair import SurfaceAtlas,augment_evidence,augment_projection
from tools.real_motion import joint_surface_checkpoint_selection as select
from tools.real_motion import joint_surface_checkpoint_evaluation as evaluation
from tools.real_motion.joint_surface_ccr_recovery import payload,restore
from tools.real_motion.static_evidence_selector_common import write_json,finite_json
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import sha256,SEM,DYN
from test_joint_surface_ccr import fixture


def actual_model():
    model,provider,_,_=fixture('cpu');provider.ccr_cache.close()
    return model


def report(value, windows=64):
    m=dict(IoU=53.,mIoU=value,MovingMacro=34.,MovingMicro=30.,
        per_horizon={h:dict(IoU=53.,mIoU=value,MovingMacro=34.,MovingMicro=30.,
            semantic_per_class={str(c):float('nan') for c in SEM},
            moving_per_class={str(c):float('nan') for c in DYN}) for h in ('1.0','2.0','3.0')})
    return dict(windows=windows,baseline=copy.deepcopy(m),variants={'joint':dict(metrics=m)})


def saved_fixture(model,contract,epoch):
    rows=[dict(epoch=i,update=i,evaluation=report(50.-select.AVERAGE_EPOCHS.index(i)
          if i in select.AVERAGE_EPOCHS else 40.)) for i in range(1,epoch+1)]
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    return copy.deepcopy(payload(model,optimizer,np.random.default_rng(4),contract,
        epoch=epoch,batch=0,updates=epoch,executed=epoch*4,
        reports=dict(train_prior=dict(positive_weights=model.columns.positive_weight.tolist()),
                     epochs=rows,**({'final_dev512':report(40.,512)} if epoch==20 else {}))))


def runs_fixture(tmp_path):
    model=actual_model()
    c=dict(protocol=select.TRAIN_PROTOCOL,model=model.configs(),initialization='RANDOM',
        history_frames=4,future_frames=6,thresholds=[.5,None],epochs=20,
        epoch_batches=[1]*20,epoch_batch_sizes=[[4]]*20,schedule_steps=20,seed=1)
    original=tmp_path/'runs'/'original';resumed=tmp_path/'runs'/'resumed'
    original.mkdir(parents=True);resumed.mkdir()
    for epoch in select.AVERAGE_EPOCHS:
        state=saved_fixture(model,c,epoch)
        state['state_dict']['transport.residual_head.bias'].fill_(epoch/10.)
        state['state_dict']['columns.static_readout.3.bias'].fill_(epoch/20.)
        torch.save(state,(original if epoch<=8 else resumed)/f'epoch_{epoch:04d}.pt')
    anchor=saved_fixture(model,c,20);torch.save(anchor,resumed/'last.pt')
    write_json(resumed/'training.json',dict(status='complete',contract=c,reports=anchor['reports']))
    write_json(original/'training.json',dict(status='stopped',contract=c,reports=saved_fixture(model,c,8)['reports']))
    return resumed,original,model,c


def test_parameter_mean_is_float64_whole_network_and_buffers_are_copied():
    model=actual_model();states=[copy.deepcopy(model.state_dict()) for _ in range(5)]
    names=set(dict(model.named_parameters()))
    for i,state in enumerate(states):
        for name in names: state[name].fill_((i+1)/3)
    result=select.average_named_parameters(states,model)
    for name,value in result.items():
        expected=(sum((s[name].double() for s in states))/5).float() if name in names else states[0][name]
        assert torch.equal(value,expected),name
    bad=copy.deepcopy(states);bad[-1]['columns.positive_weight'][0,0]=2
    with pytest.raises(RuntimeError,match='fixed buffers'):
        select.average_named_parameters(bad,model)
    bad=copy.deepcopy(states);bad[-1]['columns.encoder.0.weight'][0,0]=float('nan')
    with pytest.raises(RuntimeError,match='finiteness'):
        select.average_named_parameters(bad,model)
    with pytest.raises(RuntimeError,match='BatchNorm'):
        select.average_named_parameters([{}],torch.nn.BatchNorm1d(2))


def test_discovery_across_resume_dirs_and_readonly_average_artifact(tmp_path):
    run,original,model,c=runs_fixture(tmp_path)
    before={p:sha256(p) for p in tmp_path.rglob('*.pt')}
    audit=select.discover(run,run.parent)
    assert [s['epoch'] for s in audit['sources']]==list(select.AVERAGE_EPOCHS)
    assert audit['anchor']['sha256']==sha256(run/'last.pt')
    out=tmp_path/'comparison';out.mkdir();bundle=select.build_bundle(audit,out)
    assert set(bundle['candidates'])=={'epoch_0006','epoch_0008','epoch_0012',select.AVERAGE_NAME}
    saved,averaged=select.load_evaluation_model(bundle['candidates'][select.AVERAGE_NAME]['path'])
    assert saved['source_epochs']==list(select.AVERAGE_EPOCHS)
    assert 'optimizer' not in saved and 'torch_rng' not in saved and 'epoch' not in saved
    assert torch.allclose(averaged.transport.residual_head.bias,torch.full_like(averaged.transport.residual_head.bias,.9))
    assert torch.equal(averaged.columns.positive_weight,model.columns.positive_weight)
    with pytest.raises(RuntimeError,match='identical'):
        restore(saved,model,torch.optim.AdamW(model.parameters()),np.random.default_rng(),c)
    assert all(sha256(path)==digest for path,digest in before.items())
    # JSON conversion of absent-class NaNs must not break bundle/resume hashes.
    bundle['fingerprint']=select.stable_json_fingerprint(bundle)
    write_json(out/'bundle.json',bundle);restored=json.loads((out/'bundle.json').read_text())
    digest=restored.pop('fingerprint')
    assert select.stable_json_fingerprint(restored)==digest


def test_conflicting_same_epoch_weights_fail_closed(tmp_path):
    run,original,model,c=runs_fixture(tmp_path)
    altered=select.load_cpu_checkpoint(original/'epoch_0006.pt')
    altered=copy.deepcopy(altered);altered['state_dict']['transport.residual_head.bias'].add_(1)
    torch.save(altered,run/'epoch_0006.pt')
    with pytest.raises(RuntimeError,match='conflicting weights'):
        select.discover(run,run.parent)


def test_missing_epoch_and_changed_top5_are_never_substituted(tmp_path):
    run,original,model,c=runs_fixture(tmp_path)
    (original/'epoch_0005.pt').unlink()
    with pytest.raises(RuntimeError,match='missing retained epoch_0005'):
        select.discover(run,run.parent)
    anchor=copy.deepcopy(select.load_cpu_checkpoint(run/'last.pt'))
    anchor['reports']['epochs'][-1]['evaluation']['variants']['joint']['metrics']['mIoU']=99.
    torch.save(anchor,run/'last.pt');write_json(run/'training.json',dict(status='complete',contract=c,reports=anchor['reports']))
    with pytest.raises(RuntimeError,match='predeclared top-five'):
        select.discover(run,run.parent)


def test_save_new_weights_never_overwrites_existing_file(tmp_path):
    path=tmp_path/'mean.pt';select.save_new_weights(path,{'value':1});digest=sha256(path)
    with pytest.raises(FileExistsError):select.save_new_weights(path,{'value':2})
    assert sha256(path)==digest and not list(tmp_path.glob('*.tmp'))


def evaluation_fixture(monkeypatch,device='cpu'):
    model,p,rows,_=fixture(device);record,raw=rows[0]
    record={**record,'scene_name':'dev','t0_token':'a'}
    p.ccr_add_only_natural_bce=True;p.raw_prefetch_workers=1
    def atlas(prep):
        plain=evaluation.ccr._build_inputs_base(p,prep)
        return SurfaceAtlas(plain.world,plain.classes,plain.presence,plain.actor,prep.state['current_pose'],p.pcfg.grid)
    p.ccr_augment_evidence=lambda e,prep:augment_evidence(e,atlas(prep))
    p.ccr_augment_plan=lambda e,plan,prep:augment_projection(e,plan,prep.state['current_pose'],prep.state['world_to_future'],p.pcfg.grid)
    records=[{**record,'t0_token':str(i),'scene_name':'dev'+str(i%2)} for i in range(4)]
    calls=dict(raw=0,moving=0)
    def prefetch(provider,source,records,**kwargs):
        for r in records:
            calls['raw']+=1;yield r,copy.deepcopy(raw)
    monkeypatch.setattr(evaluation,'prefetch_raw_columns',prefetch)
    monkeypatch.setattr(evaluation.ccr,'prefetch_raw_columns',prefetch)
    monkeypatch.setattr(evaluation,'window_from_record',lambda r:SimpleNamespace(t0_token=r['t0_token'],future_tokens=list('123456')))
    def moving(*args,**kwargs):
        calls['moving']+=1
        return [(np.ones(p.pcfg.grid.shape_hwd,bool),[],[]) for _ in range(6)]
    monkeypatch.setattr(evaluation.ccr,'gt_moving_support_sequence',moving)
    actual_prepare=p.prepare_columns
    def prepare(*args,**kwargs):
        result=actual_prepare(*args,**kwargs);result.window=SimpleNamespace(t0_token='a',future_tokens=list('123456'))
        return result
    p.prepare_columns=prepare
    models={'first':model,'second':copy.deepcopy(model)}
    with torch.no_grad():
        models['first'].columns.static_readout[3].bias[0].fill_(3.)
        models['second'].columns.static_readout[3].bias[0].fill_(-3.)
        models['second'].transport.residual_head.bias.add_(.2)
    return p,SimpleNamespace(nusc=None),records,models,calls


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_shared_evaluation_matches_original_single_paths_and_resume(monkeypatch,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('actual CUDA required')
    p,source,records,models,calls=evaluation_fixture(monkeypatch,device)
    hashes={n:select.weight_fingerprint(m.state_dict()) for n,m in models.items()}
    full,performance=evaluation.evaluate_group(p,source,records,models)
    assert calls==dict(raw=4,moving=4) and performance['model_windows']==8
    assert full['first']['variants']['joint']['metrics']!=full['second']['variants']['joint']['metrics']
    for name,model in models.items():
        p.joint,p.model=model,model.transport
        reference=evaluation.ccr.evaluate(p,source,records,model,model.columns)
        for key in ('baseline','variants','action_learning'):
            assert finite_json(full[name][key])==finite_json(reference[key])
    last={};event=Event()
    def save(wi,states,totals):last.update(cursor=wi,states=copy.deepcopy(states))
    def progress(row):
        if row['event']=='all_model_window' and row['window']==2:event.set()
    with pytest.raises(InterruptedError):
        evaluation.evaluate_group(p,source,records,models,save_state=save,progress=progress,
                                   stop_event=event,checkpoint_every=1)
    assert last['cursor']==2
    resumed,_=evaluation.evaluate_group(p,source,records,models,start_window=2,saved=last['states'])
    assert finite_json(full)==finite_json(resumed)
    assert all(select.weight_fingerprint(m.state_dict())==hashes[n] for n,m in models.items())
    bad=copy.deepcopy(last['states']);bad['first']['cursor']=1
    with pytest.raises(RuntimeError,match='cursor'):
        evaluation.evaluate_group(p,source,records,models,start_window=2,saved=bad)
    p.ccr_cache.close()


def test_integer_resume_rejects_scene_and_union_corruption():
    state=evaluation.Accumulator().dump();state['base']['oi'][0]=1
    with pytest.raises(RuntimeError,match='intersection'):
        evaluation.Accumulator.restore(state,0)
    state=evaluation.Accumulator().dump();state['base']['ou'][0]=1
    with pytest.raises(RuntimeError,match='scene counts'):
        evaluation.Accumulator.restore(state,0)


def test_metric_deltas_preserve_absent_classes_and_reject_schema_changes():
    from tools.real_motion.compare_p0_f9_joint_surface_checkpoints import metric_delta
    current=report(41.)['variants']['joint']['metrics']
    reference=finite_json(report(40.)['variants']['joint']['metrics'])
    actual=metric_delta(current,reference)
    assert actual['mIoU']==1. and actual['per_horizon']['1.0']['semantic_per_class']['0'] is None
    reference['per_horizon']['1.0']['semantic_per_class'].pop('1')
    with pytest.raises(RuntimeError,match='class schemas'):
        metric_delta(current,reference)


def test_full_requires_completed_same_population_comparison_and_unchanged_sources(tmp_path):
    from tools.real_motion import eval_p0_f9_joint_surface_mean_full as full_cli
    from tools.real_motion import compare_p0_f9_joint_surface_checkpoints as cli
    run,_,_,_=runs_fixture(tmp_path)
    with pytest.raises(RuntimeError,match='no completed DEV512'):
        full_cli.find_frozen_bundle(run.parent,run)
    out=run.parent/'completed_dev512';out.mkdir()
    bundle=select.build_bundle(select.discover(run,run.parent),out)
    bundle.update(run_directory=str(run.resolve()),runs_root=str(run.parent.resolve()))
    bundle['fingerprint']=select.stable_json_fingerprint(bundle);write_json(out/'bundle.json',bundle)
    result=dict(protocol=cli.EVALUATION_PROTOCOL,status='complete',population='dev512',windows=512,
        bundle_fingerprint=bundle['fingerprint'],reports={select.AVERAGE_NAME:report(40.,512)})
    write_json(out/'comparison.json',result)
    frozen=full_cli.find_frozen_bundle(run.parent,run)
    assert frozen['population']=='full4369' and frozen['selection_frozen'] is True
    assert list(frozen['candidates'])==[select.AVERAGE_NAME]
    assert frozen['candidates'][select.AVERAGE_NAME]['path']==bundle['candidates'][select.AVERAGE_NAME]['path']
    result['windows']=4369;write_json(out/'comparison.json',result)
    with pytest.raises(RuntimeError,match='completed same-bundle DEV512'):
        full_cli.read_frozen_bundle(out,run)
    result['windows']=512;result['note']='changed after freezing';write_json(out/'comparison.json',result)
    # Changing the original report after freezing invalidates the evaluation.
    with pytest.raises(RuntimeError,match='source comparison changed'):
        cli.verify_sources(frozen)
    bad=copy.deepcopy(bundle);bad['candidates'][select.AVERAGE_NAME]['source_epochs']=[5,6,8,12,15]
    bad.pop('fingerprint');bad['fingerprint']=select.stable_json_fingerprint(bad);write_json(out/'bundle.json',bad)
    result['bundle_fingerprint']=bad['fingerprint'];write_json(out/'comparison.json',result)
    with pytest.raises(RuntimeError,match='frozen mean recipe'):
        full_cli.read_frozen_bundle(out,run)


def test_cli_real_average_shared_inference_and_readonly_interruption_resume(monkeypatch,tmp_path):
    from tools.real_motion import compare_p0_f9_joint_surface_checkpoints as cli
    from real_motion.canonical_repair_context import FixedCanonicalCache
    from real_motion.canonical_repair_execution import CanonicalCpuExecution
    run,original,model,contract=runs_fixture(tmp_path)
    p,source,records,_,_=evaluation_fixture(monkeypatch)
    files={name:tmp_path/name for name in ('config','dev_cache','dev_info','base_checkpoint','population_manifest')}
    for path in files.values():path.write_bytes(b'fixture')
    contract.update(data={name:sha256(files[name]) for name in ('dev_cache','dev_info','base_checkpoint')},
        dataroot=str(tmp_path.resolve()),runtime_config_fingerprint=select.stable_json_fingerprint({}),
        torch_version=str(torch.__version__),implementation=select.training_implementation(Path(__file__).resolve().parents[1]),
        dev_manifest_fingerprint='fixed',val_history_namespace='val',reference_execution=False)
    # Keep the complete training lineage metadata compatible with this synthetic input boundary.
    for directory in (run,original):
        info=json.loads((directory/'training.json').read_text());info['contract']=contract
        for path in directory.glob('*.pt'):
            saved=copy.deepcopy(select.load_cpu_checkpoint(path));saved['contract']=contract
            if path.name=='last.pt':
                saved['reports']['final_dev512']=report(40.,len(records));info['reports']=saved['reports']
            torch.save(saved,path)
        write_json(directory/'training.json',info)
    before={path:sha256(path) for directory in (run,original) for path in directory.iterdir() if path.is_file()}
    monkeypatch.setattr(cli,'CLEAN_SHA256',sha256(files['base_checkpoint']))
    monkeypatch.setattr(cli,'require_cuda',lambda _:torch.device('cpu'))
    monkeypatch.setattr(cli,'load_runtime_config',lambda *args:{})
    monkeypatch.setattr(cli,'make_prepare_config',lambda _:p.pcfg)
    monkeypatch.setattr(cli,'DEV64_WINDOWS',2);monkeypatch.setattr(cli,'DEV512_WINDOWS',4);monkeypatch.setattr(cli,'VAL_WINDOWS',4)
    keys=[(r['scene_name'],r['t0_token']) for r in records]
    monkeypatch.setattr(cli,'load_manifest',lambda _:({'parent_keys':keys,'manifest_fingerprint':'fixed',
                        'selected_key_fingerprint':cli.DEV64_FP},keys[:2],None))
    monkeypatch.setattr(cli,'load_cache',lambda _:({},records))
    monkeypatch.setattr(cli,'NuScenesWindowSource',lambda *args,**kwargs:source)
    monkeypatch.setattr(cli,'CachedColumnSource',lambda s,_:s)
    def provider(*args):
        p.joint=args[-2];p.model=p.joint.transport
        return p
    monkeypatch.setattr(cli,'PilotProvider',provider)
    def setup(provider,args):
        provider.ccr_cache=FixedCanonicalCache(1,neighbors=False)
        provider.ccr_execution=CanonicalCpuExecution('numpy')
        provider.ccr_val_history_cache=SimpleNamespace(namespace='val',stats=lambda:dict(writes=0),close=lambda:None)
    monkeypatch.setattr(cli.surface,'setup',setup)
    (tmp_path/'val').mkdir();(tmp_path/'val'/'manifest.json').write_text('{}')
    argv=['--run-dir',str(run),'--runs-root',str(run.parent),'--dataroot',str(tmp_path),
          '--ccr-val-history-cache',str(tmp_path),'--ccr-cpu-execution','numpy','--checkpoint-every','1']
    argv += [item for name,path in files.items() for item in ('--'+name.replace('_','-'),str(path))]
    stopped=Event();actual=cli.evaluate_group
    def interrupt(*args,**kwargs):
        log=kwargs['progress']
        def progress(row):
            log(row)
            if row['event']=='all_model_window' and row['window']==2:stopped.set()
        kwargs['progress']=progress
        return actual(*args,**kwargs)
    monkeypatch.setattr(cli,'evaluate_group',interrupt)
    interrupted=tmp_path/'interrupted'
    assert cli.main(stopped,argv+['--out-dir',str(interrupted)])==130
    state=json.loads((interrupted/'evaluation_state.json').read_text())
    assert state['completed_windows']==2
    monkeypatch.setattr(cli,'evaluate_group',actual)
    assert cli.main(argv=argv+['--out-dir',str(interrupted),'--resume'])==0
    full=tmp_path/'full'
    assert cli.main(argv=argv+['--out-dir',str(full)])==0
    a=json.loads((interrupted/'comparison.json').read_text());b=json.loads((full/'comparison.json').read_text())
    assert a['reports']==b['reports'] and a['performance']['reused_prefix_windows']==2
    assert len(a['reports'])==5 and 'epoch_0020_existing' in a['reports']
    assert a['execution']['raw_prefetch_workers']==4 and a['execution']['raw_io_workers']==1
    assert 'MovingMacro' in (full/'summary.txt').read_text() and 'horizon=3.0s' in (full/'summary.txt').read_text()
    assert all(sha256(path)==digest for path,digest in before.items())
    # A completed comparison may never be silently rerun or overwritten.
    with pytest.raises(SystemExit):cli.main(argv=argv+['--out-dir',str(full),'--resume'])
    forbidden=run/'must_not_be_created'
    with pytest.raises(SystemExit):cli.main(argv=argv+['--out-dir',str(forbidden)])
    assert not forbidden.exists()
    # Expand beyond DEV512 and evaluate ONLY the frozen mean; no re-averaging,
    # subset reference reuse or implicit single-epoch/full population mixture.
    from tools.real_motion import eval_p0_f9_joint_surface_mean_full as full_cli
    source_before={path:sha256(path) for path in full.iterdir() if path.is_file()}
    discovered=full_cli.find_frozen_bundle(tmp_path,run)
    assert list(discovered['candidates'])==[select.AVERAGE_NAME]
    outside=[{**records[0],'t0_token':'outside'+str(i),'scene_name':'outside'} for i in range(2)]
    expanded=records+outside
    monkeypatch.setattr(cli,'VAL_WINDOWS',6);monkeypatch.setattr(cli,'load_cache',lambda _:({},expanded))
    stop_full=Event()
    def interrupt_full(*args,**kwargs):
        log=kwargs['progress']
        def progress(row):
            log(row)
            if row['event']=='all_model_window' and row['window']==3:stop_full.set()
        kwargs['progress']=progress
        return actual(*args,**kwargs)
    monkeypatch.setattr(cli,'evaluate_group',interrupt_full)
    all_val=tmp_path/'all_val_interrupted'
    full_args=argv+['--source-bundle-dir',str(full)]
    assert full_cli.main(stop_full,full_args+['--out-dir',str(all_val)])==130
    prefix=json.loads((all_val/'evaluation_state.json').read_text())
    assert prefix['completed_windows']==3 and list(prefix['states'])==[select.AVERAGE_NAME]
    monkeypatch.setattr(cli,'evaluate_group',actual)
    assert full_cli.main(argv=full_args+['--out-dir',str(all_val),'--resume'])==0
    uninterrupted=tmp_path/'all_val_full'
    assert full_cli.main(argv=full_args+['--out-dir',str(uninterrupted)])==0
    restored=json.loads((all_val/'full_validation.json').read_text())
    complete=json.loads((uninterrupted/'full_validation.json').read_text())
    assert restored['reports']==complete['reports']
    assert complete['population']=='full4369' and complete['windows']==6
    assert complete['performance']['model_windows']==6 and complete['performance']['raw_windows']==6
    assert complete['reports'][select.AVERAGE_NAME]['scenes']==3
    assert 'epoch_0020_existing' not in complete['reports'] and 'best_per_metric' not in complete
    assert 'delta_joint_vs_epoch20' not in complete and not (uninterrupted/'comparison.json').exists()
    for path,digest in source_before.items():assert sha256(path)==digest
    for path,digest in before.items():assert sha256(path)==digest
    weights=json.loads((full/'bundle.json').read_text())['candidates'][select.AVERAGE_NAME]['path']
    _,only_mean=select.load_evaluation_model(weights)
    expected,_=evaluation.evaluate_group(p,source,expanded,{select.AVERAGE_NAME:only_mean})
    assert finite_json(expected)==complete['reports']
    with pytest.raises(SystemExit):full_cli.main(argv=full_args+['--out-dir',str(uninterrupted),'--resume'])
