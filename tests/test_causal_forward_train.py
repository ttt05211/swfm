"""Matched fresh initialization, real forward/cache/eval, plan/JSON and recovery."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.causal_forward_train import control,data,model,run,report
from tools.causal_forward_train.calibration import CALIBRATION
from tools.fpn_adaln_zero import model as old_model


def fake_control():
    from test_fpn_adaln_zero import fake_control as previous
    c = previous();c['contract'].update(protocol=control.CONTROL_PROTOCOL,parameters=2816122,
        model=dict(variant='FPN_UNet_ego_AdaLN_Zero_cross_self_FFN_v1',source_policy='shared_pinned',
                   experiment=dict(candidate_implementation={})),
        seed=20261011,world_size=2,learning_rates=[5e-4,3e-4],overrides=[],
        schedule='whole_1_epoch_cosine_floor0.01',precision='BF16_network_FP64_geometry_full6',
        data={},implementation={},execution=dict(torch=str(torch.__version__),cuda=torch.version.cuda,
            cpu_workers=4,prefetch_workers=2))
    return c


def test_control_resolver_no_score_or_date_selection(tmp_path):
    first = tmp_path/'fpn_adaln_zero_20pct_1ep_first';first.mkdir()
    (first/'results.json').write_text(json.dumps(fake_control()))
    assert control.resolve(tmp_path) == first.resolve()
    second = tmp_path/'fpn_adaln_zero_20pct_1ep_second';second.mkdir()
    other = fake_control();other['evaluations']['1']['variants']['occ_gt_transport']['mIoU'] = 99.
    (second/'results.json').write_text(json.dumps(other))
    with pytest.raises(RuntimeError,match='2 completed'):control.resolve(tmp_path)
    assert control.resolve(tmp_path,first) == first.resolve()


def test_same_fresh_weights_RNG_architecture_and_separate_checkpoint():
    from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
    cfg = LocalSTWMV17Config(history_frames=4,decoder_blocks=2)
    torch.manual_seed(20261011);old = old_model.make_model(cfg);rng = torch.get_rng_state().clone()
    torch.manual_seed(20261011);new = model.make_model(cfg,expected_control_spec=old.configs())
    assert torch.equal(torch.get_rng_state(),rng)
    assert sum(p.numel() for p in new.parameters()) == sum(p.numel() for p in old.parameters()) == 2816122
    for k,v in new.state_dict().items():torch.testing.assert_close(v,old.state_dict()[k],rtol=0,atol=0)
    saved = dict(protocol=run.PROTOCOL,contract=dict(protocol=run.PROTOCOL,source_policy=model.SOURCE_POLICY,
        model=new.configs()),models={'gt':new.state_dict()})
    restored = model.checkpoint_model(saved)
    assert restored.configs() == new.configs() and not any(p.requires_grad for p in restored.parameters())
    with pytest.raises(RuntimeError):old_model.checkpoint_model(saved)
    with pytest.raises(RuntimeError):model.checkpoint_model(dict(saved,protocol=control.CONTROL_PROTOCOL))


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_live_forward_matches_frozen_forward_policy_and_real_H6_backward(device):
    if device == 'cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):pytest.skip('BF16 CUDA')
    from test_fpn_adaln_zero import fixture
    from tools.causal_forward_track.adapter import TrackPredictor
    from tools.shared_history_experiments.full_gt.engine import semantic_loss
    body,base,row = fixture(device)
    predictor = data.ForwardPredictor(body,base.provider.pcfg,device,workers=1)
    raw = {k:row[1][k] for k in data.HISTORICAL}
    try:
        new = predictor.prepare(row[0],raw,times=np.arange(4)*.5)
        frozen,_ = TrackPredictor.prepare_pair(predictor,row[0],raw,times=np.arange(4)*.5)
        for key in ('kta_displacement_xy_m','anchors_xy_t0_m','source_class_id'):
            np.testing.assert_array_equal(new[0][key],frozen['forward'][0][key])
        for field in ('shared_reads','memory_reads'):
            for key,value in new[3][0][field].items():
                np.testing.assert_array_equal(value,frozen['forward'][3][0][field][key])
        # Poison old backgrounds/targets: prediction MUST build fresh Strong and
        # never consume future occupancy. Supervision is supplied only to loss.
        poisoned = dict(new[1],future_poses=row[1]['future_poses'],future_gt_occ=object(),
            _column_causal_preparation=dict(prepared_state=dict(new[3][0]['state'],column_backgrounds=object())))
        actual = data.forward_window(body,predictor.provider,(new[0],poisoned,new[2],new[3]))
        assert actual['logits'].shape[0] == 6 and actual['memory'].shape[0] == 6
        loss = semantic_loss(actual['logits'],row[1]['future_gt_occ'])[0].mean();loss.backward()
        assert torch.isfinite(loss) and body.columns.head.weight.grad.abs().sum() > 0
        assert body.transport.residual_head.weight.grad.abs().sum() > 0
        with pytest.raises(ValueError):predictor.prepare(row[0],dict(raw,future_gt_occ=object()),times=np.arange(4)*.5)
    finally:predictor.close();base.close()


def test_cached_training_path_strips_old_geometry_and_updates_record_without_mutation(monkeypatch):
    from test_fpn_adaln_zero import fixture
    from tools.source_scene_stage2 import track as old_track
    body,base,row = fixture()
    try:
        original = row[1];rec = dict(row[0],features=np.empty((len(row[0]['source_class_id']),0)))
        kta = rec['kta_displacement_xy_m'].copy();observed = []
        def cache(key,raw,builder):
            observed.append(raw)
            assert set(raw) == {*data.HISTORICAL,'future_poses'}
            return builder(),False
        tokens = list(map(str,rec['history_tokens']))
        source = SimpleNamespace(nusc=SimpleNamespace(get=lambda table,t:dict(timestamp=tokens.index(t)*500000)))
        provider = SimpleNamespace(pcfg=base.provider.pcfg,strong=base.provider.strong,
            ccr_history_cache_mode='require',ccr_val_history_cache_mode='require',
            ccr_history_cache_source=source,ccr_val_history_cache_source=object(),
            track_cache=SimpleNamespace(get_or_build=cache))
        monkeypatch.setattr(data.PilotProvider,'load_raw_columns',lambda *a,**kw:original)
        monkeypatch.setattr(old_track,'from_raw',lambda *a,**kw:pytest.fail('old matcher executed'))
        value = data.ForwardProvider.load_raw_columns(provider,source,rec,include_gt=True)
        assert set(value['_column_causal_preparation']['prepared_state']) == set(data.PURE_STATE)
        updated = data.updated_record(rec,value)
        assert updated['features'] is rec['features'] and updated['future_tokens'] is rec['future_tokens']
        np.testing.assert_array_equal(rec['kta_displacement_xy_m'],kta)
        assert original['_column_causal_preparation']['prepared_state']['column_backgrounds'] is not None
        # Use a closable generator, exactly as the real prefetch implementation.
        def batches(*a,**kw):yield [(rec,value,row[2],row[3])]
        monkeypatch.setattr(data,'original_batches',batches)
        result = list(data.batches())[0][0]
        np.testing.assert_array_equal(result[0]['kta_displacement_xy_m'],updated['kta_displacement_xy_m'])
        assert result[1]['future_gt_occ'] is original['future_gt_occ'] and len(observed)==1
    finally:base.close()


def test_actual_candidate_launch_JSON_plan_and_strict_rejection(tmp_path,monkeypatch):
    train = [dict(scene_name='TRAIN',t0_token=str(i),features=[None]) for i in range(20430)]
    dev = [dict(scene_name='VAL',t0_token=str(i),history_tokens=['past']) for i in range(4369)]
    value = fake_control();ref = tmp_path/'reference.json';ref.write_text(json.dumps(value))
    args = train,dev,ref,20261011,1
    keys,evaluation,groups = run._plan(*args)
    c = value['contract'];c.update(train_keys=[list(k) for k in keys],total_updates=len(groups[0]))
    value['updates'] = len(groups[0])
    stc = tmp_path/'VAL/past/labels.npz';stc.parent.mkdir(parents=True);stc.write_bytes(b'only stat')
    s = stc.stat();c['STC_inventory'] = [[str(stc.resolve()),s.st_size,s.st_mtime_ns]]
    path = tmp_path/'results.json';path.write_text(json.dumps(value))
    a = SimpleNamespace(control_dir=tmp_path,stc_root=tmp_path,seed=20261011,motion_lr=5e-4,repair_lr=3e-4,
        override=[],cpu_workers=4,prefetch_workers=2)
    ctx = SimpleNamespace(world_size=1)
    with pytest.raises(RuntimeError,match='ONE GPU'):run.check_settings(a,SimpleNamespace(world_size=2),value)
    monkeypatch.setattr(run,'_launch',lambda *a:run.runner.plan(*args))
    before = run.runner.plan
    assert run.launch(a,None,ctx) == (keys,evaluation,groups)
    assert run.runner.plan is before
    value['contract']['eval_keys'][:2] = value['contract']['eval_keys'][1::-1]
    value['evaluations']['1']['keys'] = copy.deepcopy(value['contract']['eval_keys'])
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError,match='eval_keys'):run.launch(a,None,ctx)
    assert run.runner.plan is before
    assert run.original.DirectProvider is not data.ForwardProvider
    from tools.fpn_unet import evaluation as original_evaluation
    assert original_evaluation.DirectPredictor is not data.ForwardPredictor
    assert run.runner.evaluate is not run.original.evaluate


def test_three_arm_report_checks_provenance_and_keeps_negative_results(tmp_path):
    from tools.causal_forward_track.run import PROTOCOL as FROZEN_PROTOCOL
    from tools.causal_forward_train.audit import PROTOCOL as AUDIT_PROTOCOL
    a = fake_control();a['contract']['STC_inventory'] = []
    control_dir = tmp_path/'control';control_dir.mkdir()
    cp = control_dir/'results.json';cp.write_text(json.dumps(a))
    ckpt = control_dir/'epoch_0001.pt';ckpt.write_bytes(b'report does not load model weights')
    keys = a['contract']['eval_keys']
    b = dict(status='complete',keys=keys,contract=dict(protocol=FROZEN_PROTOCOL,keys=keys,stc_age=False,output_calibration=CALIBRATION,
        horizon_chunk=6,source_contract=a['contract'],source=dict(checkpoint_sha256=control.sha(ckpt))),reports={})
    for name in control.NAMES:
        kind,route = name.split('_gt_',1)
        for arm,score in (('baseline',1.1),('forward',1.2)):
            b['reports'][kind+'/'+arm+'/'+route] = dict(average=dict(mIoU=score,IoU=2.,MovingMicro=3.))
            if route=='semantic_refine':
                b['reports'][kind+'/'+arm+'/'+route+'_free_minus1'] = dict(average=dict(mIoU=score+.1,IoU=2.2,MovingMicro=2.8))
    c = copy.deepcopy(a);c['contract'].update(protocol=run.PROTOCOL,source_policy=model.SOURCE_POLICY,
        world_size=1,
        model=dict(source_policy=model.SOURCE_POLICY,experiment=dict(control_results_sha256=control.sha(cp))))
    for m in c['evaluations']['1']['variants'].values():m.update(mIoU=1.5,MovingMicro=2.5)
    c['evaluations']['1']['calibration'] = CALIBRATION
    for kind in ('occ_gt','stc_gt'):
        c['evaluations']['1']['variants'][kind+'_semantic_refine_free_minus1'] = dict(mIoU=1.6,IoU=2.1,MovingMicro=2.4)
    audit = dict(status='complete',rows=[dict(key=k) for k in keys],reports={},
        contract=dict(protocol=AUDIT_PROTOCOL,keys=keys,control_results_sha256=control.sha(cp)))
    for folder,value in (('frozen_B',b),('C_train',c),('pair_audit',audit)):
        out = tmp_path/folder;out.mkdir();(out/'results.json').write_text(json.dumps(value))
    text = report.combine(control_dir,tmp_path)
    result = json.loads((tmp_path/'results.json').read_text())
    delta = result['deltas']['occ_gt_semantic_refine']
    assert delta['C_minus_B_training_policy']['mIoU'] == pytest.approx(.3)
    assert delta['C_minus_recorded_A_total']['MovingMicro'] == -.5
    assert delta['fresh_A_minus_recorded']['mIoU'] == pytest.approx(.1)
    assert result['execution_comparison']['control_training_GPUs']==2
    assert result['execution_comparison']['candidate_training_GPUs']==1
    assert result['fixed_free_minus1']['stc']['C_forward_trained']['delta']['mIoU']==pytest.approx(.1)
    assert result['fixed_free_minus1']['occ']['A_fresh']['delta']['MovingMicro']==pytest.approx(-.2)
    assert 'A_fresh' in text and 'NOT FPS' in text
    b['contract']['stc_age'] = True
    (tmp_path/'frozen_B/results.json').write_text(json.dumps(b))
    with pytest.raises(RuntimeError,match='no age'):report.combine(control_dir,tmp_path)


def test_forward_evaluator_both_modalities_and_integer_resume(tmp_path,monkeypatch):
    from test_fpn_adaln_zero import fixture
    body,base,row = fixture()
    try:
        predictor = SimpleNamespace(provider=base.provider,history=data.DirectHistory(base.provider,ram_mib=0))
        raw = {k:row[1][k] for k in data.HISTORICAL}
        new = data.ForwardPredictor.prepare(predictor,row[0],raw,times=np.arange(4)*.5)
        new = new[0],dict(new[1],future_poses=row[1]['future_poses'],future_gt_occ=row[1]['future_gt_occ']),new[2],new[3]
        for token,sem in zip(row[0]['history_tokens'],raw['history_occ']):
            folder = tmp_path/row[0]['scene_name']/str(token);folder.mkdir(parents=True)
            np.savez(folder/'labels.npz',semantics=sem)
        calls = [];actual = run.evaluation.forward_window
        def forward(*a,**kw):calls.append(1);return actual(*a,**kw)
        def moving(*a,**kw):assert len(calls)==2;return []
        monkeypatch.setattr(run.evaluation,'forward_window',forward)
        monkeypatch.setattr(run.evaluation.gt,'gt_moving_support_sequence',moving)
        monkeypatch.setattr(run.evaluation.gt,'moving_support_masks',lambda _,shape:np.ones((6,*shape),bool))
        states = []
        first = run.evaluation.evaluate(body,base.provider,SimpleNamespace(nusc=None),[[new]],stc_root=tmp_path,
            save=lambda x:states.append(copy.deepcopy(x)))
        again = run.evaluation.evaluate(body,base.provider,SimpleNamespace(nusc=None),[[new]],stc_root=tmp_path,saved=states[-1])
        assert first['keys']==again['keys'] and len(calls)==2
        assert len(first['variants'])==6 and first['calibration']==CALIBRATION
        restored,_ = run.evaluation.restore(states[-1])
        for name,fields in states[-1]['metrics'].items():
            for field,counts in fields.items():np.testing.assert_array_equal(getattr(restored[name],field),counts)
    finally:base.close()


def test_actual_cuda_runner_keeps_new_policy_resume_and_Adam(tmp_path,monkeypatch):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():pytest.skip('BF16 CUDA')
    import test_fpn_unet as baseline
    monkeypatch.setattr(baseline,'run',run.runner);monkeypatch.setattr(baseline,'model',model)
    monkeypatch.setattr(run.runner,'parser',run._parser);monkeypatch.setattr(run.runner,'launch',run._launch)
    actual = run.runner.train_step;observed = []
    def step(body,optimizer,provider,rows,ctx,**kw):
        if kw['update']==2:
            partial = torch.load(tmp_path/'new/last.pt',weights_only=False)
            state = optimizer.state_dict()['state']
            for key,values in partial['optimizers']['gt']['state'].items():
                for name,v in values.items():
                    if isinstance(v,torch.Tensor):torch.testing.assert_close(state[key][name].cpu(),v.cpu(),rtol=0,atol=0)
            torch.testing.assert_close(torch.get_rng_state(),partial['rank_rng'][0]['torch'],rtol=0,atol=0)
            observed.append(1)
        return actual(body,optimizer,provider,rows,ctx,**kw)
    monkeypatch.setattr(run.runner,'train_step',step)
    baseline.test_actual_cuda_runner_resume_and_final_eval_once(tmp_path,monkeypatch)
    final = torch.load(tmp_path/'new/last.pt',weights_only=False)
    assert final['contract']['source_policy'] == model.SOURCE_POLICY
    assert final['contract']['model']['source_policy'] == model.SOURCE_POLICY and observed==[1]
