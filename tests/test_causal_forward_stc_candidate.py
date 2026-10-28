"""Frozen candidate: exact source, same local math, no old defaults, integer merge."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.causal_forward_stc_candidate import method,recipe,common,run,parallel
from tools.causal_forward_local.geometry_screen import position_row
from tools.causal_forward_local.range_calibrate import radius as screened_radius
from test_causal_forward_local_research import velocity_fixture


def test_formal_state_and_radius_are_exactly_the_locally_screened_math():
    row=velocity_fixture();state=row[3][0]['state'];state['source_world_points']=[np.asarray([v['centroid_world']+[.1,0.,0.]]) for v in state['current']]
    raw=dict(row[1],_column_causal_preparation=dict(prepared_state=state));row=row[0],raw,row[2],row[3]
    a=position_row(row,.5,.5);b=method.state_row(row,.5,.5)
    for n,v in a[0].items():np.testing.assert_array_equal(v,b[0][n])
    for n in a[1]['_shared_source_track'].pair_velocity:
        np.testing.assert_array_equal(a[1]['_shared_source_track'].pair_velocity[n],b[1]['_shared_source_track'].pair_velocity[n])
    for name in ('source_z_t0','source_world_points','source_rel_xy'):
        np.testing.assert_array_equal(a[3][0]['state'][name],b[3][0]['state'][name])
    from real_motion.geometry import OccupancyGrid
    grid=OccupancyGrid(shape_hwd=(8,8,4));poses=torch.eye(4,dtype=torch.float64)[None].repeat(6,1,1)
    poses[:,0,3]=torch.arange(1,7)
    torch.testing.assert_close(method.radius(grid,np.eye(4),poses),screened_radius(grid,np.eye(4),poses),rtol=0,atol=0)


def test_calibration_is_full_grid_six_frames_no_inplace_mutation():
    torch.manual_seed(21);x=torch.randn(6,18,91);rho=torch.rand(6,91);before=x.clone()
    actual=method.calibrated_logits(x,rho);expected=x.clone();expected[:,17]+=-2+rho
    torch.testing.assert_close(actual,expected,rtol=0,atol=0);torch.testing.assert_close(x,before,rtol=0,atol=0)
    with pytest.raises(ValueError):method.calibrated_logits(x[:3],rho[:3])


def test_exact_downloaded_source_SHA_and_explicit_calibration_metadata(tmp_path,monkeypatch):
    from test_causal_forward_average import body_and_contract
    body,c=body_and_contract();saved=dict(protocol=c['protocol'],contract=c,epoch=2,batch=0,updates=10292,models={'gt':body.state_dict()})
    path=tmp_path/'epoch_0002.pt';torch.save(saved,path)
    monkeypatch.setattr(common.original,'select_source',lambda *a:path)
    with pytest.raises(RuntimeError,match='exact downloaded'):common.select_source(None,tmp_path)
    monkeypatch.setattr(recipe,'SOURCE_SHA256',common.base.sha256(path))
    assert common.select_source(None,tmp_path)==path
    before=copy.deepcopy(c);loaded,model=common.load_frozen(path,'cpu')
    assert c==before and loaded['evaluation_method']['calibration_scalars']==2
    assert 'GT future labels' in loaded['evaluation_method']['calibration'] and loaded['evaluation_method']['same_model_both_modalities']
    assert model.configs()==body.configs() and not any(p.requires_grad for p in model.parameters())
    for n,v in body.state_dict().items():torch.testing.assert_close(model.state_dict()[n],v,rtol=0,atol=0)


def test_same_recipe_and_integer_merge_all4369_one_GPU():
    from test_causal_forward_all import records,states_for
    keys=[[r['scene_name'],r['t0_token']] for r in records()];states=states_for(keys,'sourceSHA',8,'stc')
    for s in states:
        c=s['contract']['base'];c['protocol']=run.PROTOCOL;c['source_contract']={'evaluation_method':recipe.specification()}
    reports,raw=parallel.merge_states(states,keys,'sourceSHA',8,'stc')
    assert sum(np.asarray(raw['gt/transport']['confusion'])[0].ravel())==4369*8
    assert 'stc_gt/semantic_refine' in reports
    states[1]['contract']['base']['source_contract']['evaluation_method']['free_intercept']=-1.
    with pytest.raises(RuntimeError,match='SAME frozen'):parallel.merge_states(states,keys,'sourceSHA',8,'stc')


def test_summary_cannot_claim_partial_19_29_or_automatic_promotion():
    values=dict(per_horizon={},average=dict(mIoU=19.1,IoU=29.2,MovingMicro=18.))
    v=dict(status='running',contract=dict(windows=4369),reports={'stc_gt/semantic_refine':values})
    assert 'met=pending' in run.summary(v)
    v['status']='complete';assert 'met=True' in run.summary(v) and 'SUPERVISED OUTPUT CALIBRATION' in run.summary(v)
    v['reports']['stc_gt/semantic_refine']['average']['mIoU']=18.9;assert 'met=False' in run.summary(v)


def test_real_current_forward_all_six_future_GT_occupancy_poison_CPU():
    from test_fpn_adaln_zero import fixture
    from tools.causal_forward_full.model import make_model
    from tools.causal_forward_train.data import forward_window
    old,base,row=fixture();body=make_model(old.transport.config,feature_width=16,z_bins=4,memory_width=4).eval().requires_grad_(False)
    predictor=common.CandidatePredictor(body,base.provider.pcfg,'cpu',1)
    try:
        raw={k:row[1][k] for k in ('history_occ','history_observed','history_poses')}
        prepared=predictor.prepare(row[0],raw,times=np.arange(4)*.5)
        before={n:v.clone() for n,v in body.state_dict().items()};points=prepared[3][0]['memory_reads']['world'].copy()
        poison=prepared[0],dict(prepared[1],future_gt_occ=object()),prepared[2],prepared[3]
        actual=predictor.forward(poison,poses=row[1]['future_poses'])
        expected=forward_window(body,predictor.provider,method.state_row(poison),poses=row[1]['future_poses'],fresh=True)
        rho=method.radius(base.provider.pcfg.grid,raw['history_poses'][-1],expected['poses'])
        torch.testing.assert_close(actual['logits'],method.calibrated_logits(expected['logits'],rho),rtol=0,atol=0)
        assert actual['logits'].shape[0]==6 and not actual['logits'].requires_grad
        np.testing.assert_array_equal(prepared[3][0]['memory_reads']['world'],points)
        for n,v in before.items():torch.testing.assert_close(body.state_dict()[n],v,rtol=0,atol=0)
    finally:predictor.close();base.close()


def test_real_worker_metric_engine_stop_resume_readonly_source(tmp_path,monkeypatch):
    import test_causal_forward_all as inherited
    original_make=inherited.make_source
    def pinned_source(root):
        source,c,files=original_make(root)
        monkeypatch.setattr(recipe,'SOURCE_SHA256',common.base.sha256(source/'epoch_0002.pt'))
        return source,c,files
    monkeypatch.setattr(inherited,'run',run);monkeypatch.setattr(inherited,'make_source',pinned_source)
    inherited.test_worker_actual_metric_engine_interrupt_stc_resume_and_source_readonly(tmp_path,monkeypatch)
    contract=json.loads((tmp_path/'worker/contract.json').read_text())
    assert contract['source_contract']['evaluation_method']==recipe.specification()


def test_candidate_two_children_same_GPU_and_resume_keep_same_recipe(tmp_path,monkeypatch):
    from test_causal_forward_all import make_source,records,states_for
    source,_,files=make_source(tmp_path);checkpoint=source/'epoch_0002.pt';sha=common.base.sha256(checkpoint)
    monkeypatch.setattr(recipe,'SOURCE_SHA256',sha)
    c=common.validate_source(torch.load(checkpoint,weights_only=False));_,keys=run.select_jobs(records());p=parallel.runner;calls=[]
    monkeypatch.setattr(p.torch.cuda,'device_count',lambda:1);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr(p.base,'load_cache',lambda *a:({},records()))
    monkeypatch.setattr(p.base,'load_runtime_config',lambda *a:None)
    monkeypatch.setattr(p.base,'make_prepare_config',lambda *a:SimpleNamespace(grid=SimpleNamespace(shape_hwd=(2,2,2))))
    class Child:
        def __init__(self,cmd,**kwargs):
            calls.append((cmd,kwargs['env']['CUDA_VISIBLE_DEVICES']));self.pid=1000+len(calls);self.returncode=0
            assert 'tools.causal_forward_stc_candidate.run' in cmd
            rank=int(cmd[cmd.index('--shard-index')+1]);folder=Path(cmd[cmd.index('--out-dir')+1]);folder.mkdir(exist_ok=True)
            for kind in ('occ','stc'):
                state=states_for(keys,sha,8,kind,c)[rank];state['contract']['base']['protocol']=run.PROTOCOL
                p.base.write_json(folder/f'{kind}_state.json',state)
            p.base.write_json(folder/'contract.json',state['contract']['base'])
            p.base.write_json(folder/'evaluation.json',dict(status='complete',contract=state['contract']['base']))
        def poll(self):return self.returncode
        def wait(self):return self.returncode
        def terminate(self):self.returncode=130
    monkeypatch.setattr(p.subprocess,'Popen',Child)
    out=tmp_path/'candidate_all';args=['--source-dir',str(source),'--out-dir',str(out),'--dataroot',str(tmp_path),'--stc-root',str(tmp_path)]
    for n,path in files.items():args+=['--'+n.replace('_','-'),str(path)]
    assert parallel.main(argv=args)==0 and [gpu for _,gpu in calls]==['0','0']
    report=json.loads((out/'evaluation.json').read_text());assert report['status']=='complete' and len(report['reports'])==4
    assert report['contract']['source_contract']['evaluation_method']==recipe.specification()
    assert parallel.main(argv=args+['--resume'])==0 and all('--resume' in cmd for cmd,_ in calls[2:])
    assert p.base.sha256(checkpoint)==sha
