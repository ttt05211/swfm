"""Real Surface transport/CCR rollout plus open-loop, recovery and ABI guards."""
import copy
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from real_motion.joint_surface_ccr import JointSurfaceCCR
from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.surface_canonical_repair import augment_evidence, augment_projection
from real_motion.canonical_causal_repair import build_canonical_evidence, map_canonical_evidence
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion import joint_surface_long_rollout_common as common
from tools.real_motion import eval_p0_f9_joint_surface_long_rollout as cli
from tools.real_motion import ccr_screen_common as ccr
from tools.real_motion.static_evidence_selector_common import write_json, CLEAN_SHA256
from test_joint_long_rollout import state_fixture, window, record


def provider_fixture(empty=False, device='cpu'):
    history,pcfg,state = state_fixture(empty)
    model = JointSurfaceCCR(LocalSTWMV17Config(history_frames=4,d_model=16,semantic_dim=4,
        blocks=1,decoder_blocks=1),width=16,z_bins=4).to(device).eval().requires_grad_(False)
    # Skip only checkpoint disk loading; all model/Strong/renderer math is real.
    provider = common.SurfaceRolloutProvider.__new__(common.SurfaceRolloutProvider)
    provider.joint=model;provider.model=model.transport;provider.reference=None;provider.control=None
    provider.reference_enabled=False;provider.latents_checked=True
    provider.pcfg=pcfg;provider.strong=StrongW2DetConfig();provider.workers=2;provider.device=torch.device(device)
    w=window('dev','a');r=state['rec'];r.update(record(w))
    raw=dict(history_occ=history,history_observed=np.ones_like(history,bool),history_poses=[np.eye(4)]*4,
             future_poses=[np.eye(4)]*6,future_gt_occ=None)
    raw['_column_causal_preparation']=common.causal_geometry(raw,state,provider)
    return provider,w,r,raw


@pytest.mark.parametrize('empty',[False,True])
@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('mode',['numpy','native_parallel'])
def test_real_surface_two_blocks_and_handoff_equal_normal_surface_inference(empty,device,mode):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('actual CUDA required')
    threads=torch.get_num_threads();torch.set_num_threads(1)
    provider,w,r,raw=provider_fixture(empty,device)
    before={k:v.clone() for k,v in provider.joint.state_dict().items()}
    execution=common.SurfaceBlockExecution(provider,mode=mode,workers=2,query_workers=2,graphs=True)
    try:
        first=provider.prepare_columns(None,r,include_gt=False,raw_window=raw)
        pred1,edits,_,prob=execution.predict(first)
        common.verify_first_block(provider,r,first,pred1,prob,execution)
        # Independent normal THREE-second Surface pipeline used by the current
        # checkpoint evaluator, including its actual weighted ADD raw0.5 rule.
        ev=ccr.build_inputs(provider,first);plan=ccr.map_inputs(provider,ev,first)
        expected=ccr.probabilities(provider.joint.columns,ev,plan,first.outputs,provider.device)
        dense=ccr.compose_canonical(first.baseline,ev,plan,expected[...,0],expected[...,1],thresholds=(.5,None))
        assert np.array_equal(prob,expected)
        common.rollout.assert_dense_equal(pred1,dense)
        assert edits['removed']==0 and all(np.array_equal(a[b!=17],b[b!=17]) for a,b in zip(pred1,first.baseline))
        carry=common.handoff_from_prepared(first,pred1[-1])
        poses=[np.eye(4) for _ in range(12)]
        for hi,p in enumerate(poses):p[0,3]=.06*(hi+1);p[2,3]=.003*hi
        red=common.synthetic_preparation(pred1,raw,poses,w,provider)
        # Any accidental cache lookup / raw source read of a future t0 is fatal.
        class Bomb:
            def require(self,*a):raise AssertionError('real future cache read')
        provider.rollout_val_cache=Bomb()
        provider.load_raw_columns=lambda *a,**k: (_ for _ in ()).throw(AssertionError('future source load'))
        second=common.synthetic_preparation(pred1,raw,poses,w,provider,handoff=carry,
                                            frames=red.state['components_by_frame'])
        pred2,_,_,prob2=execution.predict(second);execution.verify(second,pred2,prob2)
        assert second.state['current'] is red.state['current']
        assert second.state['motion_handoff_audit']['memory_only_sources_added']==0
        assert np.array_equal(second.raw['history_occ'],np.stack(pred1[-4:]))
        assert second.raw['future_gt_occ'] is None and len(second.window.history_tokens)==4
        assert np.array_equal(second.state['current_pose'],poses[5])
        assert all(x.shape==provider.pcfg.grid.shape_hwd for x in pred1+pred2)
        assert all(torch.equal(before[k],v) for k,v in provider.joint.state_dict().items())
    finally:execution.close();torch.set_num_threads(threads)


def test_synthetic_surface_preparation_preserves_handed_velocity_and_skips_local_memory(monkeypatch):
    provider,w,r,raw=provider_fixture()
    first=provider.prepare_columns(None,r,include_gt=False,raw_window=raw)
    # Use unedited transport outputs to guarantee a visible unique source.
    carry=common.handoff_from_prepared(first,first.baseline[-1])
    original=common.rollout.build_four_history_state
    seen=[]
    def build(*a,**k):
        result=original(*a,**k);seen.append(result);return result
    monkeypatch.setattr(common.rollout,'build_four_history_state',build)
    import tools.real_motion.joint_column_full_common as old
    monkeypatch.setattr(old,'build_future_static_memory_only',lambda *a,**k:(_ for _ in ()).throw(AssertionError('obsolete memory')))
    prep=common.synthetic_preparation(first.baseline,raw,[np.eye(4)]*12,w,provider,handoff=carry)
    assert prep.state['velocities'] is seen[0]['velocities']
    assert prep.registrations is prep.raw['_column_causal_preparation']['registrations']
    assert prep.memory is None and prep.footprints is None
    assert prep.state['motion_handoff_audit']==seen[0]['motion_handoff_audit']


@pytest.mark.parametrize('field',['future_gt_occ','future_observed','future_mask','future_annotations'])
def test_reject_future_supervision_even_when_unused(field):
    provider,_,r,raw=provider_fixture()
    raw[field]=np.zeros((6,*provider.pcfg.grid.shape_hwd),np.uint8)
    with pytest.raises(RuntimeError,match='future'):
        provider.prepare_columns(None,r,include_gt=False,raw_window=raw)
    with pytest.raises(RuntimeError,match='supervised'):
        provider.prepare_columns(None,r,include_gt=True,raw_window=raw)


def test_early_genie_start_is_rebuilt_and_never_reads_six_history_or_gt():
    provider,w,r,raw=provider_fixture()
    early={k:r[k] for k in ('scene_name','t0_token','history_tokens','future_tokens')}
    early['history_tokens']=early['history_tokens'][-4:]
    prep=provider.prepare_columns(None,early,include_gt=False,raw_window={k:v for k,v in raw.items() if not k.startswith('_')})
    common.rollout.assert_four_inputs_equal(r,prep.state['rec'])
    assert len(prep.window.history_tokens)==4 and prep.memory is None
    assert not any(k in prep.state['rec'] for k in ('target_yaw_rad','existence','supervised_source'))


@pytest.mark.parametrize('slim',[False,True])
def test_first_real_history_only_cache_verified_support_reused_not_modified(monkeypatch,slim):
    provider,w,r,raw=provider_fixture()
    state=raw['_column_causal_preparation']
    cache_data=copy.deepcopy(state)
    if slim:
        cache_data['prepared_state']={k:v for k,v in state['prepared_state'].items()
            if k in ('current_pose','current','source_world_points','source_rel_xy','source_z_t0','world_to_future')}
    seen=[]
    class Cache:
        def require(self,key,actual):
            common.require_history_only(actual);seen.append(key);return cache_data
    provider.rollout_val_cache=Cache()
    source=SimpleNamespace(pose=lambda _:np.eye(4))
    import real_motion.prepared as prepared
    def load(src,scene,tokens,free,workers):
        assert src is source and tokens==w.history_tokens
        return raw['history_occ'],raw['history_observed']
    monkeypatch.setattr(prepared,'_load_history_semantics_and_observation',load)
    actual=provider.load_raw_columns(source,r,include_gt=False)
    keys=set(cache_data['prepared_state'])
    prep=provider.prepare_columns(source,r,include_gt=False,raw_window=actual)
    assert seen==[('dev','a')] and actual['_ccr_history_cache_hit']
    assert actual['_column_causal_preparation'] is cache_data
    assert set(cache_data['prepared_state'])==keys and provider.columns_checked
    assert prep.raw['future_gt_occ'] is None


def orchestration_fixture(monkeypatch):
    provider,w,r,raw=provider_fixture(empty=True)
    ws=[w,window('dev','b')];rows=[r,{**r,**record(ws[1])}]
    # Orchestration fake forecasts make lifecycle checks cheap; real math is
    # covered above. The source asserts all routes finished BEFORE future GT.
    calls=[];event=Event();stop=[False]
    pred=[np.full(provider.pcfg.grid.shape_hwd,17,np.uint8) for _ in range(6)]
    pred[1][0,0,0]=11
    class Execution:
        def predict(self,prep):
            calls.append('forecast')
            return copy.deepcopy(pred),{'added':1,'removed':0,'changed':1},{'head':0.},np.zeros((0,6,2),np.float32)
        def verify(self,*args):calls.append('verify_second')
    def prepare(source,r,*,include_gt,raw_window):
        assert not include_gt
        return SimpleNamespace(state={'rec':r,'current':[]},raw=raw_window)
    provider.prepare_columns=prepare
    monkeypatch.setattr(cli,'prefetch_raw_columns',lambda p,s,rr,**kw:((row,copy.deepcopy(raw)) for row in rr))
    monkeypatch.setattr(common,'verify_first_block',lambda *a:calls.append('verify_first'))
    monkeypatch.setattr(cli,'handoff_from_prepared',lambda *a,**k:object())
    monkeypatch.setattr(common,'synthetic_preparation',lambda *a,**k:SimpleNamespace(state={
        'components_by_frame':[], 'current':[], 'motion_handoff_audit':{'matched_sources':0,'memory_only_sources_added':0}}))
    def moving(nusc,t0,tokens,horizons,**kwargs):
        assert calls.count('forecast')==3*(1+len([c for c in calls if c=='moving']))
        assert t0 in ('a','b') and horizons==(1.,2.,3.,4.,5.,6.)
        calls.append('moving');return [(np.zeros(provider.pcfg.grid.shape_hwd,bool),[],{})]*6
    monkeypatch.setattr(cli,'gt_moving_support_sequence',moving)
    class Source:
        nusc=object()
        def pose(self,t):return np.eye(4)
        def load_semantics(self,scene,token):
            assert calls[-1] in ('moving','GT')
            calls.append('GT')
            if stop[0]:event.set()
            return pred[1]
    contract=dict(protocol=common.PROTOCOL,routes=['reconciled','redetect'],thresholds=[.5,None])
    return provider,Source(),list(zip(ws,rows)),Execution(),contract,event,stop,calls


def test_openloop_order_original_t0_atomic_integer_resume_equals_uninterrupted(monkeypatch):
    p,s,jobs,exe,c,event,stop,calls=orchestration_fixture(monkeypatch)
    stop[0]=True;ledger=[]
    with pytest.raises(InterruptedError):
        cli.evaluate_windows(p,s,jobs,exe,c,save=lambda v:ledger.append(copy.deepcopy(v)),stop_event=event)
    saved=ledger[-1]
    assert saved['completed_windows']==1 and saved['second_block_exactness_passed']
    event.clear();stop[0]=False
    completed=cli.evaluate_windows(p,s,jobs,exe,c,saved=saved)
    # Independent same-window baseline, including all route counts.
    p,s,jobs,exe,c,_,_,_=orchestration_fixture(monkeypatch)
    reference=cli.evaluate_windows(p,s,jobs,exe,c)
    assert completed['counts']==reference['counts'] and completed['edits']==reference['edits']
    assert completed['completed_windows']==2
    assert cli.rollout.finalize_metrics({k:np.array(v) for k,v in completed['counts']['reconciled'].items()})['average_4s_5s_6s']['mIoU']==pytest.approx(100/3)


def test_resume_corruption_changed_model_routes_thresholds_fail_closed():
    c=dict(routes=['reconciled','redetect'],thresholds=[.5,None])
    value=cli.new_state(c['routes']);value['contract_fingerprint']=stable_json_fingerprint(c)
    value['fingerprint']=stable_json_fingerprint(value)
    assert cli.restore_state(value,c,2)['completed_windows']==0
    for changed in ({**c,'thresholds':[.6,None]},{**c,'routes':['reconciled']},{**c,'model':'changed'}):
        with pytest.raises(RuntimeError,match='changed'):cli.restore_state(value,changed,2)
    bad=copy.deepcopy(value);bad['counts']['redetect']['occ_union'][0]+=1
    with pytest.raises(RuntimeError,match='changed'):cli.restore_state(bad,c,2)
    bad=copy.deepcopy(value);bad.pop('fingerprint');bad['edits']['first']['removed']=1
    bad['fingerprint']=stable_json_fingerprint(bad)
    with pytest.raises(RuntimeError,match='ADD-only'):cli.restore_state(bad,c,2)


def test_actual_frozen_mean_artifact_loads_as_surface_not_legacy_local(tmp_path):
    from tools.real_motion.joint_surface_checkpoint_selection import evaluation_payload, load_evaluation_model, weight_fingerprint, AVERAGE_EPOCHS
    from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint
    provider,_,_,_=provider_fixture()
    model=provider.joint; original=copy.deepcopy(model.state_dict())
    value=evaluation_payload(original,model.configs(),{},dict(positive_weights=model.columns.positive_weight.tolist()),
                            [dict(epoch=i) for i in AVERAGE_EPOCHS],average=True)
    path=tmp_path/'mean.pt';torch.save(value,path);before=path.read_bytes()
    saved,loaded=load_evaluation_model(path,z_bins=4)
    assert saved['source_epochs']==[5,6,8,12,14] and isinstance(loaded,JointSurfaceCCR)
    assert weight_fingerprint(original)==weight_fingerprint(loaded.state_dict())
    with pytest.raises(RuntimeError):load_joint(path,torch.device('cpu'),reference_sha=CLEAN_SHA256,config_sha='x',allow_diagnostic=True)
    assert path.read_bytes()==before and 'optimizer' not in saved


@pytest.mark.parametrize('aligned',[False,True])
def test_cli_real_mean_snapshot_readonly_resume_and_fixed_route(tmp_path,monkeypatch,aligned):
    p,s,jobs,exe,c,event,stop,calls=orchestration_fixture(monkeypatch)
    from tools.real_motion.joint_surface_checkpoint_selection import evaluation_payload, AVERAGE_NAME, weight_fingerprint
    files={name:tmp_path/name for name in ('config','dev-cache','dev-info','base-checkpoint','population-manifest')}
    for name,path in files.items():path.write_bytes(name.encode())
    real_sha=cli.sha256
    def digest(path):return CLEAN_SHA256 if Path(path)==files['base-checkpoint'] else real_sha(path)
    trained=dict(data={name:digest(files[name.replace('_','-')]) for name in ('dev_cache','dev_info','base_checkpoint')},
        dataroot=str(tmp_path.resolve()),runtime_config_fingerprint=stable_json_fingerprint({}),
        implementation=cli.training_implementation(Path(cli.__file__).resolve().parents[2]),
        torch_version=str(torch.__version__),dev_manifest_fingerprint='manifest',prior_keys=[['train','t']])
    value=evaluation_payload(p.joint.state_dict(),p.joint.configs(),trained,
        dict(positive_weights=p.joint.columns.positive_weight.tolist()),
        [dict(epoch=i) for i in cli.AVERAGE_EPOCHS],average=True)
    mean=tmp_path/'frozen_mean.pt';torch.save(value,mean);before=mean.read_bytes()
    run=tmp_path/'train';run.mkdir();source_dir=tmp_path/'source';source_dir.mkdir()
    bundle=dict(selection_frozen=True,run_directory=str(run.resolve()),source_comparison_directory=str(source_dir.resolve()),
        audit={'contract':trained},candidates={AVERAGE_NAME:dict(path=str(mean),sha256=real_sha(mean),
            weight_fingerprint=weight_fingerprint(p.joint.state_dict()),source_epochs=list(cli.AVERAGE_EPOCHS))})
    bundle['fingerprint']=stable_json_fingerprint(bundle)
    monkeypatch.setattr(cli,'sha256',digest)
    monkeypatch.setattr(cli,'find_frozen_bundle',lambda *a:copy.deepcopy(bundle))
    def verify(actual):assert real_sha(mean)==actual['candidates'][AVERAGE_NAME]['sha256']
    monkeypatch.setattr(cli,'verify_sources',verify)
    monkeypatch.setattr(cli,'load_runtime_config',lambda *a:{})
    monkeypatch.setattr(cli,'make_prepare_config',lambda *a:p.pcfg)
    monkeypatch.setattr(cli,'load_manifest',lambda *a:(dict(manifest_fingerprint='manifest',
        selected_key_fingerprint=cli.DEV64_FP,parent_keys=[['dev',str(i)] for i in range(512)]),[('dev',str(i)) for i in range(64)],None))
    monkeypatch.setattr(cli,'load_cache',lambda *a:({},[r for _,r in jobs]))
    monkeypatch.setattr(cli,'VAL_WINDOWS',2)
    monkeypatch.setattr(cli,'require_cuda',lambda *a:torch.device('cpu'))
    monkeypatch.setattr(cli,'NuScenesWindowSource',lambda *a,**k:s)
    monkeypatch.setattr(cli,'CachedColumnSource',lambda src,*a:src)
    population=dict(population='all' if aligned else 'dev512',scenes=1,eligible_windows=2,
        requested_parent_windows=2,selected_keys=[[w.scene_name,w.t0_token] for w,_ in jobs])
    monkeypatch.setattr(cli.rollout,'select_long_population',lambda *a:(jobs,population))
    s.iter_windows=lambda **k:iter(w for w,_ in jobs)
    monkeypatch.setattr(cli.genie,'select_population',lambda *a:(jobs,population))
    monkeypatch.setattr(cli.genie,'validate_grid',lambda *a:None)
    monkeypatch.setattr(cli.rollout,'validate_timestamps',lambda *a:dict(intervals_s=[.5]*15,
        relative_times_s=((np.arange(16)-3)*.5).tolist(),max_nominal_deviation_s=0.))
    monkeypatch.setattr(common,'SurfaceRolloutProvider',lambda *a:p)
    exe.head=SimpleNamespace(stats=lambda:dict(graphs_enabled=False));exe.close=lambda:None
    monkeypatch.setattr(common,'SurfaceBlockExecution',lambda *a,**k:exe)
    out=tmp_path/'evaluation'
    argv=[arg for name,path in files.items() for arg in ('--'+name,str(path))]
    argv += ['--run-dir',str(run),'--runs-root',str(tmp_path),'--dataroot',str(tmp_path),'--out-dir',str(out),
             '--population','all' if aligned else 'dev512','--ccr-cpu-execution','numpy','--no-graphs']
    if aligned:argv+=['--population-alignment','geniedrive_code10s','--geniedrive-info',str(tmp_path/'official.pkl')]
    stop[0]=True
    assert cli.main(event,argv)==130
    assert json.loads((out/'evaluation_state.json').read_text())['completed_windows']==1
    assert not (out/'evaluation.json').exists()
    # Snapshot corruption must fail before any resumed scoring.
    snapshot=out/'checkpoint_snapshot.pt';data=snapshot.read_bytes();snapshot.write_bytes(b'wrong')
    with pytest.raises(RuntimeError,match='snapshot changed'):cli.main(event,argv+['--resume'])
    snapshot.write_bytes(data)
    event.clear();stop[0]=False
    assert cli.main(event,argv+['--resume'])==0
    result=json.loads((out/'evaluation.json').read_text())
    assert result['primary_route']=='reconciled' and result['source_epochs']==[5,6,8,12,14]
    assert set(result['routes'])=={'reconciled','redetect'} and result['windows']==2
    assert mean.read_bytes()==before and snapshot.read_bytes()==before
    assert result['future_GT_prediction_inputs'] is False and result['no_automatic_route_selection']
    assert 'average_4s_5s_6s' in (out/'summary.txt').read_text()
    assert ('geniedrive_code_compatibility' in result)==aligned
    with pytest.raises(SystemExit):cli.main(event,argv+['--resume'])  # completed result is never overwritten
