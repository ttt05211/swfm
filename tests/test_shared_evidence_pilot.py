"""CPU orchestration emulation; never claims a real-data CUDA/FPS result."""
import copy
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from test_shared_column_evidence import training_fixture,model_fixture
from real_motion.shared_column_evidence import SharedHistorySession,SharedReadExecution
from real_motion.nuscenes_adapter import gt_moving_support_sequence,gt_moving_support_for_horizon
from tools.real_motion import shared_evidence_pilot_common as common
from tools.real_motion import run_p0_f9_shared_evidence_pilot as pilot
from tools.real_motion import causal_column_common as reference
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import sha256
from tools.real_motion.v18_source_interaction_common import select_population
from tools.real_motion.shared_evidence_recovery import prepare_migration_resume,PRE_MEMORY_FIX_IMPLEMENTATION


class MetricNuScenes:
    """Small annotation API, using the REAL frozen support builder, not a stub."""
    def get(self,table,token):
        if table=='sample':
            anns=['moving:'+token,'static:'+token]
            if token.startswith('f'):anns.append('birth:'+token)
            return dict(anns=anns,data=dict(LIDAR_TOP=token))
        if table=='sample_data':return dict(ego_pose_token=token)
        if table=='ego_pose':return dict(translation=[0.,0.,0.],rotation=[1.,0.,0.,0.])
        if table=='sample_annotation':
            actor,frame=token.split(':');dt=.5*(int(frame[1:])+1) if frame.startswith('f') else 0.
            center=[6.5+dt,6.5,.5] if actor=='moving' else [1.5,1.5,.5]
            return dict(instance_token=actor,category_name='vehicle.car',translation=center,
                rotation=[1.,0.,0.,0.],size=[1.,2.,1.])
        raise KeyError((table,token))


@pytest.mark.parametrize('workers',[1,3])
def test_real_moving_support_metadata_is_unpacked_without_changing_frozen_counts(workers):
    prep,grid,_,_,_,_=model_fixture();nusc=MetricNuScenes()
    future=tuple(f'f{h}' for h in range(6));horizons=tuple(.5*(h+1) for h in range(6))
    rows=gt_moving_support_sequence(nusc,'t0',future,horizons,grid=grid,workers=workers)
    assert all(len(row)==3 and row[1] and row[2]['birth_dynamic']==1 for row in rows)
    masks=common.moving_support_masks(rows,grid.shape_hwd)
    actual,expected=common.Metrics(),common.Metrics()
    for ri,h in enumerate(reference.REPORT):
        old=gt_moving_support_for_horizon(nusc,'t0',future[h],horizons[h],grid=grid)[0]
        assert masks[h] is rows[h][0] and np.array_equal(masks[h],old)
        actual.update(ri,prep.baseline[h],prep.raw['future_gt_occ'][h],masks[h])
        expected.update(ri,prep.baseline[h],prep.raw['future_gt_occ'][h],old)
    for name in ('oi','ou','si','su','mi','mu'):
        assert np.array_equal(getattr(actual,name),getattr(expected,name))
    assert actual.mu.sum()>0


@pytest.mark.parametrize('malformed', ['bare_masks','short_sequence','short_row','wrong_shape','wrong_dtype'])
def test_moving_support_rejects_bad_contract_instead_of_broadcasting(malformed):
    shape=(4,3,2);rows=[(np.ones(shape,bool),[],{}) for _ in range(6)]
    if malformed=='bare_masks':rows=[row[0] for row in rows]
    elif malformed=='short_sequence':rows=rows[:-1]
    elif malformed=='short_row':rows[0]=rows[0][:2]
    elif malformed=='wrong_shape':rows[0]=(np.ones(shape[1:],bool),[],{})
    elif malformed=='wrong_dtype':rows[0]=(np.ones(shape,np.uint8),[],{})
    with pytest.raises(ValueError,match='Moving support'):common.moving_support_masks(rows,shape)


def test_graph_reader_bounded_chunks_match_eager_and_copy_reused_buffers():
    prep,grid,output,window,teacher,student=model_fixture();student.eval()
    with torch.inference_mode():
        plan=window.candidates(1)
        eager=common.tensor_probability(student,window,1,plan,output,
            session=SharedHistorySession(student,window.labels,window.visibility),batch_size=8)
        engine=SharedReadExecution(student)
        actual=common.tensor_probability(student,window,1,plan,output,
            session=SharedHistorySession(student,window.labels,window.visibility),batch_size=8,execution=engine)
        assert torch.equal(eager,actual) and engine.eager_calls>1
        # Simulate capture output aliasing; the same eight-row allocation is
        # overwritten on every call. Earlier chunks must remain correct.
        class Reused:
            def run(self,batch,legal):
                p,ok=engine.function(batch,legal)
                if not hasattr(self,'buffer'):self.buffer=p.new_empty((8,*p.shape[1:]))
                self.buffer[:len(p)].copy_(p)
                return self.buffer[:len(p)],ok
        aliased=common.tensor_probability(student,window,1,plan,output,
            session=SharedHistorySession(student,window.labels,window.visibility),batch_size=8,execution=Reused())
        assert torch.equal(eager,aliased)


def test_same_class_wrong_source_centres_rejected_before_device_inputs():
    joint,teacher,provider,rows=training_fixture();record,raw=rows[0]
    other={**record,'source_centroid_xy_t0_m':record['source_centroid_xy_t0_m']+1}
    with pytest.raises(RuntimeError,match='centre'):common.causal_template(raw,other)


def test_one_bundle_real_evaluation_migration_stop_resume_and_summary(tmp_path,monkeypatch):
    # Only dataset loading, legacy checkpoint loading and CUDA-only timing are
    # stubbed. Actual candidate/probe/shared reader, metrics, GT+KD gradients,
    # optimizer and recovery are exercised through the new CLI orchestrator.
    joint,teacher,base_provider,rows=training_fixture();template,raw=rows[0]
    prep,grid,_,window,_,_=model_fixture()
    train=[{**copy.deepcopy(template),'scene_name':f'train{s}','t0_token':f'{s}:{i}'} for s in range(10) for i in range(4)]
    dev=[{**copy.deepcopy(template),'scene_name':'dev','t0_token':f'd{i}'} for i in range(512)]
    keys=lambda records:tuple((r['scene_name'],r['t0_token']) for r in records)
    manifest=dict(parent_keys=keys(dev),manifest_fingerprint='fixed')
    paths={k:tmp_path/k for k in ('checkpoint','train-cache','dev-cache','population-manifest','base-checkpoint','train-info','dev-info')}
    for p in paths.values():p.write_bytes(b'original-input')
    digests={str(p):sha256(p) for p in paths.values()}
    ck=dict(cursor_epoch=19,model_configs={},cache_fingerprints=dict(train=digests[str(paths['train-cache'])],dev=digests[str(paths['dev-cache'])]),
        info_fingerprints=dict(train=digests[str(paths['train-info'])],dev=digests[str(paths['dev-info'])]),
        dev_manifest_fingerprint='fixed',dev_keys=keys(dev),train_keys=keys(train))
    def make_provider(checkpoint,digest,pcfg,device,workers,model,control):
        provider=SimpleNamespace(joint=model,model=model.transport,pcfg=pcfg,device=device,workers=workers)
        def load(source,record,*,include_gt):
            data=copy.deepcopy(raw)
            if not include_gt:data['future_gt_occ']=None
            return data
        def prepare(source,record,*,include_gt,raw_window,outputs=None):
            result=copy.deepcopy(prep);result.raw=raw_window
            result.state=raw_window['_column_causal_preparation']['prepared_state']
            result.outputs=outputs if outputs is not None else provider.joint.motion(record,device)
            result.baseline,result.owners,result.fallbacks,result.components,result.targets,result.yaws=reference.render_column_layers(result.state,record,result.outputs,grid)
            result.window=SimpleNamespace(scene_name=record['scene_name'],t0_token=record['t0_token'],future_tokens=tuple(f'f{h}' for h in range(6)))
            return result
        provider.load_raw_columns=load;provider.prepare_columns=prepare
        return provider
    monkeypatch.setattr(pilot,'require_cuda',lambda _:torch.device('cpu'))
    monkeypatch.setattr(pilot,'TRAIN_WINDOWS',40)
    monkeypatch.setattr(pilot,'load_joint',lambda *a,**k:(ck,copy.deepcopy(teacher)))
    monkeypatch.setattr(pilot,'CLEAN_SHA256',digests[str(paths['base-checkpoint'])])
    monkeypatch.setattr(pilot,'make_prepare_config',lambda cfg:SimpleNamespace(grid=grid))
    monkeypatch.setattr(pilot,'load_manifest',lambda path:(manifest,keys(dev[:64]),None))
    monkeypatch.setattr(pilot,'load_cache',lambda path:({},train if str(path)==str(paths['train-cache']) else dev))
    monkeypatch.setattr(pilot,'select_population',lambda keys,scenes,**kwargs:select_population(keys,scenes,calibration_scenes=2,**kwargs))
    monkeypatch.setattr(pilot,'PilotProvider',make_provider)
    monkeypatch.setattr(pilot,'NuScenesWindowSource',lambda *a,**k:SimpleNamespace(nusc=MetricNuScenes()))
    monkeypatch.setattr(pilot,'training_speed',lambda *a,**k:dict(trials=[dict(mode=m,seconds_per_window=t) for m,t in
        (('current_joint',.1),('device_geometry_joint',.08),('shared_auto_joint',.07),('shared_dense_joint',.09),('shared_tiles_joint',.08))]))
    monkeypatch.setattr(pilot,'fps_speed',lambda *a,**k:dict(trials=[dict(mode=m,six_frame_seconds=t) for m,t in
        (('current_graph',.9),('device_geometry',.6),('shared_auto',.2))],boundary='TEST_EMULATION',excludes='NO_REAL_FPS'))
    def run(out,*,resume=None,stop=None,memory_fix=False):
        argv=['pilot','--config',str(Path(__file__).resolve().parents[1]/'configs/real_motion_occfm.yaml'),
            '--dataroot',str(tmp_path),'--out-dir',str(out),'--eval-windows','2','--fps-windows','1',
            '--speed-train-windows','4','--speed-repeats','1']
        for k,v in paths.items():argv+=['--'+k,str(v)]
        if resume:argv+=['--resume',str(resume)]
        if memory_fix:argv+=['--resume-memory-fix']
        actual=common.training_step
        def step(*a,**k):
            result=actual(*a,**k)
            if stop is not None:stop.set()
            return result
        monkeypatch.setattr(pilot,'training_step',step)
        monkeypatch.setattr('sys.argv',argv)
        return pilot.main(stop)
    whole=tmp_path/'whole';stopped=tmp_path/'stopped';resumed=tmp_path/'resumed'
    assert run(whole)==0
    assert run(stopped,stop=Event())==130
    assert run(resumed,resume=stopped/'migration_last.pt')==0
    a,b=(torch.load(p/'migration_last.pt',weights_only=False) for p in (whole,resumed))
    assert a['cursor']==b['cursor']==2 and a['executed_windows']==b['executed_windows']==8
    assert all(torch.equal(v,b['student'][k]) for k,v in a['student'].items())
    assert torch.equal(a['sampling_rng'],b['sampling_rng'])
    report=json.loads((resumed/'bundle.json').read_text(encoding='utf-8'))
    assert report['status']=='complete' and report['actual_cuda'] is False
    assert set(report['reused_identical_contract_phases'])=={'probe','training_speed','fps'}
    assert 'final_dev64' in report and 'fps_initial' in report
    assert isinstance(report['gate']['actual_joint_training_faster'],bool)
    assert isinstance(report['gate']['six_frame_latency_le_250ms'],bool)
    assert 'JOINT_TRAIN' in (resumed/'summary.txt').read_text(encoding='utf-8')
    assert all(sha256(p)==digests[str(p)] for p in paths.values())
    assert not a['deployable'] and a['transport_frozen']

    # Emulate the server's immutable 1e44288 checkpoint. Only its execution
    # fingerprint differs; actual model/optimizer/RNG/data recipe is identical.
    legacy=tmp_path/'legacy';legacy.mkdir()
    old=torch.load(stopped/'migration_last.pt',weights_only=False)
    old['contract']={**old['contract'],'implementation_fingerprint':PRE_MEMORY_FIX_IMPLEMENTATION}
    torch.save(old,legacy/'migration_last.pt')
    (legacy/'contract.json').write_text(json.dumps(old['contract']),encoding='utf-8')
    (legacy/'bundle.json').write_bytes((stopped/'bundle.json').read_bytes())
    digest=sha256(legacy/'migration_last.pt')
    fixed=tmp_path/'memory_fixed'
    assert run(fixed,resume=legacy/'migration_last.pt',memory_fix=True)==0
    repaired=torch.load(fixed/'migration_last.pt',weights_only=False)
    assert repaired['cursor']==a['cursor'] and repaired['executed_windows']==a['executed_windows']
    assert all(torch.equal(v,repaired['student'][k]) for k,v in a['student'].items())
    assert torch.equal(a['sampling_rng'],repaired['sampling_rng'])
    repaired_report=json.loads((fixed/'bundle.json').read_text(encoding='utf-8'))
    assert repaired_report['reused_identical_contract_phases']==[]
    assert repaired_report['reused_verified_accuracy_phases']==['probe']
    assert repaired_report['memory_fix_resume']['optimizer_RNG_schedule_population_preserved']
    assert 'historical_execution_diagnostics_not_current_speed' in repaired_report
    assert sha256(legacy/'migration_last.pt')==digest


def test_memory_fix_resume_requires_exact_whitelist_and_explicit_opt_in():
    contract=dict(schedule_steps=1029,teacher='frozen',window_batch=4,source_budget=128,
        implementation_fingerprint='patched',diagnostic_budgets=dict(final_dev512=True))
    old=dict(protocol=pilot.PROTOCOL,transport_frozen=True,deployable=False,cursor=672,
        successful_updates=672,executed_windows=2688,
        contract={**contract,'implementation_fingerprint':PRE_MEMORY_FIX_IMPLEMENTATION})
    with pytest.raises(RuntimeError,match='identical'):prepare_migration_resume(old,contract)
    adjusted,audit=prepare_migration_resume(old,contract,allow_memory_fix=True)
    assert adjusted['contract']==contract and adjusted['cursor']==672 and audit['math_and_architecture_unchanged']
    assert old['contract']['implementation_fingerprint']==PRE_MEMORY_FIX_IMPLEMENTATION
    with pytest.raises(RuntimeError,match='identical'):
        prepare_migration_resume({**old,'contract':{**old['contract'],'implementation_fingerprint':'unknown'}},contract,allow_memory_fix=True)
    for changed in ({**contract,'window_batch':8},{**contract,'schedule_steps':2048},
            {**contract,'diagnostic_budgets':dict(final_dev512=False)},
            {**contract,'teacher':'other'}):
        with pytest.raises(RuntimeError,match='cannot change'):prepare_migration_resume(old,changed,allow_memory_fix=True)
    with pytest.raises(RuntimeError,match='identical'):
        prepare_migration_resume({**old,'protocol':'old_full'},contract,allow_memory_fix=True)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA byte gates execute in server bundle')
def test_cuda_shared_reader_graph_same_probabilities_and_fresh_native_memory():
    prep,grid,output,window,teacher,student=model_fixture()
    student=student.cuda().eval()
    from real_motion.column_device_geometry import DeviceColumnWindow
    window=DeviceColumnWindow(prep,grid,student.config,'cuda').render(
        torch.tensor([[[6.5,6.5]]*6],device='cuda'),torch.zeros(1,6,2,device='cuda'),torch.zeros(1,6,device='cuda'))
    output={k:v.cuda() for k,v in output.items()}
    with torch.inference_mode():
        plan=window.candidates(2);engine=SharedReadExecution(student)
        expected=common.tensor_probability(student,window,2,plan,output,
            session=SharedHistorySession(student,window.labels,window.visibility),batch_size=8)
        result=common.tensor_probability(student,window,2,plan,output,
            session=SharedHistorySession(student,window.labels,window.visibility),batch_size=8,execution=engine)
        assert torch.equal(expected,result)
