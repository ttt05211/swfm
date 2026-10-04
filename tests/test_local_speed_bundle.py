"""No accuracy claims: bytes, losses/gradients, RNG and bounded streaming."""
import copy
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch

from real_motion.column_gpu_sampling import GpuColumnSampler
from real_motion.native_column_cpu import backend_name, prepare_native
from real_motion.column_inference_pipeline import InferenceFeatures
from real_motion.causal_column_sampling import ColumnFeatureSampler, ColumnHistoryIndex
from real_motion.local_supervision_fastpath import column_indices
from real_motion.causal_column_model import column_loss
from tools.real_motion import causal_column_common as col
from tools.real_motion.joint_column_common import motion_loss
from test_causal_column_sampling import fixture as sampling_fixture
from test_joint_causal_columns import fixture, provider_for, optimizers
from test_causal_columns import fake_provider, moving_fixture


# The native shared library is process-local. The shell preflight runs in a
# separate Python process, so direct pytest execution under
# SWFM_COLUMN_CPU_BACKEND=native must load the already-built artifact here too.
if backend_name() == 'native':
    prepare_native()


@pytest.mark.parametrize('empty', (False, True))
def test_motion_cpu_indices_same_losses_and_gradients(monkeypatch, empty):
    _,_,joint,_,record = fixture()
    record = {k: torch.cat([v]*3) for k,v in record.items()}
    record['supervised_source'][1] = False
    record['se2_target_valid'][0, ::2] = False
    record['yaw_enabled'][2] = False
    record['target_source_mask_tube'][0] = 0
    if empty: record['supervised_source'][:] = False
    reference = copy.deepcopy(joint)
    values = []
    for name,model in (('0',reference),('1',joint)):
        monkeypatch.setenv('SWFM_LOCAL_FAST_SUPERVISION',name)
        loss,stats = motion_loss(model.motion(record,torch.device('cpu')),record,'cpu',materialize_stats=False)
        loss.backward(); values.append((loss,stats))
    assert torch.equal(values[0][0], values[1][0])
    assert all(torch.equal(values[0][1][k], v) for k,v in values[1][1].items())
    for (k,p),(_,r) in zip(joint.named_parameters(),reference.named_parameters()):
        assert (p.grad is None) == (r.grad is None), k
        if p.grad is not None: assert torch.equal(p.grad,r.grad), k


def test_column_host_indices_same_loss_gradient_and_bad_labels():
    _,_,joint,_,_ = fixture(); model = joint.columns
    torch.manual_seed(22)
    kind=torch.tensor([0,1,1,0,1]); legal=torch.ones(5,2,3,dtype=torch.bool)
    legal[kind == 0,:,2]=False
    target=torch.tensor([[1,0],[2,1],[0,2],[0,1],[1,1]])
    weight=torch.tensor([3.,2.,.5,4.,1.])
    g=torch.randn(5,2,requires_grad=True); r=torch.randn(5,2,3,requires_grad=True)
    g2=g.detach().clone().requires_grad_(); r2=r.detach().clone().requires_grad_()
    old,_=column_loss(model,g,r,kind,legal,target,weight,materialize_stats=False)
    idx=column_indices(kind.numpy(),legal.numpy(),target.numpy(),weight.numpy(),'cpu')
    new,_=column_loss(model,g2,r2,kind,legal,target,weight,materialize_stats=False,supervision_indices=idx)
    old.backward();new.backward()
    assert torch.equal(old,new) and torch.equal(g.grad,g2.grad) and torch.equal(r.grad,r2.grad)
    bad=target.numpy().copy();bad[0,0]=2
    with pytest.raises(ValueError,match='illegal target'):column_indices(kind,legal,bad,weight,'cpu')
    with pytest.raises(ValueError,match='sampling weights'):column_indices(kind,legal,target,[-1]*5,'cpu')


def test_constant_canvas_created_in_inference_can_be_used_in_training_backward():
    from real_motion.local_st_world_model_v18_se2 import _footprint_coordinates
    _footprint_coordinates.cache_clear()
    _,_,joint,_,record = fixture()
    with torch.inference_mode():
        motion_loss(joint.motion(record,torch.device('cpu')),record,'cpu',materialize_stats=False)
    coordinates = _footprint_coordinates(60,60,.8,torch.device('cpu'),torch.float32)
    assert all(not value.is_inference() and not value.requires_grad for value in coordinates)
    loss,_ = motion_loss(joint.motion(record,torch.device('cpu')),record,'cpu',materialize_stats=False)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in joint.transport.parameters())


@pytest.mark.parametrize('frames',(4,6))
@pytest.mark.parametrize('batch',(19,127,256))
def test_full_population_gpu_stream_cpu_emulation_matches_whole_horizon(frames,batch):
    prep,grid,cfg,plan=sampling_fixture(2)
    for k in ('history_occ','history_observed','history_poses'):prep.raw[k]=prep.raw[k][-frames:]
    prep.registrations=[r[-frames:] for r in prep.registrations]
    prep.raw['future_gt_occ']=object()
    # Different output sites may share causal read anchors.
    plan.evidence_xy[:40]=plan.evidence_xy[0]
    plan.classes[:40]=11
    index=ColumnHistoryIndex(prep,grid)
    cpu=ColumnFeatureSampler(prep,3,plan,grid,cfg,col.pose_motion,workers=1,history_index=index)
    gpu=GpuColumnSampler('cpu',allow_cpu=True,chunk_queries=128)
    stream=InferenceFeatures(prep,3,plan,grid,cfg,'cpu',col.pose_motion,
        backend='gpu',history_index=index,gpu_sampler=gpu,verify=True)
    with gpu.resident_window(prep,grid,cfg):
        for start in range(0,len(plan),batch):
            small=plan.subset(slice(start,start+batch))
            a=stream.sample(small,col.sample_column_features);b=cpu.sample(small,col.sample_column_features)
            for k in col.FEATURE_KEYS:
                v=a[k].numpy() if isinstance(a[k],torch.Tensor) else a[k]
                assert np.array_equal(v,b[k]),k
        assert stream.audit['total_queries']==len(plan)
        assert stream.audit['gpu_chunks'] > 0 and stream.audit['verified_chunks']==1
    assert gpu._resident is None


def test_stream_boundary_and_budget_use_whole_original_cpu_map(monkeypatch):
    prep,grid,cfg,plan=sampling_fixture()
    gpu=GpuColumnSampler('cpu',allow_cpu=True,max_working_mib=1)
    stream=InferenceFeatures(prep,3,plan,grid,cfg,'cpu',col.pose_motion,backend='gpu',gpu_sampler=gpu)
    small=plan.subset(slice(0,40))
    a=stream.sample(small,None)
    b=ColumnFeatureSampler(prep,3,plan,grid,cfg,col.pose_motion).sample(small,None)
    assert all(np.array_equal(a[k],b[k]) for k in col.FEATURE_KEYS)
    assert stream.audit['budget_chunks']==1
    gpu.max_working_mib=256
    gather=gpu._gather
    def boundary(*args):
        h,f,g=gather(*args);g[:]=True;return h,f,g
    monkeypatch.setattr(gpu,'_gather',boundary)
    stream.sample(small,None)
    assert stream.audit['boundary_chunks']==1


def test_cpu_shared_index_four_way_metrics_unchanged_with_stream_optin():
    provider,prep,cfg,_=fake_provider()
    from real_motion.causal_column_model import CausalColumnModel
    model=CausalColumnModel(cfg)
    torch.nn.init.normal_(model.refinement.weight,std=.01)
    records=[dict(scene_name='dev',t0_token=str(i)) for i in range(2)]
    with patch.object(col,'gt_moving_support_sequence',return_value=moving_fixture(prep)):
        a=col.evaluate_columns(provider,SimpleNamespace(nusc=None),records,model,(.5,.5,.95),diagnostic_thresholds=None)
        b=col.evaluate_columns(provider,SimpleNamespace(nusc=None),records,model,(.5,.5,.95),diagnostic_thresholds=None,feature_backend='gpu')
    from tools.real_motion.static_evidence_selector_common import finite_json
    assert finite_json(a)==finite_json(b)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires actual CUDA')
def test_actual_cuda_stream_probabilities_and_compositor_exact():
    prep,grid,cfg,plan=sampling_fixture(3)
    from real_motion.causal_column_model import CausalColumnModel
    model=CausalColumnModel(cfg).cuda().eval()
    torch.nn.init.normal_(model.refinement.weight,std=.01)
    device=torch.device('cuda')
    before=col.predict_probabilities(model,prep,3,plan,grid,device,batch_size=256)
    gpu=GpuColumnSampler(device,chunk_queries=128)
    with gpu.resident_window(prep,grid,cfg):
        after=col.predict_probabilities(model,prep,3,plan,grid,device,batch_size=256,
            feature_backend='gpu',gpu_sampler=gpu,verify_features=True)
    assert np.array_equal(before,after)
    from real_motion.causal_column_completion import actions_from_probabilities
    assert np.array_equal(actions_from_probabilities(plan,before,(.5,.5,.95)),
                          actions_from_probabilities(plan,after,(.5,.5,.95)))


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(
    not torch.cuda.is_available(),reason='requires actual CUDA'))])
def test_full_adamw_three_updates_host_loss_fastpath_keeps_rng_and_parameters(monkeypatch,device):
    from real_motion.column_cpu_pipeline import HorizonCpuPool
    from tools.real_motion import joint_column_full_common as full
    prep,grid,joint,control,rec=fixture()
    reference=copy.deepcopy(joint)
    joint.to(device);reference.to(device);control.to(device)
    torch.nn.init.normal_(joint.columns.refinement.weight,std=.01)
    reference.load_state_dict(joint.state_dict())
    opt,_=optimizers(joint,control);ropt,_=optimizers(reference,control)
    p,q=provider_for(prep,grid,joint),provider_for(prep,grid,reference)
    p.device=q.device=torch.device(device)
    x,y=np.random.default_rng(53),np.random.default_rng(53)
    monkeypatch.setattr(full,'bundle_enabled',lambda:True)
    monkeypatch.setattr(full,'horizon_pipeline_enabled',lambda:False)
    from test_column_horizon_pipeline import assert_state_equal
    with HorizonCpuPool(2) as pool:
        for update in (1,2,3):
            monkeypatch.setenv('SWFM_LOCAL_FAST_SUPERVISION','0')
            a=full.train_full_batch(reference,ropt,q,None,[(rec,None)]*4,y,update,10,sampling_pool=pool)
            monkeypatch.setenv('SWFM_LOCAL_FAST_SUPERVISION','1')
            b=full.train_full_batch(joint,opt,p,None,[(rec,None)]*4,x,update,10,sampling_pool=pool)
            if device == 'cpu':
                assert a['loss']==b['loss']
                assert_state_equal(joint.state_dict(),reference.state_dict())
                assert_state_equal(opt.state_dict(),ropt.state_dict())
            else:
                # Existing CUDA grid-sample/index accumulation is not generally
                # bit deterministic. Do not require an unsupported deterministic
                # backward kernel or claim byte-identical GPU training.
                assert a['loss']==pytest.approx(b['loss'],rel=2e-5,abs=1e-6)
                for k,v in joint.state_dict().items():
                    torch.testing.assert_close(v,reference.state_dict()[k],rtol=2e-5,atol=1e-6)
            assert x.bit_generator.state==y.bit_generator.state


def test_frozen_reference_counts_reused_but_control_and_sources_are_live():
    from real_motion.causal_column_model import CausalColumnModel
    from tools.real_motion.static_evidence_selector_common import finite_json
    provider, prep, cfg, _ = fake_provider()
    provider.reference_enabled = True
    provider.frozen_metric_counts = OrderedDict()
    calls = []; control_version = [False]
    def references(prepared, record, *, skip_frozen=False):
        calls.append(skip_frozen)
        frozen = [v.copy() for v in prepared.baseline]
        control = [v.copy() for v in prepared.baseline]
        if control_version[0]:
            for v in control: v[:] = 17
        return {'paired_scratch_V18_only': control, **({} if skip_frozen else {'frozen_E14': frozen})}
    provider.reference_predictions = references
    model = CausalColumnModel(cfg)
    records = [dict(scene_name='dev', t0_token='0')]
    source = SimpleNamespace(nusc=None)
    kwargs = dict(diagnostic_thresholds=None)
    with patch.object(col, 'gt_moving_support_sequence', return_value=moving_fixture(prep)):
        a = col.evaluate_columns(provider, source, records, model, (.5,.5,.95), **kwargs)
        b = col.evaluate_columns(provider, source, records, model, (.5,.5,.95), **kwargs)
        assert finite_json(a) == finite_json(b) and calls == [False, True]
        control_version[0] = True
        c = col.evaluate_columns(provider, source, records, model, (.5,.5,.95), **kwargs)
        assert finite_json(a['all']['reference_metrics']['frozen_E14']) == finite_json(c['all']['reference_metrics']['frozen_E14'])
        assert (finite_json(a['all']['reference_metrics']['paired_scratch_V18_only']) !=
                finite_json(c['all']['reference_metrics']['paired_scratch_V18_only']))
        provider.frozen_metric_counts = None
        d = col.evaluate_columns(provider, source, records, model, (.5,.5,.95), **kwargs)
        assert finite_json(c) == finite_json(d)
        provider.frozen_metric_counts = OrderedDict()
        col.evaluate_columns(provider, source, records, model, (.5,.5,.95), **kwargs)
        col.evaluate_columns(provider, SimpleNamespace(nusc=None), records, model, (.5,.5,.95), **kwargs)
        assert calls[-2:] == [False, False]  # different dataset object invalidates reuse


def test_explicit_full_population_never_falls_back_to_prefix():
    from tools.real_motion.eval_p0_f9_joint_causal_columns import evaluation_keys
    keys = [('dev', str(i)) for i in range(4369)]
    dev64, dev512 = [list(k) for k in keys[-64:]], [list(k) for k in keys[-512:]]
    train = [('train', '0')]
    assert evaluation_keys('dev64',dev64,dev512,keys,train) == tuple(keys[-64:])
    assert evaluation_keys('dev512',dev64,dev512,keys,train) == tuple(keys[-512:])
    assert evaluation_keys('full4369',dev64,dev512,keys,train) == tuple(keys)
    for bad in (keys[:512], [keys[0]]*4369):
        with pytest.raises(RuntimeError, match='4369 unique'):
            evaluation_keys('full4369',dev64,dev512,bad,train)
    with pytest.raises(RuntimeError, match='contained'):
        evaluation_keys('full4369',dev64,[('dev', 'not-found')],keys,train)
    with pytest.raises(RuntimeError, match='scene overlap'):
        evaluation_keys('full4369',dev64,dev512,keys,[('dev','train')])


@pytest.mark.parametrize('frames', (4, 6))
def test_evaluation_fixed_geometry_lookahead_is_causal_and_same_live_renderer(frames):
    from real_motion.strong_w2det import StrongW2DetConfig
    from tools.real_motion import joint_column_full_common as full
    prep, grid, joint, _, record = fixture()
    raw = copy.deepcopy(prep.raw)
    for key in ('history_occ','history_observed','history_poses'):raw[key] = raw[key][-frames:]
    raw['history_occ'][raw['history_occ'] == 4] = 17
    for f in range(frames):raw['history_occ'][f, 1+f:4+f, 5:8, 1] = 4
    prep.window.history_tokens = tuple(f'h{i}' for i in range(frames))
    strong = StrongW2DetConfig();pcfg=SimpleNamespace(grid=grid,free_label=17,frame_dt_s=.5)
    evidence = full.prepare_causal_evidence(raw,pcfg,strong,1)
    assert len(evidence['current']) == 1
    center = torch.tensor(np.asarray([c['centroid_world'][:2] for c in evidence['current']]),dtype=torch.float32)
    record['source_centroid_xy_t0_m'] = center
    record['anchors_xy_t0_m'] = center[:,None,:].repeat(1,6,1)+record['kta_displacement_xy_m']
    provider = full.EvaluationJointColumnProvider.__new__(full.EvaluationJointColumnProvider)
    provider.pcfg,provider.strong,provider.workers,provider.device = pcfg,strong,1,torch.device('cpu')
    provider.joint,provider.model,provider.columns_checked = joint,joint.transport,False
    joint.eval();outputs = joint.motion(record,torch.device('cpu'))
    with patch.object(col,'window_from_record',return_value=prep.window), \
         patch.object(col.runtime,'window_from_record',return_value=prep.window):
        original = provider.prepare_columns(None,record,include_gt=True,raw_window=raw,outputs=outputs)
        forbidden = {**raw,'future_gt_occ':object()}
        with patch.object(col.FrozenColumns,'load_raw_columns',return_value=forbidden):
            lookahead = provider.load_raw_columns(None,record,include_gt=False)
        assert not any(key in lookahead for key in ('_causal_cache_deferred',))
        lookahead['future_gt_occ'] = raw['future_gt_occ']
        optimized = provider.prepare_columns(None,record,include_gt=True,raw_window=lookahead,outputs=outputs)
        for key in ('baseline','owners','fallbacks','targets','yaws','footprints','memory'):
            assert all(np.array_equal(a,b) for a,b in zip(getattr(original,key),getattr(optimized,key))),key
        assert original.source_audit == optimized.source_audit
        for h in range(6):
            a = col.candidate_plan(original,h,grid,joint.columns.config)
            b = col.candidate_plan(optimized,h,grid,joint.columns.config)
            assert all(np.array_equal(v,getattr(b,k)) for k,v in vars(a).items())
        previous = col.render_column_layers(optimized.state,record,outputs,grid)[0]
        light = col.render_column_layers(optimized.state,record,outputs,grid,baseline_only=True)[0]
        assert all(np.array_equal(a,b) for a,b in zip(previous,light))
        moved = {**outputs,'residual_xy_m':outputs['residual_xy_m']+1.}
        updated = provider.prepare_columns(None,record,include_gt=True,raw_window=lookahead,outputs=moved)
        assert any(not np.array_equal(a,b) for a,b in zip(updated.baseline,optimized.baseline))


def test_speed_summary_helpers_are_explicit_and_switches_do_not_mutate_contract(monkeypatch,tmp_path):
    from tools.real_motion.benchmark_p0_f9_joint_speed_bundle import path_args,switches,summary_text,train_equivalence
    contract = dict(launch_cwd=str(tmp_path),arguments=dict(config='config.yaml',window_batch_size=4))
    original = copy.deepcopy(contract)
    assert path_args(contract)['config'] == str((tmp_path/'config.yaml').resolve())
    assert contract == original
    switches(False)
    from real_motion.local_supervision_fastpath import enabled,static_roi_enabled
    assert not enabled() and not static_roi_enabled()
    switches(True)
    assert enabled() and static_roi_enabled()
    text = summary_text(dict(status='failed',error='cache miss',train={},evaluation={},gate={}))
    assert 'no saved optimizer updates' in text and 'cache miss' in text
    measurement=dict(sampling_rng_fingerprint='a',sampled_columns_by_update=[40],loss_by_update=[1.2],
        windows=4,batches=1,sources=4)
    train_equivalence(measurement,copy.deepcopy(measurement))
    bad = {**measurement,'loss_by_update':[1.]}
    with pytest.raises(RuntimeError,match='losses changed'):train_equivalence(measurement,bad)
    with pytest.raises(RuntimeError,match='RNG/population'):train_equivalence(measurement,{**measurement,'sources':5})
