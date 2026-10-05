"""Exact inference-only hoisting, compiled copies and bounded CPU map overlap."""
from concurrent.futures import Future
from threading import Event, get_ident
from unittest.mock import patch

import numpy as np
import pytest
import torch

from real_motion import native_column_cpu as native
from real_motion.causal_column_sampling import ColumnFeatureSampler, ColumnHistoryIndex
from real_motion.column_inference_pipeline import HorizonInputs, HorizonFeaturePrefetch
from tools.real_motion import causal_column_common as columns
from test_causal_column_sampling import fixture as sampler_fixture
from test_joint_checkpoint_selection import _eval_fixture
from test_causal_columns import moving_fixture
from tools.real_motion.static_evidence_selector_common import finite_json


@pytest.fixture(scope='module')
def compiled():
    try: native._compiler()
    except RuntimeError: pytest.skip('optional native compiler absent')
    native.prepare_native()
    return native._loaded


@pytest.mark.parametrize('frames',(4,6))
@pytest.mark.parametrize('static',(False,True))
def test_native_direct_patch_rows_order_duplicates_membership_and_no_partial_write(compiled,frames,static):
    rng = np.random.default_rng(18)
    history = rng.integers(0,19,(frames,12,11,4),dtype=np.uint8)
    flags = rng.integers(0,4,history.shape,dtype=np.uint8)
    starts = np.array([[3,4],[0,0],[3,4],[5,4]],np.int64)
    rows = np.array([4,0,2,1],np.int64); classes = np.array([11,13,11,4],np.uint8)
    out = np.full((6,frames,7,7,4),255,np.uint8); bits = out.copy()
    expected = out.copy(); expected_bits = bits.copy()
    for xy,row,cls in zip(starts,rows,classes):
        x,y = xy; expected[row] = history[:,x:x+7,y:y+7]
        expected_bits[row] = flags[:,x:x+7,y:y+7] | ((expected[row] == cls)*2 if static else 0)
    compiled.patch_rows(history,flags,starts,rows,classes,static,out,bits)
    assert np.array_equal(out,expected) and np.array_equal(bits,expected_bits)
    before = out.copy(); bad = starts.copy(); bad[-1,0] = 6
    with pytest.raises(ValueError,match='rejected invalid'):
        compiled.patch_rows(history,flags,bad,rows,classes,static,out,bits)
    assert np.array_equal(out,before)
    with pytest.raises(ValueError,match='contiguous'):
        compiled.patch_rows(history,flags,starts,rows,classes,static,out[:,:,::-1],bits)
    with pytest.raises(ValueError,match='alias'):
        compiled.patch_rows(history,flags,starts,rows,classes,static,out,out)


@pytest.mark.parametrize('frames',(4,6))
@pytest.mark.parametrize('limit',(0,64))
def test_compiled_patch_sampling_all_bytes_reference_and_future_gt_independent(compiled,monkeypatch,frames,limit):
    monkeypatch.setenv('SWFM_COLUMN_CPU_BACKEND','native')
    prep,grid,cfg,plan = sampler_fixture(3)
    for k in ('history_occ','history_observed','history_poses'): prep.raw[k] = prep.raw[k][-frames:]
    prep.registrations = [row[-frames:] for row in prep.registrations]
    prep.raw['future_gt_occ'] = 'never read in inference'
    old = ColumnFeatureSampler(prep,3,plan,grid,cfg,columns.pose_motion,workers=4,max_cache_mib=limit)
    new = ColumnFeatureSampler(prep,3,plan,grid,cfg,columns.pose_motion,workers=2,max_cache_mib=limit,compiled_patches=True)
    for ids in (np.arange(len(plan)),np.array([3,408,812,3,0,408]),np.empty(0,np.int64)):
        small = plan.subset(ids)
        expected = old.sample(small,None); actual = new.sample(small,None)
        assert all(np.array_equal(v,actual[k]) for k,v in expected.items())


@pytest.mark.parametrize('frames',(4,6))
@pytest.mark.parametrize('budget',(1,64*2**20))
def test_hoisted_source_and_fixed_inputs_same_rows_dtypes_and_budget(frames,budget):
    prep,grid,joint,records,source,owner = _eval_fixture(frames)
    prep = owner(joint).prepare_columns(source,records[0],include_gt=True)
    plan = columns.candidate_plan(prep,1,grid,joint.columns.config)
    with torch.inference_mode():
        cache = HorizonInputs(joint.columns,prep,1,plan,torch.device('cpu'),enabled=True,max_bytes=budget)
        assert cache.bytes <= budget
        assert cache.budget_fallback == (budget == 1)
        for start in range(0,len(plan),7):
            small = plan.subset(slice(start,start+7))
            arrays = columns.sample_column_features(prep,1,small,grid,joint.columns.config)
            actual = cache.batch(arrays,small.legal,torch.device('cpu'),start,start+len(small))
            for k,v in dict(arrays,legal=small.legal).items(): assert torch.equal(actual[k],torch.as_tensor(v))
            if cache.source is not None:
                assert torch.equal(actual['source_features'],joint.columns.source_features_for(prep,1,small,torch.device('cpu')))


@pytest.mark.parametrize('frames',(4,6))
@pytest.mark.parametrize('empty',(False,True))
def test_fast_probability_bit_exact_original_batch_and_one_source_gather(compiled,monkeypatch,frames,empty):
    monkeypatch.setenv('SWFM_COLUMN_CPU_BACKEND','native')
    prep,grid,joint,records,source,owner = _eval_fixture(frames)
    prep = owner(joint).prepare_columns(source,records[0],include_gt=True)
    plan = columns.candidate_plan(prep,1,grid,joint.columns.config)
    if empty: plan = plan.subset(np.empty(0,np.int64))
    torch.nn.init.normal_(joint.columns.refinement.weight,std=.2)
    model = joint.columns; weights = {k:v.clone() for k,v in model.state_dict().items()}
    original = columns.predict_probabilities(model,prep,1,plan,grid,torch.device('cpu'),7,optimized=False)
    model.column_probability_optimized=True; model.column_readback_optimized=False
    model.column_probability_fingerprint=True
    model.column_inference_verify_remaining=1; caller=get_ident(); batches=[]
    forward=model.forward;source_features=model.source_features_for
    def check_forward(**b):
        assert get_ident() == caller; batches.append(len(b['kind']))
        return forward(**b)
    # Explicit first-use reference verification + optimized horizon both run.
    with patch.object(model,'forward',side_effect=check_forward), \
            patch.object(model,'source_features_for',wraps=source_features) as source_calls:
        actual=columns.predict_probabilities(model,prep,1,plan,grid,torch.device('cpu'),7,optimized=True)
    chunks=[min(7,len(plan)-s) for s in range(0,len(plan),7)]
    assert batches == chunks*2 and source_calls.call_count == len(chunks)+(not empty)
    assert np.array_equal(actual,original) and model.last_prediction_profile['probability_exactness_passed']
    assert model.last_prediction_profile['source_finite_check_deferred']
    assert not model.last_prediction_profile['packed_input_upload']
    assert len(model.last_prediction_profile['probability_sha256']) == 64
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in weights.items())


def test_nonfinite_hoisted_source_cannot_escape_on_unused_static_queries():
    prep,grid,joint,records,source,owner = _eval_fixture()
    prep = owner(joint).prepare_columns(source,records[0],include_gt=True)
    plan = columns.candidate_plan(prep,1,grid,joint.columns.config)
    model = joint.columns; model.column_probability_optimized=True;model.column_readback_optimized=False
    with patch.object(model,'source_features_for',return_value=torch.full((len(plan),model.source_dim),float('nan'))), \
            patch.object(model,'forward',side_effect=lambda **b:(torch.zeros(len(b['kind']),model.config.z_bins),
                torch.zeros(len(b['kind']),model.config.z_bins,3))):
        with pytest.raises(RuntimeError,match='nonfinite'):
            columns.predict_probabilities(model,prep,1,plan,grid,torch.device('cpu'),7,optimized=True)


def test_next_horizon_cpu_map_really_overlaps_and_shutdown_is_bounded():
    ready=Event();release=Event();thread_ids=[];built=[];caller=get_ident()
    planning={h:Future() for h in (1,3,5)}
    for f in planning.values():f.set_result('plan')
    factory=HorizonFeaturePrefetch(None,(1,3,5),planning,None,None,torch.device('cpu'),None,
        history_index=None,enabled=True,workers=8)
    def build(h):
        thread_ids.append(get_ident());built.append(h)
        if h == 3:
            ready.set();assert release.wait(5)
        return 'sampler'+str(h),.01
    with patch.object(factory,'_build',side_effect=build),factory:
        assert factory.get(1)[0] == 'sampler1'
        assert ready.wait(5)  # NEXT map runs while caller processes horizon1.
        assert built == [1,3] and factory.workers == 2
        release.set()
        assert factory.get(3)[0] == 'sampler3'
        assert factory.get(5)[0] == 'sampler5'
    assert factory.pending is None and all(i != caller for i in thread_ids)
    disabled=HorizonFeaturePrefetch(None,(1,),planning,None,None,torch.device('cpu'),None,history_index=None)
    with disabled:assert disabled.get(1) == (None,0.,0.)


def test_prefetch_failure_does_not_leak_workers_or_accept_wrong_plan():
    planning={1:Future(),3:Future()};planning[1].set_exception(RuntimeError('bad original plan'))
    factory=HorizonFeaturePrefetch(None,(1,3),planning,None,None,torch.device('cpu'),None,
        history_index=None,enabled=True)
    with pytest.raises(RuntimeError,match='bad original plan'),factory:factory.get(1)
    assert factory.pending is None and factory.pool._shutdown
    prep,grid,joint,records,source,owner=_eval_fixture()
    prep=owner(joint).prepare_columns(source,records[0],include_gt=True)
    plan=columns.candidate_plan(prep,1,grid,joint.columns.config)
    joint.columns.column_probability_optimized=True
    wrong=type('WrongSampler',(),{'prepared':prep,'h':3,'plan':plan})()
    with pytest.raises(ValueError,match='current horizon'):
        columns.predict_probabilities(joint.columns,prep,1,plan,grid,torch.device('cpu'),7,
            optimized=True,feature_sampler=wrong)


def test_full_report_exact_across_horizon_prefetch_probabilities_and_counts(compiled,monkeypatch):
    monkeypatch.setenv('SWFM_COLUMN_CPU_BACKEND','native')
    prep,grid,joint,records,source,owner = _eval_fixture()
    provider=owner(joint);provider.raw_prefetch_workers=provider.raw_prefetch_depth=2
    joint.columns.column_inference_optimized=True;joint.columns.column_readback_optimized=False
    torch.nn.init.normal_(joint.columns.refinement.weight,std=.2)
    with patch.object(columns,'gt_moving_support_sequence',return_value=moving_fixture(prep)):
        old=columns.evaluate_columns(provider,source,records,joint.columns,(.5,.5,None),batch_size=7,diagnostic_thresholds=None)
        joint.columns.column_probability_optimized=True;joint.columns.column_inference_verify_remaining=3
        events=[]
        new=columns.evaluate_columns(provider,source,records,joint.columns,(.5,.5,None),batch_size=7,
            diagnostic_thresholds=None,progress=events.append)
    assert finite_json(old) == finite_json(new)
    assert all(p['inverse_map_overlaps_previous_horizon'] for e in events for p in e['prediction_seconds_by_horizon'].values())


def test_column_suite_same_counts_reverse_order_restores_flags_and_reports_hotspot(tmp_path,compiled,monkeypatch):
    from tools.real_motion import joint_eval_speed as speed
    from real_motion.column_runtime_pipeline import CachedColumnSource
    monkeypatch.setenv('SWFM_COLUMN_CPU_BACKEND','native')
    prep,grid,joint,records,source,owner=_eval_fixture();provider=owner(joint)
    weights={k:v.clone() for k,v in joint.state_dict().items()}
    with patch.object(columns,'gt_moving_support_sequence',return_value=moving_fixture(prep)):
        result=speed.benchmark_evaluation(provider,CachedColumnSource(source),records,joint.columns,(.5,.5,None),
            tmp_path,windows=3,repeats=2,batch_size=7,column_suite=True)
    assert [t['name'] for t in result['trials']] == ['parallel_raw','parallel_columns','parallel_columns_prefetch',
        'parallel_columns_prefetch','parallel_columns','parallel_raw']
    assert result['integer_counts_exact'] and len({t['counts_fingerprint'] for t in result['trials']}) == 1
    assert all(not t['buffered_readback'] for t in result['trials'])
    assert all(t['column_audit']['peak_horizon_inputs_bytes'] <= 64*2**20 for t in result['trials'])
    assert all(t['column_audit']['peak_horizon_inputs_working_bytes_bound'] <= 64*2**20 for t in result['trials'])
    assert result['probability_bytes_exact'] and len({t['probability_fingerprint'] for t in result['trials']}) == 1
    assert all(t['probability_verifications_in_timed_windows'] == 0 for t in result['trials'])
    assert all(not hasattr(joint.columns,name) for name in ('column_probability_optimized','column_map_prefetch'))
    assert all(torch.equal(v,joint.state_dict()[k]) for k,v in weights.items())
    text=speed.summary_text(result)
    assert 'column_probability_breakdown=' in text and 'column_probability_speedup=' in text and 'largest_column_host_stage=' in text
    assert 'No automatic promotion' in text and not (tmp_path/'evaluation.json').exists()


def test_same_counts_but_changed_probability_fingerprint_rejected(tmp_path,compiled,monkeypatch):
    from tools.real_motion import joint_eval_speed as speed
    from real_motion.column_runtime_pipeline import CachedColumnSource
    monkeypatch.setenv('SWFM_COLUMN_CPU_BACKEND','native')
    prep,grid,joint,records,source,owner=_eval_fixture();provider=owner(joint)
    forward=columns.predict_probabilities
    def forged(model,*args,**kwargs):
        value=forward(model,*args,**kwargs)
        if getattr(model,'column_probability_optimized',False):model.last_prediction_profile['probability_sha256']='wrong'
        return value  # SAME probabilities/counts, but intentionally corrupt evidence.
    with patch.object(columns,'gt_moving_support_sequence',return_value=moving_fixture(prep)), \
            patch.object(columns,'predict_probabilities',side_effect=forged):
        with pytest.raises(RuntimeError,match='probability BYTES'):
            speed.benchmark_evaluation(provider,CachedColumnSource(source),records,joint.columns,(.5,.5,None),
                tmp_path,windows=3,repeats=1,batch_size=7,column_suite=True)
    assert not hasattr(joint.columns,'column_probability_fingerprint') and not (tmp_path/'speed.json').exists()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA/BF16 required')
def test_cuda_same_bf16_batch_probability_bytes_and_report():
    prep,grid,joint,records,source,owner = _eval_fixture();joint.cuda();device=torch.device('cuda')
    provider=owner(joint);prep=provider.prepare_columns(source,records[0],include_gt=True)
    prep.outputs=joint.motion(records[0],device)
    plan=columns.candidate_plan(prep,1,grid,joint.columns.config)
    expected=columns.predict_probabilities(joint.columns,prep,1,plan,grid,device,7,optimized=False)
    joint.columns.column_probability_optimized=True;joint.columns.column_readback_optimized=False
    actual=columns.predict_probabilities(joint.columns,prep,1,plan,grid,device,7,optimized=True)
    assert np.array_equal(expected,actual)
