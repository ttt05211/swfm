"""Speed-only changes preserve byte probabilities, finite checks and counts."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from real_motion.column_inference_pipeline import ProbabilityReadback, inference_tensors
from real_motion.column_runtime_pipeline import CachedColumnSource
from tools.real_motion import causal_column_common as columns
from tools.real_motion import joint_eval_speed as speed
from test_joint_checkpoint_selection import _eval_fixture
from test_causal_columns import moving_fixture


@pytest.mark.parametrize('n',(0,1,7))
def test_packed_input_bytes_dtypes_alignment_shapes_and_strides(n):
    arrays=dict(history=np.arange(n*12,dtype=np.uint8).reshape(n,3,4)[:,:,::2],
        flags=np.ones((n,3,2),np.uint8),context=np.arange(n*5,dtype=np.float32).reshape(n,5)/19,
        classes=np.full(n,13,np.uint8),indices=np.arange(n,dtype=np.int64),
        bf16=torch.arange(n,dtype=torch.bfloat16),double=torch.arange(n,dtype=torch.float64))
    legal=np.zeros((n,2,3),bool);legal[...,0]=True
    expected=inference_tensors(arrays,legal,torch.device('cpu'))
    actual=inference_tensors(arrays,legal,torch.device('cpu'),packed=True)
    assert actual.keys() == expected.keys()
    for key,value in expected.items():
        assert actual[key].dtype == value.dtype and actual[key].shape == value.shape
        assert torch.equal(actual[key],value)
        assert actual[key].storage_offset()*actual[key].element_size() % actual[key].element_size() == 0


@pytest.mark.parametrize('limit', (1, 48, 200, 1000000))
def test_probability_buffer_exact_order_empty_and_byte_budget(limit):
    buffered = ProbabilityReadback(buffered=True,max_bytes=limit)
    original = ProbabilityReadback()
    for n in (2, 0, 5, 1, 7):
        value = torch.arange(n*6,dtype=torch.float32).reshape(n,2,3)/17
        original.append(value);buffered.append(value)
        assert buffered.bytes <= limit and buffered.peak_bytes <= limit
    assert np.array_equal(buffered.result((0,2,3)),original.result((0,2,3)))
    assert not buffered.pending and buffered.bytes == 0
    assert buffered.transfers <= original.transfers
    assert ProbabilityReadback(buffered=True).result((0,2,3)).shape == (0,2,3)


@pytest.mark.parametrize('buffered', (False, True))
def test_linked_probabilities_exact_training_check_not_disabled(buffered):
    prep,grid,joint,records,source,owner=_eval_fixture()
    provider=owner(joint);prepared=provider.prepare_columns(source,records[0],include_gt=True)
    plan=columns.candidate_plan(prepared,1,grid,joint.columns.config)
    torch.nn.init.normal_(joint.columns.refinement.weight,std=.2)
    expected=columns.predict_probabilities(joint.columns,prepared,1,plan,grid,torch.device('cpu'),7,optimized=False)
    joint.columns.column_readback_optimized=buffered
    actual=columns.predict_probabilities(joint.columns,prepared,1,plan,grid,torch.device('cpu'),7,optimized=True)
    assert np.array_equal(expected,actual)
    profile=joint.columns.last_prediction_profile
    assert profile['source_finite_check_deferred'] == buffered
    if buffered: assert profile['probability_readback_transfers'] == 1
    else: assert profile['probability_readback_transfers'] > 1
    small=plan.subset([0]);values=columns.sample_column_features(prepared,1,small,grid,joint.columns.config)
    batch={k:torch.as_tensor(v) for k,v in values.items()}
    batch['source_features']=torch.full((1,joint.columns.source_dim),float('nan'))
    with pytest.raises(RuntimeError,match='continuous source'):
        joint.columns(**batch)  # direct/training default still checks immediately


def test_deferred_source_finiteness_cannot_be_hidden_by_finite_logits():
    prep,grid,joint,records,source,owner=_eval_fixture()
    prepared=owner(joint).prepare_columns(source,records[0],include_gt=True)
    plan=columns.candidate_plan(prepared,1,grid,joint.columns.config)
    def zeros(**batch):
        assert batch['validate_source'] is False
        n=len(batch['kind']);z=joint.columns.config.z_bins
        return torch.zeros(n,z),torch.zeros(n,z,3)
    with patch.object(joint.columns,'source_features_for',side_effect=lambda p,h,s,d:
            torch.full((len(s),joint.columns.source_dim),float('inf'))), \
            patch.object(joint.columns,'forward',side_effect=zeros):
        with pytest.raises(RuntimeError,match='nonfinite'):
            columns.predict_probabilities(joint.columns,prepared,1,plan,grid,torch.device('cpu'),7,optimized=True)


def test_speed_population_round_robin_not_prefix_or_gt_selected():
    records=[dict(scene_name=s,t0_token=str(i)) for s in ('z','a','b') for i in range(6)]
    rows,keys=speed.speed_records(records,6)
    assert [str(r['scene_name']) for r in rows] == ['z','a','b']*2
    assert len(set(keys)) == 6
    with pytest.raises(ValueError,match='duplicate'):speed.speed_records(records+[records[0]],6)


def test_speed_runs_reverse_order_exact_counts_and_restores_settings(tmp_path):
    prep,grid,joint,records,source,owner=_eval_fixture()
    provider=owner(joint);provider.raw_prefetch_workers=provider.raw_prefetch_depth=1
    with torch.no_grad():joint.columns.generation.bias.fill_(1.5)
    original={k:v.clone() for k,v in joint.state_dict().items()}
    with patch.object(columns,'gt_moving_support_sequence',return_value=moving_fixture(prep)):
        result=speed.benchmark_evaluation(provider,CachedColumnSource(source),records,joint.columns,
            (.5,.5,None),tmp_path,windows=3,repeats=2,batch_size=7)
    assert [t['name'] for t in result['trials']] == ['serial_raw','parallel_raw','parallel_buffered',
        'parallel_buffered','parallel_raw','serial_raw']
    assert len({t['counts_fingerprint'] for t in result['trials']}) == 1
    assert result['integer_counts_exact'] and not result['actual_cuda']
    assert provider.raw_prefetch_workers == provider.raw_prefetch_depth == 1
    assert not hasattr(joint.columns,'column_readback_optimized')
    assert all(torch.equal(v,original[k]) for k,v in joint.state_dict().items())
    assert 'raw_prefetch_speedup=' in speed.summary_text(result)
    assert not (tmp_path/'evaluation.json').exists()


def test_mismatched_speed_counts_fail_closed_and_restore(tmp_path):
    provider=SimpleNamespace(workers=2,device=torch.device('cpu'),frozen_metric_counts={})
    model=SimpleNamespace(column_inference_optimized=False)
    records=[dict(scene_name='s',t0_token=str(i)) for i in range(3)]
    with patch.object(speed,'_pass',side_effect=[('a',{}),('a',{}),('b',{}),('b',{})]):
        with pytest.raises(RuntimeError,match='changed integer'):
            speed.benchmark_evaluation(provider,CachedColumnSource(None),records,model,(.5,.5,None),
                tmp_path,windows=3,repeats=1)
    assert model.column_inference_optimized is False
    assert not hasattr(model,'column_readback_optimized') and not hasattr(provider,'raw_prefetch_workers')


@pytest.mark.parametrize('frames', (4, 6))
def test_previous_strong_reuse_same_registration_bytes_and_one_less_extraction(frames):
    from real_motion.rigid_transport import rigid_source_points_world
    from real_motion.strong_w2det import StrongW2DetConfig, match_instances
    prep,grid,joint,records,source,owner=_eval_fixture()
    raw=copy.deepcopy(prep.raw)
    # Add enough connected voxels for the REAL frozen Strong threshold.
    history=np.full((frames,*grid.shape_hwd),17,np.uint8)
    for f in range(frames):history[f,1+f:4+f,5:8,1]=4
    poses=[np.eye(4)]*frames;strong=StrongW2DetConfig()
    current=columns.extract_instances_cropped_exact(history[-1],poses[-1],grid=grid,cfg=strong)
    previous=columns.extract_instances_cropped_exact(history[-2],poses[-2],grid=grid,cfg=strong)
    assert len(current) == len(previous) == 1
    state=dict(current=current,velocities=match_instances(previous,current,.5,max_speed_mps=strong.max_match_speed_mps),
        source_world_points=[rigid_source_points_world(c['voxel_indices'],poses[-1],grid=grid) for c in current])
    extract=columns.extract_instances_cropped_exact
    with patch.object(columns,'extract_instances_cropped_exact',wraps=extract) as counted:
        expected=columns.causal_source_history(history,poses,state,grid,strong,2)
        assert counted.call_count == frames-1
        counted.reset_mock()
        actual=columns.causal_source_history(history,poses,state,grid,strong,2,previous_instances=previous)
        assert counted.call_count == frames-2
    # Registrations, transformed point arrays, association and audit are exact.
    def equal(a,b):
        if isinstance(a,np.ndarray):assert np.array_equal(a,b)
        elif isinstance(a,(tuple,list)):
            assert len(a) == len(b)
            for x,y in zip(a,b):equal(x,y)
        else:assert a == b
    equal(expected,actual)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA required')
def test_real_cuda_buffered_readback_probabilities_exact():
    prep,grid,joint,records,source,owner=_eval_fixture();joint.cuda()
    provider=owner(joint);prepared=provider.prepare_columns(source,records[0],include_gt=True)
    prepared.outputs=joint.motion(records[0],torch.device('cuda'))
    plan=columns.candidate_plan(prepared,1,grid,joint.columns.config)
    expected=columns.predict_probabilities(joint.columns,prepared,1,plan,grid,torch.device('cuda'),7,optimized=False)
    actual=columns.predict_probabilities(joint.columns,prepared,1,plan,grid,torch.device('cuda'),7,optimized=True)
    assert np.array_equal(expected,actual)
    assert joint.columns.last_prediction_profile['probability_readback_transfers'] == 1
    assert joint.columns.last_prediction_profile['packed_input_upload']
