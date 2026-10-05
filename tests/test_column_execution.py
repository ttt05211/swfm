"""Execution is opt-in, causal, finite, bounded and preserves all query heads."""
from unittest.mock import patch
import numpy as np
import pytest
import torch

from real_motion.column_execution import PatchMemory, ColumnExecution, execution_session, AsyncProbabilityReadback
from tools.real_motion import causal_column_common as columns
from test_joint_checkpoint_selection import _eval_fixture
from test_causal_columns import moving_fixture


def batch_for(model,prepared,h,plan,grid):
    arrays=columns.sample_column_features(prepared,h,plan,grid,model.config)
    batch={k:torch.as_tensor(v) for k,v in arrays.items()}
    batch['source_features']=model.source_features_for(prepared,h,plan,torch.device('cpu'))
    return arrays,batch,torch.as_tensor(plan.legal)


@pytest.mark.parametrize('frames',(4,6))
@pytest.mark.parametrize('limit',(1,64*2**20))
def test_patch_cache_identical_bytes_only_bounded_query_specific_heads(frames,limit):
    prep,grid,joint,records,source,owner=_eval_fixture(frames)
    prep=owner(joint).prepare_columns(source,records[0],include_gt=True)
    plan=columns.candidate_plan(prep,1,grid,joint.columns.config).subset([0,0,0,1,1,2])
    with torch.inference_mode():
      arrays,batch,legal=batch_for(joint.columns,prep,1,plan,grid)
      # Same patch but DIFFERENT query context/base/source must not share logits.
      batch['context'][1,0]+=0.8;batch['source_features'][2]+=0.2
      torch.nn.init.normal_(joint.columns.generation.weight,std=.3)
      torch.nn.init.normal_(joint.columns.refinement.weight,std=.3)
      cache=PatchMemory(limit)
      memory,invalid=cache.encode(joint.columns,arrays,batch)
      expected=joint.columns.encode_history(batch['history'],batch['flags'])
      assert torch.equal(memory,expected[0]) and torch.equal(invalid,expected[1])
      assert cache.bytes <= limit and cache.peak <= limit
      ex=ColumnExecution(joint.columns,reuse=True)
      a,fa=ex.function(batch,legal);b,fb=ex.run(batch,legal,memory=memory,invalid=invalid)
      assert torch.equal(a,b) and fa and fb
      assert cache.hits>=2 and cache.encoded<=4
      old=memory.clone();arrays['flags'][0,0,0,0,0]^=1;batch['flags']=torch.as_tensor(arrays['flags'])
      after,_=cache.encode(joint.columns,arrays,batch)
      direct,_=joint.columns.encode_history(batch['history'],batch['flags'])
      assert torch.equal(after,direct)


@pytest.mark.parametrize('reuse',(False,True))
def test_probability_and_counts_nonconstant_head_exact_and_session_no_weight_writes(reuse):
    prep,grid,joint,records,source,owner=_eval_fixture();model=joint.columns
    with torch.no_grad():
      torch.nn.init.normal_(model.refinement.weight,std=.3);torch.nn.init.normal_(model.generation.weight,std=.3)
    provider=owner(joint);model.column_inference_optimized=True;model.column_readback_optimized=False
    weights={k:v.clone() for k,v in joint.state_dict().items()}
    with patch.object(columns,'gt_moving_support_sequence',return_value=moving_fixture(prep)):
      old=columns.evaluate_columns(provider,source,records,model,(.5,.5,.95),batch_size=7,diagnostic_thresholds=None)
      model.column_inference_verify_remaining=3;model.column_async_readback=True
      with execution_session(model,graphs=True,reuse=reuse):
        new=columns.evaluate_columns(provider,source,records,model,(.5,.5,.95),batch_size=7,diagnostic_thresholds=None)
    from tools.real_motion.static_evidence_selector_common import finite_json
    assert finite_json(old)==finite_json(new)
    assert not hasattr(model,'column_execution_session')
    assert all(torch.equal(v,joint.state_dict()[k]) for k,v in weights.items())


def test_inference_only_changed_weights_and_nonfinite_fail_closed():
    prep,grid,joint,records,source,owner=_eval_fixture()
    prep=owner(joint).prepare_columns(source,records[0],include_gt=True)
    plan=columns.candidate_plan(prep,1,grid,joint.columns.config).subset([0,1])
    ex=ColumnExecution(joint.columns)
    with pytest.raises(RuntimeError,match='inference_mode'):ex.validate()
    with torch.inference_mode():
      arrays,batch,legal=batch_for(joint.columns,prep,1,plan,grid)
      batch['source_features'][:]=float('nan')
      _,finite=ex.run(batch,legal);assert not finite
      joint.columns.generation.bias.add_(1)
      with pytest.raises(RuntimeError,match='weights/calibration'):ex.run(batch,legal)


def test_parameter_replacement_with_same_version_cannot_replay_stale_graph():
    prep,grid,joint,records,source,owner=_eval_fixture()
    ex=ColumnExecution(joint.columns)
    old=joint.columns.generation.weight
    replacement=torch.nn.Parameter(old.detach().clone())
    while replacement._version<old._version:
        with torch.no_grad():replacement.add_(0)
    joint.columns.generation.weight=replacement
    with torch.inference_mode(),pytest.raises(RuntimeError,match='weights/calibration'):ex.validate()


def test_patch_window_cannot_leak_into_next_window_or_nested_scope():
    prep,grid,joint,records,source,owner=_eval_fixture();ex=ColumnExecution(joint.columns,reuse=True)
    with ex.patch_window():
      first=ex.current_memory
      with pytest.raises(RuntimeError,match='nested'):
        with ex.patch_window():pass
    assert ex.current_memory is None
    with ex.patch_window():assert ex.current_memory is not first and not ex.current_memory.rows


def test_async_cpu_empty_and_order():
    for sizes in ((),(2,3,1)):
      out=AsyncProbabilityReadback((sum(sizes),2,3),'cpu')
      expected=[]
      for i,n in enumerate(sizes):
        a=torch.full((n,2,3),float(i));out.append(a);expected.append(a.numpy())
      actual=out.result((0,2,3));out.close()
      assert np.array_equal(actual,np.concatenate(expected) if expected else np.empty((0,2,3),np.float32))


def test_slab_lru_preserves_current_batch_hits_and_oversize_does_not_truncate():
    from real_motion.causal_column_model import CausalColumnModel
    from real_motion.causal_column_completion import ColumnConfig
    model=CausalColumnModel(ColumnConfig(width=8,heads=2,layers=1,semantic_dim=4,z_bins=2),history_frames=4).eval()
    rng=np.random.default_rng(19)
    history=rng.integers(0,19,(3,4,7,7,2),dtype=np.uint8);flags=rng.integers(0,4,history.shape,dtype=np.uint8)
    row_size=history[0].nbytes+flags[0].nbytes+4*49*8*4+4*49
    cache=PatchMemory(2*row_size)
    with torch.inference_mode():
      for ids in ([0,1],[1,2],[0,0],[0,1,2,0],[2,1]):
        arrays=dict(history=history[ids],flags=flags[ids]);batch={k:torch.as_tensor(v) for k,v in arrays.items()}
        actual=cache.encode(model,arrays,batch);expected=model.encode_history(**batch)
        assert all(torch.equal(a,b) for a,b in zip(actual,expected))
        assert cache.bytes<=cache.limit and cache.capacity==2
      assert cache.memory.shape[0]==2 and len(cache.rows)<=2


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA graph/pinned D2H required')
@pytest.mark.parametrize('reuse',(False,True))
def test_cuda_graph_async_probabilities_bytes_and_graph_overwrite(reuse):
    prep,grid,joint,records,source,owner=_eval_fixture();device=torch.device('cuda')
    provider=owner(joint);prep=provider.prepare_columns(source,records[0],include_gt=True)
    joint.cuda()
    prep.outputs=joint.motion(records[0],device)
    plan=columns.candidate_plan(prep,1,grid,joint.columns.config)
    with torch.no_grad():torch.nn.init.normal_(joint.columns.refinement.weight,std=.3)
    expected=columns.predict_probabilities(joint.columns,prep,1,plan,grid,device,7,optimized=True)
    joint.columns.column_async_readback=True;joint.columns.column_readback_optimized=False
    with execution_session(joint.columns,graphs=True,reuse=reuse) as ex:
      actual=columns.predict_probabilities(joint.columns,prep,1,plan,grid,device,7,optimized=True)
      assert ex.replays>1 and not ex.failures
    assert np.array_equal(expected,actual)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA allocation fallback required')
def test_graph_buffer_allocation_oom_falls_back_without_repeated_capture(monkeypatch):
    prep,grid,joint,records,source,owner=_eval_fixture()
    prep=owner(joint).prepare_columns(source,records[0],include_gt=True)
    plan=columns.candidate_plan(prep,1,grid,joint.columns.config).subset([0,1])
    with torch.inference_mode():
      arrays,batch,legal=batch_for(joint.columns,prep,1,plan,grid)
    joint.cuda()  # persistent buffers must retain ordinary version counters
    with torch.inference_mode():
      batch={k:v.cuda() for k,v in batch.items()};legal=legal.cuda()
      ex=ColumnExecution(joint.columns,graphs=True)
      expected,finite=ex.function(batch,legal);assert finite
      original=torch.Tensor.clone
      def oom(value,*args,**kwargs):
        if value is batch['history']:raise torch.cuda.OutOfMemoryError('synthetic graph-buffer OOM')
        return original(value,*args,**kwargs)
      monkeypatch.setattr(torch.Tensor,'clone',oom)
      for _ in range(2):
        actual,finite=ex.run(batch,legal)
        assert finite and torch.equal(expected,actual)
      assert ex.replays==0 and ex.eager_calls==2 and len(ex.failures)==1


def test_execution_scripts_are_opt_in_and_combine_speed_with_six_frame_fps():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    suite=(root/'tools/real_motion/run_p0_f9_joint_execution_speed_fps.sh').read_text(encoding='utf-8')
    assert '--speed-execution-and-fps' in suite and '--thresholds 0.5 0.5 0.95' in suite
    assert '--speed-benchmark' in suite and '--population dev64' in suite
    wrapper=(root/'tools/real_motion/run_p0_f9_joint_interim_eval.sh').read_text(encoding='utf-8')
    assert '${FULL_JOINT_EVAL_EXECUTION_BACKEND:-eager}' in wrapper
    assert '--execution-backend "$EXECUTION_BACKEND"' in wrapper


def test_fps_requires_six_frames_and_no_selection_or_fps_from_eval():
    from tools.real_motion.joint_execution_speed import _generation_fingerprint,summary_text
    profiles=[dict(queries=i,probability_sha256=str(i)) for i in range(6)]
    a=[np.zeros((2,2,2),np.uint8) for _ in range(6)]
    with pytest.raises(RuntimeError,match='six'):_generation_fingerprint(a[:3],profiles[:3])
    assert _generation_fingerprint(a,profiles)==_generation_fingerprint(a,profiles)


def test_real_six_frame_path_rebuilds_prior_and_produces_all_six_without_gt():
    from tools.real_motion import joint_execution_speed as speed
    prep,grid,joint,records,source,owner=_eval_fixture();provider=owner(joint)
    provider.strong=object();provider.pcfg.frame_dt_s=.5
    raw={**prep.raw,'future_gt_occ':None}
    state=dict(current_sem=raw['history_occ'][-1],current_pose=raw['history_poses'][-1],future_poses=raw['future_poses'],
        current=[],velocities=[],source_world_points=[],column_backgrounds='MUST_NOT_REUSE')
    raw['_column_causal_preparation']=dict(prepared_state=state)
    original=provider.prepare_columns;prepared_states=[]
    def prepare(src,rec,*,include_gt,raw_window,outputs=None):
        assert include_gt is False and raw_window['future_gt_occ'] is None
        actual=raw_window['_column_causal_preparation']['prepared_state']
        assert 'column_backgrounds' not in actual and actual['anchors'] is prep.baseline
        prepared_states.append(actual)
        return original(src,rec,include_gt=False,raw_window=raw_window,outputs=outputs)
    provider.prepare_columns=prepare
    joint.columns.column_inference_optimized=True;joint.columns.column_readback_optimized=False
    joint.columns.column_probability_fingerprint=True;joint.columns.column_sampling_workers=2
    with patch.object(speed.runtime,'_strong_all_horizons',return_value=(prep.baseline,[[] for _ in range(6)])) as prior, \
         patch.object(columns,'gt_moving_support_sequence',side_effect=AssertionError('NO GT metric')):
      pred,profiles=speed.generation_six(provider,source,records[0],raw,joint.columns,(.5,.5,.95),7)
    assert prior.call_count==1 and len(prepared_states)==1
    assert len(pred)==len(profiles)==6 and all(a.shape==grid.shape_hwd for a in pred)
    assert state['column_backgrounds']=='MUST_NOT_REUSE'  # caller's fixed state not modified
    assert len(speed._generation_fingerprint(pred,profiles)[0])==64


def test_one_execution_suite_runs_eval_and_real_six_paths_restores_state(tmp_path):
    from tools.real_motion import joint_execution_speed as speed
    from real_motion.column_runtime_pipeline import CachedColumnSource
    prep,grid,joint,records,source,owner=_eval_fixture();provider=owner(joint)
    provider.reference=joint.transport;provider.joint=joint;provider.strong=object();provider.pcfg.frame_dt_s=.5
    original=provider.load_raw_columns
    def load(src,rec,*,include_gt):
        raw=original(src,rec,include_gt=include_gt)
        if not include_gt:
            raw['future_gt_occ']=None
            raw['_column_causal_preparation']=dict(prepared_state=dict(current_sem=raw['history_occ'][-1],
                current_pose=raw['history_poses'][-1],future_poses=raw['future_poses'],
                current=[],velocities=[],source_world_points=[],column_backgrounds='old'))
        return raw
    provider.load_raw_columns=load;weights={k:v.clone() for k,v in joint.state_dict().items()}
    with patch.object(columns,'gt_moving_support_sequence',return_value=moving_fixture(prep)), \
         patch.object(columns,'window_from_record',return_value=prep.window), \
         patch.object(speed.runtime,'_strong_all_horizons',return_value=(prep.baseline,[[] for _ in range(6)])), \
         patch.object(speed.runtime,'_stage_gpu_inputs'),patch.object(speed.runtime,'_release_gpu_inputs'), \
         patch.object(speed.runtime,'_forecast_once_with_prior_rebuild',return_value=prep.baseline):
      r=speed.benchmark_execution(provider,CachedColumnSource(source),records,joint.columns,(.5,.5,.95),tmp_path,
          windows=3,repeats=2,batch_size=7,fps_windows=1)
    assert r['integer_counts_exact'] and r['six_frame_dense_outputs_exact'] and not r['rejected']
    assert len(r['trials'])==8 and len(r['fps_trials'])==8
    assert [t['name'] for t in r['trials']][:4]==list(speed.MODES)
    assert [t['name'] for t in r['trials']][4:]==list(reversed(speed.MODES))
    assert all(t['frames']==6 for t in r['fps_trials'])
    assert set(r['fps'])==set(speed.MODES)
    assert not hasattr(joint.columns,'column_execution_session')
    assert all(torch.equal(v,joint.state_dict()[k]) for k,v in weights.items())
    assert 'Eval windows/s is NOT frame FPS' in speed.summary_text(r)


def test_real_fixed_geometry_source_link_renderer_all_six_no_gt_or_prior_mock():
    import copy
    from dataclasses import replace
    from types import SimpleNamespace
    from real_motion.joint_causal_columns import JointCausalColumns
    from real_motion.strong_w2det import StrongW2DetConfig
    from tools.real_motion.joint_column_full_common import build_fixed_geometry
    from tools.real_motion import joint_execution_speed as speed
    from test_joint_causal_columns import fixture
    prep,grid,old,_,record=fixture()
    joint=JointCausalColumns(replace(old.transport.v17_config,history_frames=4),old.columns.config).eval()
    prep.window.history_tokens=tuple(f'h{i}' for i in range(4))
    raw=copy.deepcopy(prep.raw)
    for key in ('history_occ','history_observed','history_poses'):raw[key]=raw[key][-4:]
    raw['future_gt_occ']=None
    raw['history_occ'][raw['history_occ']==4]=17
    for f in range(4):raw['history_occ'][f,1+f:4+f,5:8,1]=4
    strong=StrongW2DetConfig();pcfg=SimpleNamespace(grid=grid,frame_dt_s=.5,free_label=17)
    current=columns.extract_instances_cropped_exact(raw['history_occ'][-1],raw['history_poses'][-1],grid=grid,cfg=strong)
    assert len(current)==1
    center=torch.tensor(np.array([c['centroid_world'][:2] for c in current]),dtype=torch.float32)
    record['source_centroid_xy_t0_m']=center
    record['anchors_xy_t0_m']=center[:,None,:].repeat(1,6,1)+record['kta_displacement_xy_m']
    provider=columns.FrozenColumns.__new__(columns.FrozenColumns)
    provider.pcfg=pcfg;provider.device=torch.device('cpu');provider.workers=1;provider.strong=strong
    provider.joint=joint;provider.model=joint.transport;provider.columns_checked=False
    provider.encode_record=lambda r:joint.motion(r,provider.device)
    joint.columns.column_inference_optimized=True;joint.columns.column_readback_optimized=False
    joint.columns.column_probability_fingerprint=False
    # Only record->window metadata is substituted. REAL source extraction,
    # Strong/KTA prior, renderer, history registration, candidates and network.
    with patch.object(columns,'window_from_record',return_value=prep.window), \
         patch.object(speed.runtime,'window_from_record',return_value=prep.window), \
         patch.object(columns,'gt_moving_support_sequence',side_effect=AssertionError('FPS cannot touch GT')):
      raw['_column_causal_preparation']=build_fixed_geometry(raw,record,pcfg,strong,1,joint.columns.config)
      staged=speed.runtime._gpu_inputs(record,provider.device)
      before,profiles=speed.generation_six(provider,None,record,raw,joint.columns,(.5,.5,.95),7,transport_inputs=staged)
      old_fp=speed._generation_fingerprint(before,profiles)
      with execution_session(joint.columns,graphs=True,reuse=True):
        after,profiles=speed.generation_six(provider,None,record,raw,joint.columns,(.5,.5,.95),7,transport_inputs=staged)
      assert old_fp==speed._generation_fingerprint(after,profiles)
      assert len(after)==6 and all(a.shape==grid.shape_hwd for a in after)
      strict,profiles=speed.generation_six(provider,None,record,raw,joint.columns,(.5,.5,.95),7,
          transport_inputs=staged,loaded_history=True)
      assert old_fp==speed._generation_fingerprint(strict,profiles)
