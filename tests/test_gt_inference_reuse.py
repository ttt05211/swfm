import copy
import numpy as np
import pytest
import torch
from tools.gt_inference_reuse import engine,run


def test_readout_preserves_original_chunked_sampling_and_values():
    from tools.shared_history_experiments.refine.model import SemanticRefiner
    torch.set_num_threads(1);torch.manual_seed(21)
    columns=SemanticRefiner(feature_width=16,z_bins=4,memory_width=4,width=8,chunk_size=17).eval()
    # Last layer must be nonzero, or a broken residual path could pass unnoticed.
    torch.nn.init.normal_(columns.readout[-1].weight,std=.05)
    prior=torch.rand(6,18,8*8*4);mem=torch.rand(6,4,4,4,4)
    with torch.no_grad():
        a=columns(mem,prior,range(6),(8,8,4));b=engine.readout(columns,mem,prior,range(6),(8,8,4))
    torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize('backend',['readout_reuse','geometry_reuse','combined'])
def test_complete_forward_fresh_strong_no_weight_changes_and_no_forecast_cache(backend,monkeypatch):
    from test_shared_source_track import fixture
    from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
    p,rec,raw=fixture();torch.set_num_threads(1);torch.manual_seed(21)
    try:
        torch.nn.init.normal_(p.body.columns.readout[-1].weight,std=.05)
        poses=np.tile(np.eye(4),(6,1,1));poses[:,0,3]=np.arange(6)*.03
        row=p.prepare(rec,raw,times=np.arange(4)*.5);ready=engine.prepared(p,row,poses,np.arange(4)*.5)
        fp=tensor_state_fingerprint(p.body.state_dict())
        reference=engine.fresh(p,ready,[.2]*17+[0.],'reference')
        calls=[];original=engine.backgrounds
        def counted(*a,**kw):calls.append(1);return original(*a,**kw)
        monkeypatch.setattr(engine,'backgrounds',counted)
        with torch.no_grad():
            a=engine.fresh(p,ready,[.2]*17+[0.],backend)
            b=engine.fresh(p,ready,[.2]*17+[0.],backend)
        assert calls==[1,1] and a.shape==(6,*raw['history_occ'].shape[1:])
        np.testing.assert_array_equal(a,reference);np.testing.assert_array_equal(b,reference)
        assert tensor_state_fingerprint(p.body.state_dict())==fp
        assert 'column_backgrounds' not in ready[0][0][3][0]['state']
    finally:p.close()


def test_summary_keeps_reference_repeat_and_candidate_failures_without_timing_filter():
    sample=dict(seconds={n:[[.1],[.2]] for n in engine.BACKENDS},
        changed_voxels={n:[0,1] for n in engine.BACKENDS},
        peak_allocated_mib={n:1 for n in engine.BACKENDS},peak_reserved_mib={n:2 for n in engine.BACKENDS},profile={},
        strong_audit=dict(strict_byte_parity=True,changed_voxels_by_horizon=[0]*6))
    result=run.aggregate([sample],engine.BACKENDS,1,True)
    assert result['reference']['output_frame_fps']==pytest.approx(40.)
    assert all(not v['strict_byte_parity'] for v in result.values())
    assert not result['combined']['strict_pass_gate'] and result['combined']['all_timings_retained']


def test_reused_geometry_matches_complete_prior_and_memory_tensors():
    from test_shared_source_track import fixture
    from tools.shared_history_experiments.refine.geometry import TransportPrior,aligned_memory
    p,rec,raw=fixture();torch.set_num_threads(1);torch.manual_seed(21)
    try:
        poses=np.tile(np.eye(4),(6,1,1));poses[:,0,3]=np.arange(6)*.03
        ready=engine.prepared(p,p.prepare(rec,raw,times=np.arange(4)*.5),poses,np.arange(4)*.5)
        (row,staged,times),cpu_poses=ready
        row,_=engine.old.candidate(row,p.provider,times);rec,raw,_,history=row
        state=history[0]['state'];state=dict(state,column_backgrounds=engine.backgrounds(raw,state,cpu_poses,p.provider,raw['_shared_source_track']))
        ref=TransportPrior(raw,state,p.provider.pcfg.grid,p.provider.device)
        engine.old.temporal_static_prior(dict(renderer=ref,raw=raw,poses=staged['poses']),support_only=True)
        opt=engine.ReusedTransport(raw,state,p.provider.pcfg.grid,p.provider.device,staged['poses'],staged)
        opt.static_support(staged['observed'])
        torch.testing.assert_close(ref.static,opt.static,rtol=0,atol=0)
        n=len(state['current']);targets=staged['poses'].new_tensor(np.asarray([s['centroid_world'] for s in state['current']]))[:,None].expand(-1,6,-1)
        yaws=targets.new_zeros((n,6));columns=p.body.columns
        with torch.no_grad():
            a=ref.render(staged['poses'],targets,yaws);b=opt.render(staged['poses'],targets,yaws)
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            feature=torch.rand(4,columns.lift.in_channels,raw['history_occ'].shape[1]//2,raw['history_occ'].shape[2]//2)
            lifted=columns.lift_history(feature)
            output={'future_transport_queries':torch.rand(n,6,columns.source_query.in_features)}
            a=aligned_memory(columns,lifted,output,history,raw,targets,yaws,staged['poses'],p.provider.pcfg.grid)
            b=engine.memory(columns,lifted,output,history,raw,targets,yaws,staged['poses'],p.provider.pcfg.grid,opt,range(6))
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    finally:p.close()


def test_existing_native_majority_cpu4_strong_bytes_and_provider_restored():
    from test_shared_source_track import fixture
    from real_motion.strong_majority_execution import ParallelNativeMajority
    p,rec,raw=fixture();torch.set_num_threads(1)
    try:
        poses=np.tile(np.eye(4),(6,1,1));ready=engine.prepared(p,p.prepare(rec,raw,times=np.arange(4)*.5),poses,np.arange(4)*.5)
        original=p.provider.strong_majority
        with ParallelNativeMajority(4) as majority:
            p.gt_reuse_majority=majority
            audit=engine.strong_audit(p,ready)
            assert audit['strict_byte_parity'] and p.provider.strong_majority is original
            expected=engine.fresh(p,ready,[0.]*18,'reference')
            actual=engine.fresh(p,ready,[0.]*18,'combined_cpu4')
            np.testing.assert_array_equal(actual,expected)
            assert p.provider.strong_majority is original
            assert majority.stats()['calls']==12  # six audit + six timed-path frames
    finally:p.close()
