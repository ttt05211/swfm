import numpy as np
import torch
from tools.stc_overnight.replay import Mirror, GridShift, prepare_forecast, iter_forecasts, temporal_static_prior, multiframe_velocity
from test_shared_source_track import fixture, component


def test_mirror_physical_correspondence_and_proper_poses():
    p, rec, raw = fixture()
    try:
        raw['history_poses'][:, 0, 3] = np.arange(4)
        poses = np.tile(raw['history_poses'][-1], (6, 1, 1)); poses[:, 1, 3] += np.arange(6)
        m = Mirror(True)
        _, flipped, dest = m.apply(rec, raw, poses, p.provider.pcfg.grid)
        np.testing.assert_allclose(flipped['history_poses'][-1], np.eye(4), atol=1e-14)
        np.testing.assert_allclose(np.linalg.det(dest[:, :3, :3]), 1., atol=1e-14)
        np.testing.assert_array_equal(m.restore(torch.tensor(flipped['history_occ']), raw['history_occ'].shape[1:]), raw['history_occ'])
        assert Mirror().apply(rec, raw, poses, p.provider.pcfg.grid)[1] is raw
    finally: p.close()


def test_chunked_forecast_matches_original_cpu():
    p, rec, raw = fixture()
    try:
        # Nonzero residual ensures the point decoder is tested, not just the prior.
        torch.nn.init.normal_(p.body.columns.readout[-1].weight, std=.01)
        poses = np.tile(np.eye(4), (6, 1, 1)); poses[:, 0, 3] = np.arange(6)*.1
        row = p.prepare(rec, raw, times=np.arange(4)*.5)
        expected = p.forward(row, poses=poses)
        c = prepare_forecast(p, row, poses)
        items = list(iter_forecasts(c))
        for key in ('prior', 'memory', 'logits'):
            actual = torch.cat([v[key] for _, v in items])
            torch.testing.assert_close(actual, expected[key], rtol=1e-5, atol=2e-6)
    finally: p.close()


def test_supported_temporal_probability_does_not_erase_current_occupied():
    p,rec,raw=fixture()
    try:
        raw['history_occ'][-1,1:3,5:7,0]=17
        row=p.prepare(rec,raw,times=np.arange(4)*.5)
        context=prepare_forecast(p,row,np.tile(np.eye(4),(6,1,1)))
        original=context['renderer'].static.clone()
        audit=temporal_static_prior(context,support_only=True)
        after=context['renderer'].static
        assert audit['soft_supported_free_voxels']>0
        mask=original.argmax(1)!=17
        torch.testing.assert_close(after.permute(0,2,3,4,1)[mask],original.permute(0,2,3,4,1)[mask],rtol=0,atol=0)
        torch.testing.assert_close(after.sum(1),torch.ones_like(after[:,0]))
    finally:p.close()


def test_center_filter_moves_geometry_feature_points_consistently_and_not_old_inputs():
    p,rec,raw=fixture()
    try:
        raw['history_occ'][-1][raw['history_occ'][-1]==4]=17
        raw['history_occ'][-1][tuple(component(5)['voxel_indices'].T)]=4
        row=p.prepare(rec,raw,times=np.arange(4)*.5)
        old_anchor=row[0]['anchors_xy_t0_m'].copy();old_points=[v.copy() for v in row[3][0]['state']['source_world_points']]
        changed,audit=multiframe_velocity(row,p.provider,np.arange(4)*.5,True)
        old,new=row[3][0]['state'],changed[3][0]['state']
        delta=np.asarray([b['centroid_world']-a['centroid_world'] for a,b in zip(old['current'],new['current'])])
        assert audit['center_filter'] and np.linalg.norm(delta)>0
        for i,(a,b) in enumerate(zip(old_points,new['source_world_points'])):np.testing.assert_allclose(b-a,np.broadcast_to(delta[i],a.shape),atol=1e-12)
        a=row[3][0]['memory_reads'];b=changed[3][0]['memory_reads']
        np.testing.assert_allclose(b['world']-a['world'],delta[a['actor']],atol=1e-12)
        np.testing.assert_array_equal(row[0]['anchors_xy_t0_m'],old_anchor)
        assert changed[0]['anchors_xy_t0_m'] is not row[0]['anchors_xy_t0_m']
    finally:p.close()


def test_shift_valid_crop_and_occupancy_first_decoder():
    from tools.stc_overnight.hierarchical import decode
    view=GridShift(1,1);shape=(4,4,2)
    x=torch.randn(18,*shape)
    shifted=torch.zeros_like(x);shifted[:,1:,1:]=x[:,:-1,:-1]
    restored=view.restore(shifted,shape)
    valid=torch.tensor(view.valid(shape))
    torch.testing.assert_close(restored[:,valid],x[:,valid])
    assert (restored.argmax(0)[~valid]==17).all()
    logits=torch.full((18,2,2,2),-2.);logits[17]=0
    assert (logits.argmax(0)==17).all() and (decode(logits)!=17).all()


def test_tiny_confidence_head_initial_identity_bound_and_final_loss_gradient():
    from tools.stc_overnight.occupancy_head import OccupancyCalibration,features,adjust
    head=OccupancyCalibration();shape=(8,8,4)
    prior=torch.zeros(1,18,np.prod(shape));prior[:,17]=1.
    memory=torch.randn(1,16,4,4,4)
    x=features(memory,prior,shape);beta=head(x,shape)
    logits=torch.randn_like(prior)
    torch.testing.assert_close(adjust(logits,beta),logits,atol=0,rtol=0)
    target=torch.ones((1,np.prod(shape)),dtype=torch.long)
    torch.nn.functional.cross_entropy(adjust(logits,beta),target).backward()
    assert head.net[-1].bias.grad.abs().sum()>0
    assert sum(p.numel() for p in head.parameters())==1025
    assert beta.abs().max()<=2


def test_soft_memory_consistency_keeps_geometry_and_dynamic_evidence():
    from tools.stc_overnight.replay import semantic_memory_consistency
    p,rec,raw=fixture()
    try:
        row=p.prepare(rec,raw,times=np.arange(4)*.5)
        old=row[3][0]['memory_reads'];before=old['static_valid'].copy()
        value,audit=semantic_memory_consistency(row,p.provider.pcfg.grid,'cpu')
        new=value[3][0]['memory_reads']
        assert value[:3]==row[:3]
        assert value[3][0]['state'] is row[3][0]['state']
        np.testing.assert_array_equal(old['static_valid'],before)
        np.testing.assert_array_equal(new['coverage'],old['coverage'])
        assert audit['minimum']>=.25-1e-6 and audit['maximum']<=1+1e-6
        assert (new['static_valid']<=before+1e-6).all()
    finally:p.close()


def test_relative_binary_alignment_identity_sign_and_invalid_marginals():
    import pytest
    from tools.stc_overnight.calibration import binary_bias
    p=np.ones(18);p[17]=17
    np.testing.assert_array_equal(binary_bias(p,p),np.zeros(18))
    q=p.copy();q[17]=34
    bias=binary_bias(p,q)
    assert bias[0]>bias[17]
    np.testing.assert_array_equal(bias[:17],np.full(17,bias[0]))
    with pytest.raises(ValueError):binary_bias(np.zeros(18),p)


def test_calibration_fit_only_requests_historical_inputs(monkeypatch):
    from tools.stc_overnight import calibration
    from types import SimpleNamespace
    class HistoryOnly:
        records=[dict(role=role,scene_name=f'{role}_{i}',t0_token=str(i),future_poses=np.tile(np.eye(4),(6,1,1)))
            for role,n in [('clean_train',32),('reference64',64)] for i in range(n)]
        def history(self,row,modality,known):
            assert known=='dense_valid'
            assert modality==('gt' if row['role']=='clean_train' else 'stc')
            return {},dict(modality=modality),np.arange(4)*.5
        def targets(self,*args,**kwargs):raise AssertionError('calibration must not read future labels')
    monkeypatch.setattr(calibration,'prepare_forecast',lambda p,row,poses:row)
    def forecasts(context):
        for h in range(6):
            x=torch.zeros(1,18,8);x[:,17]=2 if context['raw']['modality']=='gt' else 3
            yield [h],dict(logits=x)
    monkeypatch.setattr(calibration,'iter_forecasts',forecasts)
    predictor=SimpleNamespace(prepare=lambda rec,raw,times:dict(raw=raw))
    fit=calibration.fit(HistoryOnly(),predictor,'frozen_source_hash')
    assert len(fit['keys']['source'])==32 and len(fit['keys']['target'])==64
    assert fit['target_GT_labels_used_for_fitting'] is False
    assert fit['bias'][0]>fit['bias'][17]


def test_supported_server_routes_preserve_original_predictions():
    from tools.stc_overnight.verify_all import forecast
    p,rec,raw=fixture()
    try:
        row=p.prepare(rec,raw,times=np.arange(4)*.5);poses=np.tile(np.eye(4),(6,1,1))
        bias=np.zeros(18);bias[:17]=.2
        original=forecast(p,row,poses,bias)
        extended=forecast(p,row,poses,bias,include_support=True)
        for name,value in original.items():np.testing.assert_array_equal(value,extended[name])
        assert set(extended)==set(original)|{'supported_'+name for name in original}
        assert all(v.shape==(6,*p.provider.pcfg.grid.shape_hwd) for v in extended.values())
    finally:p.close()


def test_observer_em_recovers_independent_synthetic_rates_and_prior_mass():
    from tools.stc_overnight.temporal_em import binary_observer_posterior,apply
    rng=np.random.default_rng(21);latent=rng.random(100000)<.08
    sensitivity=np.asarray([.65,.75,.8,.7]);false_positive=np.asarray([.015,.02,.01,.025])
    occupied=rng.random((len(latent),4))<np.where(latent[:,None],sensitivity,false_positive)
    known=rng.random(occupied.shape)>.1
    code=(np.where(known,occupied.astype(int)+1,0)*(3**np.arange(4))).sum(1)
    posterior,audit=binary_observer_posterior(np.bincount(code,minlength=81))
    assert posterior[0]==0 and posterior[80]>posterior[40]
    assert abs(audit['occupied_prior']-.08)<.01
    np.testing.assert_allclose(audit['sensitivity'],sensitivity,atol=.03,rtol=0)
    np.testing.assert_allclose(audit['false_positive'],false_positive,atol=.005,rtol=0)
    p,rec,raw=fixture()
    try:
        row=p.prepare(rec,raw,times=np.arange(4)*.5)
        context=prepare_forecast(p,row,np.tile(np.eye(4),(6,1,1)))
        prior=context['renderer'].static.clone();apply(context);after=context['renderer'].static
        torch.testing.assert_close(after.sum(1),torch.ones_like(after[:,0]))
        assert torch.isfinite(after).all() and (after>=0).all()
        mask=prior.argmax(1)!=17
        assert (after.permute(0,2,3,4,1)[mask].max(-1).values>=.5).all()
    finally:p.close()
