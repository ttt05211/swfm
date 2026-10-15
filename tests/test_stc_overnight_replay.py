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
