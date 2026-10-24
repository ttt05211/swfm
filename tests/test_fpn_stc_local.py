import numpy as np
import pytest
import torch
from tools.fpn_stc_local import calibration as cal


def test_identity_ties_and_scalar_formula_no_mutation():
    torch.manual_seed(21);z=torch.randn(6,18,32);p=torch.randn_like(z)
    z[:,2,0]=z[:,17,0]=10.;old=z.clone();prior=p.clone()
    predictions=cal.predict(z,p,(4,4,2))
    np.testing.assert_array_equal(predictions[cal.name(cal.BASELINE)],z[[1,3,5]].argmax(1).reshape(3,4,4,2))
    for gain,bias in cal.CANDIDATES:
        expect=z[[1,3,5]]+(gain-1)*(z[[1,3,5]]-p[[1,3,5]])
        expect[:,17]+=bias
        np.testing.assert_array_equal(predictions[cal.name((gain,bias))],expect.argmax(1).reshape(3,4,4,2))
    torch.testing.assert_close(z,old,rtol=0,atol=0);torch.testing.assert_close(p,prior,rtol=0,atol=0)


def fixture_counts():
    v=np.zeros((64,len(cal.CANDIDATES),3,18,18),np.int64)
    v[:,:,:,4,4]=6;v[:,:,:,4,17]=4
    # Candidate1 helps fitting scenes but harms held-out ones.
    v[:32,1,:,4,4]=8;v[:32,1,:,4,17]=2
    v[32:,1,:,4,4]=3;v[32:,1,:,4,17]=7
    return v


def test_selection_never_uses_held_labels_and_preserves_failures():
    v=fixture_counts();r=cal.summarize(v)
    assert r['selected_index']==1 and not r['held32_both_improve']
    assert r['held32_delta']['mIoU']==pytest.approx(-30.)
    v[32:,2,:,4,4]=10;v[32:,2,:,4,17]=0
    assert cal.summarize(v)['selected_index']==1
    assert len(r['held32'])==2
    with pytest.raises(ValueError):cal.summarize(v[:63])


def test_full_grid_counts_and_zero_union_scores():
    pred={cal.name(c):np.full((3,4,4,2),4,np.uint8) for c in cal.CANDIDATES}
    gt=np.full((3,4,4,2),4,np.uint8);gt[:,0]=17
    c=cal.counts(pred,gt)
    assert c.shape==(12,3,18,18) and (c.sum((-1,-2))==32).all()
    assert cal.report(c[0])['average']['mIoU']==75.


def test_motion_identity_and_copy_only_shared_geometry():
    import copy
    from test_fpn_unet import fixture
    from tools.fpn_unet_all.run import AllPredictor
    from tools.fpn_stc_local import motion
    body,old,row=fixture();pcfg=old.provider.pcfg;old.close()
    body.eval().requires_grad_(False)
    with torch.no_grad():body.columns.head.weight.normal_(std=.02)
    p=AllPredictor(body,pcfg,'cpu',workers=1);p.kind='stc'
    try:
        raw={k:row[1][k] for k in ('history_occ','history_observed','history_poses')}
        prepared=p.prepare(row[0],raw,times=np.arange(4)*.5)
        track=copy.deepcopy(prepared[1]['_shared_source_track'])
        history=copy.deepcopy(prepared[3][0]['memory_reads'])
        before={k:v.clone() for k,v in body.state_dict().items()}
        fw=p.forward(prepared,poses=row[1]['future_poses'])
        identity=motion.render(body,p.provider,prepared,fw,'damped',1.)
        torch.testing.assert_close(identity,fw['logits'],rtol=0,atol=0)
        predictions=motion.predict(body,p.provider,prepared,fw)
        assert len(predictions)==len(motion.CANDIDATES)==12
        assert all(v.shape==(3,*pcfg.grid.shape_hwd) and v.dtype==np.uint8 for v in predictions.values())
        for k,v in track.pair_velocity.items():np.testing.assert_array_equal(prepared[1]['_shared_source_track'].pair_velocity[k],v)
        for k,v in history.items():np.testing.assert_array_equal(prepared[3][0]['memory_reads'][k],v)
        for k,v in before.items():torch.testing.assert_close(body.state_dict()[k],v,rtol=0,atol=0)
    finally:p.close()


def test_memory_processing_constant_identity_height_and_no_mutation():
    from tools.fpn_stc_local import memory
    x=torch.randn(6,16,8,8,4);before=x.clone()
    torch.testing.assert_close(memory.process(x,'smooth',0.),x,rtol=0,atol=0)
    assert torch.count_nonzero(memory.process(x,'gain',0.))==0
    # XY filtering must never mix different heights, including grid boundaries.
    constant=torch.arange(4.).view(1,1,1,1,4).expand_as(x).clone()
    torch.testing.assert_close(memory.process(constant,'smooth',.5),constant,rtol=0,atol=0)
    torch.testing.assert_close(x,before,rtol=0,atol=0)


def test_input_derived_free_evidence_only_no_GT_visibility():
    from tools.fpn_stc_local import evidence
    raw=dict(history_occ=np.full((4,4,4,2),17,np.uint8),history_observed=np.ones((4,4,4,2),bool),
        future_gt_occ=object(),mask_lidar=object())
    data=dict(static_valid=np.ones((4,2,2,2),np.float32),coverage=np.ones(3))
    history=(dict(memory_reads=data,shared_reads=object()),None)
    changed=evidence.change(history,raw,0.)
    assert not changed[0]['memory_reads']['static_valid'].any()
    np.testing.assert_array_equal(evidence.change(history,raw,.5)[0]['memory_reads']['static_valid'],.5)
    np.testing.assert_array_equal(data['static_valid'],1.)
    assert changed[0]['memory_reads']['coverage'] is data['coverage']
    assert changed[0]['shared_reads'] is history[0]['shared_reads']


def test_wider_scalar_candidates_still_match_same_two_parameter_formula():
    from tools.fpn_stc_local import wide
    torch.manual_seed(12);z=torch.randn(6,18,32);p=torch.randn_like(z)
    preds=cal.predict(z,p,(4,4,2),candidates=wide.CANDIDATES)
    assert len(preds)==16
    for gain,bias in wide.CANDIDATES:
        expected=z[[1,3,5]]+(gain-1)*(z[[1,3,5]]-p[[1,3,5]])
        expected[:,17]+=bias
        np.testing.assert_array_equal(preds[wide.name((gain,bias))],expected.argmax(1).reshape(3,4,4,2))
