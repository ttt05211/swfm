import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from real_motion.height_causal_field import HeightCausalField,gather_centres
from real_motion.causal_column_completion import ColumnConfig,ColumnPlan,KEEP,ADD,GENERATE,FREE
from tools.real_motion.causal_column_common import pose_motion,sample_column_features


def fixture():
    from test_source_repair_pilot import scene
    grid,prep=scene();z=grid.shape_hwd[-1]
    prep.raw['future_poses']=np.tile(np.eye(4),(6,1,1))
    config=ColumnConfig(z_bins=z)
    xy=np.array([[3,2],[5,5]],np.int32);flat=(xy[:,0:1]*8+xy[:,1:2])*z+np.arange(z)
    base=prep.baseline[0].ravel()[flat];legal=np.zeros((2,z,3),bool);legal[...,KEEP]=True;legal[...,ADD]=base==FREE
    plan=ColumnPlan(xy,np.array([1,0],np.uint8),np.array([0,-3],np.int32),np.array([4,11],np.uint8),
        flat,base,base.copy(),legal,np.zeros((2,12),np.float32))
    prep.targets[:,0]=[3.5,2.5,1.5]
    out=dict(history_source_context=torch.randn(1,8,requires_grad=True),
        future_transport_queries=torch.randn(1,6,8,requires_grad=True))
    return grid,prep,config,plan,out


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_centres_equal_original_patch_centre_and_do_not_read_gt(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('actual CUDA needed')
    grid,prep,cfg,plan,_=fixture();prep.raw['future_gt_occ']='POISON'
    sampled=gather_centres(prep,[(0,plan,None,None)],grid,cfg,device,pose_motion)
    expected=sample_column_features(prep,0,plan,grid,cfg)
    np.testing.assert_array_equal(sampled.labels.cpu(),expected['history'][:,:,3,3])
    flags=expected['flags'][:,:,3,3]
    np.testing.assert_array_equal(sampled.observed.cpu(),(flags&1)!=0)
    np.testing.assert_array_equal(sampled.owned.cpu(),(flags&2)!=0)


@pytest.mark.parametrize('mode',['temporal_gate','shared_field'])
def test_height_output_live_context_gradient_and_legal_actions(mode):
    grid,prep,cfg,plan,out=fixture()
    samples=gather_centres(prep,[(0,plan,None,None)],grid,cfg,'cpu',pose_motion)
    model=HeightCausalField(mode,z_bins=cfg.z_bins,source_dim=8)
    field=model.encode_history(torch.tensor(np.asarray(prep.raw['history_occ'])),torch.tensor(np.asarray(prep.raw['history_observed'])))
    g,r=model(field,samples,out);p=model.probabilities(g,r,samples)
    assert g.shape==(2,4) and r.shape==(2,4,3)
    assert torch.isfinite(p).all() and torch.allclose(p.sum(-1),torch.ones((2,4)))
    assert not p[...,2].any()
    (g.sum()+r.sum()).backward()
    assert out['future_transport_queries'].grad[0,0].abs().sum()>0
    assert out['history_source_context'].grad.abs().sum()>0
    if mode=='shared_field':assert model.spatial[0][0].weight.grad.abs().sum()>0


def test_native_height_is_not_collapsed_and_moving_pose_is_current():
    grid,prep,cfg,plan,out=fixture()
    first=gather_centres(prep,[(0,plan,None,None)],grid,cfg,'cpu',pose_motion)
    prep.targets[:,0,0]+=1
    second=gather_centres(prep,[(0,plan,None,None)],grid,cfg,'cpu',pose_motion)
    assert not torch.equal(first.labels[0],second.labels[0])
    assert torch.unique(first.native_height).numel()==4
    assert torch.equal(first.labels[1],second.labels[1])


def test_forward_source_field_is_causal_owned_and_has_actual_backward():
    from real_motion.height_causal_field import gather_forward_fields
    grid,prep,cfg,plan,out=fixture();prep.raw['future_gt_occ']='POISON'
    samples=gather_forward_fields(prep,[(0,plan,None,None)],grid,'cpu')
    assert samples.labels[0,:,1].eq(4).all()
    assert samples.labels[1,:,0].eq(11).all()
    assert samples.aligned_density.shape==(2,4,4)
    model=HeightCausalField('forward_field',z_bins=4,source_dim=8)
    g,r=model(None,samples,out);(g.sum()+r.sum()).backward()
    assert out['future_transport_queries'].grad.abs().sum()>0
