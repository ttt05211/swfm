from types import SimpleNamespace
import math
import numpy as np
from real_motion.motion_transport_v1.contracts import SourceRecord,SourceDecomposition
from real_motion.motion_transport_v1.targets import build_training_targets
class FakeNusc:
    def __init__(self):
        self.samples={'t0':{'anns':['a0']}};self.anns={'a0':{'instance_token':'inst','category_name':'vehicle.car','translation':[10.,0.,0.],'size':[2.,4.,2.],'rotation':[1.,0.,0.,0.]}}
        for i in range(1,7):
            tok=f'f{i}';at=f'a{i}';self.samples[tok]={'anns':[] if i==3 else [at]}
            if i!=3:
                yaw=math.pi/2 if i==1 else 0.;self.anns[at]={'instance_token':'inst','category_name':'vehicle.car','translation':[10.,float(i),0.],'size':[2.,4.,2.],'rotation':[math.cos(yaw/2),0.,0.,math.sin(yaw/2)]}
    def get(self,table,token):
        if table=='sample':return self.samples[token]
        if table=='sample_annotation':return self.anns[token]
        raise KeyError((table,token))
class FakeSource:
    def __init__(self,observed=True):self.nusc=FakeNusc();self.observed=observed
    def load_occ3d(self,scene,token,require_lidar_mask=True):
        sem=np.full((2,2,1),17,np.uint8);sem[0,0,0]=4;obs=np.ones((2,2,1),bool);obs[0,0,0]=self.observed;return sem,obs
def _fixture():
    points=np.array([[11.,0.,0.],[10.8,.2,0.],[10.7,-.2,0.],[10.9,.1,0.],[10.6,-.1,0.],[10.5,.2,0.]]);vox=np.stack([np.arange(len(points)),np.zeros(len(points),int),np.zeros(len(points),int)],1);src=SourceRecord(0,4,vox,points,points.mean(0),np.zeros(3),True,np.array([[0.,0.],[1.,1.]]),len(points));decomp=SourceDecomposition([src],np.zeros((6,1,1,1),np.uint8),np.zeros((0,3),int),np.zeros(0,np.uint8),np.zeros((0,3)),np.arange(1,7)*.5);w=SimpleNamespace(scene_name='s',t0_token='t0',future_tokens=tuple(f'f{i}' for i in range(1,7)));return decomp,w
def test_motion_target_rotates_about_gt_box_center_and_missing_is_invalid():
    decomp,w=_fixture();t=build_training_targets(FakeSource(),w,decomp).motion_targets[0];assert np.allclose(t.gt_xy_world[0,0],[10.,2.],atol=1e-6);assert bool(t.valid[0]);assert not bool(t.valid[2])
def test_future_lidar_observation_is_audit_only_not_ce_validity():
    decomp,w=_fixture();targets=build_training_targets(FakeSource(observed=False),w,decomp);assert targets.future_observed is not None;assert not bool(targets.future_observed[0,0,0,0]);assert bool(targets.future_valid[0,0,0,0]);assert int(targets.future_semantics[0,0,0,0])==4;assert targets.future_valid.all()
