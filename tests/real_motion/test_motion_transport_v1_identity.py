import numpy as np
from real_motion.geometry import OccupancyGrid
from real_motion.strong_w2det import StrongW2DetConfig,strong_w2det_sequence
from real_motion.motion_transport_v1.contracts import CausalInputs
from real_motion.motion_transport_v1.source_adapter import decompose_strong_sources
from real_motion.motion_transport_v1.compositor import hard_kta_identity

def pose(tx=0,ty=0,tz=0,yaw=0,pitch=0):
    cy,sy=np.cos(yaw),np.sin(yaw);cp,sp=np.cos(pitch),np.sin(pitch);Rz=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1.]]);Ry=np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]]);T=np.eye(4);T[:3,:3]=Rz@Ry;T[:3,3]=[tx,ty,tz];return T
def causal(hist,poses,future):return CausalInputs('s:0','s',hist,np.ones_like(hist,dtype=bool),tuple(poses),tuple(future),np.arange(6)*.5,2.5+np.arange(1,7)*.5)
def test_zero_delta_is_exact_strong_w2det_with_small_and_unmatched_and_pose():
    grid=OccupancyGrid(-4,-4,-1,(.4,.4,.4),(20,20,8));cfg=StrongW2DetConfig();hist=np.full((6,*grid.shape_hwd),17,dtype=np.uint8);hist[-2,5:8,5:7,2]=4;hist[-1,6:9,5:7,2]=4;hist[-1,13:16,13:15,2]=7;hist[-1,2:3,2:4,2]=6;hist[-1,10:12,2:4,1]=11;hist[-2,10:12,2:4,1]=11;poses=[pose(tx=.02*i,yaw=.01*i) for i in range(6)];future=[pose(tx=.12+.03*i,ty=-.02*i,yaw=.08+.02*i,pitch=.03) for i in range(6)];c=causal(hist,poses,future);dec=decompose_strong_sources(c,grid=grid,cfg=cfg,frame_dt_s=.5,crop_radius_limit_m=11.2);got=hard_kta_identity(c,dec,grid=grid);ref=strong_w2det_sequence(hist,poses,future,frame_dt_s=.5,grid=grid,cfg=cfg);assert np.array_equal(got,ref),np.count_nonzero(got!=ref)
def test_zero_delta_empty_dynamic_exact():
    grid=OccupancyGrid(-2,-2,-1,(.4,.4,.4),(10,10,5));cfg=StrongW2DetConfig();hist=np.full((6,*grid.shape_hwd),17,dtype=np.uint8);hist[:,2:5,2:5,1]=11;poses=[pose() for _ in range(6)];future=[pose(tx=.1*i,yaw=.02*i) for i in range(6)];c=causal(hist,poses,future);dec=decompose_strong_sources(c,grid=grid,cfg=cfg);assert len(dec.sources)==0;assert np.array_equal(hard_kta_identity(c,dec,grid=grid),strong_w2det_sequence(hist,poses,future,grid=grid,cfg=cfg))
