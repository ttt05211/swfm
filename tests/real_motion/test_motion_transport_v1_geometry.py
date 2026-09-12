import numpy as np
from real_motion.geometry import OccupancyGrid
from real_motion.motion_transport_v1.geometry import index_to_metric_center,metric_to_center_index,F_to_world,world_to_F,f_to_world_matrix
def test_center_index_roundtrip_and_axes():
    g=OccupancyGrid();idx=np.array([[0,0,0],[199,199,15],[17,83,4]]);p=index_to_metric_center(idx,g);assert np.allclose(metric_to_center_index(p,g),idx);assert np.allclose(p[0],[-39.8,-39.8,-.8])
def test_world_F_roundtrip_with_pitch():
    yaw=.7;pitch=.2;cy,sy=np.cos(yaw),np.sin(yaw);cp,sp=np.cos(pitch),np.sin(pitch);Rz=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1.]]);Ry=np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]]);T=np.eye(4);T[:3,:3]=Rz@Ry;T[:3,3]=[12,-4,1.7];q=np.array([[1,-2,.4],[-3,5,-.2],[0,0,0.]]);assert np.allclose(world_to_F(F_to_world(q,T),T),q,atol=1e-10);F=f_to_world_matrix(T);assert np.allclose(F[:3,2],[0,0,1])
