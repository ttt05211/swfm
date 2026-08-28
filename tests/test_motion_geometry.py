import numpy as np
from real_motion.motion import decompose_masks,PersistenceMotionConfig
from real_motion.geometry import OccupancyGrid,relative_transform,warp_semantic_grid
from real_motion.kta import causal_kta,KTAConfig


def tiny_grid():return OccupancyGrid(-2,-2,-1,(1,1,1),(4,4,2))

def test_free_is_never_confident_static():
    x=np.full((3,4,4,2),17,dtype=np.int64);x[:,1,1,0]=4;m=decompose_masks(x,PersistenceMotionConfig());assert m.confident_static[1,1,0];assert not m.confident_static[0,0,0];assert m.free[0,0,0]

def test_relative_identity_warp():
    g=tiny_grid();sem=np.full(g.shape_hwd,17,dtype=np.int64);sem[1,2,0]=4;T=np.eye(4);assert np.array_equal(warp_semantic_grid(sem,relative_transform(T,T),g,17),sem)


def test_occ3d_axis0_is_metric_x():
    # Use an asymmetric grid so an accidental [Y,X,Z] implementation cannot
    # pass by symmetry. A +1m metric-x transform must increment array axis 0.
    g=OccupancyGrid(-2,-3,-1,(1,1,1),(4,6,2))
    sem=np.full(g.shape_hwd,17,dtype=np.int64);sem[1,4,0]=4
    T=np.eye(4);T[0,3]=1.0
    warped=warp_semantic_grid(sem,T,g,17)
    assert warped[2,4,0]==4
    assert warped[1,5,0]==17
    assert g.x_max==2 and g.y_max==3


def test_kta_constant_translation():
    g=tiny_grid();hist=np.full((2,*g.shape_hwd),17,dtype=np.int64);hist[0,1,1,0]=4;hist[1,1,2,0]=4;cand=np.zeros(g.shape_hwd,dtype=bool);cand[1,2,0]=1;pred,_,comps=causal_kta(hist,cand,[1.0],g,KTAConfig(history_dt_s=1.0,max_match_distance_m=2.0));assert len(comps)==1;assert pred[0,1,3,0]==4

def test_online_prepare_never_loads_future_semantics():
    from real_motion.prepared import PrepareConfig,prepare_nuscenes_window
    from real_motion.nuscenes_adapter import WindowTokens
    class Source:
        def __init__(self):self.loaded=[]
        def load_semantics(self,scene,token):
            self.loaded.append(token);x=np.full((4,4,2),17,dtype=np.int64);x[1,1 if token=='h0' else 2,0]=4;return x
        def pose(self,token):return np.eye(4)
        def official_trajectory(self,history,future,hist_last=2,zero_prefix=0,require_info=False):return np.zeros((4,2),dtype=np.float32)
    src=Source();w=WindowTokens('s',('h0','h1'),'h1',('f0','f1'));cfg=PrepareConfig(history_frames=2,future_frames=2,frame_dt_s=0.5,tube_radii=(0,0),trajectory_length=4,trajectory_hist_last=2,trajectory_zero_prefix=0,trajectory_protocol='unit_4step',require_temporal_info=False,grid=tiny_grid(),motion=PersistenceMotionConfig(min_observed_frames=2,min_static_observations=2),kta=KTAConfig(history_dt_s=0.5,max_match_distance_m=3.0));out=prepare_nuscenes_window(src,w,cfg,include_gt=False);assert src.loaded==['h0','h1'];assert 'future_gt_occ' not in out;assert out['generation_support_occ'].shape==(2,4,4);assert out['trajectory'].shape==(4,2)

def test_component_track_marks_whole_shifted_object_moving():
    x=np.full((3,8,8,2),17,dtype=np.int64);x[0,3:5,1:3,0]=4;x[1,3:5,2:4,0]=4;x[2,3:5,3:5,0]=4;cfg=PersistenceMotionConfig(history_dt_s=0.5,voxel_size_xy_m=(0.4,0.4),moving_speed_mps=0.5,static_speed_mps=0.2);m=decompose_masks(x,cfg);cur=(x[-1]==4);assert cur.sum()==4;assert m.moving[cur].all();assert not m.confident_static[cur].any();assert not m.uncertain[cur].any()

def test_component_track_marks_persistent_object_static():
    x=np.full((3,8,8,2),17,dtype=np.int64);x[:,3:5,3:5,0]=4;m=decompose_masks(x,PersistenceMotionConfig());cur=(x[-1]==4);assert m.confident_static[cur].all();assert not m.moving[cur].any()

def test_component_without_history_is_uncertain_not_static():
    x=np.full((3,8,8,2),17,dtype=np.int64);x[-1,3:5,3:5,0]=4;m=decompose_masks(x,PersistenceMotionConfig());cur=(x[-1]==4);assert m.uncertain[cur].all();assert not m.confident_static[cur].any();assert not m.moving[cur].any()


def test_unobserved_holes_are_not_negative_static_evidence():
    x=np.full((4,6,6,1),17,dtype=np.int64)
    x[:,2,2,0]=15
    obs=np.zeros_like(x,dtype=bool)
    obs[0,2,2,0]=True
    obs[2,2,2,0]=True
    obs[3,2,2,0]=True
    cfg=PersistenceMotionConfig(min_static_observations=3,use_component_tracks=False)
    m=decompose_masks(x,cfg,history_observed=obs)
    assert m.confident_static[2,2,0]
    assert not m.moving[2,2,0]
    assert m.persistence[2,2,0]==1.0


def test_one_observation_never_becomes_confident_static():
    x=np.full((4,6,6,1),17,dtype=np.int64)
    x[-1,2,2,0]=4
    obs=np.zeros_like(x,dtype=bool)
    obs[-1,2,2,0]=True
    cfg=PersistenceMotionConfig(min_static_observations=3)
    m=decompose_masks(x,cfg,history_observed=obs)
    assert m.uncertain[2,2,0]
    assert not m.confident_static[2,2,0]
    assert not m.moving[2,2,0]


def test_noneligible_stuff_centroid_shift_cannot_become_moving():
    x=np.full((3,10,10,1),17,dtype=np.int64)
    x[0,3:6,1:4,0]=11
    x[1,3:6,2:5,0]=11
    x[2,3:6,3:6,0]=11
    obs=x!=17
    cfg=PersistenceMotionConfig(min_static_observations=3)
    m=decompose_masks(x,cfg,history_observed=obs)
    cur=x[-1]==11
    assert not m.moving[cur].any()
    # Some overlap may legitimately have enough stationary evidence, while the
    # newly exposed part stays uncertain. The key contract is: stuff jitter can
    # never be promoted to MOVING.
    assert m.uncertain[cur].any()


def test_low_persistence_alone_never_means_moving():
    x=np.full((3,6,6,1),17,dtype=np.int64)
    x[0,2,2,0]=4
    x[1,2,2,0]=7
    x[2,2,2,0]=4
    obs=np.zeros_like(x,dtype=bool)
    obs[:,2,2,0]=True
    cfg=PersistenceMotionConfig(min_static_observations=3,use_component_tracks=False)
    m=decompose_masks(x,cfg,history_observed=obs)
    assert m.persistence[2,2,0] < cfg.static_min_persistence
    assert m.uncertain[2,2,0]
    assert not m.moving[2,2,0]
