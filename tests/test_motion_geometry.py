import numpy as np
from real_motion.motion import decompose_masks, PersistenceMotionConfig, FREE
from real_motion.geometry import OccupancyGrid, relative_transform, warp_semantic_grid
from real_motion.kta import causal_kta, KTAConfig


def tiny_grid(): return OccupancyGrid(-2,-2,-1,(1,1,1),(4,4,2))


def test_free_is_never_confident_static():
    x=np.full((3,4,4,2),17,dtype=np.int64)
    x[:,1,1,0]=4
    masks=decompose_masks(x,PersistenceMotionConfig())
    assert masks.confident_static[1,1,0]
    assert not masks.confident_static[0,0,0]
    assert masks.free[0,0,0]


def test_relative_identity_warp():
    g=tiny_grid(); sem=np.full(g.shape_hwd,17,dtype=np.int64); sem[1,2,0]=4
    T=np.eye(4)
    out=warp_semantic_grid(sem,relative_transform(T,T),g,17)
    assert np.array_equal(out,sem)


def test_kta_constant_translation():
    g=tiny_grid(); hist=np.full((2,*g.shape_hwd),17,dtype=np.int64)
    hist[0,1,1,0]=4; hist[1,1,2,0]=4
    cand=np.zeros(g.shape_hwd,dtype=bool); cand[1,2,0]=1
    pred,_,comps=causal_kta(hist,cand,[1.0],g,KTAConfig(history_dt_s=1.0,max_match_distance_m=2.0))
    assert len(comps)==1
    assert pred[0,1,3,0]==4

def test_online_prepare_never_loads_future_semantics():
    from real_motion.prepared import PrepareConfig, prepare_nuscenes_window
    from real_motion.nuscenes_adapter import WindowTokens
    from real_motion.motion import PersistenceMotionConfig
    from real_motion.kta import KTAConfig
    class Source:
        def __init__(self): self.loaded=[]
        def load_semantics(self,scene,token):
            self.loaded.append(token)
            x=np.full((4,4,2),17,dtype=np.int64)
            x[1,1 if token=='h0' else 2,0]=4
            return x
        def pose(self,token): return np.eye(4)
        def official_trajectory(self,t0,future): return np.zeros((2,2),dtype=np.float32)
    src=Source(); w=WindowTokens('s',('h0','h1'),'h1',('f0','f1'))
    cfg=PrepareConfig(history_frames=2,future_frames=2,frame_dt_s=0.5,tube_radii=(0,0),
                      grid=tiny_grid(),motion=PersistenceMotionConfig(min_observed_frames=2),
                      kta=KTAConfig(history_dt_s=0.5,max_match_distance_m=3.0))
    out=prepare_nuscenes_window(src,w,cfg,include_gt=False)
    assert src.loaded==['h0','h1']
    assert 'future_gt_occ' not in out
    assert out['generation_support_occ'].shape==(2,4,4)


def test_component_track_marks_whole_shifted_object_moving():
    """A translating multi-voxel object must not be split into static interior/moving edge."""
    x=np.full((3,8,8,2),17,dtype=np.int64)
    # 2x2 car shifts one cell in +x each 0.5s. Adjacent frames overlap by 50%.
    x[0,3:5,1:3,0]=4
    x[1,3:5,2:4,0]=4
    x[2,3:5,3:5,0]=4
    cfg=PersistenceMotionConfig(history_dt_s=0.5,voxel_size_xy_m=(0.4,0.4),
                                moving_speed_mps=0.5,static_speed_mps=0.2)
    masks=decompose_masks(x,cfg)
    cur=(x[-1]==4)
    assert cur.sum()==4
    assert masks.moving[cur].all()
    assert not masks.confident_static[cur].any()
    assert not masks.uncertain[cur].any()


def test_component_track_marks_persistent_object_static():
    x=np.full((3,8,8,2),17,dtype=np.int64)
    x[:,3:5,3:5,0]=4
    masks=decompose_masks(x,PersistenceMotionConfig())
    cur=(x[-1]==4)
    assert masks.confident_static[cur].all()
    assert not masks.moving[cur].any()


def test_component_without_history_is_uncertain_not_static():
    x=np.full((3,8,8,2),17,dtype=np.int64)
    x[-1,3:5,3:5,0]=4
    masks=decompose_masks(x,PersistenceMotionConfig())
    cur=(x[-1]==4)
    assert masks.uncertain[cur].all()
    assert not masks.confident_static[cur].any()
