import numpy as np

from real_motion.nuscenes_adapter import WindowTokens
from real_motion.prepared import PrepareConfig,load_nuscenes_window_raw


class _Source:
    def load_occ3d(self,scene,token,require_lidar_mask=True):
        value=int(token[1:])
        sem=np.full((2,3,1),value%18,dtype=np.uint8)
        return sem,np.full_like(sem,value%2,dtype=bool)

    def load_semantics(self,scene,token):
        return self.load_occ3d(scene,token)[0]

    def pose(self,token):
        out=np.eye(4,dtype=np.float64); out[0,3]=int(token[1:]); return out

    def official_trajectory(self,*args,**kwargs):
        return np.zeros((12,2),dtype=np.float32)


def test_parallel_raw_window_loading_matches_serial_order():
    window=WindowTokens(
        "scene",tuple(f"t{i}" for i in range(6)),"t5",
        tuple(f"t{i}" for i in range(6,12)))
    serial=load_nuscenes_window_raw(_Source(),window,PrepareConfig(),io_workers=1)
    parallel=load_nuscenes_window_raw(_Source(),window,PrepareConfig(),io_workers=4)
    for key in ("history_occ","history_observed","future_gt_occ","trajectory"):
        assert np.array_equal(serial[key],parallel[key])
    assert all(np.array_equal(a,b) for a,b in zip(serial["history_poses"],parallel["history_poses"]))
    assert all(np.array_equal(a,b) for a,b in zip(serial["future_poses"],parallel["future_poses"]))
