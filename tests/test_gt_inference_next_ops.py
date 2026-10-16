import numpy as np
import pytest
import torch
from tools.gt_inference_reuse import engine
from tools.gt_inference_reuse.next_ops import packed_splats
from tools.joint_occ_experiments.batching.geometry import grouped_splat


def test_packed_source_addresses_weights_and_empty_rows_are_exact_on_cpu():
    from real_motion.geometry import OccupancyGrid
    grid=OccupancyGrid(x_min=0,y_min=0,z_min=0,voxel_size=(1,1,1),shape_hwd=(8,8,4))
    torch.manual_seed(21);count=6
    mapped=[torch.rand(count,n,3,dtype=torch.float64)*8-1 for n in (31,0,19)]
    # Identical source footprints MUST remain separate and ordered.
    mapped.append(mapped[0].clone())
    ids,alpha=packed_splats(mapped,grid,count)
    for v,at,w in zip(mapped,ids,alpha):
        slots=torch.arange(count)[:,None].expand(-1,v.shape[1]).flatten()
        expected_at,expected_w=grouped_splat(v.reshape(-1,3),torch.ones(count*v.shape[1]),slots,grid,'sum')
        torch.testing.assert_close(at,expected_at,rtol=0,atol=0)
        torch.testing.assert_close(w,expected_w,rtol=0,atol=0)
    assert packed_splats([],grid,count)==([],[])


def test_complete_six_frame_fresh_forward_and_native_pool_restored():
    from test_shared_source_track import fixture
    from real_motion.strong_majority_execution import ParallelNativeMajority
    p,rec,raw=fixture();torch.set_num_threads(1)
    try:
        torch.nn.init.normal_(p.body.columns.readout[-1].weight,std=.05)
        poses=np.tile(np.eye(4),(6,1,1));poses[:,0,3]=np.arange(6)*.03
        ready=engine.prepared(p,p.prepare(rec,raw,times=np.arange(4)*.5),poses,np.arange(4)*.5)
        original=p.provider.strong_majority
        with ParallelNativeMajority(4) as majority:
            p.gt_reuse_majority=majority
            expected=engine.fresh(p,ready,[.2]*17+[0.],'combined_cpu4')
            actual=engine.fresh(p,ready,[.2]*17+[0.],'packed_splat_cpu4')
            np.testing.assert_array_equal(expected,actual)
            assert majority.stats()['calls']==12 and p.provider.strong_majority is original
    finally:p.close()
