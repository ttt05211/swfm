import numpy as np
import pytest
from real_motion.nuscenes_adapter import NuScenesWindowSource,_first_step_xy_from_temporal_info


def _source_with_rows(tokens):
    src=object.__new__(NuScenesWindowSource)
    src.info_by_token={}
    for i,t in enumerate(tokens):
        # singleton dimensions mimic temporal-info variants; only the first XY
        # pair is the official per-frame one-step ego offset.
        src.info_by_token[t]={'gt_ego_fut_trajs':np.array([[[i+0.25,-i-0.5],[99.,99.]]],dtype=np.float32)}
    return src


def test_first_step_extractor_uses_first_xy_pair():
    rec={'gt_ego_fut_trajs':np.array([[[1.5,-2.0],[7.,8.]]],dtype=np.float32)}
    assert np.allclose(_first_step_xy_from_temporal_info(rec),[1.5,-2.0])


def test_occfm_fut_trajectory_is_12_rows_with_hist_last_mask():
    hist=tuple(f'h{i}' for i in range(6));fut=tuple(f'f{i}' for i in range(6));tokens=hist+fut;src=_source_with_rows(tokens)
    traj=src.official_trajectory(hist,fut,hist_last=4,zero_prefix=2,require_info=True)
    assert traj.shape==(12,2)
    assert np.array_equal(traj[:2],np.zeros((2,2),dtype=np.float32))
    for i in range(2,12):assert np.allclose(traj[i],[i+0.25,-i-0.5])


def test_occfm_fut_trajectory_requires_temporal_info_in_formal_mode():
    hist=tuple(f'h{i}' for i in range(6));fut=tuple(f'f{i}' for i in range(6));src=_source_with_rows(hist+fut);del src.info_by_token['f5']
    with pytest.raises(RuntimeError):src.official_trajectory(hist,fut,require_info=True)
