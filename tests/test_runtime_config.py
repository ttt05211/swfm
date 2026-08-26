from pathlib import Path
import pytest
from real_motion.runtime_config import load_runtime_config,get_cfg,make_prepare_config

CFG=Path(__file__).resolve().parents[1]/'configs'/'real_motion_occfm.yaml'

def test_runtime_yaml_is_source_of_truth_and_metric_frozen():
    cfg=load_runtime_config(CFG)
    assert get_cfg(cfg,'MODEL.WINDOW_HW')==[20,20]
    assert get_cfg(cfg,'MODEL.TRAJECTORY_LENGTH')==12
    assert get_cfg(cfg,'UPSTREAM.WM_VARIANT')=='occfm_fut'
    assert get_cfg(cfg,'UPSTREAM.WM_CHECKPOINT_REL').endswith('epoch=000196.ckpt')
    assert get_cfg(cfg,'EGO_PROTOCOL.NAME')=='occfm_fut_12step_v1'
    assert get_cfg(cfg,'EGO_PROTOCOL.HIST_LAST')==4
    assert get_cfg(cfg,'EGO_PROTOCOL.ZERO_PREFIX_STEPS')==2
    assert get_cfg(cfg,'METRIC.PROTOCOL')=='interval_displacement_v2'
    assert get_cfg(cfg,'TARGET.WM_TARGET')=='dynamic_semantics_inside_causal_generation_support'
    pcfg=make_prepare_config(cfg)
    assert pcfg.grid.shape_hwd==(200,200,16)
    assert pcfg.motion.static_speed_mps==0.2
    assert pcfg.tube_radii==(1,2,3,4,5,6)
    assert pcfg.trajectory_length==12 and pcfg.trajectory_zero_prefix==2

def test_runtime_override_is_explicit():
    cfg=load_runtime_config(CFG,['MODEL.MAX_WINDOWS=10']);assert get_cfg(cfg,'MODEL.MAX_WINDOWS')==10

def test_hist_trajectory_contract_is_rejected():
    with pytest.raises(ValueError):load_runtime_config(CFG,['MODEL.TRAJECTORY_LENGTH=6'])
