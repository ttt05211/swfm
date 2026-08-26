from pathlib import Path
from real_motion.runtime_config import load_runtime_config,get_cfg,make_prepare_config

def test_runtime_yaml_is_source_of_truth_and_metric_frozen():
    cfg=load_runtime_config(Path(__file__).resolve().parents[1]/'configs'/'real_motion_occfm.yaml');assert get_cfg(cfg,'MODEL.WINDOW_HW')==[20,20];assert get_cfg(cfg,'METRIC.PROTOCOL')=='interval_displacement_v2';assert get_cfg(cfg,'TARGET.WM_TARGET')=='dynamic_semantics_inside_causal_generation_support';pcfg=make_prepare_config(cfg);assert pcfg.grid.shape_hwd==(200,200,16);assert pcfg.motion.static_speed_mps==0.2;assert pcfg.tube_radii==(1,2,3,4,5,6)
def test_runtime_override_is_explicit():
    cfg=load_runtime_config(Path(__file__).resolve().parents[1]/'configs'/'real_motion_occfm.yaml',['MODEL.MAX_WINDOWS=10']);assert get_cfg(cfg,'MODEL.MAX_WINDOWS')==10
