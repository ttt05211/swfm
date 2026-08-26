import math
from real_motion.metrics.stratified import record_subset_labels,kta_difficulty

def test_subset_labels_do_not_change_metric_contract():
    rec={'delta_speed_mps':1.5,'horizon_s':2.0,'yaw0_world':0.0,'yawh_world':math.radians(30),'kta_center_error_m':2.0};labels=record_subset_labels(rec,speed_change_threshold=1.0,turn_rate_threshold_radps=math.radians(10),kta_cuts=(0.5,1.5));assert 'turning+speed-change' in labels;assert 'kta-hard' in labels
def test_kta_difficulty_frozen_cuts():
    assert kta_difficulty(0.5,(0.5,1.5))=='easy';assert kta_difficulty(1.0,(0.5,1.5))=='medium';assert kta_difficulty(2.0,(0.5,1.5))=='hard'
