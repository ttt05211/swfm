"""Fixed LOCAL development recipe; never fit/select on the server evaluation."""
SOURCE_SHA256='7760c6f4f92277aa248161cd3e90f87ac2354329babceb6e5ed3b872adde33a0'
POSITION_BLEND=.5
VELOCITY_BLEND=.5
FREE_INTERCEPT=-2.
FREE_SLOPE=1.


def specification():
    return dict(name='current_forward_CV_state_plus_range_free_v1',source_checkpoint_sha256=SOURCE_SHA256,
        position_blend=POSITION_BLEND,velocity_blend=VELOCITY_BLEND,
        free_intercept=FREE_INTERCEPT,free_slope=FREE_SLOPE,
        free_formula='free_logit += -2 + min(horizontal_range_in_last_observed_ego_m/40,1)',
        network_parameters=2816122,calibration_scalars=2,network_weights_changed=False,
        calibration='FIRST32 original reference64 scenes GT future labels; two global coefficients; supervised output calibration',
        calibration_selection='fit32 mIoU+.5IoU; GT loss<=2mIoU/3IoU on same fit32; frozen before ALL4369',
        data_use='input-only source-state estimation; no future occupancy in inference; target VAL calibration information IS used',
        same_model_both_modalities=True,training=False,old_results_unchanged=True,
        evidence='local repeatedly inspected DEV64 only; full-VAL 19/29 NOT yet established; no inherited FPS')
