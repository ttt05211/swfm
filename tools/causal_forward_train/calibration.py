"""Fixed inherited FREE -1, not a fit/search or a training-logit modification."""
from tools.fpn_stc_bias448.run import predict as fixed_predict,FREE_BIAS,NAMES as OLD_NAMES

SUFFIX = '_free_minus1'
CALIBRATION = dict(type='fixed_inherited_labeled_output_calibration',free_class=17,free_logit_bias=FREE_BIAS,
    coefficient_origin='previous FPN/STC first32 GT-label selection; NOT refitted on AdaLN',
    residual_gain=1.,shared_OCC_STC=True,all_six_horizons=True,training_loss_changed=False,
    age_decay_added=False,automatic_selection=False)


def predict(logits,shape):
    # Reuse the previously validated FP32/tie-correct fixed -1 implementation.
    values = fixed_predict(logits,shape)
    return dict(semantic_refine=values[OLD_NAMES[0]],semantic_refine_free_minus1=values[OLD_NAMES[1]])


def delta(new,old):
    names = ('mIoU','IoU','MovingMicro')
    return {k:new[k]-old[k] for k in names if new.get(k) is not None and old.get(k) is not None}
