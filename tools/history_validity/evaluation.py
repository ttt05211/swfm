"""Same real OCC/STC DEV64 engine; bind validity policy to integer state."""
from tools.perception_robustness.evaluation import evaluator as base_evaluator, ROUTES
from .data import POLICIES

PROTOCOL = 'history_known_occ_stc_gt_dev64_v1'


def evaluator(stc_root, policy, *, inventory=None):
    if policy not in POLICIES: raise ValueError('unknown history known policy')
    base = base_evaluator(stc_root, inventory=inventory)
    def evaluate(*args, saved=None, save=None, **kwargs):
        if saved and (saved.get('known_protocol') != PROTOCOL or saved.get('history_known_policy') != policy):
            raise RuntimeError('history known evaluation policy differs')
        def publish(state):
            if save: save(dict(state, known_protocol=PROTOCOL, history_known_policy=policy))
        result = base(*args, saved=saved, save=publish, **kwargs)
        result.update(history_known_policy=policy, STC_known_policy='dense_valid',
                      metric_visibility_mask=False, learned_prior_gate=False)
        return result
    return evaluate
