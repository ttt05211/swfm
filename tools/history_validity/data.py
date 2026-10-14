"""Change learned history validity only, never source geometry or GT targets."""
import numpy as np
from tools.shared_history_experiments.refine.history import ResidualHistoryInputs

POLICIES = ('lidar_known', 'dense_valid')


def policy_raw(raw, policy):
    if policy not in POLICIES: raise ValueError('unknown history known policy')
    labels = np.asarray(raw['history_occ'])
    observed = np.asarray(raw['history_observed'])
    if labels.ndim != 4 or len(labels) != 4 or observed.shape != labels.shape or observed.dtype != bool:
        raise ValueError('exact FOUR histories and boolean known required')
    if labels.dtype.kind not in 'iu': raise ValueError('integer historical semantics required')
    if policy == 'lidar_known': return raw
    # Validity is NOT semantic confidence or sensor visibility. Invalid labels
    # remain invalid; grid_sample still handles out-of-grid queries normally.
    known = (labels >= 0) & (labels < 18)
    return dict(raw, history_observed=known)


def history_inputs(policy):
    if policy not in POLICIES: raise ValueError('unknown history known policy')
    class PolicyHistoryInputs(ResidualHistoryInputs):
        def build(self, raw):
            # Called on a bounded process-local cache miss, in the CPU prefetch
            # worker. Source masks/registrations/backgrounds are unchanged;
            # validity-dependent static/dynamic memory coverage is rebuilt.
            return super().build(policy_raw(raw, policy))
    return PolicyHistoryInputs


def prefetch_adapter(base, policy):
    def batches(provider, source, groups, descriptors, **kwargs):
        iterator = base(provider, source, groups, descriptors, **kwargs)
        try:
            for rows in iterator:
                # The descriptor reader above and encoder MUST consume the
                # same convention, on TRAIN and OCC evaluation alike.
                yield [(rec, policy_raw(raw, policy), inputs, history)
                       for rec, raw, inputs, history in rows]
        finally:
            close = getattr(iterator, 'close', None)
            if close: close()
    return batches
