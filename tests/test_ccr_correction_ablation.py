import numpy as np
import torch
from types import SimpleNamespace

from tools.real_motion.diagnose_p0_f9_ccr_correction_ablation import _effective_thresholds


def test_effective_thresholds_match_weighted_logit_identity():
    head=SimpleNamespace(positive_weight=torch.tensor([[3.0,2.0],[4.0,5.0]]))
    rule=_effective_thresholds(head)
    assert abs(rule["B_equivalent_threshold_on_corrected_ADD"]["static"]-.25)<1e-12
    assert abs(rule["B_equivalent_threshold_on_corrected_ADD"]["dynamic"]-.20)<1e-12

    z=np.linspace(-8,8,1001)
    for role,w in (("static",3.0),("dynamic",4.0)):
        corrected=1/(1+np.exp(-(z-np.log(w))))
        raw=1/(1+np.exp(-z))
        threshold=rule["B_equivalent_threshold_on_corrected_ADD"][role]
        assert np.array_equal(raw>=.5,corrected>=threshold)
