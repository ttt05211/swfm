from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch

from real_motion.motion_transport_v1 import engine

class _CalNet(torch.nn.Module):
    def __init__(self):
        super().__init__();self.head2=torch.nn.Linear(1,18)
        torch.nn.init.zeros_(self.head2.weight);torch.nn.init.zeros_(self.head2.bias)

class _Pipe:
    def __init__(self):self.source_network=_CalNet()

def test_formal_calibrate_lambda_uses_probe_head_and_output_gradients():
    pipe=_Pipe();ctx=engine.DistContext();cfg={'loss':{'gradient_calibration':{'probe_epsilon':1e-3,'max_ce_antagonistic_fraction_of_motion':.5,'active_motion_grad_rel':1e-4}}}
    rec=SimpleNamespace()
    def forward_losses(pipe,rec,cfg,**kw):
        delta=pipe.source_network.head2(torch.ones((1,1))).reshape(1,6,3);zero=delta.sum()*0
        return zero,zero,0,zero,delta,None,None,{}
    def losses_for_delta(pipe,rec,cfg,delta):
        # Occupancy-like and GT-motion-like objectives deliberately have different optima.
        occ=(delta.square()).mean();mnum=((delta-.2).square()).sum();return occ,mnum,6,None,{}
    with patch.object(engine,'prepare_scene',lambda *a,**kw:rec),patch.object(engine,'forward_losses',forward_losses),patch.object(engine,'_losses_for_delta',losses_for_delta):
        ref,rows=engine.calibrate_lambda(pipe,None,[(None,None)],cfg,ctx,batches=2,seed=3407)
    assert np.isfinite(ref) and ref>0
    assert len(rows)==2
    for row in rows:
        assert row['probe_epsilon']==1e-3
        assert row['motion_pairs_global']==6
        assert np.isfinite(row['ratio']) and row['ratio']>0
        assert row['G_mot_head']>0
        assert row['head_occ_abs']['max']>0
        assert row['head_motion_abs']['max']>0
        assert row['output_occ_abs']['max']>0
        assert row['output_motion_abs']['max']>0
