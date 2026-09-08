import os
import numpy as np
import pytest
import torch

from tests.real_motion.test_motion_transport_v1_renderer import fixture,_targets_for_delta,_run_device_acceptance
from real_motion.motion_transport_v1.compositor import render_soft_ordered,compose_hard,hard_kta_identity
from real_motion.motion_transport_v1.losses import occupancy_ce_full


def _cuda_required():
    return os.environ.get('MT_V1_REQUIRE_CUDA','0') == '1'


def _require_or_skip_cuda():
    if torch.cuda.is_available():
        return
    if _cuda_required():
        pytest.fail('MT_V1_REQUIRE_CUDA=1 but torch.cuda.is_available() is False')
    pytest.skip('CUDA runner not available; use MT_V1_REQUIRE_CUDA=1 on the formal GPU acceptance run')


def _cuda_soft_probability_gradient_contract():
    g,c,d=fixture();td=torch.zeros((1,6,3),device='cuda');td[:,:,0]=.35;td[:,:,1]=-.35;_,targets=_targets_for_delta(g,c,d,td);x=torch.zeros((1,6,3),device='cuda',requires_grad=True)
    def f(v):
        scene=render_soft_ordered(c,d,v,[0],grid=g)
        for row in scene.horizons:
            assert torch.isfinite(row.probabilities).all()
            assert torch.all(row.probabilities>=0) and torch.all(row.probabilities<=1)
            assert torch.allclose(row.probabilities.sum(1),torch.ones(len(row.probabilities),device='cuda'),atol=1e-5)
        loss,_=occupancy_ce_full(scene,targets);return loss
    y=f(x);grad=torch.autograd.grad(y,x)[0];eps=1e-4
    for j in (0,1,2):
        xp=x.detach().clone();xm=x.detach().clone();xp[0,0,j]+=eps;xm[0,0,j]-=eps;fd=float(((f(xp)-f(xm))/(2*eps)).detach().cpu());assert np.isfinite(fd);assert np.isclose(float(grad[0,0,j].detach().cpu()),fd,rtol=.05,atol=5e-3),(j,float(grad[0,0,j].detach().cpu()),fd)


def _cuda_chunk_gradient_contract():
    g,c,d=fixture();a=torch.zeros((1,6,3),device='cuda',requires_grad=True);b=a.detach().clone().requires_grad_(True);a.data[:,:,0]=.11;a.data[:,:,2]=.07;b.data.copy_(a.data);sa=render_soft_ordered(c,d,a,[0],grid=g,query_chunk=7);sb=render_soft_ordered(c,d,b,[0],grid=g,query_chunk=99999)
    for x,y in zip(sa.horizons,sb.horizons):
        assert torch.equal(x.flat_indices,y.flat_indices);assert torch.allclose(x.probabilities,y.probabilities,atol=1e-7,rtol=1e-6)
    la=sum(x.probabilities.square().sum() for x in sa.horizons);lb=sum(x.probabilities.square().sum() for x in sb.horizons);la.backward();lb.backward();assert torch.allclose(a.grad,b.grad,atol=2e-6,rtol=2e-5)


def _cuda_hard_zero_kta_contract():
    g,c,d=fixture();zero=torch.zeros((1,6,3),device='cuda');hard_zero=compose_hard(c,d,zero,[0],grid=g);kta=hard_kta_identity(c,d,grid=g);assert np.array_equal(hard_zero,kta)


def test_pytorch26_cuda_complete_acceptance():
    """Formal GPU gate: same directions/schedule plus soft/hard invariants on CUDA.

    Ordinary CPU CI skips this test explicitly. The formal CUDA command MUST set
    MT_V1_REQUIRE_CUDA=1 so a missing/CPU-only CUDA stack is a failure, not a pass.
    """
    _require_or_skip_cuda()
    if _cuda_required():
        assert torch.__version__.startswith('2.6.0'),torch.__version__
    torch.cuda.empty_cache();_cuda_soft_probability_gradient_contract();_cuda_chunk_gradient_contract();_cuda_hard_zero_kta_contract();_run_device_acceptance('cuda');torch.cuda.synchronize()
