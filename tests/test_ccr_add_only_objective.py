import torch

from tools.real_motion.ccr_screen_common import _add_only_natural_loss


def test_add_only_natural_loss_ignores_remove_channel_and_positive_weight():
    logits=torch.tensor(
        [[[0.2, -7.0],[0.3, 9.0]], [[-0.4, 12.0],[0.1, -13.0]]],
        dtype=torch.float32,requires_grad=True)
    actors=torch.tensor([-2,0],dtype=torch.int64)
    target=torch.tensor(
        [[[1,1],[0,0]], [[0,1],[1,0]]],dtype=torch.bool)
    weight=torch.ones((2,2,2),dtype=torch.float32)

    class Dummy:
        positive_weight=torch.tensor([[32.,32.],[32.,32.]])
    loss=_add_only_natural_loss(Dummy(),logits,actors,target,weight)
    loss.backward()
    # REMOVE is intentionally outside the objective.
    assert torch.equal(logits.grad[...,1],torch.zeros_like(logits.grad[...,1]))
    assert torch.isfinite(loss)
    assert torch.any(logits.grad[...,0]!=0)


def test_add_only_natural_loss_respects_importance_mask():
    logits=torch.zeros((2,2,2),dtype=torch.float32,requires_grad=True)
    actors=torch.tensor([-2,0],dtype=torch.int64)
    target=torch.zeros((2,2,2),dtype=torch.bool)
    target[0,0,0]=True
    weight=torch.zeros((2,2,2),dtype=torch.float32)
    weight[0,0,0]=4.0
    weight[1,1,0]=2.0

    loss=_add_only_natural_loss(None,logits,actors,target,weight)
    # One weighted item per role; equal-role normalization => 2*log(2).
    assert torch.allclose(loss,2*torch.log(torch.tensor(2.0)),rtol=0,atol=1e-6)
