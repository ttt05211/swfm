import torch

from real_motion.context import context_plan_from_prediction, crop_prediction_and_context
from real_motion.windows import WindowPlan


def test_context_plan_contains_prediction_and_shifts_at_boundaries():
    origins = torch.tensor([[[0, 0], [15, 15], [30, 30]]], dtype=torch.long)
    valid = torch.tensor([[True, True, True]])
    plan = WindowPlan(origins, valid, (20, 20), (50, 50))
    ctx = context_plan_from_prediction(plan, context_hw=(40, 40))
    assert ctx.window_hw == (40, 40)
    assert ctx.full_hw == (50, 50)
    assert torch.equal(ctx.origins[0], torch.tensor([[0, 0], [5, 5], [10, 10]]))
    for p, c in zip(plan.origins[0], ctx.origins[0]):
        assert int(c[0]) <= int(p[0])
        assert int(c[1]) <= int(p[1])
        assert int(c[0]) + 40 >= int(p[0]) + 20
        assert int(c[1]) + 40 >= int(p[1]) + 20


def test_crop_prediction_and_context_shapes():
    x = torch.arange(1 * 6 * 2 * 50 * 50, dtype=torch.float32).reshape(1, 6, 2, 50, 50)
    plan = WindowPlan(
        torch.tensor([[[7, 9], [30, 30]]]),
        torch.tensor([[True, True]]),
        (20, 20),
        (50, 50),
    )
    local, context, ctx_plan = crop_prediction_and_context(x, plan, context_hw=(40, 40))
    assert local.shape == (1, 2, 6, 2, 20, 20)
    assert context.shape == (1, 2, 6, 2, 40, 40)
    assert ctx_plan.origins.shape == (1, 2, 2)
