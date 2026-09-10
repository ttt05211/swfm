import math

import torch
import torch.nn.functional as F

from real_motion.local_st_world_model_v17 import soft_transport_overlap_loss
from tools.real_motion.diagnose_p0_f9_v17_horizon_gradient_attribution import (
    FUTURE_FRAMES,
    head_gradient_attribution,
    overlap_horizon_term,
    position_horizon_term,
    squared_energy_shares,
)


def test_position_horizon_terms_sum_to_frozen_micro_smooth_l1():
    torch.manual_seed(3)
    pred = torch.randn(5, FUTURE_FRAMES, 2, requires_grad=True)
    target = torch.randn_like(pred)
    valid = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    total_labels = int(valid.sum())
    terms = [
        position_horizon_term(
            pred, target, valid, h, global_valid_labels=total_labels
        )
        for h in range(FUTURE_FRAMES)
    ]
    decomposed = sum(x["objective_contribution"] for x in terms)
    frozen = F.smooth_l1_loss(pred[valid], target[valid], reduction="mean", beta=1.0)
    assert torch.allclose(decomposed, frozen, atol=1e-7, rtol=1e-6)
    assert sum(int(x["labels"]) for x in terms) == total_labels


def test_overlap_horizon_terms_sum_to_frozen_micro_overlap():
    torch.manual_seed(4)
    B = 3
    pred = torch.zeros(B, FUTURE_FRAMES, 2, requires_grad=True)
    target = torch.randn_like(pred) * 0.35
    footprint = torch.zeros(B, 7, 7)
    footprint[0, 2:5, 2:5] = 1.0
    footprint[1, 1:4, 3:6] = 1.0
    # Source 2 intentionally has no footprint; its target-valid labels must not
    # enter the overlap denominator.
    valid = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    present = footprint.flatten(1).sum(1) > 0
    eligible = valid & present[:, None]
    total_labels = int(eligible.sum())

    terms = [
        overlap_horizon_term(
            pred,
            target,
            footprint,
            valid,
            h,
            global_eligible_labels=total_labels,
            patch_resolution_m=0.4,
        )
        for h in range(FUTURE_FRAMES)
    ]
    decomposed = sum(x["objective_contribution"] for x in terms)
    frozen, stats = soft_transport_overlap_loss(
        pred, target, footprint, valid, patch_resolution_m=0.4
    )
    assert int(stats["transport_overlap_labels"]) == total_labels
    assert sum(int(x["labels"]) for x in terms) == total_labels
    assert torch.allclose(decomposed, frozen, atol=2e-7, rtol=2e-6)


def test_head_projection_shares_sum_to_one_and_can_expose_conflict():
    # g0 points partly against the final update while g1 dominates it.
    vectors = [
        torch.tensor([-1.0, 0.0]),
        torch.tensor([3.0, 0.0]),
        torch.tensor([0.0, 1.0]),
    ]
    report = head_gradient_attribution(vectors)
    assert math.isclose(report["projection_share_sum"], 1.0, abs_tol=1e-12)
    assert report["rows"][0]["projection_share"] < 0.0
    assert report["rows"][1]["cosine_to_total"] > 0.0


def test_squared_gradient_energy_shares_are_normalized():
    shares = squared_energy_shares([1.0, 4.0, 0.0, 5.0])
    assert math.isclose(sum(shares), 1.0, abs_tol=1e-12)
    assert shares == [0.1, 0.4, 0.0, 0.5]


def test_position_objective_gradient_decomposes_by_disjoint_horizon_slots():
    torch.manual_seed(5)
    pred = torch.randn(4, FUTURE_FRAMES, 2, requires_grad=True)
    target = torch.randn_like(pred)
    valid = torch.ones(4, FUTURE_FRAMES, dtype=torch.bool)
    total_labels = int(valid.sum())
    grads = []
    for h in range(FUTURE_FRAMES):
        term = position_horizon_term(
            pred, target, valid, h, global_valid_labels=total_labels
        )["objective_contribution"]
        grads.append(torch.autograd.grad(term, pred, retain_graph=True)[0])
    full = F.smooth_l1_loss(pred[valid], target[valid], reduction="mean", beta=1.0)
    full_grad = torch.autograd.grad(full, pred)[0]
    assert torch.allclose(sum(grads), full_grad, atol=1e-7, rtol=1e-6)
    for h, grad in enumerate(grads):
        other = [j for j in range(FUTURE_FRAMES) if j != h]
        assert torch.count_nonzero(grad[:, other]) == 0
