from dataclasses import replace
import numpy as np
import pytest
import torch

from real_motion.canonical_causal_repair import CanonicalEvidence, RepairPlan, FEATURE_DIM
from real_motion.ccr_frozen_b import frozen_b_probabilities
from real_motion.surface_canonical_repair import SurfaceCanonicalRepairHead, SURFACE_DIM, PHASE_DIM
from real_motion.surface_ccr_execution import SurfaceExecution


def inputs(device, n=47, sources=2, role='static'):
    rng = np.random.default_rng(42)
    actors = np.full(n, -2, np.int32)
    if role == 'dynamic': actors[:] = 0
    if role == 'mixed': actors[::3] = 0
    classes = np.where(actors >= 0, 4, 11).astype(np.uint8)
    evidence = CanonicalEvidence(rng.normal(size=(n, FEATURE_DIM + SURFACE_DIM)).astype(np.float32),
        np.broadcast_to(classes[:, None], (n, 4)).copy(), actors, classes,
        np.zeros((n, 3)), np.ones((n, 4), bool), {})
    plan = RepairPlan(np.zeros((n, 6), np.int64), rng.integers(0, 18, (n, 6), dtype=np.uint8),
        rng.integers(0, 18, (n, 6), dtype=np.uint8), np.ones((n, 6, 2), bool),
        rng.normal(size=(n, 6, 8 + PHASE_DIM)).astype(np.float32))
    output = {'history_source_context': torch.randn(sources, 8, device=device),
              'future_transport_queries': torch.randn(sources, 6, 8, device=device)}
    return evidence, plan, output


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('role,sources', [('static', 0), ('static', 2), ('mixed', 2), ('dynamic', 2)])
def test_actual_graph_and_changed_live_inputs_preserve_probability_bytes(device, role, sources):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('actual CUDA required')
    device = torch.device(device)
    head = SurfaceCanonicalRepairHead(8, 16).to(device).eval().requires_grad_(False)
    # Nonzero, trained-like weights, not only zero-init identity.
    with torch.no_grad(): head.surface.weight.normal_(); head.phase.weight.normal_()
    evidence, plan, output = inputs(device, sources=sources, role=role)
    session = SurfaceExecution(head, device, graphs=True)
    try:
        for _ in range(3):
            expected = frozen_b_probabilities(head, evidence, plan, output, device)
            actual = session(head, evidence, plan, output, device)
            np.testing.assert_array_equal(actual, expected)
            evidence = replace(evidence, features=evidence.features + .125)
            plan = replace(plan, context=plan.context - .25,
                           base=(plan.base + 1) % 18, fallback=(plan.fallback + 2) % 18,
                           legal=~plan.legal)
            output = {k: v + .3 for k, v in output.items()}
        assert not session.failures
        if device.type == 'cuda' and role == 'static':
            assert session.counts['captures_verified'] == 1
            assert session.counts['graph_replays'] == 3
        else: assert session.counts['eager_chunks'] == 3
    finally: session.close()


def test_empty_population_invalid_budget_and_stale_weights_fail_closed():
    device = torch.device('cpu')
    head = SurfaceCanonicalRepairHead(8, 16).eval().requires_grad_(False)
    with pytest.raises(ValueError, match='chunk'): SurfaceExecution(head, device, chunk=4096)
    with pytest.raises(ValueError, match='bounded'): SurfaceExecution(head, device, max_graphs=5)
    session = SurfaceExecution(head, device)
    empty = inputs(device, n=0)
    assert session(head, *empty, device).shape == (0, 6, 2)
    with torch.no_grad(): head.phase.weight.add_(1)
    with pytest.raises(RuntimeError, match='head changed'): session(head, *empty, device)
    session.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='actual CUDA required')
def test_graph_eviction_bounds_memory_and_recaptures_exactly():
    device = torch.device('cuda')
    head = SurfaceCanonicalRepairHead(8, 16).to(device).eval().requires_grad_(False)
    session = SurfaceExecution(head, device, max_graphs=2)
    for n in [41, 42, 43, 41]:
        args = inputs(device, n=n)
        np.testing.assert_array_equal(session(head, *args, device),
                                      frozen_b_probabilities(head, *args, device))
        assert len(session.cache) <= 2
    assert session.counts['evictions'] == 2 and not session.failures
    session.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='actual CUDA required')
def test_optional_graph_rejection_keeps_full_eager_outputs(monkeypatch):
    device = torch.device('cuda')
    head = SurfaceCanonicalRepairHead(8, 16).to(device).eval().requires_grad_(False)
    session = SurfaceExecution(head, device)
    def reject(*args): raise RuntimeError('test capture not supported')
    monkeypatch.setattr(session, '_capture', reject)
    args = inputs(device)
    expected = frozen_b_probabilities(head, *args, device)
    for _ in range(2): np.testing.assert_array_equal(session(head, *args, device), expected)
    assert session.counts['capture_rejections'] == 1 and session.counts['eager_chunks'] == 2
    session.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='actual CUDA required')
def test_multichunk_trained_geometry_and_changed_source_count_keep_bytes():
    device = torch.device('cuda')
    head = SurfaceCanonicalRepairHead(8, 16).to(device).eval().requires_grad_(False)
    with torch.no_grad(): head.surface.weight.normal_(); head.phase.weight.normal_()
    session = SurfaceExecution(head, device)
    for sources in (2, 5):
        evidence, plan, output = inputs(device, n=8192+73, sources=sources)
        labels = evidence.labels.copy(); classes = evidence.classes.copy()
        classes[1::2] = 13; labels[1::2] = 13
        evidence = replace(evidence, labels=labels, classes=classes)
        np.testing.assert_array_equal(session(head, evidence, plan, output, device),
            frozen_b_probabilities(head, evidence, plan, output, device))
    assert session.counts['captures_verified'] == 2 and not session.failures
    session.close()
