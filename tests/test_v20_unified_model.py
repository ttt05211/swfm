from __future__ import annotations

import torch

from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
from real_motion.local_st_world_model_v18_se2 import LocalSpatialTemporalWorldModelV18SE2
from real_motion.v20_history_world import FREE_LABEL, CanonicalLattice
from real_motion.v20_unified_data import UnifiedHistoryInput, UnifiedSourceInput
from real_motion.v20_unified_model import V20UnifiedTransportCompletion
from real_motion.v20_unified_runtime import prepare_runtime_queries, transport_condition


def _model():
    cfg = LocalSTWMV17Config(
        d_model=16,
        semantic_dim=4,
        heads=4,
        blocks=1,
        decoder_blocks=1,
        tube_hw=8,
        use_representation=True,
    )
    lattice = CanonicalLattice((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (4, 4, 4))
    return V20UnifiedTransportCompletion(
        LocalSpatialTemporalWorldModelV18SE2(cfg), coarse_lattice=lattice
    )


def _history(model):
    semantic = torch.full((1, 6, 4, 4, 4), FREE_LABEL)
    observed = torch.ones_like(semantic, dtype=torch.bool)
    pose = torch.eye(4).view(1, 1, 4, 4).expand(1, 6, 4, 4).clone()
    return model.encode_history(
        UnifiedHistoryInput(semantic, observed, observed.clone(), pose)
    )


def _sources(*, requires_grad=False, empty=False):
    n = 0 if empty else 1
    q = torch.randn(n, 6, 16, requires_grad=requires_grad)
    h = torch.randn(n, 16, requires_grad=requires_grad)
    return UnifiedSourceInput(
        history_source_context=h,
        future_transport_queries=q,
        source_anchor_xyz_t0_m=torch.full((n, 3), 1.5),
        kta_displacement_xy_m=torch.zeros(n, 6, 2),
        window_index=torch.zeros(n, dtype=torch.long),
    )


def test_adapter_bypass_is_exact_and_zero_init_is_identity_when_enabled():
    torch.manual_seed(3)
    model = _model().eval()
    history = _history(model)
    source = _sources()
    bypass = model.fuse_source_scene(history, source, adapter_enabled=False)
    enabled = model.fuse_source_scene(history, source, adapter_enabled=True)
    assert torch.equal(bypass.shared_queries, source.future_transport_queries)
    assert torch.equal(enabled.shared_queries, source.future_transport_queries)
    assert torch.equal(enabled.adapter_delta, torch.zeros_like(enabled.adapter_delta))


def test_unknown_semantic_placeholder_is_invariant():
    model = _model().eval()
    semantic_a = torch.full((1, 6, 4, 4, 4), FREE_LABEL)
    semantic_b = semantic_a.clone()
    observed = torch.ones_like(semantic_a, dtype=torch.bool)
    observed_free = torch.ones_like(observed)
    observed[:, :, 1, 2, 3] = False
    observed_free[:, :, 1, 2, 3] = False
    semantic_a[:, :, 1, 2, 3] = 0
    semantic_b[:, :, 1, 2, 3] = 11
    pose = torch.eye(4).view(1, 1, 4, 4).expand(1, 6, 4, 4).clone()
    a = model.encode_history(
        UnifiedHistoryInput(semantic_a, observed, observed_free, pose)
    )
    b = model.encode_history(
        UnifiedHistoryInput(semantic_b, observed, observed_free, pose)
    )
    assert torch.equal(a.features, b.features)


def test_empty_source_path_still_decodes_completion():
    model = _model().eval()
    history = _history(model)
    source = _sources(empty=True)
    fusion = model.fuse_source_scene(history, source, adapter_enabled=True)
    current = torch.full((1, 6, 4, 4, 4), FREE_LABEL)
    pose = history.future_ego_to_t0
    queries, _ = prepare_runtime_queries(
        current,
        pose,
        coarse_lattice=model.coarse_lattice,
        native_origin_xyz_m=(0.0, 0.0, 0.0),
        native_voxel_size_xyz_m=(1.0, 1.0, 1.0),
        core_shape_xyz=(4, 4, 4),
        halo=2,
    )
    logits, report = model.decode_completion(
        history, source, fusion, transport_condition(current), queries[:1]
    )
    assert len(logits) == 1 and logits[0].shape == (4, 4, 4, 18)
    assert report.requested_points == 0


def test_initial_completion_prefers_free_but_is_not_an_exact_zero_head():
    model = _model().eval()
    assert float(model.completion_head.bias[FREE_LABEL]) == 2.0
    assert float(model.completion_head.weight.abs().sum()) > 0.0
    assert float(model.completion_head.weight.std()) < 0.01


def test_grid_sample_uses_voxel_centers_and_explicit_xyz_order():
    model = _model().eval()
    volume = torch.zeros(1, 1, 4, 4, 4)
    for x in range(4):
        for y in range(4):
            for z in range(4):
                volume[0, 0, x, y, z] = 100 * x + 10 * y + z
    points = torch.tensor([[0.5, 0.5, 0.5], [3.5, 2.5, 1.5]])
    got = model._sample_volume(volume, points, torch.zeros(2, dtype=torch.long))
    assert torch.equal(got[:, 0], torch.tensor([0.0, 321.0]))


def test_out_of_bounds_source_is_counted_and_has_no_scatter_contribution():
    model = _model().eval()
    history = _history(model)
    source = _sources()
    source = UnifiedSourceInput(
        history_source_context=source.history_source_context,
        future_transport_queries=source.future_transport_queries,
        source_anchor_xyz_t0_m=torch.tensor([[20.0, 20.0, 20.0]]),
        kta_displacement_xy_m=source.kta_displacement_xy_m,
        window_index=source.window_index,
    )
    fusion = model.fuse_source_scene(history, source, adapter_enabled=True)
    field, density, report = model._scatter_source_tokens(history, source, fusion)
    assert report.requested_points == 6 and report.in_bounds_points == 0
    assert report.out_of_bounds_points == 6
    assert torch.equal(field, torch.zeros_like(field))
    assert torch.equal(density, torch.zeros_like(density))


def test_source_scatter_position_uses_current_transport_residual():
    model = _model().eval()
    source = _sources()
    transport = {
        "residual_xy_m": torch.zeros(1, 6, 2),
        "existence_logits": torch.zeros(1, 6),
        "yaw_delta_rad": torch.zeros(1, 6),
    }
    transport["residual_xy_m"][0, 0] = torch.tensor([0.75, -0.25])
    pos = model.source_positions_from_transport(source, transport)
    expected = source.source_anchor_xyz_t0_m[0, :2] + torch.tensor(
        [0.75, -0.25]
    )
    assert torch.allclose(pos[0, 0, :2], expected)


def test_completion_gradient_reaches_shared_source_token_and_adapter_last_layer():
    torch.manual_seed(5)
    model = _model().train()
    history = _history(model)
    source = _sources(requires_grad=True)
    fusion = model.fuse_source_scene(history, source, adapter_enabled=True)
    current = torch.full((1, 6, 4, 4, 4), FREE_LABEL)
    queries, _ = prepare_runtime_queries(
        current,
        history.future_ego_to_t0,
        coarse_lattice=model.coarse_lattice,
        native_origin_xyz_m=(0.0, 0.0, 0.0),
        native_voxel_size_xyz_m=(1.0, 1.0, 1.0),
        core_shape_xyz=(4, 4, 4),
        halo=0,
    )
    logits, _ = model.decode_completion(
        history, source, fusion, transport_condition(current), queries[:1]
    )
    loss = logits[0][..., 3].mean()
    loss.backward()
    assert source.future_transport_queries.grad is not None
    assert float(source.future_transport_queries.grad.abs().sum()) > 0.0
    assert model.source_adapter[-1].weight.grad is not None
    assert float(model.source_adapter[-1].weight.grad.abs().sum()) > 0.0


def test_completion_query_chunking_is_numerically_identical():
    torch.manual_seed(12)
    model = _model().eval()
    history = _history(model)
    source = _sources()
    fusion = model.fuse_source_scene(history, source, adapter_enabled=True)
    current = torch.full((1, 6, 4, 4, 4), FREE_LABEL)
    queries, _ = prepare_runtime_queries(
        current,
        history.future_ego_to_t0,
        coarse_lattice=model.coarse_lattice,
        native_origin_xyz_m=(0.0, 0.0, 0.0),
        native_voxel_size_xyz_m=(1.0, 1.0, 1.0),
        core_shape_xyz=(2, 2, 2),
        halo=1,
    )
    with torch.no_grad():
        future, _ = model.build_future_features(
            history, source, fusion, transport_condition(current)
        )
        together = model.decode_completion_from_features(history, future, queries[:4])
        chunked = (
            model.decode_completion_from_features(history, future, queries[:2])
            + model.decode_completion_from_features(history, future, queries[2:4])
        )
    assert all(torch.equal(a, b) for a, b in zip(together, chunked))


def test_transport_loss_updates_adapter_after_zero_initialized_identity():
    torch.manual_seed(9)
    model = _model().train()
    # A real Clean-E14 checkpoint has trained transport heads.  The tiny test
    # fixture is freshly initialized (zero residual head), so emulate that
    # checkpoint property before checking the adapter gradient.
    with torch.no_grad():
        model.v18.residual_head.weight.normal_(std=0.1)
        model.v18.existence_head.weight.normal_(std=0.1)
    history = _history(model)
    source = _sources(requires_grad=True)
    fusion = model.fuse_source_scene(history, source, adapter_enabled=True)
    transport = model.decode_transport(fusion.shared_queries)
    loss = transport["residual_xy_m"].square().mean() + transport["existence_logits"].mean()
    loss.backward()
    grad = model.source_adapter[-1].weight.grad
    assert grad is not None and float(grad.abs().sum()) > 0.0
