from __future__ import annotations

import torch

from real_motion.v20_history_world import FREE_LABEL, CanonicalLattice
from real_motion.v20_unified_data import sample_training_tiles
from real_motion.v20_unified_loss import (
    completion_coupled_focal_semantic,
    completion_cross_entropy,
    completion_tiles_coupled_focal_semantic,
    completion_tiles_cross_entropy,
)
from real_motion.v20_unified_runtime import (
    assemble_completion_logits,
    completion_support,
    compose_completion_tiles,
    dense_geometry_and_transport_condition,
    prepare_runtime_queries,
)
from tools.real_motion.eval_p0_f9_v20_unified import (
    _dynamic_diagnostic_group_counts,
    _finalize,
    _new_raw,
    _update_many,
)


def test_tile_sampler_contract_and_no_support_behavior():
    support = torch.zeros(1, 6, 4, 4, 2, dtype=torch.bool)
    labels = torch.full(support.shape, FREE_LABEL)
    valid = torch.ones_like(support)
    assert sample_training_tiles(support, labels, valid, core_shape_xyz=(2, 2, 2)) == []

    support[:] = True
    labels[:, :, :2, :2, :] = 3
    g = torch.Generator().manual_seed(7)
    tiles = sample_training_tiles(
        support,
        labels,
        valid,
        core_shape_xyz=(2, 2, 2),
        draws_per_horizon=16,
        positive_draws=8,
        generator=g,
    )
    assert len(tiles) == 6 * 16
    for horizon in range(6):
        chosen = [t for t in tiles if t.horizon == horizon]
        assert len(chosen) == 16
        assert sum(t.core_start_xyz[0] == 0 and t.core_start_xyz[1] == 0 for t in chosen) >= 8


def test_compact_dynamic_diagnostics_are_distinct_from_unavailable():
    assert _dynamic_diagnostic_group_counts({}) is None
    assert _dynamic_diagnostic_group_counts(
        {"dynamic_diagnostic_groups": ["BIRTH", "DORMANT_ANCESTRAL"]}
    ) == {"BIRTH": 1, "DORMANT_ANCESTRAL": 1}
    assert _dynamic_diagnostic_group_counts(
        {"dynamic_diagnostic_counts": {"BIRTH": 3}}
    ) == {"BIRTH": 3}


def test_tile_sampler_uses_all_uniform_when_no_positive():
    support = torch.ones(1, 6, 2, 2, 1, dtype=torch.bool)
    labels = torch.full(support.shape, FREE_LABEL)
    valid = torch.ones_like(support)
    tiles = sample_training_tiles(
        support,
        labels,
        valid,
        core_shape_xyz=(1, 1, 1),
        draws_per_horizon=5,
        positive_draws=3,
        generator=torch.Generator().manual_seed(2),
    )
    assert len(tiles) == 6 * 5


def test_completion_empty_mask_is_graph_connected_zero():
    logits = torch.randn(2, 2, 1, 18, requires_grad=True)
    result = completion_cross_entropy(
        logits,
        torch.zeros(2, 2, 1, dtype=torch.long),
        torch.zeros(2, 2, 1, dtype=torch.bool),
    )
    assert result.count == 0 and float(result.mean) == 0.0
    result.mean.backward()
    assert logits.grad is not None and torch.equal(logits.grad, torch.zeros_like(logits))


def test_repeated_tile_draws_keep_exact_ce_multiplicity():
    a = torch.zeros(1, 1, 1, 18, requires_grad=True)
    b = torch.zeros(1, 1, 1, 18, requires_grad=True)
    with torch.no_grad():
        a[..., 1] = 2.0
        b[..., 1] = -2.0
    target = torch.ones(1, 1, 1, dtype=torch.long)
    mask = torch.ones_like(target, dtype=torch.bool)
    got = completion_tiles_cross_entropy([a, a, b], [target] * 3, [mask] * 3)
    expected = (
        torch.nn.functional.cross_entropy(a.reshape(-1, 18), target.reshape(-1)) * 2
        + torch.nn.functional.cross_entropy(b.reshape(-1, 18), target.reshape(-1))
    ) / 3
    assert torch.allclose(got.mean, expected)
    assert got.count == 3


def test_single_tile_and_aggregate_loss_statistics_are_identical():
    torch.manual_seed(4)
    logits = torch.randn(2, 3, 1, 18)
    target = torch.randint(0, 18, (2, 3, 1))
    mask = torch.tensor([[[True], [False], [True]], [[True], [True], [False]]])
    direct = completion_cross_entropy(logits, target, mask)
    aggregate = completion_tiles_cross_entropy([logits], [target], [mask])
    assert direct.count == aggregate.count
    assert torch.equal(direct.loss_sum, aggregate.loss_sum)
    assert torch.equal(direct.mean, aggregate.mean)


def test_deferred_completion_statistics_preserve_loss_and_count():
    torch.manual_seed(5)
    logits = torch.randn(2, 3, 1, 18)
    target = torch.randint(0, 18, (2, 3, 1))
    mask = torch.ones_like(target, dtype=torch.bool)
    regular = completion_tiles_cross_entropy([logits], [target], [mask])
    deferred = completion_tiles_cross_entropy(
        [logits], [target], [mask], materialize_stats=False
    )
    assert torch.equal(deferred.mean, regular.mean)
    assert int(deferred.count) == regular.count


def test_coupled_completion_decomposes_ce_at_gamma_zero_and_original_mass():
    torch.manual_seed(13)
    logits = torch.randn(4, 18, requires_grad=True)
    target = torch.tensor([FREE_LABEL, 2, FREE_LABEL, 5])
    mask = torch.ones(4, dtype=torch.bool)
    # With gamma=0 and semantic weight equal to the occupied mass, the
    # factorization is exactly the original 18-way cross entropy.
    got = completion_coupled_focal_semantic(
        logits,
        target,
        mask,
        presence_focal_gamma=0.0,
        semantic_weight=0.5,
    )
    expected = torch.nn.functional.cross_entropy(logits, target)
    assert torch.allclose(got.mean, expected, atol=1.0e-6, rtol=1.0e-6)
    assert got.count == 4
    assert got.free_count == 2
    assert got.occupied_count == 2
    assert got.class_counts[FREE_LABEL] == 2
    assert got.class_counts[2] == 1
    assert got.class_counts[5] == 1
    got.mean.backward()
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad[:, FREE_LABEL].abs().sum()) > 0.0
    assert float(logits.grad[:, :FREE_LABEL].abs().sum()) > 0.0


def test_coupled_completion_separately_normalizes_positive_semantics():
    logits = torch.zeros(3, 18, requires_grad=True)
    target = torch.tensor([FREE_LABEL, FREE_LABEL, 4])
    mask = torch.ones(3, dtype=torch.bool)
    got = completion_coupled_focal_semantic(
        logits,
        target,
        mask,
        presence_focal_gamma=2.0,
        semantic_weight=1.0,
    )
    presence_logit = torch.log(torch.tensor(17.0))
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        presence_logit.expand(3),
        torch.tensor([0.0, 0.0, 1.0]),
        reduction="none",
    )
    expected_presence = ((1.0 - torch.exp(-bce)).square() * bce).mean()
    expected_semantic = torch.log(torch.tensor(17.0))
    assert torch.allclose(got.presence_mean, expected_presence)
    assert torch.allclose(got.semantic_mean, expected_semantic)
    assert torch.allclose(got.mean, expected_presence + expected_semantic)


def test_coupled_tile_loss_preserves_draw_multiplicity_and_class_counts():
    torch.manual_seed(14)
    occupied_logits = torch.randn(1, 2, 1, 18, requires_grad=True)
    free_logits = torch.randn(1, 2, 1, 18, requires_grad=True)
    occupied_target = torch.tensor([[[3], [7]]])
    free_target = torch.full((1, 2, 1), FREE_LABEL)
    mask = torch.ones_like(occupied_target, dtype=torch.bool)
    aggregate = completion_tiles_coupled_focal_semantic(
        [occupied_logits, occupied_logits, free_logits],
        [occupied_target, occupied_target, free_target],
        [mask, mask, mask],
    )
    direct = completion_coupled_focal_semantic(
        torch.cat([occupied_logits, occupied_logits, free_logits], dim=0),
        torch.cat([occupied_target, occupied_target, free_target], dim=0),
        torch.cat([mask, mask, mask], dim=0),
    )
    assert torch.allclose(aggregate.mean, direct.mean)
    assert torch.allclose(aggregate.presence_mean, direct.presence_mean)
    assert torch.allclose(aggregate.semantic_mean, direct.semantic_mean)
    assert aggregate.count == direct.count == 6
    assert aggregate.occupied_count == direct.occupied_count == 4
    assert aggregate.free_count == direct.free_count == 2
    assert aggregate.class_counts == direct.class_counts


def test_coupled_completion_empty_mask_is_graph_connected_zero():
    logits = torch.randn(2, 18, requires_grad=True)
    result = completion_coupled_focal_semantic(
        logits,
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.bool),
    )
    assert result.count == 0
    assert result.occupied_count == 0
    assert float(result.mean) == 0.0
    result.mean.backward()
    assert logits.grad is not None
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_coupled_completion_deferred_statistics_match_materialized():
    torch.manual_seed(15)
    logits = torch.randn(2, 3, 1, 18)
    target = torch.tensor([[[FREE_LABEL], [2], [4]], [[7], [FREE_LABEL], [9]]])
    mask = torch.tensor(
        [[[True], [True], [False]], [[True], [True], [True]]]
    )
    regular = completion_tiles_coupled_focal_semantic(
        [logits], [target], [mask]
    )
    deferred = completion_tiles_coupled_focal_semantic(
        [logits], [target], [mask], materialize_stats=False
    )
    assert torch.equal(deferred.mean, regular.mean)
    assert torch.equal(deferred.presence_mean, regular.presence_mean)
    assert torch.equal(deferred.semantic_mean, regular.semantic_mean)
    assert int(deferred.count) == regular.count
    assert int(deferred.free_count) == regular.free_count
    assert int(deferred.occupied_count) == regular.occupied_count
    assert tuple(int(value) for value in deferred.class_counts) == regular.class_counts


def test_transport_condition_is_spatial_proportions_plus_coverage():
    lattice = CanonicalLattice(
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2, 1, 1)
    )
    current = torch.full((1, 6, 2, 1, 1), FREE_LABEL)
    current[:, :, 0] = 4
    pose = torch.eye(4).view(1, 1, 4, 4).expand(1, 6, 4, 4).clone()
    valid, cond = dense_geometry_and_transport_condition(
        current,
        pose,
        coarse_lattice=lattice,
        native_origin_xyz_m=(0.0, 0.0, 0.0),
        native_voxel_size_xyz_m=(1.0, 1.0, 1.0),
        chunk_shape_xyz=(2, 1, 1),
    )
    assert bool(valid.all())
    assert cond.shape == (1, 6, 19, 2, 1, 1)
    assert torch.allclose(
        cond[:, :, 4, 0], torch.ones_like(cond[:, :, 4, 0])
    )
    assert torch.allclose(
        cond[:, :, FREE_LABEL, 1],
        torch.ones_like(cond[:, :, FREE_LABEL, 1]),
    )
    assert torch.allclose(cond[:, :, -1], torch.ones_like(cond[:, :, -1]))
    assert not cond.requires_grad


def test_halo_queries_write_core_only():
    lattice = CanonicalLattice((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (4, 4, 2))
    current = torch.full((1, 6, 4, 4, 2), FREE_LABEL)
    pose = torch.eye(4).view(1, 1, 4, 4).expand(1, 6, 4, 4).clone()
    queries, report = prepare_runtime_queries(
        current,
        pose,
        coarse_lattice=lattice,
        native_origin_xyz_m=(0.0, 0.0, 0.0),
        native_voxel_size_xyz_m=(1.0, 1.0, 1.0),
        core_shape_xyz=(2, 2, 2),
        halo=1,
    )
    assert report.out_of_bounds_voxels == 0
    logits = []
    for i, q in enumerate(queries):
        x = torch.zeros(*q.tile.halo_shape_xyz, 18)
        x[..., i % 17] = 5.0
        logits.append(x)
    dense = assemble_completion_logits(logits, queries, output_shape=current.shape)
    assert dense.shape == (*current.shape, 18)
    assert torch.isfinite(dense).all()


def test_free_logit_offset_changes_only_diagnostic_argmax_threshold():
    lattice = CanonicalLattice((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2, 2, 1))
    current = torch.full((1, 6, 2, 2, 1), FREE_LABEL)
    pose = torch.eye(4).view(1, 1, 4, 4).expand(1, 6, 4, 4).clone()
    queries, _ = prepare_runtime_queries(
        current,
        pose,
        coarse_lattice=lattice,
        native_origin_xyz_m=(0.0, 0.0, 0.0),
        native_voxel_size_xyz_m=(1.0, 1.0, 1.0),
        core_shape_xyz=(2, 2, 1),
        halo=0,
    )
    logits = []
    for query in queries:
        value = torch.zeros(*query.tile.halo_shape_xyz, 18)
        value[..., 4] = 1.0
        value[..., FREE_LABEL] = 2.0
        logits.append(value)

    unchanged = compose_completion_tiles(current, logits, queries)
    below_threshold = compose_completion_tiles(
        current, logits, queries, free_logit_offset=0.5
    )
    above_threshold = compose_completion_tiles(
        current, logits, queries, free_logit_offset=1.0
    )

    assert torch.equal(unchanged, current)
    assert torch.equal(below_threshold, current)
    assert torch.all(above_threshold == 4)


def test_runtime_queries_reuse_dense_masks_without_changing_contract():
    lattice = CanonicalLattice((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (4, 4, 2))
    current = torch.full((1, 6, 4, 4, 2), FREE_LABEL)
    current[:, :, 0, 0, 0] = 3
    pose = torch.eye(4).view(1, 1, 4, 4).expand(1, 6, 4, 4).clone()
    valid, _ = dense_geometry_and_transport_condition(
        current,
        pose,
        coarse_lattice=lattice,
        native_origin_xyz_m=(0.0, 0.0, 0.0),
        native_voxel_size_xyz_m=(1.0, 1.0, 1.0),
        chunk_shape_xyz=(4, 4, 2),
    )
    support = completion_support(current, valid)
    kwargs = dict(
        coarse_lattice=lattice,
        native_origin_xyz_m=(0.0, 0.0, 0.0),
        native_voxel_size_xyz_m=(1.0, 1.0, 1.0),
        core_shape_xyz=(2, 2, 2),
        halo=1,
    )
    plain, plain_report = prepare_runtime_queries(current, pose, **kwargs)
    cached, cached_report = prepare_runtime_queries(
        current,
        pose,
        dense_geometry_valid=valid,
        dense_completion_support=support,
        **kwargs,
    )
    assert cached_report == plain_report
    assert len(cached) == len(plain)
    for left, right in zip(cached, plain):
        assert left.tile == right.tile
        assert torch.equal(left.points_xyz_t0_m, right.points_xyz_t0_m)
        assert torch.equal(left.geometry_valid, right.geometry_valid)
        assert torch.equal(left.support, right.support)

    repeated, repeated_report = prepare_runtime_queries(
        current,
        pose,
        tiles=[plain[0].tile, plain[0].tile],
        dense_geometry_valid=valid,
        dense_completion_support=support,
        **kwargs,
    )
    assert repeated_report.requested_voxels == 2 * plain[0].support.numel()
    assert repeated[0].points_xyz_t0_m.data_ptr() == repeated[1].points_xyz_t0_m.data_ptr()


def test_evaluator_uses_exact_dataset_accumulated_metric_semantics():
    names = {"perfect": _new_raw(), "free": _new_raw()}
    gt = torch.full((2, 2, 1), FREE_LABEL, dtype=torch.uint8).numpy()
    gt[0, 0, 0] = 4
    perfect = gt.copy()
    free = torch.full((2, 2, 1), FREE_LABEL, dtype=torch.uint8).numpy()
    moving = torch.zeros(2, 2, 1, dtype=torch.bool).numpy()
    moving[0, 0, 0] = True
    for horizon in range(6):
        _update_many(
            names,
            horizon,
            {"perfect": perfect, "free": free},
            gt,
            moving,
            FREE_LABEL,
        )
    perfect_result = _finalize(names["perfect"])
    free_result = _finalize(names["free"])
    assert perfect_result["IoU"] == 100.0
    assert perfect_result["MovingMicro"] == 100.0
    assert free_result["IoU"] == 0.0
    assert free_result["MovingMicro"] == 0.0
