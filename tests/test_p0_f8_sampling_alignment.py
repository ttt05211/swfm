import math

import torch

from real_motion.edit_repair import CLEAR, KEEP, WRITE_OFFSET
from real_motion.edit_repair_v2 import (
    full_edit_supervision,
    select_stratified_balanced_edit_supervision,
    split_population_edit_loss,
)
from tools.real_motion.p0_f8_train_impl_v2 import (
    _merge_latest_train_v2_fields,
    aggregate_edit_validation_infos,
    edit_loss_for_endpoint_v2,
)


def _record(*, num_dynamic_keeps=4, num_background_keeps=4):
    ne = 4
    edit_idx = torch.arange(ne, dtype=torch.int32)
    edit_actions = torch.tensor(
        [CLEAR, WRITE_OFFSET, WRITE_OFFSET + 1, WRITE_OFFSET], dtype=torch.uint8
    )
    edit_anchor = torch.tensor([1, 0, 1, 0], dtype=torch.uint8)
    edit_result = torch.tensor([0, 1, 2, 1], dtype=torch.uint8)
    edit_moving = torch.tensor([True, True, False, False])

    dyn_idx = torch.arange(100, 100 + num_dynamic_keeps, dtype=torch.int32)
    bg_idx = torch.arange(200, 200 + num_background_keeps, dtype=torch.int32)
    keep_idx = torch.cat([dyn_idx, bg_idx], dim=0)
    dyn_anchor = torch.tensor(
        [1 + (i % 2) for i in range(num_dynamic_keeps)], dtype=torch.uint8
    )
    bg_anchor = torch.zeros(num_background_keeps, dtype=torch.uint8)
    keep_anchor = torch.cat([dyn_anchor, bg_anchor], dim=0)
    # Dynamic KEEP has hard priority, background is priority 0 in the v1 sidecar.
    dyn_priority = torch.tensor(
        [2 if i % 2 == 0 else 1 for i in range(num_dynamic_keeps)], dtype=torch.uint8
    )
    bg_priority = torch.zeros(num_background_keeps, dtype=torch.uint8)

    return {
        "sample_id": "scene:sample",
        "scene_name": "scene",
        "edit_flat_indices": edit_idx,
        "edit_actions": edit_actions,
        "edit_anchor_slots": edit_anchor,
        "edit_result_slots": edit_result,
        "edit_moving": edit_moving,
        "keep_flat_indices": keep_idx,
        "keep_anchor_slots": keep_anchor,
        "keep_priority": torch.cat([dyn_priority, bg_priority], dim=0),
    }


def test_stratified_sampler_forces_dynamic_and_background_keep():
    rec = _record(num_dynamic_keeps=4, num_background_keeps=4)
    sel = select_stratified_balanced_edit_supervision(
        rec,
        keep_ratio=1.0,
        dynamic_keep_fraction=0.5,
        deterministic=True,
    )
    assert sel["num_edits"] == 4
    assert sel["num_keeps"] == 4
    assert sel["num_dynamic_keeps"] == 2
    assert sel["num_background_keeps"] == 2
    assert sel["actions"][:4].ne(KEEP).all()
    assert sel["actions"][4:].eq(KEEP).all()


def test_stratified_sampler_fills_shortfall_from_other_stratum():
    rec = _record(num_dynamic_keeps=6, num_background_keeps=1)
    sel = select_stratified_balanced_edit_supervision(
        rec,
        keep_ratio=1.0,
        dynamic_keep_fraction=0.5,
        deterministic=True,
    )
    assert sel["num_keeps"] == 4
    assert sel["num_background_keeps"] == 1
    assert sel["num_dynamic_keeps"] == 3
    assert torch.unique(sel["flat_indices"]).numel() == sel["flat_indices"].numel()


def test_balanced_positions_are_exact_subset_of_full_pool():
    rec = _record(num_dynamic_keeps=4, num_background_keeps=4)
    pool = full_edit_supervision(rec)
    sel = select_stratified_balanced_edit_supervision(
        rec,
        keep_ratio=1.0,
        dynamic_keep_fraction=0.5,
        deterministic=True,
    )
    selected_from_pool = pool["flat_indices"].index_select(0, sel["pool_positions"])
    assert torch.equal(selected_from_pool, sel["flat_indices"])


def test_full_pool_lovasz_penalizes_unsampled_background_false_write():
    rec = _record(num_dynamic_keeps=4, num_background_keeps=4)
    pool = full_edit_supervision(rec)
    sel = select_stratified_balanced_edit_supervision(
        rec,
        keep_ratio=1.0,
        dynamic_keep_fraction=0.5,
        deterministic=True,
    )

    n = int(pool["actions"].numel())
    num_actions = 10
    good = torch.full((n, num_actions), -6.0)
    good[torch.arange(n), pool["actions"].long()] = 6.0

    # Keep the balanced CE subset identical, but make one KEEP outside that
    # subset predict WRITE. Balanced CE cannot see this error; full-pool Lovasz
    # and the pool false-edit metric must see it.
    selected = set(sel["pool_positions"].tolist())
    unsampled_keep = [
        i for i in range(int(pool["num_edits"]), n) if i not in selected
    ]
    assert unsampled_keep
    bad = good.clone()
    bad[unsampled_keep[0]] = -6.0
    bad[unsampled_keep[0], WRITE_OFFSET] = 6.0

    good_loss, good_info = split_population_edit_loss(
        good,
        pool["anchor_slots"],
        pool["result_slots"],
        pool["actions"],
        sel["pool_positions"],
        sel["actions"],
        lovasz_weight=0.5,
    )
    bad_loss, bad_info = split_population_edit_loss(
        bad,
        pool["anchor_slots"],
        pool["result_slots"],
        pool["actions"],
        sel["pool_positions"],
        sel["actions"],
        lovasz_weight=0.5,
    )

    assert abs(good_info["ce"] - bad_info["ce"]) < 1e-7
    assert bad_info["lovasz"] > good_info["lovasz"]
    assert bad_loss > good_loss
    assert good_info["pool_false_edit_rate"] == 0.0
    assert bad_info["pool_false_edit_rate"] > 0.0
    assert bad_info["balanced_false_edit_rate"] == 0.0
    assert good_info["num_pool_predicted_edits"] == 4
    assert bad_info["num_pool_predicted_edits"] == 5
    assert math.isclose(good_info["pool_predicted_edit_fraction"], 4 / 12)
    assert math.isclose(bad_info["pool_predicted_edit_fraction"], 5 / 12)


class _FakeEditCache:
    def __init__(self, record):
        self.record = record

    def get_batch(self, sample_ids):
        assert list(sample_ids) == ["scene:sample"]
        return [self.record]


class _FakeVAE:
    def decode_logits_at_flat_indices(self, endpoint, indices_per_sample):
        scalar = endpoint.mean()
        out = []
        for idx in indices_per_sample:
            n = int(idx.numel())
            # Create differentiable semantic evidence whose class ordering varies
            # with the endpoint value, so both CE and Lovasz have a gradient path.
            basis = torch.arange(18, device=endpoint.device, dtype=endpoint.dtype)
            out.append(scalar * basis.unsqueeze(0).expand(n, -1))
        return out


class _FakeModel:
    def edit_head(self, semantic_logits, anchor_slots, horizons):
        del anchor_slots, horizons
        # Preserve a differentiable path from semantic evidence to ten actions.
        return semantic_logits[:, :10]


def test_v2_edit_loss_wires_full_pool_decoder_once_and_balanced_ce_subset():
    rec = _record(num_dynamic_keeps=4, num_background_keeps=4)
    endpoint = torch.ones((1, 6, 1, 1, 1), requires_grad=True)
    loss, info = edit_loss_for_endpoint_v2(
        _FakeModel(),
        endpoint,
        sample_ids=["scene:sample"],
        edit_cache=_FakeEditCache(rec),
        vae=_FakeVAE(),
        keep_ratio=1.0,
        keep_when_no_edit=64,
        lovasz_weight=0.5,
        deterministic=True,
        selection_seed=7,
    )
    assert torch.isfinite(loss)
    assert info["num_edits"] == 4
    assert info["num_keeps"] == 4
    assert info["num_dynamic_keeps"] == 2
    assert info["num_background_keeps"] == 2
    assert info["num_pool_keeps"] == 8
    assert info["num_pool_dynamic_keeps"] == 4
    assert info["num_pool_background_keeps"] == 4
    assert info["num_supervised_voxels"] == 8
    assert info["num_lovasz_voxels"] == 12
    loss.backward()
    assert endpoint.grad is not None
    assert torch.isfinite(endpoint.grad).all()


def test_validation_aggregation_uses_each_population_size():
    infos = [
        {
            "num_supervised_voxels": 10,
            "num_lovasz_voxels": 100,
            "num_edits": 5,
            "num_keeps": 5,
            "num_dynamic_keeps": 3,
            "num_background_keeps": 2,
            "num_pool_keeps": 80,
            "num_pool_dynamic_keeps": 20,
            "num_pool_background_keeps": 60,
            "num_moving_edits": 2,
            "ce": 1.0,
            "lovasz": 0.2,
            "accuracy": 0.8,
            "edit_accuracy": 0.6,
            "balanced_false_edit_rate": 0.4,
            "pool_false_edit_rate": 0.1,
            "false_edit_rate": 0.1,
            "num_ce_predicted_edits": 4,
            "num_pool_predicted_edits": 10,
        },
        {
            "num_supervised_voxels": 90,
            "num_lovasz_voxels": 10,
            "num_edits": 45,
            "num_keeps": 45,
            "num_dynamic_keeps": 20,
            "num_background_keeps": 25,
            "num_pool_keeps": 5,
            "num_pool_dynamic_keeps": 2,
            "num_pool_background_keeps": 3,
            "num_moving_edits": 10,
            "ce": 3.0,
            "lovasz": 0.8,
            "accuracy": 0.4,
            "edit_accuracy": 0.2,
            "balanced_false_edit_rate": 0.2,
            "pool_false_edit_rate": 0.9,
            "false_edit_rate": 0.9,
            "num_ce_predicted_edits": 9,
            "num_pool_predicted_edits": 2,
        },
    ]
    out = aggregate_edit_validation_infos(infos, lovasz_weight=0.5)

    expected_ce = (1.0 * 10 + 3.0 * 90) / 100
    expected_lovasz = (0.2 * 100 + 0.8 * 10) / 110
    expected_pool_false = (0.1 * 80 + 0.9 * 5) / 85
    expected_balanced_false = (0.4 * 5 + 0.2 * 45) / 50

    assert math.isclose(out["edit_ce"], expected_ce, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(
        out["result_lovasz"], expected_lovasz, rel_tol=0, abs_tol=1e-12
    )
    assert math.isclose(
        out["edit_loss"], expected_ce + 0.5 * expected_lovasz,
        rel_tol=0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        out["pool_false_edit_rate"], expected_pool_false,
        rel_tol=0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        out["balanced_false_edit_rate"], expected_balanced_false,
        rel_tol=0,
        abs_tol=1e-12,
    )
    assert out["num_supervised_voxels"] == 100
    assert out["num_lovasz_voxels"] == 110
    assert out["num_pool_keeps"] == 85
    assert out["num_ce_predicted_edits"] == 13
    assert math.isclose(out["ce_predicted_edit_fraction"], 13 / 100)
    assert out["num_pool_predicted_edits"] == 12
    assert math.isclose(out["pool_predicted_edit_fraction"], 12 / 110)


def test_v2_fields_are_persisted_into_latest_training_record():
    history = [{"step": 200, "train": {"objective": 1.0}, "val": {}}]
    fields = {
        "balanced_false_edit_rate": 0.2,
        "pool_false_edit_rate": 0.1,
        "dynamic_keep_fraction_realized": 0.5,
        "num_lovasz_voxels": 123,
        "num_dynamic_keeps": 10,
        "num_background_keeps": 10,
    }
    _merge_latest_train_v2_fields(history, fields)
    for key, value in fields.items():
        assert history[-1]["train"][key] == value
