import torch

from tools.real_motion.eval_p0_f8_teacher_endpoint import decision_gate
from tools.real_motion.p0_f8_train_impl_v2 import F8_PROTOCOL
from tools.real_motion.smoke_p0_f8_v2 import (
    REQUIRED_V2_FIELDS,
    validate_smoke_checkpoint,
)
from tools.real_motion.train_p0_f8_teacher_endpoint import (
    ENDPOINT_SOURCE,
    TeacherEndpointEditModel,
    teacher_endpoint_from_batch,
)


def _population_fields():
    return {
        "balanced_false_edit_rate": 0.2,
        "pool_false_edit_rate": 0.1,
        "dynamic_keep_fraction_realized": 0.5,
        "num_lovasz_voxels": 40,
        "num_dynamic_keeps": 5,
        "num_background_keeps": 5,
        "num_pool_keeps": 30,
        "num_pool_dynamic_keeps": 10,
        "num_pool_background_keeps": 20,
    }


def test_smoke_checkpoint_requires_saved_v2_population_contract():
    fields = _population_fields()
    ck = {
        "architecture": {"protocol": F8_PROTOCOL},
        "step": 20,
        "edit_lambda": 0.5,
        "state_dict": {"edit_head.fc2.bias": torch.zeros(10)},
        "training_history": [{
            "step": 20,
            "train": {**fields},
            "val": {**fields, "num_supervised_voxels": 20},
        }],
    }
    report = validate_smoke_checkpoint(ck, expected_steps=20)
    assert report["status"] == "PASS"
    assert report["p0_f8_protocol"] == F8_PROTOCOL
    assert set(REQUIRED_V2_FIELDS).issubset(ck["training_history"][-1]["train"])


def test_teacher_endpoint_is_exact_cached_repair_target():
    repair = torch.randn(2, 6, 16, 50, 50)
    batch = {
        "repair_target_latent": repair,
        "anchor_future_latent": torch.zeros_like(repair),
        "full_history_latent": torch.ones_like(repair),
    }
    endpoint = teacher_endpoint_from_batch(batch, torch.device("cpu"))
    assert endpoint.data_ptr() == repair.data_ptr()
    assert torch.equal(endpoint, repair)


def test_teacher_model_contains_only_edit_head_parameters():
    model = TeacherEndpointEditModel()
    names = [name for name, _ in model.named_parameters()]
    assert names
    assert all(name.startswith("edit_head.") for name in names)
    assert not hasattr(model, "transition")


def _metric_report(moving, moving_1s, overall):
    return {
        "overall": {"mIoU": float(overall)},
        "moving": {
            "mIoU": float(moving),
            "per_horizon": {
                1.0: {"mIoU": float(moving_1s)},
                2.0: {"mIoU": float(moving)},
                3.0: {"mIoU": float(moving)},
            },
        },
    }


def test_teacher_decision_gate_uses_declared_overall_moving_and_1s_limits():
    report = {
        "strong_w2det_anchor": _metric_report(21.0, 35.0, 40.0),
        "teacher_endpoint_ceiling": _metric_report(25.0, 34.7, 40.1),
        "delta_Overall_vs_strong_anchor": 0.1,
        "delta_Moving_vs_strong_anchor": 4.0,
        "endpoint_source": ENDPOINT_SOURCE,
    }
    gate = decision_gate(
        report,
        min_delta_overall=0.0,
        min_delta_moving=3.0,
        min_delta_moving_1s=-0.5,
    )
    assert gate["status"] == "PASS"
    report["teacher_endpoint_ceiling"]["moving"]["per_horizon"][1.0]["mIoU"] = 34.0
    gate = decision_gate(
        report,
        min_delta_overall=0.0,
        min_delta_moving=3.0,
        min_delta_moving_1s=-0.5,
    )
    assert gate["status"] == "FAIL"
    assert not gate["checks"]["delta_Moving_1s"]["pass"]
