import copy

import pytest
import torch
import torch.nn as nn

from real_motion.models.p0_f8 import AnchorRelativeEditHead
from real_motion.windows import WindowPlan
from tools.real_motion.p0_f8_train_impl_v2 import F8_PROTOCOL
from tools.real_motion.train_p0_f6_decoder_aware_wm import REPAIR_CONTRACT
from tools.real_motion.train_p0_f8_frozen_causal_endpoint_probe import (
    ENDPOINT_SOURCE,
    PROBE_PROTOCOL,
    causal_endpoint_from_prepared,
    load_probe_head_into_model,
    reset_head_and_freeze_transition,
    validate_causal_checkpoint,
    validate_probe_checkpoint,
)


def _causal_checkpoint():
    return {
        "step": 200,
        "state_dict": {"transition.weight": torch.ones(1)},
        "architecture": {
            "protocol": F8_PROTOCOL,
            "repair_endpoint_contract": REPAIR_CONTRACT,
            "sample_steps": 10,
            "source_noise_std": 0.0,
            "keep_bias": 2.0,
        },
    }


def test_causal_checkpoint_requires_deterministic_v2_deployment_contract():
    contract = validate_causal_checkpoint(_causal_checkpoint())
    assert contract["step"] == 200
    assert contract["sample_steps"] == 10

    wrong_protocol = copy.deepcopy(_causal_checkpoint())
    wrong_protocol["architecture"]["protocol"] = "p0_f8_v1"
    with pytest.raises(RuntimeError, match="not P0-F8 v2"):
        validate_causal_checkpoint(wrong_protocol)

    noisy = copy.deepcopy(_causal_checkpoint())
    noisy["architecture"]["source_noise_std"] = 0.1
    with pytest.raises(RuntimeError, match="source_noise_std=0"):
        validate_causal_checkpoint(noisy)


class _TinyProbeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transition = nn.Linear(2, 2)
        self.edit_head = AnchorRelativeEditHead(keep_bias=-3.0)
        self.sample_calls = []

    def sample(
        self,
        history,
        anchor,
        *,
        history_context,
        trajectory,
        window_origins,
    ):
        self.sample_calls.append({
            "history": history,
            "anchor": anchor,
            "history_context": history_context,
            "trajectory": trajectory,
            "window_origins": window_origins,
        })
        return anchor + 1.0


def test_reset_discards_old_head_and_freezes_only_transition():
    model = _TinyProbeModel()
    old_head = model.edit_head
    reset_head_and_freeze_transition(model, keep_bias=2.0)
    assert model.edit_head is not old_head
    assert not any(p.requires_grad for p in model.transition.parameters())
    assert all(p.requires_grad for p in model.edit_head.parameters())
    assert model.transition.training is False
    assert model.edit_head.training is True
    assert float(model.edit_head.fc2.bias[0]) == pytest.approx(2.0)


def test_causal_endpoint_uses_deployment_inputs_and_is_detached():
    model = _TinyProbeModel()
    history = torch.zeros(1, 6, 16, 20, 20, requires_grad=True)
    anchor = torch.zeros(1, 6, 16, 20, 20, requires_grad=True)
    anchor_full = torch.zeros(1, 6, 16, 50, 50, requires_grad=True)
    plan = WindowPlan(
        torch.tensor([[[0, 0]]]),
        torch.tensor([[True]]),
        (20, 20),
        (50, 50),
    )
    prepared = {
        "history": history,
        "anchor": anchor,
        "context": torch.zeros(1, 6, 16, 40, 40),
        "trajectory": torch.zeros(1, 12, 2),
        "origins": torch.tensor([[0, 0]]),
        "plan": plan,
        "effective": torch.tensor([True]),
        "batch_size": 1,
        "topk": 1,
        "anchor_full": anchor_full,
        # A sentinel target is intentionally present but must never be read.
        "target": object(),
    }
    endpoint = causal_endpoint_from_prepared(model, prepared)
    assert len(model.sample_calls) == 1
    assert model.sample_calls[0]["history"] is history
    assert model.sample_calls[0]["anchor"] is anchor
    assert not endpoint.requires_grad
    assert tuple(endpoint.shape) == (1, 6, 16, 50, 50)
    assert torch.all(endpoint[..., :20, :20] == 1.0)
    assert torch.all(endpoint[..., 20:, 20:] == 0.0)


def _probe_checkpoint(model):
    return {
        "step": 400,
        "causal_checkpoint_sha256": "causal-sha",
        "vae_checkpoint_sha256": "vae-sha",
        "edit_head_state_dict": copy.deepcopy(model.edit_head.state_dict()),
        "architecture": {
            "protocol": PROBE_PROTOCOL,
            "endpoint_source": ENDPOINT_SOURCE,
            "transition_trainable": False,
            "source_edit_head_reused": False,
            "source_causal_checkpoint_step": 200,
        },
    }


def test_probe_overlay_is_bound_to_exact_causal_checkpoint_and_vae():
    source = _TinyProbeModel()
    with torch.no_grad():
        source.edit_head.fc2.bias.fill_(0.75)
    probe = _probe_checkpoint(source)
    target = _TinyProbeModel()
    arch = load_probe_head_into_model(
        target,
        probe,
        causal_sha256="causal-sha",
        vae_sha256="vae-sha",
    )
    assert arch["protocol"] == PROBE_PROTOCOL
    assert torch.equal(target.edit_head.fc2.bias, source.edit_head.fc2.bias)

    with pytest.raises(RuntimeError, match="different causal checkpoint"):
        validate_probe_checkpoint(
            probe,
            causal_sha256="wrong",
            vae_sha256="vae-sha",
        )
    with pytest.raises(RuntimeError, match="different VAE"):
        validate_probe_checkpoint(
            probe,
            causal_sha256="causal-sha",
            vae_sha256="wrong",
        )
