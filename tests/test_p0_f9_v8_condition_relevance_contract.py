from pathlib import Path


SCRIPT = Path("tools/real_motion/diagnose_p0_f9_v8_condition_relevance.py")


def test_condition_relevance_variants_are_single_factor_ablations():
    text = SCRIPT.read_text(encoding="utf-8")
    assert '"full": {"physics_condition": True, "context_condition": True}' in text
    assert '"no_context": {"physics_condition": True, "context_condition": False}' in text
    assert '"no_physics": {"physics_condition": False, "context_condition": True}' in text


def test_condition_relevance_is_no_training_diagnostic():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "optimizer.step" not in text
    assert ".backward(" not in text
    assert "torch.inference_mode" in text
    assert "apply_dynamic_repair" in text
    assert "context_value_full_minus_no_context" in text
    assert "physics_value_full_minus_no_physics" in text
