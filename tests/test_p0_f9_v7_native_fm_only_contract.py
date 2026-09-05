from __future__ import annotations

import ast
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "tools" / "real_motion" / "train_p0_f9_v7_native_fm_only.py"


def _tree():
    return ast.parse(SOURCE.read_text(encoding="utf-8"))


def _call_name(node: ast.Call) -> str | None:
    fn = node.func
    if isinstance(fn, ast.Name):
        return fn.id
    if isinstance(fn, ast.Attribute):
        return fn.attr
    return None


def test_v7_has_no_semantic_or_vae_decoder_training_calls():
    names = {_call_name(node) for node in ast.walk(_tree()) if isinstance(node, ast.Call)}
    assert "semantic_loss_for_endpoint" not in names
    assert "absolute_future_semantic_loss" not in names
    assert "class_weights_from_edit_cache" not in names
    assert "load_official_vae" not in names
    assert "decode_logits_at_flat_indices" not in names


def test_v7_total_loss_is_exact_native_fm_loss():
    assignments = [node for node in ast.walk(_tree()) if isinstance(node, ast.Assign)]
    found = False
    for node in assignments:
        if not any(isinstance(t, ast.Name) and t.id == "total_loss" for t in node.targets):
            continue
        found = isinstance(node.value, ast.Name) and node.value.id == "fm_loss"
        if found:
            break
    assert found, "v7 must optimize total_loss = fm_loss with no auxiliary term"


def test_v7_flow_loss_does_not_request_one_step_endpoint():
    calls = [
        node
        for node in ast.walk(_tree())
        if isinstance(node, ast.Call) and _call_name(node) == "flow_loss"
    ]
    assert calls
    for call in calls:
        kw = {item.arg: item.value for item in call.keywords if item.arg is not None}
        assert "return_endpoint" in kw
        assert isinstance(kw["return_endpoint"], ast.Constant)
        assert kw["return_endpoint"].value is False
