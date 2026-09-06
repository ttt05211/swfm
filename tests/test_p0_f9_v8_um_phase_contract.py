from __future__ import annotations

import ast
from pathlib import Path

import torch

from real_motion.motion_mask_sidecar import (
    MOTION_MASK_PROTOCOL,
    MOTION_MASK_SIDECAR_VERSION,
    MotionMaskSidecar,
    effective_motion_weight_mass,
    normalized_motion_weighted_mse,
)


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "tools" / "real_motion" / "train_p0_f9_v8_um_phase.py"


def test_lambda_zero_is_uniform_fm_mean():
    g = torch.Generator().manual_seed(7)
    err = torch.rand((3, 6, 16, 20, 20), generator=g)
    mask = torch.rand((3, 6, 20, 20), generator=g) > 0.8
    got = normalized_motion_weighted_mse(err, mask, 0.0)
    expected = err.mean()
    assert torch.allclose(got, expected, atol=1e-7, rtol=1e-7)


def test_lambda_two_weight_mass_matches_closed_form():
    p = 0.06120
    got = effective_motion_weight_mass(p, 2.0)
    expected = 3.0 * p / (1.0 + 2.0 * p)
    assert abs(got - expected) < 1e-12
    assert 0.16 < got < 0.17


def test_motion_sidecar_roundtrip(tmp_path):
    path = tmp_path / "masks.pt"
    rows = []
    for sid in ("a", "b"):
        mask = torch.zeros((6, 50, 50), dtype=torch.bool)
        mask[:, 10:12, 20:23] = True
        rows.append({"sample_id": sid, "scene_name": "scene", "motion_mask_latent": mask})
    torch.save(
        {
            "version": MOTION_MASK_SIDECAR_VERSION,
            "metadata": {
                "motion_mask_protocol": MOTION_MASK_PROTOCOL,
                "source_train_cache_index_sha256": "x",
            },
            "records": rows,
        },
        path,
    )
    sidecar = MotionMaskSidecar(path)
    sidecar.validate_sample_ids(["a", "b"])
    batch = sidecar.get_batch(["b", "a"])
    assert tuple(batch.shape) == (2, 6, 50, 50)
    assert batch.dtype == torch.bool
    assert int(batch.sum()) == 2 * 6 * 2 * 3


def test_experimental_trainer_only_writes_milestone_checkpoint_names():
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
    exact_strings = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "latest.pt" not in exact_strings
    assert "best.pt" not in exact_strings
    assert "last.pt" not in exact_strings
    assert "step_0000.pt" not in exact_strings

    text = TRAINER.read_text(encoding="utf-8")
    assert 'f"step_{phase_step:04d}.pt"' in text
    assert 'model.load_state_dict(parent["ema"]["state_dict"], strict=True)' in text
    assert 'if a.variant == "U":' in text
    assert "total_loss = fm_loss" in text
    assert "total_loss = weighted_preview" in text
    assert '"optimizer_state_dict": optimizer.state_dict()' in text
    assert '"rng_state": _capture_payload_rng()' in text
    assert '"sampling_progress": {' in text
    assert "_restore_rng_state(ck[\"rng_state\"])" in text
