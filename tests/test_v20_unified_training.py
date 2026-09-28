from __future__ import annotations

import json
import random
from unittest.mock import patch

import numpy as np
import pytest
import torch

from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
from real_motion.local_st_world_model_v18_se2 import LocalSpatialTemporalWorldModelV18SE2
from real_motion.v20_history_world import CanonicalLattice
from real_motion.v20_unified_model import V20UnifiedTransportCompletion
from real_motion.v20_unified_training import (
    TrainerProgress,
    build_optimizer,
    checkpoint_payload,
    configure_training_phase,
    load_model_checkpoint,
    restore_training_checkpoint,
    save_checkpoint,
    set_optimizer_phase_lrs,
    verify_resume_inputs,
)
from tools.real_motion.train_p0_f9_v20_unified import (
    _autocast,
    _load_monitor_history,
)


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
    return V20UnifiedTransportCompletion(
        LocalSpatialTemporalWorldModelV18SE2(cfg),
        coarse_lattice=CanonicalLattice((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (4, 4, 4)),
    )


def test_optimizer_groups_exist_once_and_phase_switch_does_not_rebuild():
    model = _model()
    optimizer = build_optimizer(model)
    identity = id(optimizer)
    assert [g["name"] for g in optimizer.param_groups] == ["new", "v18"]
    enabled = configure_training_phase(model, "warmup")
    set_optimizer_phase_lrs(
        optimizer, phase="warmup", new_lr=2e-4, v18_lr=2e-5
    )
    assert not enabled and optimizer.param_groups[1]["lr"] == 0.0
    assert not any(p.requires_grad for p in model.v18.parameters())
    enabled = configure_training_phase(model, "joint")
    set_optimizer_phase_lrs(optimizer, phase="joint", new_lr=2e-4, v18_lr=2e-5)
    assert enabled and optimizer.param_groups[1]["lr"] == 2e-5
    assert all(p.requires_grad for p in model.v18.parameters())
    assert id(optimizer) == identity


def test_training_autocast_uses_bfloat16():
    sentinel = object()
    with patch(
        "tools.real_motion.train_p0_f9_v20_unified.torch.autocast",
        return_value=sentinel,
    ) as autocast:
        assert _autocast(torch.device("cuda"), True) is sentinel
    autocast.assert_called_once_with(device_type="cuda", dtype=torch.bfloat16)


def test_checkpoint_resume_restores_counters_model_optimizer_and_rng(tmp_path):
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    model = _model()
    optimizer = build_optimizer(model)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    base = tmp_path / "base.pt"
    manifest = tmp_path / "index.json"
    torch.save({"base": True}, base)
    manifest.write_text('{"protocol":"test"}', encoding="utf-8")
    progress = TrainerProgress(7, 7, "warmup")
    payload = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler_state={"successful_updates": 7},
        scaler=scaler,
        progress=progress,
        base_checkpoint=base,
        manifest_paths={"train": manifest},
        config={"warmup_updates": 128},
        repository_root=tmp_path,
    )
    path = tmp_path / "resume.pt"
    save_checkpoint(path, payload)
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))

    loaded_model, checkpoint = load_model_checkpoint(path)
    loaded_optimizer = build_optimizer(loaded_model)
    loaded_scaler = torch.cuda.amp.GradScaler(enabled=False)
    got_progress = restore_training_checkpoint(
        checkpoint,
        model=loaded_model,
        optimizer=loaded_optimizer,
        scaler=loaded_scaler,
    )
    actual = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert actual == expected
    assert got_progress == progress
    for a, b in zip(model.parameters(), loaded_model.parameters()):
        assert torch.equal(a, b)


def test_resume_contract_normalizes_json_lattice_and_rejects_drift(tmp_path):
    model = _model()
    optimizer = build_optimizer(model)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    base = tmp_path / "base.pt"
    manifest = tmp_path / "index.json"
    torch.save({"base": True}, base)
    manifest.write_text('{"protocol":"test"}', encoding="utf-8")
    contract = {
        "seed": 7,
        "warmup_updates": 128,
        "screen1024": True,
        "amp_dtype": "bfloat16",
    }
    payload = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler_state={"successful_updates": 8},
        scaler=scaler,
        progress=TrainerProgress(8, 8, "warmup"),
        base_checkpoint=base,
        manifest_paths={"train": manifest},
        config={"warmup_updates": 128},
        repository_root=tmp_path,
        resume_contract=contract,
    )
    json_lattice = {
        "origin_xyz_m": [0.0, 0.0, 0.0],
        "voxel_size_xyz_m": [1.0, 1.0, 1.0],
        "shape_xyz": [4, 4, 4],
    }
    verify_resume_inputs(
        payload,
        base_checkpoint=base,
        manifest_paths={"train": manifest},
        resume_contract=contract,
        coarse_lattice=json_lattice,
    )
    with pytest.raises(RuntimeError, match="run contract differs"):
        verify_resume_inputs(
            payload,
            base_checkpoint=base,
            manifest_paths={"train": manifest},
            resume_contract={**contract, "warmup_updates": 1},
            coarse_lattice=json_lattice,
        )
    with pytest.raises(RuntimeError, match="run contract differs"):
        verify_resume_inputs(
            payload,
            base_checkpoint=base,
            manifest_paths={"train": manifest},
            resume_contract={**contract, "amp_dtype": "float16"},
            coarse_lattice=json_lattice,
        )
    with pytest.raises(RuntimeError, match="coarse lattice mismatch"):
        verify_resume_inputs(
            payload,
            base_checkpoint=base,
            manifest_paths={"train": manifest},
            resume_contract=contract,
            coarse_lattice={**json_lattice, "shape_xyz": [5, 4, 4]},
        )


def test_resume_monitor_history_preserves_rows_and_protects_later_run(tmp_path):
    old_out = tmp_path / "old"
    old_out.mkdir()
    rows = [
        {"successful_updates": 128, "value": "kept"},
        {"successful_updates": 256, "value": "future"},
    ]
    (old_out / "monitor.json").write_text(
        json.dumps(rows), encoding="utf-8"
    )
    checkpoint = {
        "config": {"out_dir": str(old_out)},
        "monitor_history": [],
    }
    fresh_out = tmp_path / "fresh"
    fresh_out.mkdir()
    assert _load_monitor_history(fresh_out, checkpoint, 128) == rows[:1]
    with pytest.raises(RuntimeError, match="newer than the resume checkpoint"):
        _load_monitor_history(old_out, checkpoint, 128)
