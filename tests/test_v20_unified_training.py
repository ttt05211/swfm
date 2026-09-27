from __future__ import annotations

import random

import numpy as np
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
