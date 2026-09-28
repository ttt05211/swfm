from __future__ import annotations

import json
import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
from real_motion.local_st_world_model_v18_se2 import LocalSpatialTemporalWorldModelV18SE2
from real_motion.motion_transport import FEATURE_DIM, FUTURE_FRAMES, HISTORY_FRAMES
from real_motion.v20_history_world import FREE_LABEL, CanonicalLattice
from real_motion.v20_unified_data import UnifiedHistoryInput
from real_motion.v20_unified_model import (
    V20UnifiedConfig,
    V20UnifiedTransportCompletion,
)
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
from tools.real_motion.v20_unified_common import (
    first_stage_forward,
    first_stage_forward_batch,
    training_completion_inputs_batch,
)


def _model(unified_config: V20UnifiedConfig = V20UnifiedConfig()):
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
        config=unified_config,
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


def _prepared_fixture(seed: int, sources: int):
    generator = torch.Generator().manual_seed(seed)
    semantic = torch.randint(
        0, 18, (1, HISTORY_FRAMES, 4, 4, 4), generator=generator
    )
    observed = torch.ones_like(semantic, dtype=torch.bool)
    history = UnifiedHistoryInput(
        semantic=semantic,
        observed=observed,
        observed_free=observed.clone(),
        future_ego_to_t0=torch.eye(4)
        .view(1, 1, 4, 4)
        .expand(1, FUTURE_FRAMES, 4, 4)
        .clone(),
    )
    tube = torch.randint(
        0, 18, (sources, HISTORY_FRAMES, 8, 8), generator=generator
    )
    gpu = {
        "features": torch.randn(sources, FEATURE_DIM, generator=generator),
        "tube": tube,
        "kta": torch.randn(
            sources, FUTURE_FRAMES, 2, generator=generator
        ),
        "frame_motion": torch.randn(
            sources, HISTORY_FRAMES, 5, generator=generator
        ),
        "source_mask": torch.randint(
            0, 2, tube.shape, generator=generator
        ),
    }
    return SimpleNamespace(
        history=history,
        state={"gpu": gpu},
        source_anchor_xyz_t0_m=torch.rand(
            sources, 3, generator=generator
        )
        * 3.0,
        future_semantic=torch.full(
            (1, FUTURE_FRAMES, 4, 4, 4), FREE_LABEL
        ),
        formal_valid=torch.ones(
            (1, FUTURE_FRAMES, 4, 4, 4), dtype=torch.bool
        ),
    )


def test_first_stage_window_batch_matches_independent_forwards():
    torch.manual_seed(23)
    model = _model().eval()
    prepared = [_prepared_fixture(31, 2), _prepared_fixture(32, 3)]
    with torch.no_grad():
        separate = [
            first_stage_forward(model, row, adapter_enabled=True)
            for row in prepared
        ]
        together = first_stage_forward_batch(
            model, prepared, adapter_enabled=True
        )
    assert torch.equal(
        together["sources"].window_index,
        torch.tensor([0, 0, 1, 1, 1]),
    )
    for key in ("residual_xy_m", "existence_logits", "yaw_delta_rad"):
        expected = torch.cat([row["transport"][key] for row in separate])
        assert torch.allclose(
            together["transport"][key], expected, atol=1e-6, rtol=1e-6
        )


def test_completion_window_batch_preserves_per_window_draw_accounting():
    torch.manual_seed(24)
    model = _model().eval()
    prepared = [_prepared_fixture(41, 2), _prepared_fixture(42, 1)]
    with torch.no_grad():
        first = first_stage_forward_batch(
            model, prepared, adapter_enabled=True
        )
        current = torch.full(
            (2, FUTURE_FRAMES, 4, 4, 4), FREE_LABEL
        )
        logits, targets, masks, report = training_completion_inputs_batch(
            model,
            prepared,
            first,
            current,
            native_grid={
                "origin_xyz_m": (0.0, 0.0, 0.0),
                "voxel_size_xyz_m": (1.0, 1.0, 1.0),
            },
            generator=torch.Generator().manual_seed(7),
            collect_distribution_stats=True,
        )
    assert len(logits) == len(targets) == len(masks) == 2 * 6 * 16
    assert report["tiles_by_window"] == [96, 96]
    assert report["unique_tiles_by_window"] == [6, 6]
    assert report["window_indices"].count(0) == 96
    assert report["window_indices"].count(1) == 96
    distribution = report["distribution"]
    assert int(distribution["natural_class_counts"].sum()) == 2 * 6 * 4 * 4 * 4
    assert int(distribution["natural_class_counts"][FREE_LABEL]) == 2 * 6 * 4 * 4 * 4
    assert int(distribution["natural_windows"]) == 2
    assert int(distribution["natural_windows_without_positive"]) == 2


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


def test_checkpoint_roundtrip_preserves_completion_execution_config(tmp_path):
    unified_config = V20UnifiedConfig(
        tile_decode_batch_size=16,
        runtime_query_chunk=256,
        checkpoint_completion_tiles=False,
    )
    model = _model(unified_config)
    optimizer = build_optimizer(model)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    base = tmp_path / "base.pt"
    manifest = tmp_path / "index.json"
    torch.save({"base": True}, base)
    manifest.write_text('{"protocol":"test"}', encoding="utf-8")
    contract = {
        "tile_decode_batch_size": 16,
        "runtime_query_chunk": 256,
        "checkpoint_completion_tiles": False,
    }
    payload = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler_state={"successful_updates": 0},
        scaler=scaler,
        progress=TrainerProgress(),
        base_checkpoint=base,
        manifest_paths={"train": manifest},
        config={},
        repository_root=tmp_path,
        resume_contract=contract,
    )
    path = tmp_path / "fast.pt"
    save_checkpoint(path, payload)

    loaded_model, checkpoint = load_model_checkpoint(path)
    assert loaded_model.config.tile_decode_batch_size == 16
    assert loaded_model.config.runtime_query_chunk == 256
    assert not loaded_model.config.checkpoint_completion_tiles
    verify_resume_inputs(
        checkpoint,
        base_checkpoint=base,
        manifest_paths={"train": manifest},
        resume_contract=contract,
        coarse_lattice=model.coarse_lattice,
    )
    with pytest.raises(RuntimeError, match="run contract differs"):
        verify_resume_inputs(
            checkpoint,
            base_checkpoint=base,
            manifest_paths={"train": manifest},
            resume_contract={**contract, "tile_decode_batch_size": 8},
            coarse_lattice=model.coarse_lattice,
        )

    # V1 checkpoints remain readable for evaluation/diagnostics, while the
    # expanded resume contract prevents continuing them under the V2 loss.
    legacy_payload = dict(payload)
    legacy_payload["protocol"] = (
        "p0_f9_v20_unified_transport_completion_train_v1"
    )
    legacy_path = tmp_path / "legacy.pt"
    save_checkpoint(legacy_path, legacy_payload)
    legacy_model, legacy_checkpoint = load_model_checkpoint(legacy_path)
    assert legacy_checkpoint["protocol"].endswith("train_v1")
    assert legacy_model.config == loaded_model.config


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
