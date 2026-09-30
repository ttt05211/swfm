"""Paired end-to-end training contracts; synthetic geometry, not nuScenes gains."""
from __future__ import annotations
import copy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch

from real_motion.motion_transport import FEATURE_DIM
from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
from real_motion.local_st_world_model_v18_se2 import LocalSpatialTemporalWorldModelV18SE2
from real_motion.v18_source_interaction import (ARMS, InteractionConfig, SourceInteractionV18,
                                               neighbor_graph, causal_forward)
from tools.real_motion import v18_source_interaction_common as common
from tools.real_motion import train_p0_f9_v18_source_interaction as trainer


CFG = LocalSTWMV17Config(d_model=16, semantic_dim=8, heads=4, blocks=1, decoder_blocks=1)


def record(n=3, scene="train", token="t0"):
    generator = torch.Generator().manual_seed(37)
    kta = torch.zeros(n, 6, 2)
    target = torch.ones(n, 6, 2)*.25
    center = torch.stack((torch.arange(n).float()+1.5, torch.ones(n)*1.5), dim=1)
    motion = torch.randn(n, 6, 5, generator=generator)*.1; motion[..., 4] = 1
    return dict(scene_name=scene, t0_token=token, future_tokens=tuple(f"future{i}" for i in range(6)),
        features=torch.randn(n, FEATURE_DIM, generator=generator),
        local_semantic_tube=torch.randint(0, 18, (n, 6, 20, 20), generator=generator, dtype=torch.uint8),
        frame_motion_features=motion, kta_displacement_xy_m=kta,
        source_centroid_xy_t0_m=center, anchors_xy_t0_m=center[:, None]+kta,
        source_class_id=torch.full((n,), 4),
        target_source_mask_tube=torch.ones(n, 6, 20, 20, dtype=torch.uint8),
        supervised_source=torch.ones(n, dtype=torch.bool), existence=torch.ones(n, 6),
        se2_target_valid=torch.ones(n, 6, dtype=torch.bool),
        target_source_residual_xy_m=target, target_source_displacement_xy_m=target+kta,
        target_yaw_rad=torch.ones(n, 6)*.1, yaw_enabled=torch.ones(n, dtype=torch.bool),
        yaw_label_valid=torch.ones(n, 6, dtype=torch.bool))


def base_checkpoint():
    torch.manual_seed(25)
    model = LocalSpatialTemporalWorldModelV18SE2(CFG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    batch = common.make_batch([record()], "cpu")
    loss, _ = common.original_objective(causal_forward(model, batch), batch)
    loss.backward(); optimizer.step()
    return {"model_config": asdict(CFG), "state_dict": trainer.cpu_weights(model),
            "optimizer": copy.deepcopy(optimizer.state_dict())}


def test_twenty_percent_exact_balanced_temporal_and_scene_disjoint():
    keys = [(f"scene{i:03d}", f"t{t:03d}") for i in range(700) for t in range(29)]
    keys += [(f"scene{i:03d}", "t029") for i in range(130)]
    assert len(keys) == 20430
    train, cal = common.select_population(keys, {"dev"})
    assert len(train) == len(set(train)) == 4086 and len(cal) == 64
    assert not {s for s, _ in train} & {s for s, _ in cal}
    assert len({s for s, _ in train}) == 668
    assert max(int(t[1:]) for _, t in train) >= 26
    assert min(int(t[1:]) for _, t in train) <= 2
    assert (train, cal) == common.select_population(keys, {"dev"})
    with pytest.raises(RuntimeError, match="duplicate"):
        common.select_population(keys+[keys[0]], set())
    with pytest.raises(RuntimeError, match="overlap"):
        common.select_population(keys, {"scene000"})


def test_whole_window_packing_covers_every_record_no_source_truncation():
    rows = [record(n=i+1, token=str(i)) for i in range(8)]
    packed = list(common.window_batches(rows, seed=10, epoch=1, source_budget=5, window_budget=2))
    assert sorted(i for batch in packed for i in batch) == list(range(8))
    assert any(len(batch) == 1 and len(rows[batch[0]]["features"]) > 5 for batch in packed)
    assert packed == list(common.window_batches(rows, seed=10, epoch=1, source_budget=5, window_budget=2))


def test_graph_no_cross_window_no_self_no_future_label_inputs_translation_invariant():
    a, b = record(), record(scene="other")
    batch = common.make_batch([a, b], "cpu")
    idx, valid = batch["neighbor_indices"], batch["neighbor_valid"]
    for i in range(6):
        assert all((int(j)//3 == i//3 and int(j) != i) for j in idx[i][valid[i]])
    changed = copy.deepcopy(a)
    for key in common.LABEL_KEYS:
        changed[key] = torch.full_like(changed[key], False if changed[key].dtype == torch.bool else 123)
    x = common.make_batch([a], "cpu", labels=False)
    y = common.make_batch([changed], "cpu", labels=False)
    assert all(torch.equal(x[k], y[k]) for k in x)
    changed["source_centroid_xy_t0_m"] += 50
    y = common.make_batch([changed], "cpu", labels=False)
    assert all(torch.equal(x[k], y[k]) for k in x)


def test_initial_output_exact_v18_and_optimizer_moments_preserved_without_aliasing():
    ck = base_checkpoint(); before = copy.deepcopy(ck["optimizer"])
    models, optimizers = common.build_pair(ck, torch.device("cpu"))
    batch = common.make_batch([record()], "cpu")
    for m in models.values(): m.eval()
    with torch.inference_mode():
        a, b = [causal_forward(models[arm], batch) for arm in ARMS]
    assert all(torch.equal(a[key], b[key]) for key in a)
    assert all(p.requires_grad for m in models.values() for p in m.parameters())
    assert len(optimizers[ARMS[0]].param_groups) == 1 and len(optimizers[ARMS[1]].param_groups) == 2
    for opt in optimizers.values():
        for original, restored in zip(before["state"].values(), list(opt.state.values())[:len(before["state"])]):
            for key in ("step", "exp_avg", "exp_avg_sq"):
                assert torch.equal(original[key], restored[key])
    p = next(models[ARMS[0]].parameters())
    optimizers[ARMS[0]].state[p]["exp_avg"].zero_()
    first = next(iter(before["state"].values()))["exp_avg"]
    assert torch.equal(next(iter(ck["optimizer"]["state"].values()))["exp_avg"], first)
    malformed = {**ck, "optimizer": {**before, "param_groups": before["param_groups"]*2}}
    with pytest.raises(RuntimeError, match="optimizer"):
        common.build_pair(malformed, "cpu")


def test_end_to_end_original_encoder_xy_yaw_and_interaction_really_train():
    models, opts = common.build_pair(base_checkpoint(), "cpu")
    model, opt = models[ARMS[1]], opts[ARMS[1]]
    b = common.make_batch([record()], "cpu")
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        loss, _ = common.original_objective(causal_forward(model, b), b)
        loss.backward()
        assert model.spatial_stem[0].weight.grad.abs().sum() > 0
        assert model.residual_head.weight.grad.abs().sum() > 0
        assert model.yaw_head.weight.grad.abs().sum() > 0
        assert model.source_interaction.attention.out_proj.weight.grad.abs().sum() > 0
        if step == 1:
            assert model.source_interaction.attention.in_proj_weight.grad.abs().sum() > 0
        opt.step()


def test_unsupervised_context_retained_nan_missing_labels_safe_and_loss_same_as_original():
    rows = record()
    model = LocalSpatialTemporalWorldModelV18SE2(CFG)
    b = common.make_batch([rows], "cpu")
    out = causal_forward(model, b)
    original, _ = common.se2_objective_loss(out, b, yaw_weight=19., shape_weight=.25, patch_resolution_m=.8)
    ours, _ = common.original_objective(out, b)
    torch.testing.assert_close(original, ours)
    rows["supervised_source"][0] = False
    for key in ("target_source_residual_xy_m", "target_source_displacement_xy_m", "target_yaw_rad", "existence"):
        rows[key][0] = float("nan")
    rows["se2_target_valid"][1, 0] = False
    rows["target_source_residual_xy_m"][1, 0] = float("nan")
    b = common.make_batch([rows], "cpu")
    assert len(b["features"]) == 3
    loss, _ = common.original_objective(causal_forward(model, b), b)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    b["target_source_residual_xy_m"][2, 0] = float("nan")
    with pytest.raises(RuntimeError, match="nonfinite"):
        common.original_objective(causal_forward(model, b), b)


@pytest.mark.parametrize("n", [0, 1, 3])
def test_empty_singleton_and_out_of_radius_correction_is_zero_even_after_training(n):
    model = SourceInteractionV18(CFG).eval()
    with torch.no_grad():
        model.source_interaction.attention.out_proj.weight.fill_(.1)
        model.source_interaction.attention.out_proj.bias.fill_(.2)
    rows = record(n); rows["source_centroid_xy_t0_m"] *= 100
    batch = common.make_batch([rows], "cpu")
    with torch.inference_mode():
        latent = LocalSpatialTemporalWorldModelV18SE2.forward(model, *[batch[k] for k in common.INPUT_KEYS])
        interacted = causal_forward(model, batch)
    assert all(torch.equal(latent[k], interacted[k]) for k in latent)


def test_window_permutation_and_neighbor_context_independent_of_other_windows():
    torch.manual_seed(1)
    model = SourceInteractionV18(CFG).eval()
    with torch.no_grad(): model.source_interaction.attention.out_proj.weight.normal_(std=.05)
    a, b = record(), record(scene="other")
    b["features"] *= 20
    with torch.inference_mode():
        alone = causal_forward(model, common.make_batch([a], "cpu"))
        together = causal_forward(model, common.make_batch([a, b], "cpu"))
        perm = torch.tensor([2, 0, 1])
        shuffled = {k: v[perm] if torch.is_tensor(v) and v.ndim and len(v) == 3 else v for k, v in a.items()}
        reordered = causal_forward(model, common.make_batch([shuffled], "cpu"))
    for key in alone:
        torch.testing.assert_close(alone[key], together[key][:3], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(alone[key][perm], reordered[key], atol=2e-6, rtol=2e-5)


def test_calibration_selection_is_real_miou_not_ade_and_keeps_update0_on_failure():
    model = LocalSpatialTemporalWorldModelV18SE2(CFG)
    prev = {"score": 40., "update": 0}
    assert trainer.select_checkpoint(prev, {"metrics": {"mIoU": 39.9}}, model, 12) is prev
    chosen = trainer.select_checkpoint(prev, {"metrics": {"mIoU": 40.2}}, model, 12)
    assert chosen["update"] == 12 and chosen["score"] == 40.2
    with pytest.raises(RuntimeError, match="nonfinite"):
        trainer.select_checkpoint(prev, {"metrics": {"mIoU": float("nan")}}, model, 12)


def test_checkpoint_loader_guards_full_model_role_and_contract(tmp_path):
    models, _ = common.build_pair(base_checkpoint(), "cpu")
    identity = dict(protocol=trainer.PROTOCOL, arm=ARMS[1], mode="screen", base_checkpoint_sha256="a"*64,
                    runtime_config_fingerprint="b"*64, interaction_config=asdict(InteractionConfig()))
    ck = trainer.checkpoint_payload(identity, models[ARMS[1]], role="selected_full_v18_candidate", update=10, screen_pass=True)
    path = tmp_path/"candidate.pt"; torch.save(ck, path)
    _, loaded, _ = common.load_candidate(path, "cpu", base_sha="a"*64, config_sha="b"*64)
    assert isinstance(loaded, SourceInteractionV18)
    for change in ({"screen_pass": False}, {"selected_update": 0}, {"mode": "smoke"}, {"checkpoint_role": "last_full_v18_diagnostic"}):
        torch.save({**ck, **change}, path)
        with pytest.raises(RuntimeError, match="cannot be deployed"):
            common.load_candidate(path, "cpu", base_sha="a"*64, config_sha="b"*64)
    torch.save(ck, path)
    with pytest.raises(RuntimeError, match="contract"):
        common.load_candidate(path, "cpu", base_sha="wrong", config_sha="b"*64)


def test_record_contract_rejects_box_center_labels_or_anchor_drift():
    rows = record(); common.validate_records([rows])
    rows["anchors_xy_t0_m"] += .1
    with pytest.raises(RuntimeError, match="anchor"):
        common.validate_records([rows])
    rows = record(); rows["target_source_displacement_xy_m"] += .1
    with pytest.raises(RuntimeError, match="residual target"):
        common.validate_records([rows])


def render_fixture():
    from real_motion.geometry import OccupancyGrid
    from real_motion.rigid_transport import RasterizedRigidComponent
    grid = OccupancyGrid(x_min=0, y_min=0, z_min=0, voxel_size=(1, 1, 1), shape_hwd=(12, 12, 2))
    pcfg = SimpleNamespace(grid=grid, free_label=17, frame_dt_s=.5)
    comp = {"class_id": 4, "voxel_indices": np.array([[1, 1, 0]]), "centroid_world": np.array([1.5, 1.5, .5])}
    prior = RasterizedRigidComponent(4, comp["voxel_indices"], 1)
    b = np.full(grid.shape_hwd, 17, np.uint8); b[1, 1, 0] = 4; b[11, 11, 0] = 11
    state = dict(current=[comp], current_pose=np.eye(4), future_poses=[np.eye(4)]*6,
        source_world_points=[np.array([[1.5, 1.5, .5]])], source_rel_xy=[np.zeros((1, 2))],
        world_to_future=[np.eye(4)]*6, source_z_t0=np.array([.5]), gpu={}, anchors=[b]*6,
        baseline_by_hi=[[prior]]*6, baseline_clear_flat_by_hi=[np.array([26])]*6)
    truth = b.copy(); truth[1, 1, 0] = 17; truth[2, 1, 0] = 4
    return state, pcfg, dict(future_gt_occ=[truth]*6)


def test_actual_six_horizon_deployment_requests_no_future_gt_and_keeps_static_background():
    state, pcfg, raw = render_fixture()
    rows = record(1, "dev"); flags = []
    model = LocalSpatialTemporalWorldModelV18SE2(CFG)
    with torch.no_grad():
        model.residual_head.weight.zero_(); model.residual_head.bias[:] = torch.tensor([1., 0.])
    def prepare(source, r, include_gt):
        flags.append(include_gt)
        return None, raw, {**state, "rec": r}, None
    provider = SimpleNamespace(device=torch.device("cpu"), model=None, pcfg=pcfg, strong=None, prepare=prepare)
    _, pred = common.forecast(provider, None, rows, model)
    assert flags == [False] and len(pred) == 6
    assert pred[5][2, 1, 0] == 4 and pred[5][1, 1, 0] == 17 and pred[5][11, 11, 0] == 11


def test_full_cli_synthetic_paired_smoke_persists_models_calibration_and_real_metrics(tmp_path):
    state, pcfg, raw = render_fixture(); config = InteractionConfig()
    ck = base_checkpoint()
    base = LocalSpatialTemporalWorldModelV18SE2(CFG); base.load_state_dict(ck["state_dict"])
    train = [record(1, f"train{i}", f"t{j}") for i in range(20) for j in range(3)]
    dev = [record(1, "dev", f"d{i}") for i in range(64)]
    flags = []
    def prepare(source, r, include_gt):
        flags.append(include_gt)
        return (SimpleNamespace(scene_name=r["scene_name"], t0_token=r["t0_token"], future_tokens=r["future_tokens"]),
                raw, {**state, "rec": r}, common.predict(base, r, "cpu", config))
    provider = SimpleNamespace(device=torch.device("cpu"), model=base, pcfg=pcfg, strong=None,
        workers=1, sha="a"*64, prepare=prepare, encode_record=lambda r: common.predict(base, r, "cpu", config))
    files = {k: tmp_path/k for k in ("train-cache", "dev-cache", "population-manifest", "base-checkpoint", "train-info", "dev-info")}
    for p in files.values(): p.touch()
    torch.save(ck, files["base-checkpoint"])
    manifest = {"parent_keys": [("dev", str(i)) for i in range(512)],
                "selected_key_fingerprint": trainer.DEV64_FP, "manifest_fingerprint": "b"*64}
    argv = ["train", "--config", str(Path(__file__).resolve().parents[1]/"configs/real_motion_occfm.yaml"),
        "--dataroot", str(tmp_path), "--out-dir", str(tmp_path/"result"), "--device", "cpu", "--mode", "smoke",
        "--source-budget", "4", "--window-budget", "2"]
    for k, p in files.items(): argv += ["--"+k, str(p)]
    with patch("sys.argv", argv), patch.object(trainer, "make_prepare_config", return_value=pcfg), \
         patch.object(trainer, "load_manifest", return_value=(manifest, [("dev", f"d{i}") for i in range(64)], None)), \
         patch.object(trainer, "load_cache", side_effect=[({}, train), ({}, dev)]), \
         patch.object(trainer, "FrozenXYV18", return_value=provider), \
         patch.object(trainer, "NuScenesWindowSource", return_value=SimpleNamespace(nusc=None)), \
         patch.object(common.runtime, "_stage_gpu_inputs"), patch.object(common.runtime, "_release_gpu_inputs"), \
         patch.object(common, "assert_forward_exact"), patch.object(common.runtime, "_exactness_check"), \
         patch.object(common, "gt_moving_support_sequence", return_value=[(np.ones_like(raw["future_gt_occ"][0], bool), None)]*6):
        trainer.main()
    result = tmp_path/"result"
    summary = json.loads((result/"summary.json").read_text(encoding="utf-8"))
    assert summary["train_windows"] == 8 and summary["updates_per_arm"] == 4
    assert summary["epochs_per_arm"] == 1 and len(summary["dev"]["variants"]) == 4
    assert all(flags)  # GT used only in explicit evaluation calls, not training forward.
    assert len(list(result.glob("*.pt"))) == 4
    assert summary["route"] == "smoke_only_not_effectiveness_evidence"
    assert not any(arm["screen_pass"] for arm in summary["arms"].values())
    rows = [json.loads(line) for line in (result/"progress.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["update"] for r in rows if r["event"] == "train" and r["arm"] == ARMS[0]] == list(range(1, 5))
    assert [r["update"] for r in rows if r["event"] == "train" and r["arm"] == ARMS[1]] == list(range(1, 5))
    for path in result.glob("*.pt"):
        with pytest.raises(RuntimeError, match="cannot be deployed"):
            common.load_candidate(path, "cpu", base_sha="a"*64,
                config_sha=torch.load(path, weights_only=False)["runtime_config_fingerprint"])
