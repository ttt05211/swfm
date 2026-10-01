"""No real nuScenes gain claims: frozen predictions, loss masks, paired CLI."""
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
from tools.real_motion import v18_xy_specialist_common as common
from tools.real_motion import train_p0_f9_v18_xy_specialist as trainer
from tools.real_motion import v18_source_interaction_common as evaluator

CFG = LocalSTWMV17Config(d_model=16, semantic_dim=8, heads=4, blocks=1, decoder_blocks=1)


def record(n=3, scene="train", token="t0"):
    rng = torch.Generator().manual_seed(37)
    kta = torch.zeros(n, 6, 2)
    target = torch.ones(n, 6, 2)*.25
    center = torch.stack((torch.arange(n).float()+1.5, torch.ones(n)*1.5), dim=1)
    motion = torch.randn(n, 6, 5, generator=rng)*.1; motion[..., 4] = 1
    mask = torch.zeros(n, 6, 20, 20, dtype=torch.uint8); mask[:, :, 7:14, 9:12] = 1
    return dict(scene_name=scene, t0_token=token, future_tokens=tuple(f"future{i}" for i in range(6)),
        features=torch.randn(n, FEATURE_DIM, generator=rng),
        local_semantic_tube=torch.randint(0, 18, (n, 6, 20, 20), generator=rng, dtype=torch.uint8),
        frame_motion_features=motion, kta_displacement_xy_m=kta,
        source_centroid_xy_t0_m=center, anchors_xy_t0_m=center[:, None]+kta, source_class_id=torch.full((n,), 4),
        target_source_mask_tube=mask, supervised_source=torch.ones(n, dtype=torch.bool), existence=torch.ones(n, 6),
        se2_target_valid=torch.ones(n, 6, dtype=torch.bool), target_source_residual_xy_m=target,
        target_source_displacement_xy_m=target+kta, target_yaw_rad=torch.ones(n, 6)*.4,
        yaw_enabled=torch.ones(n, dtype=torch.bool), yaw_label_valid=torch.ones(n, 6, dtype=torch.bool))


def base_checkpoint():
    torch.manual_seed(25)
    model = LocalSpatialTemporalWorldModelV18SE2(CFG)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    b = common.make_batch([record()], "cpu")
    loss, _ = evaluator.original_objective(model(*[b[k] for k in common.INPUT_KEYS]), {**b, "existence": torch.ones(3, 6)})
    loss.backward(); opt.step()
    return {"model_config": asdict(CFG), "state_dict": trainer.cpu_weights(model), "optimizer": copy.deepcopy(opt.state_dict())}


def base_outputs(model, rec):
    b = common.make_batch([rec], "cpu", labels=False)
    model.eval()
    with torch.inference_mode():
        return model(*[b[k] for k in common.INPUT_KEYS])


def test_restore_tail_lr_moments_exact_initial_xy_and_no_state_aliasing():
    ck = base_checkpoint()
    before = copy.deepcopy(ck["optimizer"])
    models, opts = common.build_pair(ck, "cpu")
    for arm in common.ARMS:
        model = models[arm]; base = base_outputs(model, record())
        initial = common.predict(model, record(), "cpu", base=base)
        assert all(torch.equal(initial[k], base[k]) for k in base)
        assert initial["yaw_delta_rad"] is base["yaw_delta_rad"]
        assert initial["existence_logits"] is base["existence_logits"]
        assert not any(p.requires_grad for p in model.yaw_head.parameters())
        assert not any(p.requires_grad for p in model.existence_head.parameters())
        assert model.spatial_stem[0].weight.requires_grad and model.residual_head.weight.requires_grad
        assert opts[arm].param_groups[0]["lr"] == 5e-5
        for old, restored in zip(before["state"].values(), opts[arm].state.values()):
            for key in ("step", "exp_avg", "exp_avg_sq"):
                assert torch.equal(old[key], restored[key])
    p = next(models[common.ARMS[0]].parameters())
    opts[common.ARMS[0]].state[p]["exp_avg"].zero_()
    assert torch.equal(next(iter(ck["optimizer"]["state"].values()))["exp_avg"], next(iter(before["state"].values()))["exp_avg"])
    with pytest.raises(RuntimeError, match="optimizer"):
        common.build_pair({**ck, "optimizer": None}, "cpu")


@pytest.mark.parametrize("teacher_prob", [0., .5, 1.])
def test_core_xy_updates_but_frozen_heads_and_actual_predictions_do_not(teacher_prob):
    ck = base_checkpoint(); models, opts = common.build_pair(ck, "cpu")
    model, opt = models[common.ARMS[0]], opts[common.ARMS[0]]
    rows = record(); frozen = {k: v.clone() for k, v in base_outputs(model, rows).items()}
    b = common.make_batch([rows], "cpu")
    yaw = frozen["yaw_delta_rad"].clone().requires_grad_()
    loss, _ = common.xy_objective(common.forward_xy(model, b), b, yaw, teacher_prob=teacher_prob, seed=37)
    loss.backward()
    assert model.spatial_stem[0].weight.grad.abs().sum() > 0
    assert model.decoder[0].self_attn.in_proj_weight.grad.abs().sum() > 0
    assert model.residual_head.weight.grad.abs().sum() > 0
    assert yaw.grad is None
    opt.step(); common.assert_frozen_heads(model, ck["state_dict"])
    after = common.predict(model, rows, "cpu", base=frozen)
    assert not torch.equal(after["residual_xy_m"], frozen["residual_xy_m"])
    assert after["yaw_delta_rad"] is frozen["yaw_delta_rad"]
    assert after["existence_logits"] is frozen["existence_logits"]


def test_teacher_uses_only_legal_gt_yaw_and_losses_change_without_forward_change():
    rows = record(4)
    rows["supervised_source"][1] = False
    rows["yaw_enabled"][2] = False
    rows["yaw_label_valid"][3, :3] = False
    b = common.make_batch([rows], "cpu")
    pred = b["target_source_residual_xy_m"].clone().requires_grad_()
    frozen_yaw = torch.zeros(4, 6)
    ordinary, a = common.xy_objective(pred, b, frozen_yaw, teacher_prob=0.)
    taught, z = common.xy_objective(pred, b, frozen_yaw, teacher_prob=1.)
    assert a["teacher_gt_yaw_labels"] == 0
    assert z["teacher_gt_yaw_labels"] == 9 and z["teacher_eligible_yaw_labels"] == 9
    assert taught < ordinary
    assert z["translation_smooth_l1"] == a["translation_smooth_l1"] == 0.
    for seed in (1, 9):
        a = common.xy_objective(pred, b, frozen_yaw, teacher_prob=.5, seed=seed)[1]
        z = common.xy_objective(pred, b, frozen_yaw, teacher_prob=.5, seed=seed)[1]
        assert a == z


def test_invalid_or_unsupervised_nan_labels_sanitized_valid_labels_fail_closed():
    rows = record()
    rows["supervised_source"][0] = False
    rows["se2_target_valid"][1, 0] = False
    rows["yaw_label_valid"][2, 0] = False
    for key in ("target_source_residual_xy_m", "target_source_displacement_xy_m", "target_yaw_rad"):
        rows[key][0] = float("nan")
        rows[key][1, 0] = float("nan")
    rows["target_yaw_rad"][2, 0] = float("nan")
    b = common.make_batch([rows], "cpu")
    pred = torch.zeros(3, 6, 2, requires_grad=True)
    loss, stats = common.xy_objective(pred, b, torch.zeros(3, 6), teacher_prob=.5)
    loss.backward()
    assert stats["valid_xy_labels"] == 11 and torch.isfinite(pred.grad).all()
    b["target_source_residual_xy_m"][2, 1] = float("nan")
    with pytest.raises(RuntimeError, match="nonfinite valid"):
        common.xy_objective(pred, b, torch.zeros(3, 6))


@pytest.mark.parametrize("n", [0, 1, 3])
def test_forward_whitelist_inference_independent_of_all_future_labels(n):
    rows = record(n); model = LocalSpatialTemporalWorldModelV18SE2(CFG)
    base = base_outputs(model, rows)
    altered = copy.deepcopy(rows)
    for key in common.LABEL_KEYS:
        altered.pop(key)
    a = common.predict(model, rows, "cpu", base=base)
    z = common.predict(model, altered, "cpu", base=base)
    assert all(torch.equal(a[k], z[k]) for k in base)
    with pytest.raises(RuntimeError, match="requires frozen"):
        common.predict(model, rows, "cpu")


def test_teacher_schedule_monotonic_zero_final_third_and_empty_labels_not_updates():
    for total in (1, 2, 3, 4, 17, 1535):
        p = [common.teacher_probability(i, total) for i in range(total)]
        assert all(a >= b for a, b in zip(p, p[1:]))
        assert all(x == 0. for x in p[2*total//3:])
    assert common.teacher_probability(0, 1535) == .5
    rows = [record(token=str(i)) for i in range(4)]
    rows[0]["se2_target_valid"][:] = False
    use, empty = common.epoch_batches(rows, seed=0, epoch=1, source_budget=1, window_budget=1)
    assert len(use) == 3 and empty == [[0]]
    assert sorted(i for batch in use+empty for i in batch) == list(range(4))
    with pytest.raises(ValueError):
        common.teacher_probability(5, 5)


def test_compact_cache_is_causal_normal_tensor_and_fail_closed_budget_identity():
    rows = record(); model = LocalSpatialTemporalWorldModelV18SE2(CFG)
    provider = SimpleNamespace(encode_record=lambda r: base_outputs(model, r))
    cache, mib = common.cache_baseline_outputs(provider, [rows])
    assert set(cache[common.record_key(rows)]) == {"yaw_delta_rad", "existence_logits"}
    assert 0 < mib < .001
    yaw = cache[common.record_key(rows)]["yaw_delta_rad"]
    assert not yaw.is_inference() and not yaw.requires_grad
    loss, _ = common.xy_objective(torch.zeros(3, 6, 2, requires_grad=True), common.make_batch([rows], "cpu"), yaw)
    loss.backward()
    with pytest.raises(RuntimeError, match="duplicate"):
        common.cache_baseline_outputs(provider, [rows, rows])
    with pytest.raises(RuntimeError, match="budget"):
        common.cache_baseline_outputs(provider, [rows], max_mib=0)


def test_checkpoint_loader_requires_two_forward_contract_and_rejects_failed_models(tmp_path):
    model = LocalSpatialTemporalWorldModelV18SE2(CFG)
    identity = dict(protocol=common.PROTOCOL, arm=common.ARMS[1], mode="screen", base_checkpoint_sha256="a"*64,
                    runtime_config_fingerprint="b"*64,
                    deployment_contract="specialist_XY_plus_frozen_CleanE14_yaw_existence_two_forward_v1")
    ck = trainer.checkpoint_payload(identity, model, role="selected_xy_specialist_candidate", update=10, screen_pass=True)
    path = tmp_path/"candidate.pt"; torch.save(ck, path)
    _, loaded = common.load_candidate(path, "cpu", base_sha="a"*64, config_sha="b"*64)
    assert not any(p.requires_grad for p in loaded.parameters())
    for change in ({"screen_pass": False}, {"selected_update": 0}, {"mode": "smoke"}, {"checkpoint_role": "last_xy_specialist_diagnostic"}):
        torch.save({**ck, **change}, path)
        with pytest.raises(RuntimeError, match="cannot be deployed"):
            common.load_candidate(path, "cpu", base_sha="a"*64, config_sha="b"*64)
    for change in ({"deployment_contract": "own_yaw"}, {"protocol": "old_v18"}, {"base_checkpoint_sha256": "wrong"}):
        torch.save({**ck, **change}, path)
        with pytest.raises(RuntimeError, match="contract mismatch"):
            common.load_candidate(path, "cpu", base_sha="a"*64, config_sha="b"*64)


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


def test_real_six_horizon_forecast_never_requests_gt_and_preserves_static_background():
    state, pcfg, _ = render_fixture(); rows = record(1, "dev"); flags = []
    model = LocalSpatialTemporalWorldModelV18SE2(CFG)
    with torch.no_grad():
        model.residual_head.weight.zero_(); model.residual_head.bias[:] = torch.tensor([1., 0.])
        model.yaw_head.bias.fill_(3.)
    base = {"residual_xy_m": torch.zeros(1, 6, 2), "yaw_delta_rad": torch.zeros(1, 6), "existence_logits": torch.zeros(1, 6)}
    def prepare(source, r, include_gt):
        flags.append(include_gt)
        return None, {}, {**state, "rec": r}, base
    provider = SimpleNamespace(device=torch.device("cpu"), model=None, pcfg=pcfg, strong=None, prepare=prepare)
    _, pred = common.forecast(provider, None, rows, model)
    assert flags == [False] and len(pred) == 6
    assert pred[5][2, 1, 0] == 4 and pred[5][1, 1, 0] == 17 and pred[5][11, 11, 0] == 11


def test_full_cli_paired_synthetic_smoke_saves_calibrated_candidates_and_real_metrics(tmp_path):
    state, pcfg, raw = render_fixture(); ck = base_checkpoint()
    base = LocalSpatialTemporalWorldModelV18SE2(CFG); base.load_state_dict(ck["state_dict"])
    train = [record(1, f"train{i}", f"t{j}") for i in range(20) for j in range(3)]
    dev = [record(1, "dev", f"d{i}") for i in range(64)]
    def prepare(source, r, include_gt):
        assert include_gt is True  # Explicit evaluation only; training uses cached tensors.
        return (SimpleNamespace(scene_name=r["scene_name"], t0_token=r["t0_token"], future_tokens=r["future_tokens"]),
                raw, {**state, "rec": r}, base_outputs(base, r))
    provider = SimpleNamespace(device=torch.device("cpu"), model=base, pcfg=pcfg, strong=None,
        workers=1, sha="a"*64, prepare=prepare, encode_record=lambda r: base_outputs(base, r))
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
         patch.object(evaluator.runtime, "_stage_gpu_inputs"), patch.object(evaluator.runtime, "_release_gpu_inputs"), \
         patch.object(evaluator, "assert_forward_exact"), patch.object(evaluator.runtime, "_exactness_check"), \
         patch.object(evaluator, "gt_moving_support_sequence", return_value=[(np.ones_like(raw["future_gt_occ"][0], bool), None)]*6):
        trainer.main()
    result = tmp_path/"result"
    summary = json.loads((result/"summary.json").read_text(encoding="utf-8"))
    assert summary["train_windows"] == 8 and summary["updates_per_arm"] == 4
    assert summary["epochs_per_arm"] == 1 and len(summary["dev"]["variants"]) == 4
    assert len(list(result.glob("*.pt"))) == 4 and summary["fixed_tail_lr"] == 5e-5
    assert summary["route"] == "smoke_only_not_effectiveness_evidence"
    assert not any(arm["screen_pass"] for arm in summary["arms"].values())
    rows = [json.loads(line) for line in (result/"progress.jsonl").read_text(encoding="utf-8").splitlines()]
    for arm in common.ARMS:
        logs = [r for r in rows if r["event"] == "train" and r["arm"] == arm]
        assert [r["update"] for r in logs] == list(range(1, 5))
        assert logs[-1]["teacher_probability"] == logs[-1]["teacher_gt_yaw_labels"] == 0
        model = LocalSpatialTemporalWorldModelV18SE2(CFG)
        weights = torch.load(result/(arm+"_last.pt"), weights_only=False)["state_dict"]
        model.load_state_dict(weights); model.yaw_head.requires_grad_(False); model.existence_head.requires_grad_(False)
        common.assert_frozen_heads(model, ck["state_dict"])
    for path in result.glob("*.pt"):
        ck_saved = torch.load(path, weights_only=False)
        with pytest.raises(RuntimeError, match="cannot be deployed"):
            common.load_candidate(path, "cpu", base_sha="a"*64, config_sha=ck_saved["runtime_config_fingerprint"])
