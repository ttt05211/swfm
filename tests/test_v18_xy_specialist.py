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
from tools.real_motion import eval_p0_f9_v18_xy_specialist as expanded

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


def test_expanded_population_adds_scenes_not_just_windows_and_preserves_identities():
    rows = [dict(scene_name=f"dev{s}", t0_token=f"d{s}_{i}")
            for s in range(150) for i in range(29+(s < 19))]
    assert len(rows) == 4369
    parent = [(r["scene_name"], r["t0_token"]) for r in rows if int(r["scene_name"][3:]) < 18][:512]
    dev64 = parent[::8][:64]
    manifest = {"parent_keys": parent}
    selected, keys, groups = expanded.plan_populations(rows, manifest, dev64, population="full4369")
    assert len(selected) == len(set(keys)) == 4369 and keys[:64] == tuple(dev64)
    assert len({s for s, _ in groups["new_scenes_only"]}) == 132
    assert not {s for s, _ in groups["new_scenes_only"]} & {s for s, _ in parent}
    _, small, small_groups = expanded.plan_populations(rows, manifest, dev64, population="dev512")
    assert len(small) == 512 and small[:64] == tuple(dev64)
    assert list(small_groups) == ["dev64_reproduction"]
    with pytest.raises(RuntimeError, match="duplicate"):
        expanded.plan_populations(rows+[rows[0]], manifest, dev64, population="full4369")
    with pytest.raises(RuntimeError, match="missing"):
        expanded.plan_populations(rows[1:], manifest, dev64, population="full4369")
    with pytest.raises(RuntimeError, match="150-scene"):
        expanded.plan_populations(rows[:-1], manifest, dev64, population="full4369")


def evaluation_fixture():
    state, pcfg, raw = render_fixture()
    base = LocalSpatialTemporalWorldModelV18SE2(CFG)
    with torch.no_grad():
        base.residual_head.weight.zero_(); base.residual_head.bias.zero_()
    calls = []
    def prepare(source, r, include_gt):
        calls.append(common.record_key(r))
        assert include_gt is True
        return (SimpleNamespace(scene_name=r["scene_name"], t0_token=r["t0_token"], future_tokens=r["future_tokens"]),
                raw, {**state, "rec": r}, base_outputs(base, r))
    provider = SimpleNamespace(device=torch.device("cpu"), model=base, pcfg=pcfg, strong=None,
                               workers=1, sha="a"*64, prepare=prepare)
    return provider, raw, calls


def patch_renderer(raw):
    from contextlib import ExitStack
    stack = ExitStack()
    stack.enter_context(patch.object(evaluator.runtime, "_stage_gpu_inputs"))
    stack.enter_context(patch.object(evaluator.runtime, "_release_gpu_inputs"))
    stack.enter_context(patch.object(evaluator, "assert_forward_exact"))
    stack.enter_context(patch.object(evaluator.runtime, "_exactness_check"))
    stack.enter_context(patch.object(evaluator, "gt_moving_support_sequence",
        return_value=[(np.ones_like(raw["future_gt_occ"][0], bool), None)]*6))
    return stack


def test_population_reports_equal_separate_evaluation_without_repeated_prepare():
    provider, raw, calls = evaluation_fixture()
    rows = [record(1, f"dev{i%2}", f"d{i}") for i in range(4)]
    candidate = copy.deepcopy(provider.model)
    with torch.no_grad(): candidate.residual_head.bias[0] = 1.
    completed = []
    keys = tuple(common.record_key(r) for r in rows)
    with patch_renderer(raw):
        combined = evaluator.evaluate_models(provider, SimpleNamespace(nusc=None), rows, {"last": candidate}, None,
            prediction_fn=common.predict, population_groups={"first": keys[:2], "overlap": keys[1:]},
            population_complete_fn=lambda g, r: completed.append((g, len(calls))))
        assert calls == list(keys) and completed == [("first", 2), ("all", 4), ("overlap", 4)]
        separate = evaluator.evaluate_models(provider, SimpleNamespace(nusc=None), rows[:2], {"last": candidate}, None,
                                              prediction_fn=common.predict)
    assert combined["populations"]["first"]["windows"] == 2
    assert combined["populations"]["overlap"]["windows"] == 3
    from tools.real_motion.static_evidence_selector_common import finite_json
    first = {k: v for k, v in combined["populations"]["first"].items() if k not in ("windows", "scenes")}
    assert finite_json(first) == finite_json(separate)
    assert first["variants"]["last"]["delta_vs_v18_pp"]["mIoU"] > 0
    for groups in ({"all": keys}, {"empty": []}, {"duplicate": [keys[0]]*2}, {"missing": [("other", "absent")]}):
        with pytest.raises(RuntimeError, match="population"):
            evaluator.evaluate_models(provider, None, rows, {}, None, population_groups=groups)


def test_expanded_cli_reproduces_dev64_in_single_512_pass_and_leaves_checkpoints_unchanged(tmp_path):
    from real_motion.v21_source_induction import stable_json_fingerprint
    from real_motion.runtime_config import load_runtime_config
    from tools.real_motion.static_evidence_selector_common import write_json
    provider, raw, calls = evaluation_fixture()
    rows = [record(1, "dev", f"d{i}") for i in range(512)]
    keys = [common.record_key(r) for r in rows]
    manifest = {"parent_keys": keys, "selected_key_fingerprint": trainer.DEV64_FP,
                "manifest_fingerprint": "b"*64}
    config = Path(__file__).resolve().parents[1]/"configs/real_motion_occfm.yaml"
    config_sha = stable_json_fingerprint(load_runtime_config(config, []))
    train = [("train", f"t{i}") for i in range(4086)]
    cal = [("cal", f"c{i}") for i in range(64)]
    identity = dict(protocol=common.PROTOCOL, mode="screen", train_keys=train, calibration_keys=cal, dev_keys=keys[:64],
                    population_fingerprint=stable_json_fingerprint({"train": train, "calibration": cal}),
                    dev_manifest_fingerprint=manifest["manifest_fingerprint"], base_checkpoint_sha256=provider.sha,
                    runtime_config_fingerprint=config_sha,
                    deployment_contract="specialist_XY_plus_frozen_CleanE14_yaw_existence_two_forward_v1")
    model_dir = tmp_path/"original"; model_dir.mkdir()
    write_json(model_dir/"execution_contract.json", identity)
    models = {}
    for arm in common.ARMS:
        last = copy.deepcopy(provider.model)
        with torch.no_grad(): last.residual_head.bias[0] = 1.
        models[arm+"_last"] = last
        for suffix, model, role, update in (
            ("best", provider.model, "selected_xy_specialist_candidate", 0),
            ("last", last, "last_xy_specialist_diagnostic", 3)):
            torch.save(trainer.checkpoint_payload({**identity, "arm": arm}, model, role=role, update=update),
                       model_dir/(arm+"_"+suffix+".pt"))
    with patch_renderer(raw):
        previous_dev = evaluator.evaluate_models(provider, SimpleNamespace(nusc=None), rows[:64], models, None,
                                                prediction_fn=common.predict)
    aliases = {arm+"_selected": "V18_BASE" for arm in common.ARMS}
    expanded.attach_baseline_aliases(previous_dev, aliases, keys[:64])
    previous = dict(protocol=common.PROTOCOL, mode="screen", updates_per_arm=3, dev=previous_dev,
                    arms={arm: {"selected_update": 0} for arm in common.ARMS})
    write_json(model_dir/"summary.json", previous)
    digests = {p.name: expanded.sha256(p) for p in model_dir.iterdir()}
    calls.clear()
    inputs = {k: tmp_path/k for k in ("dev-cache", "population-manifest", "base-checkpoint", "dev-info")}
    for path in inputs.values(): path.touch()
    argv = ["evaluate", "--config", str(config), "--model-dir", str(model_dir), "--out-dir", str(tmp_path/"expanded"),
            "--dataroot", str(tmp_path), "--device", "cpu", "--population", "dev512"]
    for k, path in inputs.items(): argv += ["--"+k, str(path)]
    with patch("sys.argv", argv), patch_renderer(raw), \
         patch.object(expanded, "make_prepare_config", return_value=provider.pcfg), \
         patch.object(expanded, "load_manifest", return_value=(manifest, keys[:64], None)), \
         patch.object(expanded, "load_cache", return_value=({}, rows)), \
         patch.object(expanded, "FrozenXYV18", return_value=provider), \
         patch.object(expanded, "NuScenesWindowSource", return_value=SimpleNamespace(nusc=None)):
        expanded.main()
    result = json.loads((tmp_path/"expanded/summary.json").read_text(encoding="utf-8"))
    assert result["windows"] == len(calls) == len(set(calls)) == 512
    assert result["dev64_reproduction"]["max_abs_difference_pp"] == 0.
    assert result["reports"]["dev64_reproduction"]["windows"] == 64
    assert sum(v["baseline_alias_verified"] for v in result["checkpoint_audit"].values()) == 2
    assert not list((tmp_path/"expanded").glob("*.pt"))
    assert {p.name: expanded.sha256(p) for p in model_dir.iterdir()} == digests
    assert "no_retraining_or_checkpoint_reselection" in result["route"]
    broken = copy.deepcopy(previous)
    broken["dev"]["baseline"]["mIoU"] += .001
    with pytest.raises(RuntimeError, match="reproduction mismatch"):
        expanded.check_dev64_reproduction(previous_dev, broken)
    bad_path = model_dir/(common.ARMS[0]+"_best.pt")
    ck = torch.load(bad_path, weights_only=False)
    ck["state_dict"]["residual_head.bias"][0] += 1.
    torch.save(ck, bad_path)
    with pytest.raises(RuntimeError, match="update0 checkpoint weights differ"):
        expanded.read_bundle(model_dir, provider, config_sha, manifest, keys[:64])
