"""Mechanics/synthetic learning only; these tests are NOT nuScenes evidence."""
from dataclasses import replace
import numpy as np
import pytest
import torch

from real_motion.geometry import OccupancyGrid
from real_motion.sparse_emergence import (
    EmergenceConfig, EmergenceInputs, history_in_query_frame, prepare_inputs,
    target_sets, sample_rows, compose_points, nondegradation_gate,
)
from real_motion.sparse_emergence_model import SparseEmergenceDecoder, set_loss
from tools.real_motion.sparse_emergence_common import evaluate_points, EvalRow, select_calibration_threshold, load_emergence
from tools.real_motion.train_p0_f9_sparse_emergence import select_train_calibration, train_variants, training_fit, summary_text
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import Metrics, DYN


def scene():
    grid = OccupancyGrid(x_min=0,y_min=0,z_min=0,voxel_size=(1.,1.,1.),shape_hwd=(8,8,4))
    hist = np.full((6,8,8,4),17,np.uint8); hist[:,1,1,1] = 11
    base = hist[-1].copy(); gt = base.copy(); gt[2,2,2] = 13
    cfg = EmergenceConfig(context_cells=4,width=16,points_per_query=4,target_points=16)
    inp = prepare_inputs(hist,base,grid,1.,np.eye(4),cfg)
    return grid,hist,base,gt,cfg,inp


def test_history_inverse_alignment_and_unknown():
    grid,hist,*_ = scene()
    identity = history_in_query_frame(hist,[np.eye(4)]*6,np.eye(4),grid)
    assert np.array_equal(identity,hist)
    pose = np.eye(4); pose[0,3] = 1
    shifted = history_in_query_frame(hist,[np.eye(4)]*6,pose,grid)
    assert (shifted[:,0,1,1] == 11).all()
    assert (shifted[:,-1] == 18).all()


def test_candidates_have_no_future_gt_argument_and_targets_do_not_modify_inputs():
    grid,hist,base,gt,cfg,inp = scene()
    before = inp.history.copy(),inp.query.copy(),inp.patch_xy.copy()
    targets = target_sets(inp,base,gt,cfg)
    assert targets["mask"].sum() == 1
    assert all(np.array_equal(a,b) for a,b in zip(before,(inp.history,inp.query,inp.patch_xy)))
    other = target_sets(inp,base,np.full(gt.shape,17,np.uint8),cfg)
    assert not other["mask"].any()
    # Budgeted spatial shortlist is identical regardless of target positives.
    again = prepare_inputs(hist,base,grid,1.,np.eye(4),cfg)
    assert np.array_equal(again.patch_xy,inp.patch_xy)


def test_sampling_importance_weights_recover_full_patch_population():
    target = {"dynamic":np.r_[np.ones(3,bool),np.zeros(27,bool)],"count":np.r_[np.ones(9),np.zeros(21)]}
    ids,w = sample_rows(target,9,np.random.default_rng(2))
    assert len(ids) == len(set(ids)) == 9
    assert w.sum() == 30
    assert np.isclose((w*(target["count"][ids]>0)).sum()/w.sum(),.3)


def test_compositor_collision_boundary_identity_and_protection():
    _,_,base,_,cfg,inp = scene(); n = len(inp.patch_xy); k = 16
    xyz = np.zeros((n,k,3),np.float32); sem = np.full((n,k),13,np.int64); conf = np.ones((n,k),np.float32)
    assert np.array_equal(compose_points(base,inp,xyz,sem,conf,None,cfg),base)
    pred = compose_points(base,inp,xyz,sem,conf,0,cfg)
    assert (pred[base < 17] == base[base < 17]).all()
    assert ((pred != base) & (pred == 13)).any()
    # Highest confidence wins, deterministic point-order tie handling.
    sem[:,0] = 11; conf[:,0] = .9; conf[:,1:] = .8
    pred = compose_points(base,inp,xyz,sem,conf,0,cfg)
    assert (pred[pred != base] == 11).all()
    xyz[:] = 1.  # exactly upper boundary -> outside patch, never spills
    assert np.array_equal(compose_points(base,inp,xyz,sem,conf,0,cfg),base)
    xyz[0,0,0] = np.nan
    with pytest.raises(ValueError,match="nonfinite"): compose_points(base,inp,xyz,sem,conf,0,cfg)


@pytest.mark.parametrize("refinement",[False,True])
def test_positive_set_training_gradients_and_empty_presence_only(refinement):
    torch.set_num_threads(1)
    _,_,base,gt,cfg,inp = scene(); cfg = replace(cfg,refinement=refinement)
    model = SparseEmergenceDecoder(cfg)
    targets = target_sets(inp,base,gt,cfg)
    out = model(torch.from_numpy(inp.history),torch.from_numpy(inp.query))
    loss,_ = set_loss(out,torch.from_numpy(targets["xyz"]),torch.from_numpy(targets["labels"]),
        torch.from_numpy(targets["mask"]),torch.ones(len(inp.history)),(4,4,4))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.xyz_head.parameters())
    assert model.xyz_head.weight.grad.abs().sum() > 0
    assert model.semantic_head.weight.grad.abs().sum() > 0
    if refinement: assert model.geometry_feedback.weight.grad.abs().sum() > 0
    model.zero_grad(); out = model(torch.from_numpy(inp.history),torch.from_numpy(inp.query))
    loss,details = set_loss(out,torch.from_numpy(targets["xyz"]),torch.from_numpy(targets["labels"]),
        torch.zeros_like(torch.from_numpy(targets["mask"])),torch.ones(len(inp.history)),(4,4,4))
    loss.backward()
    assert details["geometry"] == details["semantic"] == 0
    assert model.semantic_head.weight.grad.abs().sum() == 0
    assert model.presence_head.bias.grad.abs().sum() > 0


def test_same_parameter_budget_and_training_initialization():
    _,_,base,gt,cfg,inp = scene(); target = target_sets(inp,base,gt,cfg)
    bank = {"history":inp.history,"query":inp.query,**{k:target[k] for k in ("xyz","labels","mask")},"weight":np.ones(len(inp.history),np.float32)}
    models,details = train_variants(bank,{"direct":cfg,"refined":replace(cfg,refinement=True)},torch.device("cpu"),
        updates=2,batch_size=2,seed=3)
    assert sum(p.numel() for p in models["direct"].parameters()) == sum(p.numel() for p in models["refined"].parameters())
    assert all(np.isfinite(r["loss"]) for r in details["last"].values())
    fit = training_fit(models,bank,torch.device("cpu"),3)
    assert all(0 <= r["patch_hit_rate"] <= 1 and np.isfinite(r["mean_presence_probability"]) for r in fit.values())


def test_tiny_overfit_can_decode_shape_absent_from_history():
    torch.set_num_threads(1); torch.manual_seed(3)
    _,hist,base,gt,cfg,inp = scene(); targets = target_sets(inp,base,gt,cfg)
    selected = np.flatnonzero(targets["count"])[0]
    x = torch.from_numpy(inp.history[selected:selected+1]); q = torch.from_numpy(inp.query[selected:selected+1])
    xyz = torch.from_numpy(targets["xyz"][selected:selected+1]); labels = torch.from_numpy(targets["labels"][selected:selected+1])
    mask = torch.from_numpy(targets["mask"][selected:selected+1])
    model = SparseEmergenceDecoder(cfg); opt = torch.optim.Adam(model.parameters(),lr=.01)
    losses = []
    for _ in range(100):
        opt.zero_grad(); out = model(x,q); loss,_ = set_loss(out,xyz,labels,mask,torch.ones(1),(4,4,4))
        loss.backward(); opt.step(); losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]*.25
    assert not (hist == 13).any()  # no source/prototype for generated class
    from tools.real_motion.sparse_emergence_common import predict_points
    pred = compose_points(base,inp,*predict_points(model,inp,torch.device("cpu")),0,cfg)
    assert ((pred == gt) & (pred != base) & (gt == 13)).any()


def fake_report(delta=0.,generated=1,added=1):
    metrics = {k:delta for k in ("mIoU","IoU","MovingMacro","MovingMicro")}
    return {"delta_vs_v18_pp":{**metrics,"per_horizon":{str(h):dict(metrics) for h in (1.,2.,3.)}},
            "quality":{"added":added,"unseen_static_semantic_tp":generated}}


def test_gate_rejects_zero_generation_degradation_and_nonfinite():
    assert nondegradation_gate(fake_report())["pass"]
    assert not nondegradation_gate(fake_report(generated=0,added=0))["pass"]
    assert not nondegradation_gate(fake_report(delta=-.001))["pass"]
    r = fake_report(); r["delta_vs_v18_pp"]["mIoU"] = float("nan")
    assert not nondegradation_gate(r)["pass"]
    r = fake_report(); r["delta_vs_v18_pp"]["per_horizon"]["3.0"]["IoU"] = -.1
    assert not nondegradation_gate(r)["pass"]


def test_calibration_is_active_and_not_zero_edit_baseline():
    reports = {}
    for t,gen,d in ((.1,5,-.01),(.2,2,.01),(.4,1,0.)):
        r = fake_report(delta=d,generated=gen); r["threshold"] = t; r["gate"] = nondegradation_gate(r); reports[t] = r
    assert select_calibration_threshold(reports) == .2
    assert select_calibration_threshold({.1:reports[.1]}) is None


def test_full_and_changed_cell_metrics_match_for_generated_points():
    _,_,base,gt,cfg,inp = scene(); model = SparseEmergenceDecoder(cfg)
    novel = np.zeros(gt.shape,np.uint8); novel[gt == 13] = 1
    moving = np.ones(gt.shape,bool)
    rows = [EvalRow("scene",i,inp,base,gt,moving,novel,Metrics.counts(base,gt,moving,17)) for i in range(3)]
    reports = evaluate_points(model,rows,torch.device("cpu"),(None,0.))
    assert reports[None]["quality"]["added"] == 0
    assert not reports[None]["gate"]["pass"]
    from real_motion.source_evidence_audit import metric_count_delta
    from tools.real_motion.sparse_emergence_common import predict_points
    pred = compose_points(base,inp,*predict_points(model,inp,torch.device("cpu")),0,cfg)
    exact = Metrics.counts(pred,gt,moving,17)
    sparse = metric_count_delta(rows[0].base_counts,base,pred,gt,moving,DYN)
    assert all(np.array_equal(x,y) for x,y in zip(exact,sparse))


def test_calibration_entire_scenes_are_excluded_from_training():
    keys = [(f"scene-{i}",f"token-{i}-{j}") for i in range(8) for j in range(5)]
    train,cal = select_train_calibration(keys,8,2,{"dev"})
    assert not {k[0] for k in train} & {k[0] for k in cal}
    assert (train,cal) == select_train_calibration(keys,8,2,{"dev"})
    with pytest.raises(RuntimeError,match="overlap"): select_train_calibration(keys,8,2,{"scene-1"})
    with pytest.raises(RuntimeError,match="duplicate"): select_train_calibration(keys+[keys[0]],8,2,{"dev"})


def test_failed_checkpoint_and_identity_mismatch_are_not_deployable(tmp_path):
    from dataclasses import asdict
    from real_motion.sparse_emergence import PROTOCOL,FEATURE_PROTOCOL
    cfg = scene()[4]; model = SparseEmergenceDecoder(cfg)
    ck = {"protocol":PROTOCOL,"feature_protocol":FEATURE_PROTOCOL,"base_checkpoint_sha256":"base",
          "runtime_config_fingerprint":"config","model_config":asdict(cfg),"state_dict":model.state_dict(),"screen_pass":False,
          "checkpoint_role":"auxiliary_generation_only","threshold":None}
    path = tmp_path/"aux.pt"; torch.save(ck,path)
    with pytest.raises(RuntimeError,match="failed screen"): load_emergence(path,torch.device("cpu"),base_sha="base",config_sha="config")
    with pytest.raises(RuntimeError,match="contract mismatch"): load_emergence(path,torch.device("cpu"),base_sha="other",config_sha="config")
    _,restored = load_emergence(path,torch.device("cpu"),base_sha="base",config_sha="config",allow_failed_diagnostic=True)
    assert not restored.training


def test_candidate_scope_oracle_is_separate_and_uses_same_bounded_candidates():
    from tools.real_motion.sparse_emergence_common import candidate_scope_ceiling
    _,_,base,gt,cfg,inp = scene()
    novel = np.zeros(gt.shape,np.uint8); novel[gt == 13] = 1
    moving = np.zeros(gt.shape,bool)
    rows = [EvalRow("scene",i,inp,base,gt,moving,novel,Metrics.counts(base,gt,moving,17)) for i in range(3)]
    report = candidate_scope_ceiling(rows,cfg)
    assert report["audit_only"] and not report["used_for_dev_tuning"]
    assert report["scope_gt_delta_pp"]["mIoU"] > 0
    assert report["novel_quality"]["unseen_static_scope_tp"] == 3
    assert report["novel_quality"]["unseen_static_budget_tp"] == 3


def test_deployment_never_requests_future_gt():
    from tools.real_motion.sparse_emergence_common import forecast_with_emergence
    _,_,base,_,cfg,inp = scene(); model = SparseEmergenceDecoder(cfg)
    class Provider:
        device = torch.device("cpu")
        def prepare(self, source, record, *, include_gt, horizons):
            assert include_gt is False and tuple(horizons) == tuple(range(6))
            return "window",None,[base]*6,{h:inp for h in horizons}
    window,pred = forecast_with_emergence(Provider(),None,None,model,None)
    assert window == "window" and all(np.array_equal(p,base) for p in pred)


def test_missing_moving_support_can_pass_only_if_both_predictions_are_undefined():
    r = fake_report()
    r["baseline"] = {"MovingMacro":float("nan"),"MovingMicro":float("nan")}
    r["selected"] = dict(r["baseline"])
    for key in ("MovingMacro","MovingMicro"): r["delta_vs_v18_pp"][key] = float("nan")
    assert nondegradation_gate(r)["pass"]
    r["selected"]["MovingMicro"] = 0.
    assert not nondegradation_gate(r)["pass"]
