"""XY-only learned improvements: causal input, real training and real A1 tests."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch
from real_motion.v18_xy_trajectory import (
    INPUT_KEYS, PROTOCOL, INPUT_PROTOCOL, TrajectoryConfig, make_xy_model,
    replace_xy_outputs, xy_objective, xy_screen_gate,
)
from tools.real_motion import v18_xy_trajectory_common as common
from tools.real_motion import train_p0_f9_v18_xy_trajectory as trainer


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def inputs(n=4, dim=4):
    return dict(query=torch.randn(n,6,dim),context=torch.randn(n,dim),
        frame_motion=torch.ones(n,6,5),kta=torch.zeros(n,6,2),base_xy=torch.zeros(n,6,2))


def labels(n=4, hw=3):
    return dict(target_xy=torch.full((n,6,2),.3),valid=torch.ones(n,6,dtype=torch.bool),
        supervised=torch.ones(n,dtype=torch.bool),footprint=torch.ones(n,hw,hw),
        target_yaw=torch.zeros(n,6),base_yaw=torch.zeros(n,6),
        yaw_enabled=torch.ones(n,dtype=torch.bool),yaw_valid=torch.ones(n,6,dtype=torch.bool))


@pytest.mark.parametrize("kind",trainer.KINDS)
@pytest.mark.parametrize("n",(0,3))
def test_zero_initialization_is_exact_actual_bfloat_v18(kind,n):
    b = inputs(n); b["base_xy"] = torch.randn(n,6,2).bfloat16()
    m = make_xy_model(kind,TrajectoryConfig(dim=4,hidden=8))
    assert torch.equal(m(**b),b["base_xy"].float())


def test_joint_control_matches_integrated_capacity_and_initial_weights():
    cfg = TrajectoryConfig(dim=4,hidden=8)
    position,integrated = (make_xy_model(k,cfg) for k in trainer.KINDS[1:])
    integrated.load_state_dict(position.state_dict())
    assert sum(p.numel() for p in position.parameters()) == sum(p.numel() for p in integrated.parameters())
    with torch.no_grad():
        position.velocity.bias.fill_(2); integrated.velocity.bias.fill_(2)
    p,i = position(**inputs()),integrated(**inputs())
    assert torch.equal(p,torch.full_like(p,2))
    assert torch.equal(i,torch.arange(1,7)[None,:,None].expand(4,6,2).float())


def test_no_gt_inputs_and_missing_history_is_sanitized():
    cfg = TrajectoryConfig(dim=4,hidden=8); m = make_xy_model(trainer.KINDS[-1],cfg)
    with torch.no_grad(): m.velocity.weight.fill_(.1)
    b = inputs(); b["frame_motion"][...,4] = 0
    other = copy.deepcopy(b); other["frame_motion"][...,:4] = 1234
    assert torch.equal(m(**b),m(**other))
    with pytest.raises(TypeError): m(**b,target_xy=torch.zeros(4,6,2))
    other["query"][0,0,0] = float("nan")
    with pytest.raises(ValueError,match="finite"): m(**other)


def test_only_xy_changes_no_clamp_or_future_gt_gate():
    base = {"residual_xy_m":torch.zeros(2,6,2),"yaw_delta_rad":torch.randn(2,6),"existence_logits":torch.randn(2,6)}
    learned = torch.full((2,6,2),30.)
    out = replace_xy_outputs(base,learned)
    assert out["yaw_delta_rad"] is base["yaw_delta_rad"]
    assert out["existence_logits"] is base["existence_logits"]
    assert out["residual_xy_m"] is learned
    assert not base["residual_xy_m"].any()


def test_original_masked_loss_is_finite_and_differentiable_with_missing_gt():
    b,t = inputs(),labels(); t["valid"][0,1] = False
    t["target_xy"][0,1] = float("nan"); t["target_yaw"][0,1] = float("nan")
    m = make_xy_model(trainer.KINDS[-1],TrajectoryConfig(dim=4,hidden=8))
    loss,parts = xy_objective(m(**b),b,t); loss.backward()
    assert torch.isfinite(loss) and parts["valid_labels"] == 23
    assert torch.isfinite(m.velocity.weight.grad).all() and m.velocity.weight.grad.abs().sum() > 0
    t["valid"][:] = False
    with pytest.raises(ValueError,match="no valid"): xy_objective(m(**b),b,t)


def test_real_three_arm_training_reduces_synthetic_fit_and_selects_on_calibration():
    torch.manual_seed(9); b,t = inputs(),labels()
    bank = {k:v.numpy() for k,v in {**b,**t}.items()}
    baseline = common.error_summary(bank["base_xy"],bank["target_xy"],bank["valid"])
    models,best,history = trainer.train_variants(bank,bank,TrajectoryConfig(dim=4,hidden=8),
        torch.device("cpu"),updates=96,batch_size=4,seed=7)
    assert len(history) == 3
    for k,m in models.items():
        fit = trainer.label_fit(m,bank,torch.device("cpu"))
        assert fit["selection_score_m"] < baseline["selection_score_m"]
        assert best[k]["update"] == 96
        assert best[k]["calibration"]["selection_score_m"] == fit["selection_score_m"]
    # Truly frozen weights are not in any XY checkpoint.
    ck = trainer.checkpoint_payload({},models[trainer.KINDS[-1]],role="last_xy_diagnostic",update=96)
    assert set(ck["state_dict"]) == set(models[trainer.KINDS[-1]].state_dict())


def test_scene_split_and_metric_finite_contract():
    keys = [(f"train{i}",f"t{j}") for i in range(8) for j in range(2)]
    train,cal = trainer.select_train_calibration(keys,4,2,{"dev"})
    assert not {k[0] for k in train} & {k[0] for k in cal}
    with pytest.raises(RuntimeError,match="overlap"): trainer.select_train_calibration(keys,4,2,{"train0"})
    b = inputs(); target = b["base_xy"].clone(); valid = torch.ones(4,6,dtype=torch.bool)
    target[0,0] = float("nan"); valid[0,0] = False
    assert common.error_summary(b["base_xy"],target,valid)["all_six_ADE"]["count"] == 23
    valid[0,0] = True
    with pytest.raises(ValueError,match="nonfinite"): common.error_summary(b["base_xy"],target,valid)


def test_gate_rejects_zero_and_any_horizon_regression():
    row = dict(IoU=.1,mIoU=.4,MovingMicro=1.2,MovingMacro=.1)
    report = {"delta_vs_v18_pp":{**row,"per_horizon":{str(h):dict(row) for h in (1.,2.,3.)}}}
    assert xy_screen_gate(report)["pass"]
    report["delta_vs_v18_pp"]["per_horizon"]["3.0"]["MovingMacro"] = -.01
    assert not xy_screen_gate(report)["pass"]
    report["delta_vs_v18_pp"]["per_horizon"]["3.0"]["MovingMacro"] = None
    assert not xy_screen_gate(report)["pass"]
    row = dict.fromkeys(row,0.)
    assert not xy_screen_gate({"delta_vs_v18_pp":{**row,"per_horizon":{}}})["pass"]


def test_checkpoint_contract_rejects_failed_last_smoke_update0_and_wrong_baseline(tmp_path):
    cfg = TrajectoryConfig(dim=4,hidden=8)
    m = make_xy_model(trainer.KINDS[-1],cfg)
    identity = dict(protocol=PROTOCOL,input_protocol=INPUT_PROTOCOL,base_checkpoint_sha256="a"*64,
                    runtime_config_fingerprint="b"*64,mode="screen")
    ck = trainer.checkpoint_payload(identity,m,role="selected_xy_candidate",update=256,screen_pass=True)
    path = tmp_path/"xy.pt"; torch.save(ck,path)
    _,loaded = common.load_xy_adapter(path,"cpu",base_sha="a"*64,config_sha="b"*64)
    assert torch.equal(loaded(**inputs()),torch.zeros(4,6,2))
    for change in ({"screen_pass":False},{"checkpoint_role":"last_xy_diagnostic"},{"mode":"smoke"},{"selected_update":0}):
        torch.save({**ck,**change},path)
        with pytest.raises(RuntimeError,match="cannot be deployed"): common.load_xy_adapter(path,"cpu",base_sha="a"*64,config_sha="b"*64)
    torch.save(ck,path)
    with pytest.raises(RuntimeError,match="contract"): common.load_xy_adapter(path,"cpu",base_sha="c"*64,config_sha="b"*64)


def scene_record(scene="train0",token="t0"):
    n = 1; b = inputs(n); t = labels(n,20)
    center = torch.tensor([[1.5,1.5]])
    rec = dict(scene_name=scene,t0_token=token,future_tokens=tuple(f"f{i}" for i in range(6)),
        source_centroid_xy_t0_m=center,anchors_xy_t0_m=center[:,None]+b["kta"],
        kta_displacement_xy_m=b["kta"],frame_motion_features=b["frame_motion"],
        target_source_residual_xy_m=t["target_xy"],target_source_displacement_xy_m=t["target_xy"]+b["kta"],
        supervised_source=t["supervised"],se2_target_valid=t["valid"],
        target_source_mask_tube=torch.ones(n,6,20,20,dtype=torch.uint8),target_yaw_rad=t["target_yaw"],
        yaw_enabled=t["yaw_enabled"],yaw_label_valid=t["yaw_valid"],source_class_id=torch.full((n,),4))
    out = dict(residual_xy_m=b["base_xy"],yaw_delta_rad=t["base_yaw"],existence_logits=torch.zeros(n,6),
        future_transport_queries=b["query"],history_source_context=b["context"])
    return rec,out


def test_bank_prepares_once_and_whitelists_causal_features():
    rec,out = scene_record(); calls = []
    provider = SimpleNamespace(encode_record=lambda r:(calls.append(r) or out))
    bank = common.prepare_bank(provider,[rec])
    assert len(calls) == 1 and set(bank) == set(INPUT_KEYS) | set(common.LABEL_KEYS)
    assert set(common.causal_inputs(rec,out,"cpu")) == set(INPUT_KEYS)
    invalid = copy.deepcopy(rec); invalid["anchors_xy_t0_m"] += .1
    with pytest.raises(RuntimeError,match="anchor"): common.prepare_bank(provider,[invalid])
    with pytest.raises(RuntimeError,match="budget"): common.prepare_bank(provider,[rec],max_mib=0)


def test_real_frozen_encoder_latents_preserve_forward_and_disable_all_base_gradients(tmp_path):
    from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
    from real_motion.local_st_world_model_v18_se2 import LocalSpatialTemporalWorldModelV18SE2
    from real_motion.motion_transport import FEATURE_DIM
    cfg = LocalSTWMV17Config(d_model=16,semantic_dim=8,heads=4,blocks=1,decoder_blocks=1,tube_hw=20)
    model = LocalSpatialTemporalWorldModelV18SE2(cfg)
    rec,_ = scene_record()
    rec["features"] = torch.randn(1,FEATURE_DIM)
    rec["local_semantic_tube"] = torch.randint(0,18,(1,6,20,20),dtype=torch.uint8)
    device = torch.device("cpu")
    with patch.object(common.full,"_load_model",return_value=({},model,cfg)), \
         patch.object(common,"validate_clean_e14_checkpoint",return_value="a"*64):
        provider = common.FrozenXYV18(tmp_path/"base.pt","a"*64,None,device,1)
    out = provider.encode_record(rec)
    assert provider.latents_checked and out["future_transport_queries"].shape == (1,6,16)
    normal = common.runtime._model_forward(model,common.runtime._gpu_inputs(rec,device),device)
    assert all(torch.equal(v,out[k]) for k,v in normal.items())
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    # Bank owns normal copied arrays, not inference tensors that cannot be
    # saved for the new trainable head's backward.
    bank = common.prepare_bank(provider,[rec])
    m = make_xy_model(trainer.KINDS[-1],TrajectoryConfig(dim=16,hidden=8))
    b = {k:torch.from_numpy(v) for k,v in bank.items()}
    inp = {k:b[k] for k in INPUT_KEYS}
    loss,_ = xy_objective(m(**inp),inp,b); loss.backward()
    assert m.velocity.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())


def render_fixture():
    from real_motion.geometry import OccupancyGrid
    from real_motion.rigid_transport import RasterizedRigidComponent
    grid = OccupancyGrid(x_min=0,y_min=0,z_min=0,voxel_size=(1,1,1),shape_hwd=(12,12,2))
    pcfg = SimpleNamespace(grid=grid,free_label=17,frame_dt_s=.5)
    rec,out = scene_record("dev")
    comp = {"class_id":4,"voxel_indices":np.array([[1,1,0]]),"centroid_world":np.array([1.5,1.5,.5])}
    prior = RasterizedRigidComponent(4,comp["voxel_indices"],1)
    b = np.full(grid.shape_hwd,17,np.uint8); b[1,1,0] = 4; b[11,11,0] = 11
    state = dict(rec=rec,current=[comp],current_pose=np.eye(4),future_poses=[np.eye(4)]*6,
        source_world_points=[np.array([[1.5,1.5,.5]])],source_rel_xy=[np.zeros((1,2))],
        world_to_future=[np.eye(4)]*6,source_z_t0=np.array([.5]),gpu={},anchors=[b]*6,
        baseline_by_hi=[[prior]]*6,baseline_clear_flat_by_hi=[np.array([26])]*6)
    truth = b.copy(); truth[1,1,0] = 17; truth[2,1,0] = 4
    return rec,out,state,pcfg,dict(future_gt_occ=[truth]*6)


def test_actual_renderer_zero_identity_motion_only_and_deployment_no_future_gt():
    rec,out,state,pcfg,raw = render_fixture(); device = torch.device("cpu")
    provider = SimpleNamespace(device=device,pcfg=pcfg,strong=None,workers=1,model=None)
    observed_gt_flags = []
    def prepare(source,r,*,include_gt):
        observed_gt_flags.append(include_gt)
        return SimpleNamespace(scene_name="dev",t0_token="t0",future_tokens=rec["future_tokens"]),raw,state,out
    provider.prepare = prepare
    m = make_xy_model(trainer.KINDS[0],TrajectoryConfig(dim=4,hidden=8))
    predictions = common.render_outputs(state,pcfg,out)
    baseline = common.runtime._forecast_once(None,state,pcfg,None,device,precomputed_out=out)
    assert all(np.array_equal(predictions[i],baseline[h]) for i,h in enumerate((1,3,5)))
    with torch.no_grad(): m.delta.bias[:] = torch.tensor([1.,0.])
    _,learned = common.forecast_with_xy(provider,None,rec,m)
    assert observed_gt_flags == [False] and len(learned) == 6
    assert learned[5][2,1,0] == 4 and learned[5][1,1,0] == 17
    assert learned[5][11,11,0] == 11
    # Actual A1 compositor, raw metrics and MotionErrors, only external data
    # adapters / reference checks are mocked in this small synthetic fixture.
    with patch.object(common.runtime,"_stage_gpu_inputs"),patch.object(common.runtime,"_release_gpu_inputs"), \
         patch.object(common,"assert_forward_exact"),patch.object(common.runtime,"_exactness_check"), \
         patch.object(common,"gt_moving_support_sequence",return_value=[(np.ones_like(learned[0],bool),None)]*6):
        report = common.evaluate_dev(provider,SimpleNamespace(nusc=None),[rec],{"xy":m})
    assert observed_gt_flags == [False,True]
    assert report["variants"]["xy"]["delta_vs_v18_pp"]["MovingMicro"] > 0


def test_training_cli_smoke_writes_complete_bounded_summary_and_six_small_checkpoints(tmp_path):
    # Real train/calibration selection, bank, optimizer, loss, checkpoints,
    # actual renderer and summaries. Only workstation-unavailable datasets,
    # frozen network loader and first-window external checks are substituted.
    rec,out,state,pcfg,raw = render_fixture(); device = torch.device("cpu")
    train = [scene_record(f"train{i}",f"t{i}")[0] for i in range(8)]
    dev = [rec]
    provider = SimpleNamespace(dim=4,device=device,pcfg=pcfg,strong=None,workers=1,model=None,sha="a"*64,
        encode_record=lambda r:out,
        prepare=lambda source,r,include_gt:(SimpleNamespace(scene_name="dev",t0_token="t0",future_tokens=rec["future_tokens"]),raw,state,out))
    files = {k:tmp_path/k for k in ("train-cache","dev-cache","population-manifest","base-checkpoint","dev-info")}
    for p in files.values(): p.touch()
    manifest = {"parent_keys":[("dev",str(i)) for i in range(512)],
                "selected_key_fingerprint":trainer.DEV64_FP,"manifest_fingerprint":"b"*64}
    argv = ["train","--config",str(Path(__file__).resolve().parents[1]/"configs/real_motion_occfm.yaml"),
        "--dataroot",str(tmp_path),"--out-dir",str(tmp_path/"result"),"--device","cpu","--mode","smoke","--batch-size","2"]
    for k,p in files.items(): argv += ["--"+k,str(p)]
    with patch("sys.argv",argv),patch.object(trainer,"make_prepare_config",return_value=pcfg), \
         patch.object(trainer,"load_manifest",return_value=(manifest,[("dev","t0")]*64,None)), \
         patch.object(trainer,"load_cache",side_effect=[({},train),({},dev)]), \
         patch.object(trainer,"align_records",side_effect=lambda rows,keys:[next(r for r in rows if (r["scene_name"],r["t0_token"])==k) for k in keys]), \
         patch.object(trainer,"FrozenXYV18",return_value=provider),patch.object(trainer,"NuScenesWindowSource",return_value=SimpleNamespace(nusc=None)), \
         patch.object(common.runtime,"_stage_gpu_inputs"),patch.object(common.runtime,"_release_gpu_inputs"), \
         patch.object(common,"assert_forward_exact"),patch.object(common.runtime,"_exactness_check"), \
         patch.object(common,"gt_moving_support_sequence",return_value=[(np.ones_like(raw["future_gt_occ"][0],bool),None)]*6):
        trainer.main()
    result = tmp_path/"result"
    summary = json.loads((result/"summary.json").read_text(encoding="utf-8"))
    assert summary["updates"] == 8 and len(summary["dev"]["variants"]) == 6
    assert len(list(result.glob("*.pt"))) == 6
    assert not any(a["screen_pass"] for a in summary["arms"].values())
    assert "ONLY XY changed" in (result/"summary.txt").read_text(encoding="utf-8")
    for ck in result.glob("*.pt"):
        with pytest.raises(RuntimeError,match="cannot be deployed"):
            common.load_xy_adapter(ck,"cpu",base_sha="a"*64,config_sha=summary["runtime_config_fingerprint"])
