"""Frozen V18 latent bank, real geometry evaluation and XY-only deployment."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import torch

from real_motion.v18_xy_trajectory import (
    PROTOCOL,INPUT_PROTOCOL,INPUT_KEYS,TrajectoryConfig,make_xy_model,replace_xy_outputs,xy_screen_gate,
)
from real_motion.v18_motion_gap import MotionErrors,numpy
from real_motion.motion_transport import world_points_to_t0
from real_motion.source_evidence_audit import metric_count_delta,edit_quality
from real_motion.nuscenes_adapter import gt_moving_support_sequence
from real_motion.prepared import load_nuscenes_window_raw
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.v21_source_induction import select_scene_balanced_round_robin
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
from tools.real_motion import eval_p0_f9_v18_full_validation as full
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import Metrics,DYN,delta,assert_forward_exact,validate_clean_e14_checkpoint
from tools.real_motion.eval_p0_f9_source_evidence_audit import _render_current,_scene_delta
from tools.real_motion.train_p0_f9_v18_se2_clean import PROTOCOL as CLEAN_PROTOCOL


def select_train_calibration(keys,train_count,cal_count,dev_scenes):
    """Same frozen round-robin split rule, independent of generation code."""
    keys = tuple((str(a),str(b)) for a,b in keys)
    if len(keys) != len(set(keys)): raise RuntimeError("duplicate train record keys")
    if {k[0] for k in keys} & set(dev_scenes): raise RuntimeError("train/dev scene overlap")
    cal = select_scene_balanced_round_robin(keys,cal_count)
    cal_scenes = {c[0] for c in cal}
    rest = tuple(k for k in keys if k[0] not in cal_scenes)
    if len(rest) < train_count: raise RuntimeError("not enough scene-disjoint train/calibration windows")
    return select_scene_balanced_round_robin(rest,train_count),cal


class FrozenXYV18:
    def __init__(self,checkpoint,expected_sha,pcfg,device,workers):
        ck,self.model,cfg = full._load_model(checkpoint,CLEAN_PROTOCOL,device)
        self.sha = validate_clean_e14_checkpoint(ck,checkpoint,expected_sha)
        self.model.eval().requires_grad_(False)
        self.dim,self.pcfg,self.device,self.workers = cfg.d_model,pcfg,device,workers
        self.strong = StrongW2DetConfig(free_label=17)
        self.latents_checked = False

    def encode_record(self,record):
        # Per-window BF16 exactly as frozen deployment; never concatenate
        # multiple windows into a different V18 forward batch.
        inputs = runtime._gpu_inputs(record,self.device)
        out = runtime._model_forward(self.model,inputs,self.device,return_latents=True)
        if not self.latents_checked:
            assert_forward_exact(self.model,{"gpu":inputs},self.device)
            normal = runtime._model_forward(self.model,inputs,self.device)
            if any(not torch.equal(out[k],v) for k,v in normal.items()):
                raise RuntimeError("exposing frozen V18 latents changed its predictions")
            self.latents_checked = True
        return out

    def prepare(self,source,record,*,include_gt):
        window = window_from_record(record)
        raw = load_nuscenes_window_raw(source,window,self.pcfg,include_gt=include_gt,io_workers=self.workers)
        state = runtime._prepare_record(record,source,self.pcfg,self.strong,self.device,raw_window=raw)
        center = world_points_to_t0(np.asarray([c["centroid_world"] for c in state["current"]]).reshape(-1,3),state["current_pose"])[:,:2]
        if not np.allclose(center,numpy(record["source_centroid_xy_t0_m"]),rtol=0,atol=2e-4):
            raise RuntimeError("cache/actual source-centre identity mismatch")
        return window,raw,state,self.encode_record(record)


def causal_inputs(record,outputs,device):
    """Explicit whitelist; labels/validity/source supervision are NOT inputs."""
    def tensor(x): return torch.as_tensor(x,device=device).float()
    return {"query":tensor(outputs["future_transport_queries"]),"context":tensor(outputs["history_source_context"]),
            "frame_motion":tensor(record["frame_motion_features"]),"kta":tensor(record["kta_displacement_xy_m"]),
            "base_xy":tensor(outputs["residual_xy_m"])}


LABEL_KEYS = ("target_xy","valid","supervised","footprint","target_yaw","base_yaw","yaw_enabled","yaw_valid")


def prepare_bank(provider,records,*,progress=None,max_mib=256):
    chunks = defaultdict(list); size = 0
    for wi,rec in enumerate(records,1):
        outputs = provider.encode_record(rec)
        sup = numpy(rec["supervised_source"]).astype(bool)
        valid = numpy(rec["se2_target_valid"]).astype(bool)
        kta = numpy(rec["kta_displacement_xy_m"])
        center = numpy(rec["source_centroid_xy_t0_m"])
        if not np.allclose(numpy(rec["anchors_xy_t0_m"]),center[:,None]+kta,rtol=0,atol=2e-4):
            raise RuntimeError("XY cache anchor/source-centre contract mismatch")
        if not np.allclose((numpy(rec["target_source_residual_xy_m"])+kta)[valid],
                           numpy(rec["target_source_displacement_xy_m"])[valid],rtol=0,atol=2e-4):
            raise RuntimeError("XY cache residual/source-centre target contract mismatch")
        tube = numpy(rec["target_source_mask_tube"])
        if tube.shape[1:] != (6,20,20) or not np.isin(tube,(0,1)).all():
            raise RuntimeError("requires binary frozen V18 6x20x20 source footprint")
        take = sup & valid.any(1)
        inp = causal_inputs(rec,outputs,torch.device("cpu"))
        row = {k:numpy(v)[take].astype(np.float32,copy=True) for k,v in inp.items()}
        row.update({"target_xy":numpy(rec["target_source_residual_xy_m"])[take].astype(np.float32,copy=True),
            "valid":valid[take],"supervised":sup[take],
            "footprint":tube[take,-1].astype(np.uint8,copy=True),
            "target_yaw":numpy(rec["target_yaw_rad"])[take].astype(np.float32,copy=True),
            "base_yaw":numpy(outputs["yaw_delta_rad"])[take].astype(np.float32,copy=True),
            "yaw_enabled":numpy(rec["yaw_enabled"])[take].astype(bool,copy=True),
            "yaw_valid":numpy(rec["yaw_label_valid"])[take].astype(bool,copy=True)})
        size += sum(v.nbytes for v in row.values())
        if size > max_mib*2**20: raise RuntimeError("compact XY latent bank RAM budget exceeded")
        for k,v in row.items(): chunks[k].append(v)
        if wi == 1 or wi % 16 == 0 or wi == len(records):
            print(f"latent_bank={wi}/{len(records)} mib={size/2**20:.2f}",flush=True)
        if progress: progress({"event":"latent_bank","window":wi,"windows":len(records),"mib":size/2**20})
    bank = {k:np.concatenate(v) for k,v in chunks.items()}
    if not len(bank["query"]): raise RuntimeError("no valid supervised sources in selected population")
    return bank


def error_summary(pred,target,valid):
    pred,target,valid = numpy(pred),numpy(target),numpy(valid).astype(bool)
    if pred.shape != target.shape or valid.shape != pred.shape[:2]: raise ValueError("trajectory metric shape mismatch")
    if not np.isfinite(pred).all() or not np.isfinite(target[valid]).all():
        raise ValueError("nonfinite prediction/valid XY label in trajectory metric")
    e = np.linalg.norm(pred-target,axis=-1)
    def stats(v):
        return {"count":len(v),"mean_m":float(v.mean()) if len(v) else None,
                "median_m":float(np.median(v)) if len(v) else None,"p90_m":float(np.quantile(v,.9)) if len(v) else None}
    report = {"all_six_ADE":stats(e[valid]),"FDE_3s":stats(e[:,5][valid[:,5]]),
              "per_horizon":{str(.5*(h+1)):stats(e[:,h][valid[:,h]]) for h in range(6)}}
    report["selection_score_m"] = ((report["all_six_ADE"]["mean_m"]+report["FDE_3s"]["mean_m"])/2
        if report["FDE_3s"]["count"] else report["all_six_ADE"]["mean_m"])
    return report


def predict_bank(model,bank,device,batch_size=256):
    model.eval(); chunks = []
    with torch.inference_mode():
        for start in range(0,len(bank["query"]),batch_size):
            inputs = {k:torch.as_tensor(bank[k][start:start+batch_size],device=device) for k in INPUT_KEYS}
            chunks.append(model(**inputs).cpu().numpy())
    return np.concatenate(chunks)


def evaluate_dev(provider,source,records,models,*,progress=None):
    """ONE raw/V18/support pass, all selected/last learned variants rendered.

    There is no GT intervention, no dev checkpoint choice, no new source.
    """
    names = ("V18_BASE",*models)
    metrics = {k:Metrics() for k in names}
    scenes = defaultdict(lambda:{k:Metrics() for k in names})
    errors = {k:MotionErrors() for k in names}; quality = {k:defaultdict(int) for k in models}
    for wi,rec in enumerate(records,1):
        print(f"xy_dev={wi}/{len(records)} stage=shared_prepare_render",flush=True)
        window,raw,state,base = provider.prepare(source,rec,include_gt=True)
        if wi == 1:
            runtime._stage_gpu_inputs(state,provider.device)
            try:
                assert_forward_exact(provider.model,state,provider.device)
                runtime._exactness_check(provider.model,state,provider.pcfg,provider.strong,provider.device)
            finally: runtime._release_gpu_inputs(state)
        baseline = runtime._forecast_once(provider.model,state,provider.pcfg,provider.strong,provider.device,precomputed_out=base)
        all_outputs = {"V18_BASE":base}
        inputs = causal_inputs(rec,base,provider.device)
        with torch.inference_mode():
            for name,model in models.items():
                model.eval(); all_outputs[name] = replace_xy_outputs(base,model(**inputs))
            if wi == 1:
                # Full source population, not just supervised sources. Empty
                # correction is the exact baseline, including CLEAR/WRITE.
                zero = replace_xy_outputs(base,base["residual_xy_m"].float())
                identity = render_outputs(state,provider.pcfg,zero)
                if any(not np.array_equal(identity[i],baseline[h]) for i,h in enumerate((1,3,5))):
                    raise RuntimeError("zero XY correction is not voxel-exact V18")
        moving = gt_moving_support_sequence(source.nusc,window.t0_token,window.future_tokens,
            tuple(.5*(h+1) for h in range(6)),grid=provider.pcfg.grid,workers=provider.workers)
        before = [Metrics.counts(baseline[h],raw["future_gt_occ"][h],moving[h][0],17) for h in (1,3,5)]
        for name,out in all_outputs.items():
            if not torch.equal(out["yaw_delta_rad"],base["yaw_delta_rad"]) or not torch.equal(out["existence_logits"],base["existence_logits"]):
                raise RuntimeError("XY branch modified frozen yaw/existence")
            predictions = [baseline[h] for h in (1,3,5)] if name == "V18_BASE" else render_outputs(state,provider.pcfg,out)
            errors[name].update(rec,out)
            for ri,h in enumerate((1,3,5)):
                gt,pred,b = raw["future_gt_occ"][h],predictions[ri],baseline[h]
                counts = metric_count_delta(before[ri],b,pred,gt,moving[h][0],DYN)
                if wi == 1 and any(not np.array_equal(a,z) for a,z in zip(counts,Metrics.counts(pred,gt,moving[h][0],17))):
                    raise RuntimeError("XY changed-cell/full-grid counts differ")
                metrics[name].update(ri,counts=counts); scenes[window.scene_name][name].update(ri,counts=counts)
                if name in quality:
                    for k,v in edit_quality(b,pred,gt).items(): quality[name][k] += v
        if progress: progress({"event":"dev","window":wi,"windows":len(records)})
    baseline = metrics["V18_BASE"].compute()
    reports = {}
    for name in models:
        report = {"baseline":baseline,"metrics":metrics[name].compute(),"delta_vs_v18_pp":delta(metrics[name].compute(),baseline),
                  "scene_delta":_scene_delta(scenes,name),"motion_errors":errors[name].compute(),"edit_quality":dict(quality[name])}
        report["gate"] = xy_screen_gate(report); reports[name] = report
    return {"baseline":baseline,"baseline_motion_errors":errors["V18_BASE"].compute(),"variants":reports}


def render_outputs(state,pcfg,outputs):
    center = numpy(state["rec"]["anchors_xy_t0_m"])+numpy(outputs["residual_xy_m"])
    yaw = numpy(outputs["yaw_delta_rad"]).copy()
    enabled = numpy(state["rec"]["yaw_enabled"]).astype(bool)
    yaw[~enabled] = 0
    centers = {h:[runtime._target_world_from_xy_cached(xy,state["source_z_t0"][i],state["current_pose"])
                  for i,xy in enumerate(center[:,h])] for h in (1,3,5)}
    return _render_current(state,pcfg,centers,{h:yaw[:,h] for h in (1,3,5)})[0]


def load_xy_adapter(path,device,*,base_sha,config_sha,allow_failed_diagnostic=False):
    ck = torch.load(path,map_location="cpu",weights_only=False)
    if (ck.get("protocol") != PROTOCOL or ck.get("input_protocol") != INPUT_PROTOCOL
            or ck.get("base_checkpoint_sha256") != base_sha or ck.get("runtime_config_fingerprint") != config_sha
            or ck.get("checkpoint_role") not in ("selected_xy_candidate","last_xy_diagnostic")):
        raise RuntimeError("XY adapter/V18/input checkpoint contract mismatch")
    if (not ck.get("screen_pass") or ck["checkpoint_role"] != "selected_xy_candidate"
            or ck.get("mode") != "screen" or int(ck.get("selected_update",0)) <= 0) and not allow_failed_diagnostic:
        raise RuntimeError("failed/last XY diagnostic cannot be deployed as a passing module")
    model = make_xy_model(ck["kind"],TrajectoryConfig(**ck["model_config"])).to(device)
    model.load_state_dict(ck["state_dict"],strict=True); model.eval()
    return ck,model


def forecast_with_xy(provider,source,record,adapter):
    """Formal six-horizon inference: no future GT occupancy or annotation."""
    window,_,state,base = provider.prepare(source,record,include_gt=False)
    adapter.eval()
    with torch.inference_mode(): out = replace_xy_outputs(base,adapter(**causal_inputs(record,base,provider.device)))
    return window,runtime._forecast_once(provider.model,state,provider.pcfg,provider.strong,provider.device,precomputed_out=out)
