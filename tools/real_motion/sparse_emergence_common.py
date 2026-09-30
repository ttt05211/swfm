"""Shared single-pass preparation, calibration and frozen-V18 deployment.

GT responsibilities are audit-only; candidate support and model input are
constructed before target/annotation attribution. No V20 completion is used.
"""
from __future__ import annotations
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import numpy as np
import torch

from real_motion.sparse_emergence import (
    EmergenceConfig, PROTOCOL, FEATURE_PROTOCOL, REPORT, history_in_query_frame,
    prepare_inputs, target_sets, sample_rows, compose_points, nondegradation_gate,
)
from real_motion.sparse_emergence_model import SparseEmergenceDecoder
from real_motion.nuscenes_adapter import gt_moving_support_sequence
from real_motion.prepared import load_nuscenes_window_raw
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.source_evidence_audit import metric_count_delta
from real_motion.v21_source_induction import build_v21_targets
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
from tools.real_motion import eval_p0_f9_v18_full_validation as full
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import (
    Metrics, DYN, delta, assert_forward_exact, validate_clean_e14_checkpoint, future_shape_cache,
)
from tools.real_motion.train_p0_f9_v18_se2_clean import PROTOCOL as CLEAN_PROTOCOL


class FrozenGenerationV18:
    def __init__(self, checkpoint, expected_sha, pcfg, device, workers, config):
        ck, self.model, _ = full._load_model(checkpoint, CLEAN_PROTOCOL, device)
        self.sha = validate_clean_e14_checkpoint(ck, checkpoint, expected_sha)
        self.model.eval().requires_grad_(False)
        self.pcfg, self.device, self.workers, self.config = pcfg, device, workers, config
        self.strong = StrongW2DetConfig(free_label=17)
        self.checked = False

    def prepare(self, source, record, *, include_gt, horizons=REPORT):
        window = window_from_record(record)
        raw = load_nuscenes_window_raw(source, window, self.pcfg, include_gt=include_gt, io_workers=self.workers)
        state = runtime._prepare_record(record, source, self.pcfg, self.strong, self.device, raw_window=raw)
        runtime._stage_gpu_inputs(state, self.device)
        try:
            with torch.inference_mode():
                if not self.checked:
                    assert_forward_exact(self.model, state, self.device)
                    runtime._exactness_check(self.model, state, self.pcfg, self.strong, self.device)
                    self.checked = True
                outputs = runtime._model_forward(self.model, state["gpu"], self.device)
                baseline = runtime._forecast_once(self.model, state, self.pcfg, self.strong, self.device, precomputed_out=outputs)
        finally:
            runtime._release_gpu_inputs(state)
        def one(h):
            aligned = history_in_query_frame(raw["history_occ"], raw["history_poses"], raw["future_poses"][h], self.pcfg.grid)
            return h, prepare_inputs(aligned, baseline[h], self.pcfg.grid, .5*(h+1),
                np.linalg.inv(raw["history_poses"][-1]) @ raw["future_poses"][h], self.config)
        with ThreadPoolExecutor(max_workers=min(len(horizons), self.workers)) as pool:
            inputs = dict(pool.map(one, horizons))
        return window, raw, baseline, inputs


@dataclass
class EvalRow:
    scene: str
    horizon_index: int
    inputs: object
    baseline: np.ndarray
    gt: np.ndarray
    moving: np.ndarray
    novel: np.ndarray  # 1=unseen static; 2=certified birth; 3=certified dormant
    base_counts: tuple


def row_bytes(row):
    return sum(x.nbytes for x in (row.inputs.history, row.inputs.query, row.inputs.patch_xy,
        row.inputs.t0_labels, row.inputs.history_static_seen, row.baseline, row.gt, row.moving, row.novel))


def prepare_population(provider, source, records, *, training, rng, patches_per_horizon=24,
                       memory_limit_mib=2048, progress=None):
    rows, chunks = [], defaultdict(list)
    total_bytes = 0
    for wi, record in enumerate(records, 1):
        print(f"prepare={'train' if training else 'audit'} window={wi}/{len(records)}", flush=True)
        window, raw, base, inputs = provider.prepare(source, record, include_gt=True)
        if not training:
            moving = gt_moving_support_sequence(source.nusc, window.t0_token, window.future_tokens,
                tuple(.5*(h+1) for h in range(6)), grid=provider.pcfg.grid, workers=provider.workers)
            targets, _ = build_v21_targets(source, window, raw["history_occ"], raw["history_observed"],
                raw["history_poses"], grid=provider.pcfg.grid, strong_cfg=provider.strong, match_max_distance_m=4.)
            attrs, _ = future_shape_cache(source, window, raw, targets, provider.pcfg, 4., workers=provider.workers)
            responsibility = {t.instance_token: t.responsibility for t in targets}
        for ri, h in enumerate(REPORT):
            inp, gt = inputs[h], np.asarray(raw["future_gt_occ"][h], np.uint8)
            if training:
                target = target_sets(inp, base[h], gt, provider.config)
                ids, weight = sample_rows(target, patches_per_horizon, rng)
                pack = {"history": inp.history[ids], "query": inp.query[ids], "xyz": target["xyz"][ids],
                        "labels": target["labels"][ids].astype(np.int16), "mask": target["mask"][ids], "weight": weight}
                total_bytes += sum(a.nbytes for a in pack.values())
                for name, a in pack.items(): chunks[name].append(a)
            else:
                novel = np.zeros(gt.shape, np.uint8)
                novel[(gt < 17) & ~np.isin(gt, DYN) & ~inp.history_static_seen] = 1
                for token, at in attrs[h].items():
                    if at.ambiguous or at.voxel_indices is None: continue
                    code = 2 if responsibility[token] == "BIRTH" else 3
                    idx = np.asarray(at.voxel_indices, np.int64)
                    novel[tuple(idx.T)] = code
                # Own only the three evaluated frames: do not retain backing
                # arrays for unreported horizons behind a numpy view.
                b = np.asarray(base[h], np.uint8).copy()
                gt = gt.copy()
                counts = Metrics.counts(b, gt, moving[h][0], 17)
                rows.append(EvalRow(window.scene_name, ri, inp, b, gt, np.asarray(moving[h][0],bool).copy(), novel, counts))
                total_bytes += row_bytes(rows[-1])
            if total_bytes > memory_limit_mib*2**20:
                raise RuntimeError(f"population RAM budget exceeded: {total_bytes/2**20:.1f} MiB; do not create a dense disk cache")
        if progress: progress({"event": "prepare", "training": training, "window": wi, "windows": len(records), "mib": total_bytes/2**20})
    if not training: return rows
    bank = {key: np.concatenate(value) for key,value in chunks.items()}
    if not len(bank["history"]): raise RuntimeError("no causal train candidates")
    return bank


def predict_points(model, inputs, device, batch_size=64):
    if batch_size < 1: raise ValueError("invalid prediction batch size")
    model.eval()
    xyz, sem, score = [], [], []
    with torch.inference_mode():
        for start in range(0, len(inputs.history), batch_size):
            out = model(torch.as_tensor(inputs.history[start:start+batch_size], device=device),
                        torch.as_tensor(inputs.query[start:start+batch_size], device=device))
            prob, cls = out["semantic_logits"].softmax(-1).max(-1)
            xyz.append(out["xyz"].cpu().numpy()); sem.append(cls.cpu().numpy())
            score.append((prob * out["presence_logits"].sigmoid()[:,None]).cpu().numpy())
    k = model.config.queries*model.config.points_per_query
    return ((np.concatenate(xyz),np.concatenate(sem),np.concatenate(score)) if xyz else
            (np.empty((0,k,3),np.float32),np.empty((0,k),np.int64),np.empty((0,k),np.float32)))


def evaluate_points(model, rows, device, thresholds, *, batch_size=64):
    """Decode ONCE per row; reuse predictions for fixed threshold candidates.

    Count only changed cells, not 17 full-grid scans for each threshold.
    GT is used after deployment-identical composition, never to hide FP.
    """
    accum = {t: (Metrics(), Metrics(), defaultdict(int), defaultdict(lambda: (Metrics(),Metrics()))) for t in thresholds}
    for row in rows:
        points = predict_points(model, row.inputs, device, batch_size)
        empty = compose_points(row.baseline, row.inputs, *points, None, model.config)
        if not np.array_equal(empty, row.baseline): raise RuntimeError("empty proposal != frozen V18")
        for t,(baseline,selected,q,scenes) in accum.items():
            pred = compose_points(row.baseline, row.inputs, *points, t, model.config)
            counts = metric_count_delta(row.base_counts, row.baseline, pred, row.gt, row.moving, DYN)
            baseline.update(row.horizon_index, counts=row.base_counts); selected.update(row.horizon_index, counts=counts)
            sb,sp = scenes[row.scene]; sb.update(row.horizon_index,counts=row.base_counts); sp.update(row.horizon_index,counts=counts)
            added = (pred != row.baseline)
            correct = added & (pred == row.gt)
            q["added"] += int(added.sum()); q["semantic_tp"] += int(correct.sum())
            q["occupied_tp"] += int((added & (row.gt < 17)).sum())
            q["damaged_occupied_v18"] += int((added & (row.baseline < 17)).sum())
            for code,name in ((1,"unseen_static"),(2,"birth"),(3,"dormant")):
                q[name+"_semantic_tp"] += int((correct & (row.novel == code)).sum())
    reports = {}
    for t,(baseline,selected,q,scenes) in accum.items():
        b,s = baseline.compute(),selected.compute()
        scene_delta = {scene: sp.compute()["mIoU"]-sb.compute()["mIoU"] for scene,(sb,sp) in scenes.items()}
        quality = dict(q)
        quality["semantic_precision"] = quality["semantic_tp"]/quality["added"] if quality["added"] else None
        quality["occupancy_precision"] = quality["occupied_tp"]/quality["added"] if quality["added"] else None
        report = {"threshold": t,"baseline":b,"selected":s,"delta_vs_v18_pp":delta(s,b),"quality":quality,
                  "scene_delta":{"positive":sum(x>1e-9 for x in scene_delta.values()),
                                 "negative":sum(x< -1e-9 for x in scene_delta.values()),"by_scene":scene_delta}}
        report["gate"] = nondegradation_gate(report); reports[t] = report
    return reports


def select_calibration_threshold(reports):
    candidates = [r for t,r in reports.items() if t is not None and r["gate"]["pass"]]
    if not candidates: return None
    # Rank train-calibration ONLY. At a tie prefer the stricter threshold.
    return max(candidates,key=lambda r:(r["gate"]["generated_semantic_tp"],r["delta_vs_v18_pp"]["mIoU"],r["threshold"]))["threshold"]


def candidate_scope_ceiling(rows, config):
    """Audit-only simultaneous scope/budget check; NOT inference proposals.

    Local residual scope includes errors on observed-source transport too;
    novel quality is therefore reported separately, not mislabeled birth.
    This ceiling is not used to tune candidates or choose a dev threshold.
    """
    from dataclasses import replace
    baseline,scope,budget = Metrics(),Metrics(),Metrics()
    q = defaultdict(int)
    budget_cfg = replace(config,target_points=config.queries*config.points_per_query)
    for row in rows:
        p = row.baseline.copy(); cells = config.patch_cells
        for x,y in row.inputs.patch_xy:
            view = p[x:x+cells,y:y+cells]; gt = row.gt[x:x+cells,y:y+cells]
            keep = (view == 17) & (gt < 17); view[keep] = gt[keep]
        target = target_sets(row.inputs,row.baseline,row.gt,budget_cfg)
        # Mask padding through confidence=0 and threshold=1. Ground truth
        # point coordinates are used ONLY in this named oracle diagnostic.
        small = compose_points(row.baseline,row.inputs,target["xyz"],target["labels"],
            target["mask"].astype(np.float32),1.,config)
        baseline.update(row.horizon_index,counts=row.base_counts)
        scope.update(row.horizon_index,counts=metric_count_delta(row.base_counts,row.baseline,p,row.gt,row.moving,DYN))
        budget.update(row.horizon_index,counts=metric_count_delta(row.base_counts,row.baseline,small,row.gt,row.moving,DYN))
        for code,name in ((1,"unseen_static"),(2,"birth"),(3,"dormant")):
            q[name+"_scope_tp"] += int(((p != row.baseline) & (row.novel == code)).sum())
            q[name+"_budget_tp"] += int(((small != row.baseline) & (row.novel == code)).sum())
    b = baseline.compute()
    return {"scope_gt_delta_pp":delta(scope.compute(),b),"budget_gt_delta_pp":delta(budget.compute(),b),
            "novel_quality":dict(q),"audit_only":True,"used_for_dev_tuning":False}


def load_emergence(path, device, *, base_sha, config_sha, allow_failed_diagnostic=False):
    ck = torch.load(path,map_location="cpu",weights_only=False)
    if (ck.get("protocol") != PROTOCOL or ck.get("feature_protocol") != FEATURE_PROTOCOL
            or ck.get("checkpoint_role") != "auxiliary_generation_only"
            or ck.get("base_checkpoint_sha256") != base_sha or ck.get("runtime_config_fingerprint") != config_sha):
        raise RuntimeError("emergence/V18/input checkpoint contract mismatch")
    if not ck.get("screen_pass") and not allow_failed_diagnostic:
        raise RuntimeError("failed screen checkpoint cannot be used as a successful emergence module")
    if ck.get("screen_pass") and ck.get("threshold") is None:
        raise RuntimeError("accepted emergence checkpoint cannot have an abstain threshold")
    t = ck.get("threshold")
    if t is not None and (not isinstance(t,(int,float)) or not np.isfinite(t) or not 0 <= t <= 1):
        raise RuntimeError("invalid deployment confidence threshold")
    model = SparseEmergenceDecoder(EmergenceConfig(**ck["model_config"])).to(device)
    model.load_state_dict(ck["state_dict"],strict=True); model.eval()
    return ck,model


def forecast_with_emergence(provider, source, record, model, threshold, batch_size=64):
    """All six future states; include_gt=False, no future annotation access."""
    window,_,base,inputs = provider.prepare(source,record,include_gt=False,horizons=tuple(range(6)))
    return window,[compose_points(base[h],inputs[h],*predict_points(model,inputs[h],provider.device,batch_size),
                                  threshold,model.config) for h in range(6)]
