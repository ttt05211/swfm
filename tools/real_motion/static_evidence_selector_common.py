"""Shared frozen-V18 preparation and formal selector evaluation."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from real_motion.nuscenes_adapter import gt_moving_support_sequence
from real_motion.prepared import load_nuscenes_window_raw
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.static_evidence_selector import (
    PROTOCOL, FEATURE_PROTOCOL, THRESHOLD, SelectorInputs, prepare_selector_inputs,
    compose_selected_static, sparse_static_counts, acceptance_gate,
)
from real_motion.static_evidence_selector_model import SelectorConfig, StaticEvidenceSelector
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
from tools.real_motion import eval_p0_f9_v18_full_validation as full
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import (
    Metrics, delta, assert_forward_exact, validate_clean_e14_checkpoint,
)
from tools.real_motion.train_p0_f9_v18_se2_clean import PROTOCOL as CLEAN_PROTOCOL

REPORT = (1, 3, 5)
CLEAN_SHA256 = "ffaa54a76d07b4581e68046508d841bac798396533155162a5223ba155b3af73"


def finite_json(x):
    if isinstance(x, dict): return {str(k): finite_json(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [finite_json(v) for v in x]
    if isinstance(x, np.generic): return finite_json(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(finite_json(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def atomic_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


class FrozenV18:
    def __init__(self, checkpoint, expected_sha, pcfg, device, workers):
        ck, self.model, _ = full._load_model(checkpoint, CLEAN_PROTOCOL, device)
        self.sha = validate_clean_e14_checkpoint(ck, checkpoint, expected_sha)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.pcfg, self.device, self.workers = pcfg, device, workers
        self.strong = StrongW2DetConfig(free_label=pcfg.free_label)
        self.checked = False

    def prepare(self, source, record, *, include_gt):
        window = window_from_record(record)
        raw = load_nuscenes_window_raw(source, window, self.pcfg, include_gt=include_gt, io_workers=self.workers)
        state = runtime._prepare_record(record, source, self.pcfg, self.strong, self.device, raw_window=raw)
        runtime._stage_gpu_inputs(state, self.device)
        try:
            if not self.checked:
                assert_forward_exact(self.model, state, self.device)
                runtime._exactness_check(self.model, state, self.pcfg, self.strong, self.device)
                self.checked = True
            outputs = runtime._model_forward(self.model, state["gpu"], self.device)
            baseline = runtime._forecast_once(self.model, state, self.pcfg, self.strong, self.device, precomputed_out=outputs)
        finally:
            runtime._release_gpu_inputs(state)
        inputs, memory = prepare_selector_inputs(raw["history_occ"], raw["history_observed"],
            raw["history_poses"], raw["future_poses"], baseline, grid=self.pcfg.grid, workers=self.workers)
        return window, raw, baseline, inputs, memory


@dataclass
class DevRow:
    scene: str
    horizon_index: int
    inputs: SelectorInputs
    gt_sparse: np.ndarray
    base_counts: tuple


def prepare_dev(provider, source, records, progress=None):
    rows = []
    for wi, record in enumerate(records, 1):
        print(f"prepare_dev={wi}/{len(records)} stage=load_forecast_features", flush=True)
        window, raw, baseline, inputs, _ = provider.prepare(source, record, include_gt=True)
        moving = gt_moving_support_sequence(source.nusc, window.t0_token, window.future_tokens,
            tuple(.5 * (h + 1) for h in range(6)), grid=provider.pcfg.grid, workers=provider.workers)
        for ri, h in enumerate(REPORT):
            gt = np.asarray(raw["future_gt_occ"][h], np.uint8)
            counts = Metrics.counts(baseline[h], gt, moving[h][0], 17)
            if wi == 1:
                # All-reject is deployment identity; sparse counts equal full
                # renderer counts for ALL patches, including mistakes and FP.
                no_edit = compose_selected_static(baseline[h], inputs[h], np.zeros(len(inputs[h].patch_ids)))
                if not np.array_equal(no_edit, baseline[h]):
                    raise RuntimeError("all-reject selector differs from V18")
                all_add = compose_selected_static(baseline[h], inputs[h], np.ones(len(inputs[h].patch_ids)))
                exact = Metrics.counts(all_add, gt, moving[h][0], 17)
                sparse = sparse_static_counts(counts, inputs[h].labels, gt.reshape(-1)[inputs[h].voxel_flat],
                                              np.ones(len(inputs[h].labels), bool))
                if any(not np.array_equal(a, b) for a, b in zip(exact, sparse)):
                    raise RuntimeError("static sparse metrics differ from full renderer")
            rows.append(DevRow(window.scene_name, ri, inputs[h], gt.reshape(-1)[inputs[h].voxel_flat].copy(), counts))
        if progress:
            progress({"event": "prepare_dev", "window": wi, "windows": len(records)})
    return rows


def predict_probabilities(model, inputs, device, batch_size=512):
    if batch_size < 1:
        raise ValueError("inference batch size must be positive")
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(inputs.patch_ids), batch_size):
            history = torch.from_numpy(inputs.history[start:start + batch_size]).to(device)
            context = torch.from_numpy(inputs.context[start:start + batch_size]).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(history, context)
            result.append(torch.sigmoid(logits.float()).cpu().numpy())
    probability = np.concatenate(result) if result else np.empty(0, np.float32)
    if not np.isfinite(probability).all():
        raise RuntimeError("nonfinite selector output")
    return probability


def evaluate_selector(model, rows, device, batch_size=512):
    baseline, selected = Metrics(), Metrics()
    scenes = defaultdict(lambda: (Metrics(), Metrics()))
    quality = defaultdict(int)
    for row in rows:
        probability = predict_probabilities(model, row.inputs, device, batch_size)
        patch_take = probability >= THRESHOLD
        voxel_take = patch_take[row.inputs.voxel_patch_rows]
        counts = sparse_static_counts(row.base_counts, row.inputs.labels, row.gt_sparse, voxel_take)
        if any(not np.array_equal(a, b) for a, b in zip(counts[4:], row.base_counts[4:])):
            raise RuntimeError("selector changed frozen Moving counts")
        baseline.update(row.horizon_index, counts=row.base_counts)
        selected.update(row.horizon_index, counts=counts)
        sb, ss = scenes[row.scene]
        sb.update(row.horizon_index, counts=row.base_counts); ss.update(row.horizon_index, counts=counts)
        quality["candidate_patches"] += len(probability)
        quality["accepted_patches"] += int(patch_take.sum())
        quality["added"] += int(voxel_take.sum())
        quality["added_occ_tp"] += int((voxel_take & (row.gt_sparse != 17)).sum())
        quality["added_semantic_tp"] += int((voxel_take & (row.gt_sparse == row.inputs.labels)).sum())
        quality["damaged"] += int((voxel_take & (row.gt_sparse == 17)).sum())
    b, s = baseline.compute(), selected.compute()
    d = delta(s, b)
    scene_delta = {k: ss.compute()["mIoU"] - sb.compute()["mIoU"] for k, (sb, ss) in scenes.items()}
    q = dict(quality)
    q["addition_semantic_precision"] = q.get("added_semantic_tp", 0) / q["added"] if q.get("added") else None
    q["addition_occ_precision"] = q.get("added_occ_tp", 0) / q["added"] if q.get("added") else None
    return {"baseline": b, "selected": s, "delta_vs_v18_pp": d, "gate": acceptance_gate(d),
            "quality": q, "scene_delta": {"scenes": len(scenes), "positive": sum(v > 0 for v in scene_delta.values()),
                "negative": sum(v < 0 for v in scene_delta.values()), "zero": sum(v == 0 for v in scene_delta.values()),
                "by_scene": scene_delta}}


def load_selector(path, device, *, base_sha=None, config_sha=None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("protocol") != PROTOCOL or ck.get("feature_protocol") != FEATURE_PROTOCOL or ck.get("threshold") != THRESHOLD:
        raise RuntimeError("selector checkpoint contract mismatch")
    if base_sha is not None and ck.get("base_checkpoint_sha256") != base_sha:
        raise RuntimeError("selector/V18 checkpoint mismatch")
    if config_sha is not None and ck.get("runtime_config_fingerprint") != config_sha:
        raise RuntimeError("selector/runtime configuration mismatch")
    model = StaticEvidenceSelector(SelectorConfig(**ck["model_config"])).to(device)
    model.load_state_dict(ck["state_dict"], strict=True)
    return ck, model


def forecast_with_selector(provider, source, record, selector, batch_size=512):
    """Formal six-horizon deployment: never reads future occupancy labels."""
    window, _, baseline, inputs, _ = provider.prepare(source, record, include_gt=False)
    predictions = [compose_selected_static(b, x, predict_probabilities(selector, x, provider.device, batch_size))
                   for b, x in zip(baseline, inputs)]
    return window, predictions


def bank_fingerprint(arrays):
    digest = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        a = np.ascontiguousarray(value)
        digest.update(key.encode()); digest.update(str((a.shape, a.dtype.str)).encode())
        digest.update(memoryview(a).cast("B"))
    return digest.hexdigest()
