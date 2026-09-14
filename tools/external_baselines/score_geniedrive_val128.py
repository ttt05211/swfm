#!/usr/bin/env python3
"""Score GenieDrive val128 under the frozen Moving-mIoU v2 contract.

This entrypoint is intentionally Python 3.8 / PyTorch 1.13 compatible so the
external baseline does not need the main SWFM training environment.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.external_baselines.geniedrive_contract import load_manifest, validate_prediction

DYNAMIC_CLASSES = (2, 3, 4, 5, 6, 7, 9, 10)
REPORT = ((1.0, 1), (2.0, 3), (3.0, 5))


class IoUAccumulator(object):
    def __init__(self, classes):
        self.classes = tuple(int(value) for value in classes)
        self.intersection = {value: 0 for value in self.classes}
        self.union = {value: 0 for value in self.classes}

    def update(self, prediction, target, mask=None):
        prediction = np.asarray(prediction)
        target = np.asarray(target)
        if prediction.shape != target.shape:
            raise ValueError("prediction and target shapes differ")
        if mask is None:
            mask = np.ones(target.shape, dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != target.shape:
                raise ValueError("metric mask shape differs from occupancy shape")
        for class_id in self.classes:
            pred_class = (prediction == class_id) & mask
            target_class = (target == class_id) & mask
            self.intersection[class_id] += int((pred_class & target_class).sum())
            self.union[class_id] += int((pred_class | target_class).sum())

    def compute(self):
        per_class = {}
        values = []
        for class_id in self.classes:
            union = self.union[class_id]
            if union:
                value = 100.0 * self.intersection[class_id] / union
                values.append(value)
                per_class[str(class_id)] = value
            else:
                per_class[str(class_id)] = None
        return {
            "mIoU": float(np.mean(values)) if values else None,
            "per_class": per_class,
        }


def compose_kta(sample, frame_index):
    static = np.asarray(sample["static_future_occ"])[frame_index]
    kta = np.asarray(sample["kta_future_occ"])[frame_index]
    protected = np.asarray(sample["confident_static_future_mask"])[frame_index].astype(bool)
    support = np.asarray(sample["generation_support_occ"])[frame_index].astype(bool)
    if support.ndim == static.ndim - 1:
        support = np.broadcast_to(support[..., None], static.shape)
    if protected.shape != static.shape:
        protected = np.broadcast_to(protected, static.shape)
    dynamic_write = np.isin(kta, np.asarray(DYNAMIC_CLASSES))
    writable = (~protected) & support & dynamic_write
    output = static.copy()
    output[writable] = kta[writable]
    return output


def evaluation_arrays(sample):
    """Normalize full-prepared and compact P0-F9 evaluation payloads."""
    if "future_gt_occ" in sample and "gt_moving_support" in sample:
        return {
            "target": np.asarray(sample["future_gt_occ"]),
            "moving_support": np.asarray(sample["gt_moving_support"]).astype(bool),
            "baseline_name": "KTA_composed_baseline",
            "baseline": np.stack(
                [compose_kta(sample, frame_index) for frame_index in range(6)], axis=0
            ),
        }
    if "eval_future_gt_occ" in sample and "eval_gt_moving_support" in sample:
        baseline = sample.get("eval_strong_anchor_occ")
        return {
            "target": np.asarray(sample["eval_future_gt_occ"]),
            "moving_support": np.asarray(sample["eval_gt_moving_support"]).astype(bool),
            "baseline_name": (
                "Strong-W2Det_baseline" if baseline is not None else None
            ),
            "baseline": np.asarray(baseline) if baseline is not None else None,
        }
    raise KeyError(
        "reference cache sample lacks full prepared targets and compact P0-F9 eval payload"
    )


def summarized(accumulators):
    per_horizon = {
        str(horizon): accumulators[horizon].compute() for horizon, _ in REPORT
    }
    values = [per_horizon[str(horizon)]["mIoU"] for horizon, _ in REPORT]
    return {"mIoU": float(np.mean(values)), "per_horizon": per_horizon}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-cache", "--prepared", dest="reference_cache", required=True
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reference_root = Path(args.reference_cache)
    reference_index = json.loads(
        (reference_root / "index.json").read_text(encoding="utf-8")
    )
    manifest = load_manifest(args.manifest)
    manifest_ids = [row["sample_id"] for row in manifest["entries"]]
    reference_ids = [str(row["sample_id"]) for row in reference_index["entries"]]
    if reference_ids != manifest_ids:
        raise RuntimeError("reference cache index no longer matches the frozen val128 manifest")

    prediction_root = Path(args.pred_dir)
    prediction_index = json.loads(
        (prediction_root / "index.json").read_text(encoding="utf-8")
    )
    prediction_files = {
        str(row["sample_id"]): prediction_root / row["file"]
        for row in prediction_index["entries"]
    }
    missing = [sample_id for sample_id in manifest_ids if sample_id not in prediction_files]
    if missing:
        raise RuntimeError("missing predictions: %s" % missing[:10])

    overall = {horizon: IoUAccumulator(range(17)) for horizon, _ in REPORT}
    dynamic = {horizon: IoUAccumulator(DYNAMIC_CLASSES) for horizon, _ in REPORT}
    moving = {horizon: IoUAccumulator(DYNAMIC_CLASSES) for horizon, _ in REPORT}
    baseline_overall = {horizon: IoUAccumulator(range(17)) for horizon, _ in REPORT}
    baseline_dynamic = {horizon: IoUAccumulator(DYNAMIC_CLASSES) for horizon, _ in REPORT}
    baseline_moving = {horizon: IoUAccumulator(DYNAMIC_CLASSES) for horizon, _ in REPORT}
    baseline_name = None

    cached_shard_name = None
    cached_shard = None
    for entry in reference_index["entries"]:
        if entry["shard"] != cached_shard_name:
            cached_shard = torch.load(
                str(reference_root / entry["shard"]), map_location="cpu"
            )
            cached_shard_name = entry["shard"]
        sample = cached_shard[entry["index"]]
        sample_id = str(sample["sample_id"])
        prediction_payload = torch.load(
            str(prediction_files[sample_id]), map_location="cpu"
        )
        prediction = validate_prediction(prediction_payload["pred_occ"])
        arrays = evaluation_arrays(sample)
        if arrays["baseline_name"] is not None:
            if baseline_name is None:
                baseline_name = arrays["baseline_name"]
            elif baseline_name != arrays["baseline_name"]:
                raise RuntimeError("reference cache mixes baseline payload contracts")
        for horizon, frame_index in REPORT:
            target = arrays["target"][frame_index]
            support = arrays["moving_support"][frame_index]
            overall[horizon].update(prediction[frame_index], target)
            dynamic[horizon].update(prediction[frame_index], target)
            moving[horizon].update(prediction[frame_index], target, support)
            if arrays["baseline"] is not None:
                baseline = arrays["baseline"][frame_index]
                baseline_overall[horizon].update(baseline, target)
                baseline_dynamic[horizon].update(baseline, target)
                baseline_moving[horizon].update(baseline, target, support)

    report = {
        "version": "swfm_geniedrive_val128_score_v1",
        "num_predictions_used": len(manifest_ids),
        "complete_prediction_set": True,
        "GenieDrive": {
            "overall": summarized(overall),
            "dynamic": summarized(dynamic),
            "Moving-mIoU_v2": summarized(moving),
        },
        "reference_cache": {
            "root": str(reference_root.resolve()),
            "version": reference_index.get("version"),
        },
        "prediction_provenance": {
            key: value
            for key, value in prediction_index.items()
            if key != "entries"
        },
    }
    if baseline_name is not None:
        report[baseline_name] = {
            "overall": summarized(baseline_overall),
            "dynamic": summarized(baseline_dynamic),
            "Moving-mIoU_v2": summarized(baseline_moving),
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["GenieDrive"], indent=2))


if __name__ == "__main__":
    main()
