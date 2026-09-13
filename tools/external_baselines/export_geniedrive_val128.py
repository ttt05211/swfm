#!/usr/bin/env python3
"""Run official GenieDrive sequentially and stream only SWFM val128 predictions.

This script intentionally runs in the separate ``geniedrive-occ`` environment.
It avoids the official ``--out`` path, which retains every full-resolution result.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

SWFM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SWFM_ROOT))

from tools.external_baselines.geniedrive_contract import (
    EXPECTED_SHAPE,
    load_manifest,
    save_prediction,
    token_from_occ_path,
)

REPORT_FRAMES = (1, 3, 5)
REPORT_HORIZONS = (1.0, 2.0, 3.0)
OFFICIAL_MIOU_REFERENCE = (50.47, 41.47, 35.83)
OFFICIAL_VALID_SAMPLES = 4669
TESTED_GENIEDRIVE_REVISION = "da48a529ffbe14136688e9b7a56f5d1061c366c5"


def confusion_matrix(prediction, target, classes):
    pred = np.asarray(prediction).reshape(-1).astype(np.int64, copy=False)
    gt = np.asarray(target).reshape(-1).astype(np.int64, copy=False)
    valid = (gt >= 0) & (gt < classes) & (pred >= 0) & (pred < classes)
    return np.bincount(
        classes * gt[valid] + pred[valid], minlength=classes * classes
    ).reshape(classes, classes)


def per_class_iou(confusion):
    intersection = np.diag(confusion).astype(np.float64)
    union = confusion.sum(0) + confusion.sum(1) - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        return intersection / union


def official_semantic_miou(confusion):
    # GenieDrive's Metric_mIoU excludes free (17), and treats exact-zero IoU
    # as absent when forming the reported mean. Match that implementation.
    iou = per_class_iou(confusion)[:17]
    iou[iou == 0] = np.nan
    return float(np.nanmean(iou) * 100.0)


def official_binary_iou(confusion):
    # Official count_iou returns the non-empty class (index 1), not binary mIoU.
    return float(per_class_iou(confusion)[1] * 100.0)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geniedrive-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--config", default="occ_gen/configs/world_model/vae_e2e.py")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--native-reference-tolerance", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GenieDrive inference requires a CUDA GPU")

    genie_root = Path(args.geniedrive_root).resolve()
    occ_root = genie_root / "occ_gen"
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = genie_root / config_path
    checkpoint = Path(args.checkpoint).resolve()
    ann_file = Path(args.ann_file).resolve()
    for path in (occ_root, config_path, checkpoint, ann_file):
        if not path.exists():
            raise FileNotFoundError(path)
    genie_revision = subprocess.check_output(
        ["git", "-C", str(genie_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if genie_revision != TESTED_GENIEDRIVE_REVISION:
        raise RuntimeError(
            f"GenieDrive revision {genie_revision} is not the tested revision "
            f"{TESTED_GENIEDRIVE_REVISION}"
        )

    # Official configs and annotation paths are relative to occ_gen.
    os.chdir(str(occ_root))
    sys.path.insert(0, str(occ_root))

    import mmdet
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet.apis import set_random_seed
    from mmdet3d.datasets import build_dataloader, build_dataset
    from mmdet3d.models import build_model
    if mmdet.__version__ > "2.23.0":
        from mmdet.utils import compat_cfg, setup_multi_processes
    else:
        from mmdet3d.utils import compat_cfg, setup_multi_processes

    manifest = load_manifest(args.manifest)
    token_to_sample = {row["token"]: row["sample_id"] for row in manifest["entries"]}
    if len(token_to_sample) != manifest["num_samples"]:
        raise RuntimeError("manifest tokens are not unique")

    cfg = compat_cfg(Config.fromfile(str(config_path)))
    setup_multi_processes(cfg)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    if "test_mode" in cfg.model:
        cfg.model.test_mode = True
    cfg.data.test.test_mode = True
    cfg.data.test.ann_file = str(ann_file)
    cfg.gpu_ids = [args.gpu_id]
    # Match the official test.py default; deterministic cuDNN is opt-in there.
    set_random_seed(args.seed, deterministic=False)

    dataset = build_dataset(cfg.data.test)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    official_token_to_info = {str(info["token"]): info for info in dataset.data_infos}
    missing_from_annotation = sorted(set(token_to_sample) - set(official_token_to_info))
    if missing_from_annotation:
        raise RuntimeError(
            f"{len(missing_from_annotation)} val128 tokens are absent from the official "
            f"GenieDrive val annotation: {missing_from_annotation[:10]}"
        )
    missing_occ_files = []
    for token in token_to_sample:
        label_path = Path(official_token_to_info[token]["occ_path"]) / "labels.npz"
        if not label_path.is_file():
            missing_occ_files.append(str(label_path))
    if missing_occ_files:
        raise FileNotFoundError(
            f"missing Occ3D labels for {len(missing_occ_files)} selected samples; "
            f"first paths={missing_occ_files[:5]}"
        )
    selection_audit = {
        "passed": True,
        "official_annotation_samples": len(dataset),
        "selected_tokens": len(token_to_sample),
        "selected_tokens_found": len(token_to_sample),
        "selected_occ_labels_found": len(token_to_sample),
        "manifest_sha256": manifest["sample_ids_sha256"],
    }
    (output_dir / "selection_audit.json").write_text(
        json.dumps(selection_audit, indent=2), encoding="utf-8"
    )
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers,
        dist=False,
        shuffle=False,
        runner_type="IterBasedRunnerEval",
    )
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint_meta = load_checkpoint(model, str(checkpoint), map_location="cpu")
    model.CLASSES = checkpoint_meta.get("meta", {}).get("CLASSES", dataset.CLASSES)
    model = MMDataParallel(model, device_ids=[args.gpu_id])
    model.eval()

    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_sha = sha256_file(checkpoint)
    seen_tokens = set()
    saved = {}
    duplicate_hits = []
    processed = 0
    native_valid_samples = 0
    native_confusions = np.zeros((3, 18, 18), dtype=np.int64)
    native_binary_confusions = np.zeros((3, 2, 2), dtype=np.int64)
    started = time.time()

    with torch.no_grad():
        for data in data_loader:
            result_list = model(return_loss=False, rescale=True, **data)
            for result in result_list:
                predictions = np.asarray(result["pred_futu_semantics"])
                targets = np.asarray(result["targ_futu_semantics"])
                occ_paths = list(result["occ_path"])
                if predictions.ndim == 4:
                    predictions = predictions[None]
                    targets = targets[None]
                if predictions.shape[0] != len(occ_paths):
                    raise RuntimeError(
                        f"batch mismatch: predictions={predictions.shape}, occ_paths={len(occ_paths)}"
                    )
                for batch_index, occ_path in enumerate(occ_paths):
                    token = token_from_occ_path(occ_path)
                    if token in seen_tokens:
                        duplicate_hits.append(token)
                    seen_tokens.add(token)
                    occ_indices = result["occ_index"][batch_index]
                    if len(occ_indices) == len(set(occ_indices)):
                        native_valid_samples += 1
                        for report_index, frame_index in enumerate(REPORT_FRAMES):
                            native_confusions[report_index] += confusion_matrix(
                                predictions[batch_index, frame_index],
                                targets[batch_index, frame_index],
                                18,
                            )
                            pred_binary = (
                                predictions[batch_index, frame_index] != 17
                            ).astype(np.uint8)
                            target_binary = (
                                targets[batch_index, frame_index] != 17
                            ).astype(np.uint8)
                            native_binary_confusions[report_index] += confusion_matrix(
                                pred_binary, target_binary, 2
                            )
                    sample_id = token_to_sample.get(token)
                    if sample_id is None:
                        continue
                    pred = predictions[batch_index]
                    if tuple(pred.shape) != EXPECTED_SHAPE:
                        raise RuntimeError(
                            f"{token}: GenieDrive output shape {pred.shape} != {EXPECTED_SHAPE}"
                        )
                    filename = save_prediction(
                        prediction_dir,
                        sample_id,
                        pred,
                        metadata={
                            "checkpoint_sha256": checkpoint_sha,
                            "checkpoint": str(checkpoint),
                            "official_config": str(config_path),
                            "protocol": "gt_future_ego_plan_allowed",
                        },
                        overwrite=args.overwrite,
                    )
                    saved[sample_id] = filename
            processed += len(result_list)
            if processed % 100 == 0:
                print(
                    f"processed={processed}/{len(dataset)} selected={len(saved)}/{manifest['num_samples']}",
                    flush=True,
                )
            # GenieDrive's infinite eval sampler pads its iterator. The official
            # dataset length is the exact one-pass boundary; crossing it repeats
            # a scene and contaminates the stateful traversal.
            if processed >= len(dataset):
                break

    missing = [row for row in manifest["entries"] if row["sample_id"] not in saved]
    if duplicate_hits:
        raise RuntimeError(f"official val traversal repeated tokens: {duplicate_hits[:10]}")
    if missing:
        raise RuntimeError(
            f"only matched {len(saved)}/{manifest['num_samples']} val128 samples; "
            f"first missing={missing[:5]}"
        )

    native_miou = [official_semantic_miou(matrix) for matrix in native_confusions]
    native_iou = [official_binary_iou(matrix) for matrix in native_binary_confusions]
    native_deltas = [
        measured - reference
        for measured, reference in zip(native_miou, OFFICIAL_MIOU_REFERENCE)
    ]
    native_check_passed = all(
        abs(delta) <= args.native_reference_tolerance for delta in native_deltas
    ) and native_valid_samples == OFFICIAL_VALID_SAMPLES
    native_sanity = {
        "valid_samples": native_valid_samples,
        "horizons_seconds": list(REPORT_HORIZONS),
        "semantic_mIoU": native_miou,
        "binary_IoU": native_iou,
        "official_semantic_mIoU_reference": list(OFFICIAL_MIOU_REFERENCE),
        "official_valid_samples_reference": OFFICIAL_VALID_SAMPLES,
        "semantic_mIoU_delta": native_deltas,
        "tolerance_pp": args.native_reference_tolerance,
        "passed": native_check_passed,
    }
    (output_dir / "native_metric_sanity.json").write_text(
        json.dumps(native_sanity, indent=2), encoding="utf-8"
    )
    if not native_check_passed:
        raise RuntimeError(
            "native GenieDrive mIoU does not reproduce the checkpoint reference; "
            f"see {output_dir / 'native_metric_sanity.json'}"
        )

    entries = [
        {"sample_id": row["sample_id"], "token": row["token"], "file": saved[row["sample_id"]]}
        for row in manifest["entries"]
    ]
    index = {
        "version": "swfm_geniedrive_val128_predictions_v1",
        "source": "GenieDrive",
        "num_samples": len(entries),
        "manifest_sha256": manifest["sample_ids_sha256"],
        "checkpoint_sha256": checkpoint_sha,
        "geniedrive_revision": genie_revision,
        "official_val_samples_processed": processed,
        "elapsed_seconds": time.time() - started,
        "protocol": {
            "grid": [200, 200, 16],
            "labels": "Occ3D-nuScenes-18",
            "future_frames": 6,
            "frame_dt_seconds": 0.5,
            "future_ego_input": "official GenieDrive ground-truth future ego plan",
            "execution": "single-GPU official sequential sampler",
        },
        "native_metric_sanity": native_sanity,
        "entries": entries,
    }
    (prediction_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    (output_dir / "inference_summary.json").write_text(
        json.dumps({k: v for k, v in index.items() if k != "entries"}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in index.items() if k != "entries"}, indent=2))


if __name__ == "__main__":
    main()
