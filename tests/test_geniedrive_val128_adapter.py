import json

import numpy as np
import pytest
import torch

from tools.external_baselines.geniedrive_contract import (
    build_manifest,
    load_manifest,
    prediction_filename,
    save_prediction,
    split_sample_id,
    token_from_occ_path,
    validate_prediction,
)
from tools.external_baselines.export_geniedrive_val128 import (
    confusion_matrix,
    official_binary_iou,
    official_semantic_miou,
)


def fake_index(count=3):
    return {
        "version": "prepared-test",
        "metadata": {"selection_contract": "scene-disjoint-test"},
        "entries": [
            {"sample_id": f"scene-{i:04d}:token{i:04d}", "shard": "x.pt", "index": i}
            for i in range(count)
        ],
    }


def test_manifest_preserves_sample_ids_and_tokens(tmp_path):
    manifest = build_manifest(fake_index(), expected_count=3)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = load_manifest(path)
    assert loaded["num_samples"] == 3
    assert loaded["num_scenes"] == 3
    assert loaded["entries"][1] == {
        "sample_id": "scene-0001:token0001",
        "scene_name": "scene-0001",
        "token": "token0001",
    }


def test_manifest_rejects_wrong_count_and_reused_scene():
    with pytest.raises(ValueError, match="expected 128"):
        build_manifest(fake_index(), expected_count=128)
    index = fake_index(2)
    index["entries"][1]["sample_id"] = "scene-0000:another-token"
    with pytest.raises(ValueError, match="one window per scene"):
        build_manifest(index, expected_count=2)


def test_occ_path_and_sample_id_parsing():
    assert split_sample_id("scene-0001:abc") == ("scene-0001", "abc")
    assert token_from_occ_path("data/nuscenes/gts/scene-0001/abc") == "abc"
    assert token_from_occ_path(r"data\gts\scene-0001\abc.npz") == "abc"


def test_prediction_validation_and_atomic_output(tmp_path):
    pred = np.zeros((6, 200, 200, 16), dtype=np.uint8)
    assert validate_prediction(pred).shape == pred.shape
    filename = save_prediction(tmp_path, "scene-1:token", pred, {"ckpt": "sha"})
    assert filename == prediction_filename("scene-1:token")
    payload = torch.load(tmp_path / filename, map_location="cpu", weights_only=False)
    assert payload["sample_id"] == "scene-1:token"
    assert tuple(payload["pred_occ"].shape) == (6, 200, 200, 16)
    assert not list(tmp_path.glob("*.tmp"))


def test_prediction_rejects_bad_shape_and_label():
    with pytest.raises(ValueError, match="shape"):
        validate_prediction(np.zeros((3, 200, 200, 16), dtype=np.uint8))
    bad = np.zeros((6, 200, 200, 16), dtype=np.int16)
    bad[0, 0, 0, 0] = 18
    with pytest.raises(ValueError, match="range"):
        validate_prediction(bad)


def test_native_metric_sanity_matches_official_class_conventions():
    gt = np.array([0, 0, 1, 1, 17, 17], dtype=np.uint8)
    pred = np.array([0, 1, 1, 1, 17, 0], dtype=np.uint8)
    semantic = confusion_matrix(pred, gt, 18)
    # IoU(class 0)=1/3, class 1=2/3; zero/absent classes and free are excluded.
    assert official_semantic_miou(semantic) == pytest.approx(50.0)
    binary = confusion_matrix(pred != 17, gt != 17, 2)
    assert official_binary_iou(binary) == pytest.approx(80.0)
