import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from real_motion.cache_pipeline import bounded_ordered_parallel_map
from real_motion.msp_wm_cache import (
    MSP_WM_CACHE_VERSION_V3,
    MSPWorldModelCacheDataset,
)


def test_bounded_parallel_map_preserves_input_order():
    def work(x):
        # Finish later inputs earlier to ensure completion order differs.
        time.sleep((5 - x) * 0.001)
        return x * 10

    out = list(bounded_ordered_parallel_map(
        work,
        range(6),
        max_workers=3,
        max_in_flight=4,
    ))
    assert out == [0, 10, 20, 30, 40, 50]


def test_bounded_parallel_map_serial_fallback_and_validation():
    assert list(bounded_ordered_parallel_map(
        lambda x: x + 1,
        [1, 2, 3],
        max_workers=1,
        max_in_flight=1,
    )) == [2, 3, 4]
    with pytest.raises(ValueError):
        list(bounded_ordered_parallel_map(lambda x: x, [1], max_workers=0, max_in_flight=1))
    with pytest.raises(ValueError):
        list(bounded_ordered_parallel_map(lambda x: x, [1], max_workers=1, max_in_flight=0))


def _tiny_sample(sid):
    z = torch.zeros(1, 1, 2, 2)
    return {
        "sample_id": sid,
        "scene_name": "scene",
        "full_history_latent": z.clone(),
        "anchor_future_latent": z.clone(),
        "repair_target_latent": z.clone(),
        "window_origins": torch.zeros(2, 2, dtype=torch.long),
        "window_valid": torch.ones(2, dtype=torch.bool),
        "msp_write_support_latent": torch.zeros(1, 2, 2, dtype=torch.bool),
        "trajectory": torch.zeros(12, 2),
    }


def test_cache_dataset_shard_cache_is_thread_safe(tmp_path):
    samples = [_tiny_sample(f"s{i}") for i in range(4)]
    torch.save(samples[:2], tmp_path / "shard_00000.pt")
    torch.save(samples[2:], tmp_path / "shard_00001.pt")
    entries = [
        {"shard": "shard_00000.pt", "index": 0, "sample_id": "s0", "scene_name": "scene"},
        {"shard": "shard_00000.pt", "index": 1, "sample_id": "s1", "scene_name": "scene"},
        {"shard": "shard_00001.pt", "index": 0, "sample_id": "s2", "scene_name": "scene"},
        {"shard": "shard_00001.pt", "index": 1, "sample_id": "s3", "scene_name": "scene"},
    ]
    (tmp_path / "index.json").write_text(json.dumps({
        "version": MSP_WM_CACHE_VERSION_V3,
        "metadata": {"topk": 2, "latent_hw": [2, 2], "trajectory_length": 12},
        "num_samples": 4,
        "entries": entries,
    }))
    ds = MSPWorldModelCacheDataset(tmp_path)

    order = [0, 2, 1, 3] * 50
    with ThreadPoolExecutor(max_workers=8) as pool:
        got = list(pool.map(lambda i: ds[i]["sample_id"], order))
    assert got == [f"s{i}" for i in order]
