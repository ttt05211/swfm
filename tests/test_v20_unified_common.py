from __future__ import annotations

import hashlib
import json
import sys

import pytest
import torch

from tools.real_motion.v20_unified_common import (
    STAGE1_PROTOCOL,
    Stage1RowStore,
    stage1_manifest_paths,
)
from tools.real_motion.compact_p0_f9_v20_stage1_for_unified import (
    main as compact_stage1_main,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage1_store_rejects_index_shard_row_identity_mismatch(tmp_path):
    shard = tmp_path / "shard_00000.pt"
    torch.save(
        {
            "protocol": STAGE1_PROTOCOL,
            "rows": [{"scene_name": "other", "t0_token": "row"}],
        },
        shard,
    )
    index = {
        "shards": [
            {
                "file": shard.name,
                "count": 1,
                "bytes": shard.stat().st_size,
                "sha256": _sha256(shard),
                "keys": [["scene", "token"]],
            }
        ]
    }
    store = Stage1RowStore(tmp_path, index)
    with pytest.raises(RuntimeError, match="row order/key mismatch"):
        store[("scene", "token")]


def test_stage1_store_rejects_changed_shard_and_manifest_lists_every_file(tmp_path):
    shard = tmp_path / "shard_00000.pt"
    row = {"scene_name": "scene", "t0_token": "token"}
    torch.save({"protocol": STAGE1_PROTOCOL, "rows": [row]}, shard)
    index = {
        "shards": [
            {
                "file": shard.name,
                "count": 1,
                "bytes": shard.stat().st_size,
                "sha256": "0" * 64,
                "keys": [["scene", "token"]],
            }
        ]
    }
    store = Stage1RowStore(tmp_path, index)
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        store[("scene", "token")]
    paths = stage1_manifest_paths("train_stage1", tmp_path / "index.json", index)
    assert paths == {
        "train_stage1_index": tmp_path / "index.json",
        "train_stage1_shard:shard_00000.pt": shard,
    }


def test_compactor_keeps_only_responsibility_counts_and_writes_verified_index(
    tmp_path, monkeypatch
):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    shard = src / "shard_00000.pt"
    row = {
        "scene_name": "scene",
        "t0_token": "token",
        "static_supervision": [{"large": torch.ones(8)}],
        "dynamic_supervision": [
            {
                "responsibility_name": "BIRTH",
                "instance_token": "secret-id",
                "trajectory_xyz_yaw_t0": [[0.0] * 4] * 6,
            }
        ],
    }
    torch.save({"protocol": STAGE1_PROTOCOL, "rows": [row]}, shard)
    (src / "index.json").write_text(
        json.dumps(
            {
                "protocol": STAGE1_PROTOCOL,
                "shards": [{"file": shard.name, "count": 1}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["compact", "--input-dir", str(src), "--output-dir", str(dst)],
    )
    compact_stage1_main()
    index = json.loads((dst / "index.json").read_text(encoding="utf-8"))
    assert index["dynamic_diagnostic_groups_stored"] is True
    assert index["dynamic_diagnostic_counts_stored"] is True
    store = Stage1RowStore(dst, index)
    compact = store[("scene", "token")]
    assert compact["dynamic_diagnostic_counts"] == {"BIRTH": 1}
    assert "static_supervision" not in compact
    assert "dynamic_supervision" not in compact
