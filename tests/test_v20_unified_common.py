from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from tools.real_motion.v20_unified_common import (
    STAGE1_PROTOCOL,
    Stage1RowStore,
    align_v18_records_to_stage1,
    deduplicate_completion_queries,
    stage1_manifest_paths,
)
from real_motion.v20_unified_data import CompletionTileQuery, make_completion_tile
from tools.real_motion.compact_p0_f9_v20_stage1_for_unified import (
    main as compact_stage1_main,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_repeated_completion_query_is_decoded_once_but_keeps_inverse_draws():
    tile = make_completion_tile(
        window_index=0,
        horizon=2,
        core_start_xyz=(0, 0, 0),
        core_shape_xyz=(2, 2, 1),
        native_shape_xyz=(2, 2, 1),
        halo=2,
    )
    points = torch.zeros(*tile.halo_shape_xyz, 3)
    valid = torch.ones(tile.halo_shape_xyz, dtype=torch.bool)
    support = torch.ones_like(valid)
    query = CompletionTileQuery(tile, points, valid, support)
    unique, inverse = deduplicate_completion_queries([query, query, query])
    assert len(unique) == 1 and unique[0] is query
    assert inverse == [0, 0, 0]

    # Equal values with distinct tensor identities are deliberately not merged.
    distinct = CompletionTileQuery(
        tile, points.clone(), valid.clone(), support.clone()
    )
    unique, inverse = deduplicate_completion_queries([query, distinct])
    assert len(unique) == 2
    assert inverse == [0, 1]


def test_v20_entrypoint_forces_repository_tools_ahead_of_shadow_package(tmp_path):
    shadow = tmp_path / "tools"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("# conflicting tools package\n")
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    pythonpath = [str(tmp_path), str(root)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    result = subprocess.run(
        [
            sys.executable,
            str(root / "tools/real_motion/v20_validate_run_inputs.py"),
            "--help",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--dev-v18-cache" in result.stdout


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


def _stage1_store_with_keys(tmp_path, keys):
    shard = tmp_path / "ordered.pt"
    rows = [
        {"scene_name": scene, "t0_token": token}
        for scene, token in keys
    ]
    torch.save({"protocol": STAGE1_PROTOCOL, "rows": rows}, shard)
    index = {
        "num_windows": len(keys),
        "shards": [
            {
                "file": shard.name,
                "count": len(keys),
                "keys": [list(key) for key in keys],
            }
        ],
    }
    return Stage1RowStore(tmp_path, index)


def test_v18_population_is_strictly_aligned_to_stage1_order(tmp_path):
    store = _stage1_store_with_keys(
        tmp_path, [("scene-b", "t2"), ("scene-a", "t1")]
    )
    records = [
        {"scene_name": "scene-a", "t0_token": "t1", "value": 1},
        {"scene_name": "unused", "t0_token": "t3", "value": 3},
        {"scene_name": "scene-b", "t0_token": "t2", "value": 2},
    ]
    aligned, report = align_v18_records_to_stage1(
        records, store, population_name="dev512"
    )
    assert [row["value"] for row in aligned] == [2, 1]
    assert report == {
        "population": "dev512",
        "v18_source_records": 3,
        "stage1_frozen_keys": 2,
        "matched": 2,
        "unique": 2,
        "missing": 0,
        "duplicate": 0,
    }


def test_v18_population_alignment_rejects_missing_and_duplicate_keys(tmp_path):
    store = _stage1_store_with_keys(tmp_path, [("scene", "t1")])
    with pytest.raises(RuntimeError, match="misses frozen Stage1 identities"):
        align_v18_records_to_stage1(
            [{"scene_name": "other", "t0_token": "t2"}],
            store,
            population_name="dev512",
        )
    duplicate = {"scene_name": "scene", "t0_token": "t1"}
    with pytest.raises(RuntimeError, match="duplicate V18 identities"):
        align_v18_records_to_stage1(
            [duplicate, dict(duplicate)], store, population_name="dev512"
        )


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
