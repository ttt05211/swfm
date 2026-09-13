"""Small, dependency-light contract helpers for the GenieDrive val128 adapter."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple, Union

import numpy as np
import torch

MANIFEST_VERSION = "swfm_geniedrive_val128_manifest_v1"
PREDICTION_VERSION = "swfm_external_prediction_v1"
EXPECTED_SHAPE = (6, 200, 200, 16)
FREE_LABEL = 17


def split_sample_id(sample_id: str) -> Tuple[str, str]:
    """Return (scene_name, t0 token) from the frozen SWFM sample id."""
    scene, sep, token = str(sample_id).rpartition(":")
    if not sep or not scene or not token:
        raise ValueError(f"invalid SWFM sample_id {sample_id!r}; expected scene:token")
    return scene, token


def token_from_occ_path(path: str) -> str:
    """Extract the globally unique nuScenes token from a GenieDrive occ path."""
    normalized = str(path).replace("\\", "/").rstrip("/")
    token = normalized.rsplit("/", 1)[-1]
    if token.endswith(".npz"):
        token = token[:-4]
    if not token or token == "labels":
        raise ValueError(f"cannot extract sample token from occ_path {path!r}")
    return token


def manifest_digest(sample_ids: Iterable[str]) -> str:
    payload = "\n".join(str(x) for x in sample_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_manifest(index: Mapping[str, Any], expected_count: int = 128) -> Dict[str, Any]:
    entries = list(index.get("entries", []))
    sample_ids = [str(entry["sample_id"]) for entry in entries]
    if len(sample_ids) != expected_count:
        raise ValueError(
            f"prepared set contains {len(sample_ids)} samples, expected {expected_count}"
        )
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("prepared index contains duplicate sample_id values")

    rows = []
    tokens = set()
    scenes = set()
    for sample_id in sample_ids:
        scene, token = split_sample_id(sample_id)
        if token in tokens:
            raise ValueError(f"duplicate nuScenes token in manifest: {token}")
        tokens.add(token)
        scenes.add(scene)
        rows.append({"sample_id": sample_id, "scene_name": scene, "token": token})

    if len(scenes) != expected_count:
        raise ValueError(
            f"val128 must contain one window per scene; got {len(scenes)} scenes"
        )
    return {
        "version": MANIFEST_VERSION,
        "prepared_version": index.get("version"),
        "selection_contract": index.get("metadata", {}).get(
            "selection_contract", "scene_disjoint_midpoint_one_window_per_scene_v1"
        ),
        "num_samples": len(rows),
        "num_scenes": len(scenes),
        "sample_ids_sha256": manifest_digest(sample_ids),
        "entries": rows,
    }


def load_manifest(path: Union[os.PathLike, str]) -> Dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError(f"unsupported manifest version {manifest.get('version')!r}")
    entries = manifest.get("entries", [])
    if manifest.get("num_samples") != len(entries):
        raise ValueError("manifest num_samples does not match entries")
    return manifest


def validate_prediction(prediction: Any) -> np.ndarray:
    if torch.is_tensor(prediction):
        prediction = prediction.detach().cpu().numpy()
    array = np.asarray(prediction)
    if tuple(array.shape) != EXPECTED_SHAPE:
        raise ValueError(f"prediction shape {array.shape} != {EXPECTED_SHAPE}")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"prediction dtype must be integer, got {array.dtype}")
    if array.size and (int(array.min()) < 0 or int(array.max()) > FREE_LABEL):
        raise ValueError("prediction labels must be in Occ3D range [0, 17]")
    return array.astype(np.uint8, copy=False)


def prediction_filename(sample_id: str) -> str:
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:16]
    return f"geniedrive_{digest}.pt"


def save_prediction(
    output_dir: Union[os.PathLike, str],
    sample_id: str,
    prediction: Any,
    metadata: Mapping[str, Any],
    overwrite: bool = False,
) -> str:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    array = validate_prediction(prediction)
    filename = prediction_filename(sample_id)
    target = root / filename
    if target.exists() and not overwrite:
        existing = torch.load(target, map_location="cpu")
        if str(existing.get("sample_id")) != sample_id:
            raise RuntimeError(f"existing prediction has wrong sample_id: {target}")
        validate_prediction(existing.get("pred_occ"))
        existing_metadata = existing.get("metadata", {})
        requested_checkpoint = metadata.get("checkpoint_sha256")
        if requested_checkpoint and existing_metadata.get("checkpoint_sha256") != requested_checkpoint:
            raise RuntimeError(
                f"existing prediction came from another checkpoint: {target}; "
                "use --overwrite or a new output directory"
            )
        return filename
    payload = {
        "version": PREDICTION_VERSION,
        "sample_id": sample_id,
        "pred_occ": torch.from_numpy(array.copy()),
        "source": "GenieDrive",
        "metadata": dict(metadata),
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)
    return filename
