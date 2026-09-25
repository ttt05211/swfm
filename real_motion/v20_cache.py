"""Serializable cache protocol for V20 stages 1--5."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

CACHE_PROTOCOL = "p0_f9_v20_history_cache_v1"

INFERENCE_KEYS = (
    "history_semantic",
    "history_observed",
    "history_observed_free",
    "future_ego_to_world",
    "query_mask",
)

SUPERVISION_KEYS = (
    "static_target",
    "static_valid",
    "static_history_seen_t0_missing",
    "static_never_seen",
    "dynamic_partition",
)


def save_v20_cache(path: str | Path, payload: Mapping[str, object], *, metadata: Mapping[str, object]) -> None:
    obj = {
        "protocol": CACHE_PROTOCOL,
        "metadata": dict(metadata),
        "payload": dict(payload),
    }
    torch.save(obj, Path(path))


def load_v20_cache(path: str | Path) -> dict[str, object]:
    obj = torch.load(Path(path), map_location="cpu", weights_only=False)
    if obj.get("protocol") != CACHE_PROTOCOL:
        raise RuntimeError(f"unexpected V20 cache protocol: {obj.get('protocol')}")
    if "payload" not in obj or "metadata" not in obj:
        raise RuntimeError("malformed V20 cache")
    return obj


def inference_view(cache: Mapping[str, object]) -> dict[str, object]:
    payload = cache["payload"]
    missing = [k for k in INFERENCE_KEYS if k not in payload]
    if missing:
        raise KeyError(f"V20 cache missing inference keys: {missing}")
    # Construct a new dictionary so supervision cannot accidentally travel
    # through the model call by virtue of sharing one giant batch object.
    return {k: payload[k] for k in INFERENCE_KEYS}


def supervision_view(cache: Mapping[str, object]) -> dict[str, object]:
    payload = cache["payload"]
    return {k: payload[k] for k in SUPERVISION_KEYS if k in payload}
