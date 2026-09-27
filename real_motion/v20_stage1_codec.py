"""Compact, lossless codec for V20 Stage-1 cache rows.

The codec reduces disk and I/O without changing model inputs or supervision:
* history semantic values are stored only at observed occupied coarse cells;
* those labels (0..16) are packed at 5 bits/value;
* static future supervision stores one packed validity mask plus 4-bit remapped
  static/free labels in native C-order;
* decoding reconstructs exactly the dense/history or sparse/static tensors
  consumed by training.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from .metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS

FREE_LABEL = 17
_DYNAMIC = frozenset(int(x) for x in DYNAMIC_CLASS_IDS)
STATIC_LABELS = tuple(i for i in range(18) if i not in _DYNAMIC)
_STATIC_TO_CODE = np.full(18, 255, dtype=np.uint8)
for _code, _label in enumerate(STATIC_LABELS):
    _STATIC_TO_CODE[_label] = _code
_CODE_TO_STATIC = np.asarray(STATIC_LABELS, dtype=np.uint8)
if len(STATIC_LABELS) > 16:
    raise RuntimeError("static/free label alphabet no longer fits 4 bits")


def pack_bool(mask: np.ndarray) -> torch.Tensor:
    arr = np.asarray(mask, dtype=np.uint8)
    return torch.from_numpy(
        np.packbits(arr.reshape(-1), bitorder="little").copy()
    )


def unpack_bool(bits: torch.Tensor | np.ndarray, shape: Sequence[int]) -> np.ndarray:
    arr = np.asarray(
        bits.cpu() if isinstance(bits, torch.Tensor) else bits,
        dtype=np.uint8,
    )
    n = int(np.prod(tuple(int(x) for x in shape)))
    return np.unpackbits(
        arr.reshape(-1), bitorder="little", count=n
    ).reshape(tuple(int(x) for x in shape)).astype(bool)


def _pack_5bit(values: np.ndarray) -> torch.Tensor:
    vals = np.asarray(values, dtype=np.uint8).reshape(-1)
    if vals.size and int(vals.max()) > 31:
        raise ValueError("5-bit pack received value >31")
    if vals.size == 0:
        return torch.empty(0, dtype=torch.uint8)
    shifts = np.arange(5, dtype=np.uint8)
    bits = ((vals[:, None] >> shifts[None]) & 1).astype(np.uint8)
    packed = np.packbits(bits.reshape(-1), bitorder="little")
    return torch.from_numpy(packed.copy())


def _unpack_5bit(packed: torch.Tensor | np.ndarray, count: int) -> np.ndarray:
    count = int(count)
    if count == 0:
        return np.empty(0, dtype=np.uint8)
    arr = np.asarray(
        packed.cpu() if isinstance(packed, torch.Tensor) else packed,
        dtype=np.uint8,
    )
    bits = np.unpackbits(
        arr.reshape(-1), bitorder="little", count=count * 5
    ).reshape(count, 5)
    weights = (1 << np.arange(5, dtype=np.uint8))[None]
    return (bits * weights).sum(axis=1, dtype=np.uint16).astype(np.uint8)


def pack_history_semantic(
    semantic: np.ndarray,
    observed: np.ndarray,
    observed_free: np.ndarray,
    *,
    free_label: int = FREE_LABEL,
) -> dict[str, object]:
    sem = np.asarray(semantic, dtype=np.uint8)
    obs = np.asarray(observed, dtype=bool)
    free = np.asarray(observed_free, dtype=bool)
    if sem.shape != obs.shape or obs.shape != free.shape:
        raise ValueError("history semantic/observed/free shape mismatch")
    if bool((free & ~obs).any()):
        raise ValueError("observed_free must be a subset of observed")
    occupied = obs & ~free
    values = sem[occupied]
    if values.size and bool((values == int(free_label)).any()):
        raise ValueError("occupied history cell carries free label")
    return {
        "history_semantic_5bit": _pack_5bit(values),
        "history_semantic_count": int(values.size),
    }


def unpack_history_semantic(
    row: dict,
    observed: np.ndarray,
    observed_free: np.ndarray,
    *,
    free_label: int = FREE_LABEL,
) -> np.ndarray:
    obs = np.asarray(observed, dtype=bool)
    free = np.asarray(observed_free, dtype=bool)
    occupied = obs & ~free
    count = int(row["history_semantic_count"])
    if count != int(occupied.sum()):
        raise RuntimeError(
            "compressed history semantic count does not match observed occupied mask"
        )
    values = _unpack_5bit(row["history_semantic_5bit"], count)
    if values.size and bool((values > 16).any()):
        raise RuntimeError("decoded history semantic label outside 0..16")
    sem = np.full(obs.shape, int(free_label), dtype=np.uint8)
    sem[occupied] = values
    return sem


def _pack_nibbles(codes: np.ndarray) -> torch.Tensor:
    c = np.asarray(codes, dtype=np.uint8).reshape(-1)
    if c.size and int(c.max()) > 15:
        raise ValueError("nibble pack received code >15")
    if c.size % 2:
        c = np.concatenate((c, np.zeros(1, dtype=np.uint8)))
    if c.size == 0:
        return torch.empty(0, dtype=torch.uint8)
    packed = c[0::2] | (c[1::2] << 4)
    return torch.from_numpy(packed.copy())


def _unpack_nibbles(packed: torch.Tensor | np.ndarray, count: int) -> np.ndarray:
    count = int(count)
    if count == 0:
        return np.empty(0, dtype=np.uint8)
    p = np.asarray(
        packed.cpu() if isinstance(packed, torch.Tensor) else packed,
        dtype=np.uint8,
    ).reshape(-1)
    out = np.empty(p.size * 2, dtype=np.uint8)
    out[0::2] = p & 0x0F
    out[1::2] = p >> 4
    return out[:count]


def pack_static_supervision(
    gt: np.ndarray,
    observed: np.ndarray,
    *,
    dynamic_class_ids=tuple(int(x) for x in DYNAMIC_CLASS_IDS),
    free_label: int = FREE_LABEL,
) -> dict[str, object]:
    y = np.asarray(gt, dtype=np.uint8)
    obs = np.asarray(observed, dtype=bool)
    if y.shape != obs.shape:
        raise ValueError("static supervision GT/observation shape mismatch")
    dyn = np.isin(y, np.asarray(dynamic_class_ids, dtype=np.uint8))
    valid = obs & ~dyn
    labels = y[valid]
    codes = _STATIC_TO_CODE[labels] if labels.size else np.empty(0, dtype=np.uint8)
    if codes.size and bool((codes == 255).any()):
        raise RuntimeError("dynamic/unknown label entered static supervision")
    return {
        "valid_bits": pack_bool(valid),
        "semantic_4bit": _pack_nibbles(codes),
        "semantic_count": int(labels.size),
        "observed_count": int(valid.sum()),
        "observed_free_count": int((valid & (y == int(free_label))).sum()),
    }


def unpack_static_labels(sup: dict) -> np.ndarray:
    count = int(sup["semantic_count"])
    codes = _unpack_nibbles(sup["semantic_4bit"], count)
    if codes.size and int(codes.max()) >= len(_CODE_TO_STATIC):
        raise RuntimeError("invalid packed static semantic code")
    return _CODE_TO_STATIC[codes] if codes.size else np.empty(0, dtype=np.uint8)


def unpack_static_indices_and_labels(
    sup: dict,
    native_shape_xyz: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    shape = tuple(int(x) for x in native_shape_xyz)
    valid = unpack_bool(sup["valid_bits"], shape)
    labels = unpack_static_labels(sup)
    if int(valid.sum()) != int(labels.size):
        raise RuntimeError("static mask/label count mismatch")
    return np.argwhere(valid).astype(np.int64), labels.astype(np.int64, copy=False)
