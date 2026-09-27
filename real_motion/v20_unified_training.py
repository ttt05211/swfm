"""Trainer state, phase control and exact-resume checkpoints for V20 unified."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import random
import subprocess
from typing import Mapping, Sequence

import numpy as np
import torch

from .local_st_world_model_v17 import config_from_mapping_v17
from .local_st_world_model_v18_se2 import LocalSpatialTemporalWorldModelV18SE2
from .v20_history_world import CanonicalLattice
from .v20_unified_model import (
    V20_UNIFIED_MODEL_PROTOCOL,
    V20UnifiedConfig,
    V20UnifiedTransportCompletion,
)

TRAIN_PROTOCOL = "p0_f9_v20_unified_transport_completion_train_v1"


@dataclass
class TrainerProgress:
    attempted_updates: int = 0
    successful_updates: int = 0
    phase: str = "warmup"


@lru_cache(maxsize=4096)
def _file_sha256_cached(path: str, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return _file_sha256_cached(str(resolved), int(stat.st_size), int(stat.st_mtime_ns))


def current_git_sha(root: str | Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(root), text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def training_phase(successful_updates: int, warmup_updates: int) -> str:
    return "warmup" if int(successful_updates) < int(warmup_updates) else "joint"


def configure_training_phase(
    model: V20UnifiedTransportCompletion, phase: str
) -> bool:
    """Apply phase state and return whether the source adapter is enabled."""
    if phase not in {"warmup", "joint"}:
        raise ValueError("phase must be warmup or joint")
    model.train()
    joint = phase == "joint"
    for parameter in model.v18.parameters():
        parameter.requires_grad_(joint)
    if joint:
        model.v18.train()
    else:
        model.v18.eval()
    return joint


def build_optimizer(
    model: V20UnifiedTransportCompletion,
    *,
    new_lr: float = 2.0e-4,
    v18_lr: float = 2.0e-5,
    weight_decay: float = 1.0e-4,
) -> torch.optim.Optimizer:
    v18_ids = {id(p) for p in model.v18.parameters()}
    new_parameters = [p for p in model.parameters() if id(p) not in v18_ids]
    return torch.optim.AdamW(
        [
            {"params": new_parameters, "lr": float(new_lr), "name": "new"},
            {
                "params": list(model.v18.parameters()),
                "lr": 0.0,
                "target_lr": float(v18_lr),
                "name": "v18",
            },
        ],
        weight_decay=float(weight_decay),
    )


def set_optimizer_phase_lrs(
    optimizer: torch.optim.Optimizer,
    *,
    phase: str,
    new_lr: float,
    v18_lr: float,
    scale: float = 1.0,
) -> None:
    for group in optimizer.param_groups:
        if group.get("name") == "new":
            group["lr"] = float(new_lr) * float(scale)
        elif group.get("name") == "v18":
            group["lr"] = (float(v18_lr) * float(scale)) if phase == "joint" else 0.0
        else:
            raise RuntimeError("unexpected optimizer parameter group")


def scaler_step_succeeded(
    scaler: torch.cuda.amp.GradScaler,
    optimizer: torch.optim.Optimizer,
) -> bool:
    """Step once and report AMP overflow without advancing successful state."""
    before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    return float(scaler.get_scale()) >= before


def checkpoint_payload(
    *,
    model: V20UnifiedTransportCompletion,
    optimizer: torch.optim.Optimizer,
    scheduler_state: Mapping,
    scaler: torch.cuda.amp.GradScaler,
    progress: TrainerProgress,
    base_checkpoint: str | Path,
    manifest_paths: Mapping[str, str | Path],
    config: Mapping,
    repository_root: str | Path,
    resume_contract: Mapping | None = None,
    monitor_history: Sequence[Mapping] | None = None,
    frozen_reference_raw: Mapping | None = None,
) -> dict:
    manifests = {
        str(name): {
            "path": str(Path(path).resolve()),
            "sha256": file_sha256(path),
        }
        for name, path in manifest_paths.items()
    }
    lattice = model.coarse_lattice
    lattice_dict = asdict(lattice)
    grid_sha256 = hashlib.sha256(
        json.dumps(lattice_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "protocol": TRAIN_PROTOCOL,
        "model_protocol": V20_UNIFIED_MODEL_PROTOCOL,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": dict(scheduler_state),
        "scaler": scaler.state_dict(),
        "attempted_updates": int(progress.attempted_updates),
        "successful_updates": int(progress.successful_updates),
        "phase": str(progress.phase),
        "rng_state": capture_rng_state(),
        "base_checkpoint": str(Path(base_checkpoint).resolve()),
        "base_checkpoint_sha256": file_sha256(base_checkpoint),
        "manifests": manifests,
        "config": dict(config),
        "resume_contract": (
            dict(resume_contract) if resume_contract is not None else None
        ),
        "monitor_history": [dict(row) for row in (monitor_history or ())],
        "frozen_reference_raw": (
            dict(frozen_reference_raw)
            if frozen_reference_raw is not None
            else None
        ),
        "v18_model_config": asdict(model.v18.config),
        "unified_model_config": asdict(model.config),
        "coarse_lattice": lattice_dict,
        "coarse_lattice_sha256": grid_sha256,
        "git_sha": current_git_sha(repository_root),
    }


def verify_resume_inputs(
    checkpoint: Mapping,
    *,
    base_checkpoint: str | Path,
    manifest_paths: Mapping[str, str | Path],
    resume_contract: Mapping | None = None,
    coarse_lattice: Mapping | CanonicalLattice | None = None,
) -> None:
    if checkpoint.get("base_checkpoint_sha256") != file_sha256(base_checkpoint):
        raise RuntimeError("resume/base checkpoint hash mismatch")
    saved = checkpoint.get("manifests") or {}
    for name, path in manifest_paths.items():
        entry = saved.get(str(name))
        if entry is None:
            raise RuntimeError(f"resume checkpoint lacks manifest {name!r}")
        current = file_sha256(path)
        if str(entry.get("sha256")) != current:
            raise RuntimeError(
                f"resume manifest hash mismatch for {name}: "
                f"{entry.get('sha256')} != {current}"
            )
    extra = sorted(set(saved) - {str(name) for name in manifest_paths})
    if extra:
        raise RuntimeError(
            "current run omits resume manifests: " + ", ".join(extra[:8])
        )
    if resume_contract is not None:
        previous = checkpoint.get("resume_contract")
        if previous is None:
            raise RuntimeError("resume checkpoint lacks the exact-run contract")
        keys = sorted(set(previous) | set(resume_contract))
        different = [
            key for key in keys
            if previous.get(key) != resume_contract.get(key)
        ]
        if different:
            detail = ", ".join(
                f"{key}={previous.get(key)!r}->{resume_contract.get(key)!r}"
                for key in different[:8]
            )
            raise RuntimeError(f"resume run contract differs: {detail}")
    if coarse_lattice is not None:
        saved_lattice = _normalized_lattice(checkpoint.get("coarse_lattice"))
        current_lattice = _normalized_lattice(coarse_lattice)
        if saved_lattice != current_lattice:
            raise RuntimeError("resume/current Stage1 coarse lattice mismatch")


def _normalized_lattice(
    value: Mapping | CanonicalLattice | None,
) -> dict | None:
    """Normalize JSON lists and dataclass tuples before identity checks."""
    if value is None:
        return None
    raw = asdict(value) if isinstance(value, CanonicalLattice) else dict(value)
    return {
        "origin_xyz_m": tuple(float(x) for x in raw["origin_xyz_m"]),
        "voxel_size_xyz_m": tuple(float(x) for x in raw["voxel_size_xyz_m"]),
        "shape_xyz": tuple(int(x) for x in raw["shape_xyz"]),
    }


def save_checkpoint(path: str | Path, payload: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def load_model_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[V20UnifiedTransportCompletion, dict]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if checkpoint.get("protocol") != TRAIN_PROTOCOL:
        raise RuntimeError(f"unexpected unified checkpoint: {checkpoint.get('protocol')}")
    v18_cfg = config_from_mapping_v17(checkpoint["v18_model_config"])
    unified_cfg = V20UnifiedConfig(**checkpoint["unified_model_config"])
    lattice = CanonicalLattice(**checkpoint["coarse_lattice"])
    model = V20UnifiedTransportCompletion(
        LocalSpatialTemporalWorldModelV18SE2(v18_cfg),
        coarse_lattice=lattice,
        config=unified_cfg,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model, checkpoint


def restore_training_checkpoint(
    checkpoint: Mapping,
    *,
    model: V20UnifiedTransportCompletion,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
) -> TrainerProgress:
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    restore_rng_state(checkpoint["rng_state"])
    progress = TrainerProgress(
        attempted_updates=int(checkpoint["attempted_updates"]),
        successful_updates=int(checkpoint["successful_updates"]),
        phase=str(checkpoint["phase"]),
    )
    if progress.phase != training_phase(
        progress.successful_updates,
        int(checkpoint.get("config", {}).get("warmup_updates", 128)),
    ):
        raise RuntimeError("checkpoint phase/counter mismatch")
    return progress
