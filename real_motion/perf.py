"""Hardware/runtime helpers for SWFM.

The default profile is tuned for NVIDIA L40S (Ada, 48 GB) without introducing
an FP8-only dependency. The sparse CFM uses BF16/TF32 by default. The frozen
OccFM VAE intentionally stays FP32 unless ``RUNTIME.VAE_AMP.ENABLED`` is
explicitly enabled after a reconstruction/parity check.
"""
from contextlib import nullcontext
import warnings
import torch

from .runtime_config import get_cfg


def cuda_device_summary(device):
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"device": str(device), "cuda": False}
    idx = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return {
        "device": str(device),
        "cuda": True,
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_gb": props.total_memory / (1024 ** 3),
        "multi_processor_count": props.multi_processor_count,
    }


def configure_cuda_runtime(cfg, device):
    """Apply safe high-throughput CUDA knobs before model construction."""
    device = torch.device(device)
    info = cuda_device_summary(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return info

    allow_tf32 = bool(get_cfg(cfg, "RUNTIME.TF32", True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = bool(get_cfg(cfg, "RUNTIME.CUDNN_BENCHMARK", True))
    torch.set_float32_matmul_precision(str(get_cfg(cfg, "RUNTIME.MATMUL_PRECISION", "high")))

    flash = bool(get_cfg(cfg, "RUNTIME.FLASH_SDP", True))
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(flash)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)

    requested = str(get_cfg(cfg, "RUNTIME.GPU_PROFILE", "auto")).lower()
    if requested == "l40s" and "l40s" not in info.get("name", "").lower():
        warnings.warn(
            f"RUNTIME.GPU_PROFILE=L40S but CUDA device is {info.get('name')!r}; "
            "performance settings remain valid but should be re-profiled.",
            RuntimeWarning,
        )
    return info


def _dtype_from_name(name):
    name = str(name).lower()
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }
    if name not in aliases:
        raise ValueError(f"unsupported AMP dtype: {name}")
    return aliases[name]


def amp_enabled(cfg, device, override=None):
    enabled = bool(override) if override is not None else bool(get_cfg(cfg, "RUNTIME.AMP.ENABLED", True))
    return enabled and torch.device(device).type == "cuda"


def amp_dtype(cfg):
    return _dtype_from_name(get_cfg(cfg, "RUNTIME.AMP.DTYPE", "bfloat16"))


def autocast_context(cfg, device, override=None):
    if not amp_enabled(cfg, device, override):
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype(cfg), enabled=True)


def vae_autocast_context(cfg, device):
    """Autocast context for the frozen VAE; FP32 is the conservative default."""
    enabled = bool(get_cfg(cfg, "RUNTIME.VAE_AMP.ENABLED", False))
    if not enabled or torch.device(device).type != "cuda":
        return nullcontext()
    dtype = _dtype_from_name(get_cfg(cfg, "RUNTIME.VAE_AMP.DTYPE", "bfloat16"))
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def needs_grad_scaler(cfg, device, override=None):
    return amp_enabled(cfg, device, override) and amp_dtype(cfg) == torch.float16


def dataloader_kwargs(cfg, workers):
    workers = int(workers)
    out = {"pin_memory": bool(get_cfg(cfg, "RUNTIME.DATALOADER.PIN_MEMORY", True))}
    if workers > 0:
        out["persistent_workers"] = bool(get_cfg(cfg, "RUNTIME.DATALOADER.PERSISTENT_WORKERS", True))
        out["prefetch_factor"] = int(get_cfg(cfg, "RUNTIME.DATALOADER.PREFETCH_FACTOR", 4))
    return out


def maybe_compile(model, cfg, phase):
    """Compile a runtime view only when explicitly enabled and benchmarked."""
    phase = str(phase).upper()
    enabled = bool(get_cfg(cfg, f"RUNTIME.COMPILE.{phase}", False))
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        warnings.warn("torch.compile requested but unavailable; using eager mode")
        return model
    mode = str(get_cfg(cfg, "RUNTIME.COMPILE.MODE", "reduce-overhead"))
    dynamic = bool(get_cfg(cfg, "RUNTIME.COMPILE.DYNAMIC", True))
    return torch.compile(model, mode=mode, dynamic=dynamic, fullgraph=False)
