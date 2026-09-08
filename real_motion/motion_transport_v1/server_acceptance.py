from __future__ import annotations

import torch
import torch.distributed as dist

from .config import get


def expected_gpu_token(cfg):
    raw=str(get(cfg,'training.gpu_type','') or '').strip()
    if not raw:return ''
    return raw.split('_',1)[0]


def formal_server_environment(ctx,cfg,*,require_world_size=False,require_bf16=True,require_gpu_type=False):
    """Validate the hardware actually used for a formal profile/train launch.

    MT-V1 supports either one or two GPUs. By default we require CUDA/BF16 but do
    not require the config's advisory `training.gpus` or `training.gpu_type` to
    match. A caller may opt into those checks for a deliberately fixed launch.
    """
    expected_world=get(cfg,'training.gpus')
    if require_world_size:
        if expected_world is None:raise RuntimeError('training.gpus is null but an exact WORLD_SIZE check was requested')
        if int(ctx.world_size)!=int(expected_world):
            raise RuntimeError(f'formal server gate requires WORLD_SIZE={int(expected_world)}, got {ctx.world_size}')
    if int(ctx.world_size) not in (1,2):
        raise RuntimeError(f'MT-V1 supports WORLD_SIZE 1 or 2, got {ctx.world_size}')
    if ctx.device.type!='cuda' or not torch.cuda.is_available():
        raise RuntimeError('formal server gate requires CUDA on every rank')
    if require_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError('formal server gate requires CUDA BF16 support')
    if ctx.world_size>1 and dist.is_initialized() and dist.get_backend()!='nccl':
        raise RuntimeError(f'multi-GPU formal server gate requires NCCL, got {dist.get_backend()}')
    local_name=torch.cuda.get_device_name(ctx.device)
    names=[local_name]
    if ctx.world_size>1:
        names=[None]*ctx.world_size
        dist.all_gather_object(names,local_name)
    token=expected_gpu_token(cfg)
    if require_gpu_type and token:
        bad=[n for n in names if token.lower() not in str(n).lower()]
        if bad:raise RuntimeError(f'formal server gate expected GPU containing {token!r}, got {names}')
    return {
        'torch':torch.__version__,
        'cuda_build':torch.version.cuda,
        'cudnn':torch.backends.cudnn.version(),
        'world_size':int(ctx.world_size),
        'backend':dist.get_backend() if ctx.world_size>1 and dist.is_initialized() else None,
        'bf16_supported':bool(torch.cuda.is_bf16_supported()),
        'gpu_names':list(map(str,names)),
        'configured_gpu_count':None if expected_world is None else int(expected_world),
        'configured_gpu_token':token,
    }
