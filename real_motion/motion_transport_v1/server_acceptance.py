from __future__ import annotations

import torch
import torch.distributed as dist

from .config import get


def expected_gpu_token(cfg):
    raw=str(get(cfg,'training.gpu_type','') or '').strip()
    if not raw:return ''
    return raw.split('_',1)[0]


def formal_server_environment(ctx,cfg,*,require_world_size=True,require_bf16=True,require_gpu_type=True):
    expected_world=int(get(cfg,'training.gpus',2))
    if require_world_size and int(ctx.world_size)!=expected_world:
        raise RuntimeError(f'formal server gate requires WORLD_SIZE={expected_world}, got {ctx.world_size}')
    if ctx.device.type!='cuda' or not torch.cuda.is_available():
        raise RuntimeError('formal server gate requires CUDA on every rank')
    if require_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError('formal server gate requires CUDA BF16 support')
    if ctx.world_size>1 and dist.is_initialized() and dist.get_backend()!='nccl':
        raise RuntimeError(f'formal server gate requires NCCL, got {dist.get_backend()}')
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
        'expected_gpu_token':token,
    }
