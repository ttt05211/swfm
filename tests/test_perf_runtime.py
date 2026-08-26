import torch
from real_motion.perf import amp_enabled, amp_dtype, dataloader_kwargs, needs_grad_scaler, vae_autocast_context


def test_l40s_bf16_profile_does_not_require_grad_scaler_on_cpu_or_bf16():
    cfg={'RUNTIME':{'AMP':{'ENABLED':True,'DTYPE':'bfloat16'},'VAE_AMP':{'ENABLED':False},
                    'DATALOADER':{'PIN_MEMORY':True,'PERSISTENT_WORKERS':True,'PREFETCH_FACTOR':4}}}
    assert amp_dtype(cfg)==torch.bfloat16
    assert not amp_enabled(cfg,'cpu')
    assert not needs_grad_scaler(cfg,'cpu')
    with vae_autocast_context(cfg,'cpu'):
        x=torch.tensor([1.0])
    assert x.dtype==torch.float32


def test_dataloader_perf_knobs_only_add_worker_options_when_workers_exist():
    cfg={'RUNTIME':{'DATALOADER':{'PIN_MEMORY':True,'PERSISTENT_WORKERS':True,'PREFETCH_FACTOR':4}}}
    assert dataloader_kwargs(cfg,0)=={'pin_memory':True}
    assert dataloader_kwargs(cfg,8)=={'pin_memory':True,'persistent_workers':True,'prefetch_factor':4}
