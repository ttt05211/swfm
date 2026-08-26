"""Fail-closed compatibility checks for formal SWFM training assets."""
from pathlib import Path
import torch
from .runtime_config import config_fingerprint,get_cfg


def _meta_fp(meta):
    if meta.get('cache_contract_sha256'): return str(meta['cache_contract_sha256'])
    cfg=meta.get('resolved_config')
    if cfg is None: raise RuntimeError('cache metadata lacks resolved_config/cache_contract_sha256')
    return config_fingerprint(cfg,'cache')


def validate_cache_metadata(name,meta,cfg):
    required=('vae_ckpt_sha256','latent_mode','latent_extra_radius','vae_amp_enabled')
    missing=[k for k in required if k not in meta]
    if missing: raise RuntimeError(f'{name} cache metadata missing {missing}; rebuild formal cache')
    expected_fp=config_fingerprint(cfg,'cache'); got_fp=_meta_fp(meta)
    if got_fp!=expected_fp: raise RuntimeError(f'{name} cache contract mismatch: {got_fp} != {expected_fp}')
    if str(meta['latent_mode'])!=str(get_cfg(cfg,'CACHE.VAE_LATENT_MODE')):
        raise RuntimeError(f'{name} latent mode does not match current config')
    if int(meta['latent_extra_radius'])!=int(get_cfg(cfg,'MOTION.LATENT_EXTRA_RADIUS')):
        raise RuntimeError(f'{name} latent support radius does not match current config')
    if bool(meta['vae_amp_enabled'])!=bool(get_cfg(cfg,'RUNTIME.VAE_AMP.ENABLED',False)):
        raise RuntimeError(f'{name} VAE AMP convention does not match current config')
    return got_fp


def validate_cache_pair(train_meta,val_meta,cfg):
    tfp=validate_cache_metadata('train',train_meta,cfg); vfp=validate_cache_metadata('val',val_meta,cfg)
    for key in ('vae_ckpt_sha256','latent_mode','latent_extra_radius','vae_amp_enabled','vae_seed_contract'):
        if train_meta.get(key)!=val_meta.get(key):
            raise RuntimeError(f'train/val cache mismatch at {key}: {train_meta.get(key)!r} != {val_meta.get(key)!r}')
    if tfp!=vfp: raise RuntimeError('train/val cache contract fingerprints differ')
    return tfp


def load_empty_asset(path):
    obj=torch.load(Path(path),map_location='cpu',weights_only=True)
    if not isinstance(obj,dict) or 'empty_latent' not in obj:
        raise RuntimeError('formal training requires metadata-rich empty_latent.pt; rebuild it with build_latent_cache.py')
    tensor=obj['empty_latent']
    if not torch.is_tensor(tensor) or tensor.ndim!=3: raise ValueError('empty latent must be [C,H,W]')
    return tensor,obj


def validate_empty_asset(meta,cache_meta,cache_fp):
    required=('vae_ckpt_sha256','mode','vae_amp_enabled','cache_contract_sha256')
    missing=[k for k in required if k not in meta]
    if missing: raise RuntimeError(f'empty_latent metadata missing {missing}; rebuild it')
    checks={
        'vae_ckpt_sha256':cache_meta.get('vae_ckpt_sha256'),
        'mode':cache_meta.get('latent_mode'),
        'vae_amp_enabled':cache_meta.get('vae_amp_enabled'),
        'cache_contract_sha256':cache_fp,
    }
    for key,expected in checks.items():
        if meta.get(key)!=expected: raise RuntimeError(f'empty_latent mismatch at {key}: {meta.get(key)!r} != {expected!r}')


def validate_resume_checkpoint(ck,cfg,cache_fp,upstream_sha,empty_tensor):
    saved_resume=ck.get('resume_contract_sha256')
    if saved_resume is None and ck.get('resolved_config') is not None:
        saved_resume=config_fingerprint(ck['resolved_config'],'resume')
    current_resume=config_fingerprint(cfg,'resume')
    if saved_resume!=current_resume: raise RuntimeError('resume checkpoint config contract differs from current run')
    saved_cache=ck.get('cache_contract_sha256')
    if saved_cache is None and ck.get('cache_metadata') is not None:
        saved_cache=_meta_fp(ck['cache_metadata'])
    if saved_cache!=cache_fp: raise RuntimeError('resume checkpoint cache contract differs from current cache')
    if ck.get('upstream_ckpt_sha256')!=upstream_sha: raise RuntimeError('resume checkpoint upstream initialization fingerprint differs')
    saved_empty=ck.get('empty_latent')
    if not torch.is_tensor(saved_empty) or not torch.equal(saved_empty.cpu(),empty_tensor.cpu()):
        raise RuntimeError('resume checkpoint empty latent differs from supplied empty_latent.pt')
