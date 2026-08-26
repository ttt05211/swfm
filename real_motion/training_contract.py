"""Fail-closed compatibility checks for formal SWFM training assets."""
from pathlib import Path
import torch
from .runtime_config import config_fingerprint,get_cfg


def _meta_fp(meta):
    if meta.get('cache_contract_sha256'):return str(meta['cache_contract_sha256'])
    cfg=meta.get('resolved_config')
    if cfg is None:raise RuntimeError('cache metadata lacks resolved_config/cache_contract_sha256')
    return config_fingerprint(cfg,'cache')


def validate_cache_metadata(name,meta,cfg):
    required=('vae_ckpt_sha256','latent_mode','latent_extra_radius','vae_amp_enabled','trajectory_protocol','trajectory_length','upstream_wm_variant','upstream_init_variant')
    missing=[k for k in required if k not in meta]
    if missing:raise RuntimeError(f'{name} cache metadata missing {missing}; rebuild formal cache for OccFM-fut 196')
    expected_fp=config_fingerprint(cfg,'cache');got_fp=_meta_fp(meta)
    if got_fp!=expected_fp:raise RuntimeError(f'{name} cache contract mismatch: {got_fp} != {expected_fp}')
    checks={
        'latent_mode':str(get_cfg(cfg,'CACHE.VAE_LATENT_MODE')),
        'latent_extra_radius':int(get_cfg(cfg,'MOTION.LATENT_EXTRA_RADIUS')),
        'vae_amp_enabled':bool(get_cfg(cfg,'RUNTIME.VAE_AMP.ENABLED',False)),
        'trajectory_protocol':str(get_cfg(cfg,'EGO_PROTOCOL.NAME')),
        'trajectory_length':int(get_cfg(cfg,'EGO_PROTOCOL.TRAJECTORY_LENGTH',12)),
        'upstream_wm_variant':'occfm_fut',
        'upstream_init_variant':'fut_traj_196',
    }
    for key,expected in checks.items():
        actual=meta.get(key)
        if key=='latent_mode':actual=str(actual)
        elif key in ('latent_extra_radius','trajectory_length'):actual=int(actual)
        elif key=='vae_amp_enabled':actual=bool(actual)
        if actual!=expected:raise RuntimeError(f'{name} cache mismatch at {key}: {actual!r} != {expected!r}')
    return got_fp


def validate_cache_pair(train_meta,val_meta,cfg):
    tfp=validate_cache_metadata('train',train_meta,cfg);vfp=validate_cache_metadata('val',val_meta,cfg)
    for key in ('vae_ckpt_sha256','latent_mode','latent_extra_radius','vae_amp_enabled','vae_seed_contract','trajectory_protocol','trajectory_length','upstream_wm_variant','upstream_init_variant'):
        if train_meta.get(key)!=val_meta.get(key):raise RuntimeError(f'train/val cache mismatch at {key}: {train_meta.get(key)!r} != {val_meta.get(key)!r}')
    if tfp!=vfp:raise RuntimeError('train/val cache contract fingerprints differ')
    return tfp


def load_empty_asset(path):
    obj=torch.load(Path(path),map_location='cpu',weights_only=True)
    if not isinstance(obj,dict) or 'empty_latent' not in obj:raise RuntimeError('formal training requires metadata-rich empty_latent.pt; rebuild it with build_latent_cache.py')
    tensor=obj['empty_latent']
    if not torch.is_tensor(tensor) or tensor.ndim!=3:raise ValueError('empty latent must be [C,H,W]')
    return tensor,obj


def validate_empty_asset(meta,cache_meta,cache_fp):
    required=('vae_ckpt_sha256','mode','vae_amp_enabled','cache_contract_sha256','trajectory_protocol','trajectory_length')
    missing=[k for k in required if k not in meta]
    if missing:raise RuntimeError(f'empty_latent metadata missing {missing}; rebuild it for OccFM-fut 196')
    checks={'vae_ckpt_sha256':cache_meta.get('vae_ckpt_sha256'),'mode':cache_meta.get('latent_mode'),'vae_amp_enabled':cache_meta.get('vae_amp_enabled'),'cache_contract_sha256':cache_fp,'trajectory_protocol':cache_meta.get('trajectory_protocol'),'trajectory_length':cache_meta.get('trajectory_length')}
    for key,expected in checks.items():
        if meta.get(key)!=expected:raise RuntimeError(f'empty_latent mismatch at {key}: {meta.get(key)!r} != {expected!r}')


def validate_resume_checkpoint(ck,cfg,cache_fp,upstream_sha,empty_tensor):
    saved_resume=ck.get('resume_contract_sha256')
    if saved_resume is None and ck.get('resolved_config') is not None:saved_resume=config_fingerprint(ck['resolved_config'],'resume')
    current_resume=config_fingerprint(cfg,'resume')
    if saved_resume!=current_resume:raise RuntimeError('resume checkpoint config contract differs from current run')
    saved_cache=ck.get('cache_contract_sha256')
    if saved_cache is None and ck.get('cache_metadata') is not None:saved_cache=_meta_fp(ck['cache_metadata'])
    if saved_cache!=cache_fp:raise RuntimeError('resume checkpoint cache contract differs from current cache')
    if ck.get('upstream_ckpt_sha256')!=upstream_sha:raise RuntimeError('resume checkpoint upstream initialization fingerprint differs')
    if ck.get('upstream_variant')!='occfm_fut_epoch196':raise RuntimeError('resume checkpoint is not from the OccFM-fut-196 SWFM contract')
    saved_empty=ck.get('empty_latent')
    if not torch.is_tensor(saved_empty) or not torch.equal(saved_empty.cpu(),empty_tensor.cpu()):raise RuntimeError('resume checkpoint empty latent differs from supplied empty_latent.pt')
