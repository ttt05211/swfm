import copy
import torch
from real_motion.runtime_config import load_runtime_config,config_fingerprint
from real_motion.training_contract import validate_cache_pair,validate_empty_asset


def _meta(cfg):
    return {
        'vae_ckpt_sha256':'abc','latent_mode':cfg['CACHE']['VAE_LATENT_MODE'],
        'latent_extra_radius':cfg['MOTION']['LATENT_EXTRA_RADIUS'],
        'vae_amp_enabled':cfg['RUNTIME']['VAE_AMP']['ENABLED'],
        'vae_seed_contract':'per_sample_index_v1',
        'trajectory_protocol':cfg['EGO_PROTOCOL']['NAME'],
        'trajectory_length':cfg['EGO_PROTOCOL']['TRAJECTORY_LENGTH'],
        'upstream_wm_variant':'occfm_fut',
        'upstream_init_variant':'fut_traj_196',
        'cache_contract_sha256':config_fingerprint(cfg,'cache'),
        'resolved_config':cfg,
    }


def test_cache_contract_detects_motion_mismatch():
    cfg=load_runtime_config();train=_meta(cfg);val=_meta(cfg)
    assert validate_cache_pair(train,val,cfg)==config_fingerprint(cfg,'cache')
    bad=copy.deepcopy(cfg);bad['MOTION']['KTA_TUBE_RADII']=[2,2,3,4,5,6]
    val_bad=_meta(bad)
    try:
        validate_cache_pair(train,val_bad,cfg)
    except RuntimeError:
        pass
    else:
        raise AssertionError('motion/support mismatch must fail closed')


def test_empty_asset_must_match_cache():
    cfg=load_runtime_config();meta=_meta(cfg);fp=config_fingerprint(cfg,'cache')
    empty_meta={'vae_ckpt_sha256':'abc','mode':cfg['CACHE']['VAE_LATENT_MODE'],
                'vae_amp_enabled':cfg['RUNTIME']['VAE_AMP']['ENABLED'],
                'cache_contract_sha256':fp,
                'trajectory_protocol':cfg['EGO_PROTOCOL']['NAME'],
                'trajectory_length':cfg['EGO_PROTOCOL']['TRAJECTORY_LENGTH']}
    validate_empty_asset(empty_meta,meta,fp)
    empty_meta['vae_ckpt_sha256']='wrong'
    try:
        validate_empty_asset(empty_meta,meta,fp)
    except RuntimeError:
        pass
    else:
        raise AssertionError('VAE mismatch must fail closed')
