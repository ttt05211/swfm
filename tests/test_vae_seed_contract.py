import torch
from real_motion.occfm_io import _seeded_eps_like


def test_per_sample_vae_eps_is_batch_grouping_invariant():
    seeds=[101,202]
    mu=torch.zeros(2*3,4,2,2)
    together=_seeded_eps_like(mu,seeds,logical_batch=2,frames_per_sample=3)
    first=_seeded_eps_like(mu[:3],[101],logical_batch=1,frames_per_sample=3)
    second=_seeded_eps_like(mu[3:],[202],logical_batch=1,frames_per_sample=3)
    assert torch.equal(together[:3],first)
    assert torch.equal(together[3:],second)


def test_scalar_seed_keeps_legacy_whole_batch_behavior():
    mu=torch.zeros(6,2,2,2)
    a=_seeded_eps_like(mu,123,logical_batch=2,frames_per_sample=3)
    b=_seeded_eps_like(mu,123,logical_batch=2,frames_per_sample=3)
    assert torch.equal(a,b)
