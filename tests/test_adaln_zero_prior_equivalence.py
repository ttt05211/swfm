import torch
import torch.nn as nn

from real_motion.models.blocks import _factorized_tokenwise_modulation


def test_factorized_tokenwise_modulation_matches_official_base_for_repeated_condition():
    torch.manual_seed(7)
    b, n, d, h = 3, 11, 16, 8
    mod = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * h, bias=True))
    c = torch.randn(b, d)
    official = mod(c)[:, None, :].expand(b, n, 6 * h)
    tokenwise = _factorized_tokenwise_modulation(mod, c[:, None, :].expand(b, n, d))
    # Sequence-wise and token-wise SiLU kernels can differ at the last few fp32 bits
    # even for repeated inputs.  The end-to-end transition gate remains strict and
    # must reproduce official OccFM-fut exactly under the real zero-prior path.
    torch.testing.assert_close(tokenwise, official, rtol=1e-6, atol=1e-7)


def test_factorized_tokenwise_modulation_matches_direct_tokenwise_math():
    torch.manual_seed(11)
    b, n, d, h = 2, 5, 12, 6
    mod = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * h, bias=True))
    c = torch.randn(b, n, d)
    direct = mod(c)
    factorized = _factorized_tokenwise_modulation(mod, c)
    torch.testing.assert_close(factorized, direct, rtol=2e-6, atol=2e-6)
