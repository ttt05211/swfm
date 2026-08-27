import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def _factorized_tokenwise_modulation(adaLN_modulation, c):
    """Evaluate Linear(SiLU(c)) while preserving the official 2-D base path.

    For token-wise ``c=[B,N,D]`` we factor the affine map around the first token:

        L(SiLU(c_n)) = L(SiLU(c_0)) + W @ (SiLU(c_n)-SiLU(c_0))

    where ``L(z)=Wz+b``.  This is mathematically identical to applying the
    original AdaLN MLP to every token.  The important numerical property is that
    when all token conditions are identical (the zero-prior initialization), the
    delta is exactly zero and the base modulation is computed with the same
    ``[B,D]`` GEMM shape as the official OccFM DiTBlock.  That keeps the frozen
    OccFM transition-equivalence gate strict instead of relaxing its tolerance.
    """
    if c.ndim != 3:
        raise ValueError("factorized token-wise modulation expects [B,N,D]")
    act, linear = adaLN_modulation[0], adaLN_modulation[1]
    base_c = c[:, 0, :]
    base_act = act(base_c)
    token_act = act(c)
    base_mod = linear(base_act)
    delta_act = token_act - base_act[:, None, :]
    delta_mod = F.linear(delta_act, linear.weight, bias=None)
    return base_mod[:, None, :] + delta_mod


class SpatialAdaLNDiTBlock(nn.Module):
    """State-dict-compatible AdaLN-Zero block with token-wise condition."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, cond_size=None,
                 attention_cls=None, mlp_cls=None, **block_kwargs):
        super().__init__()
        if attention_cls is None or mlp_cls is None:
            from forecast.models.modules.base.attn_base import DiTAttention
            from forecast.models.modules.base.others import Mlp
            attention_cls, mlp_cls = DiTAttention, Mlp
        self.norm1=nn.LayerNorm(hidden_size,elementwise_affine=False,eps=1e-6)
        self.attn=attention_cls(hidden_size,num_heads=num_heads,qkv_bias=True,**block_kwargs)
        self.norm2=nn.LayerNorm(hidden_size,elementwise_affine=False,eps=1e-6)
        self.mlp=mlp_cls(in_features=hidden_size,hidden_features=int(hidden_size*mlp_ratio),act_layer=lambda:nn.GELU(approximate="tanh"),drop=0)
        self.adaLN_modulation=nn.Sequential(nn.SiLU(),nn.Linear(cond_size if cond_size else hidden_size,6*hidden_size,bias=True))
        nn.init.constant_(self.adaLN_modulation[-1].weight,0); nn.init.constant_(self.adaLN_modulation[-1].bias,0)

    def forward(self,x,c):
        if c.ndim not in (2,3): raise ValueError("condition must be [B,D] or [B,N,D]")
        # Keep the official OccFM DiTBlock path literally 2-D when conditioning
        # is sequence-wise.  Token-wise conditioning uses an algebraically exact
        # factorization so the zero-prior case collapses to this same base path.
        modulation = self.adaLN_modulation(c) if c.ndim == 2 else _factorized_tokenwise_modulation(self.adaLN_modulation, c)
        shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp=modulation.chunk(6,dim=-1)
        if c.ndim==2:
            shift_msa=shift_msa[:,None]; scale_msa=scale_msa[:,None]; gate_msa=gate_msa[:,None]
            shift_mlp=shift_mlp[:,None]; scale_mlp=scale_mlp[:,None]; gate_mlp=gate_mlp[:,None]
        x=x+gate_msa*self.attn(modulate(self.norm1(x),shift_msa,scale_msa))
        x=x+gate_mlp*self.mlp(modulate(self.norm2(x),shift_mlp,scale_mlp))
        return x
