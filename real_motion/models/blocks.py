import torch
import torch.nn as nn

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

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
        shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp=self.adaLN_modulation(c).chunk(6,dim=-1)
        if c.ndim==2:
            shift_msa=shift_msa[:,None]; scale_msa=scale_msa[:,None]; gate_msa=gate_msa[:,None]
            shift_mlp=shift_mlp[:,None]; scale_mlp=scale_mlp[:,None]; gate_mlp=gate_mlp[:,None]
        x=x+gate_msa*self.attn(modulate(self.norm1(x),shift_msa,scale_msa))
        x=x+gate_mlp*self.mlp(modulate(self.norm2(x),shift_mlp,scale_mlp))
        return x
