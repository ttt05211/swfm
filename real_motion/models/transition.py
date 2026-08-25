"""Window-compatible OccFM transition model with aligned prior modulation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from forecast.models.modules.transition_models.FlowMatchingV2 import FLOW_MATCHING_DOWN_X4_DiT
from forecast.models.modules.base.postion_embed import timestep_embedding
from .blocks import SpatialAdaLNDiTBlock

class MotionWindowFlowMatching(FLOW_MATCHING_DOWN_X4_DiT):
    """Reuse OccFM Conv/UNet/attention weights on fixed motion windows."""
    def __init__(self,*args,prior_channels=32,**kwargs):
        super().__init__(*args,**kwargs)
        self.prior_proj=nn.Linear(prior_channels,self.model_channels,bias=True)
        nn.init.zeros_(self.prior_proj.weight); nn.init.zeros_(self.prior_proj.bias)
        self._upgrade_dit_blocks()

    def _convert_block(self,old):
        new=SpatialAdaLNDiTBlock(hidden_size=old.norm1.normalized_shape[0],num_heads=old.attn.num_heads,
            cond_size=old.adaLN_modulation[-1].in_features,attention_mode=getattr(old.attn,"attention_mode","flash"))
        new.load_state_dict(old.state_dict(),strict=True)
        return new

    def _upgrade_dit_blocks(self):
        for group in self.downs:
            group[2]=self._convert_block(group[2]); group[3]=self._convert_block(group[3])
        for group in self.ups:
            group[2]=self._convert_block(group[2]); group[3]=self._convert_block(group[3])

    @staticmethod
    def _resize_prior(prior_bfchw,hw):
        b,f,c,h,w=prior_bfchw.shape
        y=F.interpolate(prior_bfchw.reshape(b*f,c,h,w),size=hw,mode="bilinear",align_corners=False)
        return y.reshape(b,f,c,*hw)

    def _spatial_cond(self,emb,prior,hw):
        p=rearrange(self._resize_prior(prior,hw),'b f c h w -> (b f) (h w) c')
        return self.prior_proj(p)+repeat(emb,'b d -> (b f) n d',f=prior.shape[1],n=hw[0]*hw[1])

    def _temporal_cond(self,emb,prior,hw):
        p=rearrange(self._resize_prior(prior,hw),'b f c h w -> (b h w) f c')
        return self.prior_proj(p)+repeat(emb,'b d -> (b n) f d',n=hw[0]*hw[1],f=prior.shape[1])

    def forward_single(self,x,timesteps=None,trajectory=None,prior_condition=None,**kwargs):
        if prior_condition is None:
            prior_condition=torch.zeros(x.shape[0],x.shape[1],self.prior_proj.in_features,x.shape[-2],x.shape[-1],dtype=x.dtype,device=x.device)
        if prior_condition.shape[:2]!=x.shape[:2] or prior_condition.shape[-2:]!=x.shape[-2:]:
            raise ValueError("prior_condition must align with [B,F,...,H,W] of x")
        x=rearrange(x,'b f c h w -> b c f h w')
        if x.shape[2]>self.temp_embed.shape[1]:
            raise ValueError(f"sequence length {x.shape[2]} exceeds inherited temp_embed length {self.temp_embed.shape[1]}")
        time_rel_pos_bias=self.time_rel_pos_bias(x.shape[2],device=x.device)
        x=self.init_conv(x); x=x+self.init_temporal_attn(x,pos_bias=time_rel_pos_bias); r=x.clone()
        t_emb=timestep_embedding(timesteps,self.model_channels,repeat_only=False); emb=self.t_embedder(t_emb)
        if trajectory is not None:
            trajectory=trajectory[:,:self.traj_length,:]; emb=emb+self.encode_pose(trajectory)
        hskip=[]
        for idx,(block1,block2,spatial_attn,temporal_attn,identity,downsample) in enumerate(self.downs):
            x=block1(x,emb); x=block2(x,emb); hh,ww=x.shape[-2:]
            xs=rearrange(x,'b c f h w -> (b f) (h w) c')
            if idx==0:
                if self.pos_embed.shape[1]!=hh*ww: raise RuntimeError(f"pos_embed has {self.pos_embed.shape[1]} tokens but window has {hh*ww}")
                xs=xs+self.pos_embed
            xs=spatial_attn(xs,self._spatial_cond(emb,prior_condition,(hh,ww)))
            xt=rearrange(xs,'(b f) n c -> (b n) f c',b=r.shape[0],f=x.shape[2])
            if idx==0:
                te=self.temp_embed[:,:xt.shape[1],:]; xt=xt+te.repeat(r.shape[0]*hh*ww,1,1)
            xt=temporal_attn(xt,self._temporal_cond(emb,prior_condition,(hh,ww)))
            x=rearrange(xt,'(b h w) f c -> b c f h w',b=r.shape[0],h=hh,w=ww)
            hskip.append(x); x=downsample(x)
        x=self.mid_block1(x,emb); x=self.mid_spatial_attn(x); x=self.mid_temporal_attn(x,pos_bias=time_rel_pos_bias); x=self.mid_block2(x,emb)
        for block1,block2,spatial_attn,temporal_attn,upsample in self.ups:
            x=torch.cat((x,hskip.pop()),dim=1); x=block1(x,emb); x=block2(x,emb); hh,ww=x.shape[-2:]
            xs=rearrange(x,'b c f h w -> (b f) (h w) c'); xs=spatial_attn(xs,self._spatial_cond(emb,prior_condition,(hh,ww)))
            xt=rearrange(xs,'(b f) n c -> (b n) f c',b=r.shape[0],f=x.shape[2]); xt=temporal_attn(xt,self._temporal_cond(emb,prior_condition,(hh,ww)))
            x=rearrange(xt,'(b h w) f c -> b c f h w',b=r.shape[0],h=hh,w=ww); x=upsample(x)
        x=self.final_conv(torch.cat((x,r),dim=1))
        return rearrange(x,'b c f h w -> b f c h w')

    def forward(self,batch_dict):
        batch_dict['predicted_latent']=self.forward_single(batch_dict['noised_sequence'],batch_dict['timesteps'],batch_dict.get('trajectory'),prior_condition=batch_dict.get('prior_condition'))
        return batch_dict
