from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .contracts import CropBatch


def _groups(channels: int, max_groups: int = 8) -> int:
    for g in range(min(int(max_groups), int(channels)), 0, -1):
        if channels % g == 0:
            return g
    return 1


class Conv2dGN(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.norm = nn.GroupNorm(_groups(cout), cout)
    def forward(self, x):
        return F.relu(self.norm(self.conv(x)), inplace=False)


class TemporalGN(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, (3,1,1), padding=0, bias=False)
        self.norm = nn.GroupNorm(_groups(channels), channels)
    def forward(self, x):
        return F.relu(self.norm(self.conv(x)), inplace=False)


class STPNMotionNetwork(nn.Module):
    """MT-V1-SPEC-2 STPN adaptation. Output is cumulative KTA residual [M,6,3]."""
    def __init__(self, class_tokens: int = 19, embedding_dim: int = 4, metadata_dim: int = 19,
                 future_frames: int = 6, activation_checkpointing: bool = False):
        super().__init__()
        self.class_tokens = int(class_tokens)
        self.embedding_dim = int(embedding_dim)
        self.future_frames = int(future_frames)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.embedding = nn.Embedding(self.class_tokens, self.embedding_dim)
        self.stem1 = Conv2dGN(84, 32);self.stem2 = Conv2dGN(32, 32)
        self.s1a = Conv2dGN(32, 64, stride=2);self.s1b = Conv2dGN(64, 64);self.t1 = TemporalGN(64)
        self.s2a = Conv2dGN(64, 128, stride=2);self.s2b = Conv2dGN(128, 128);self.t2 = TemporalGN(128)
        self.s3a = Conv2dGN(128, 256, stride=2);self.s3b = Conv2dGN(256, 256)
        self.d2a = Conv2dGN(256 + 128, 128);self.d2b = Conv2dGN(128, 128)
        self.d1a = Conv2dGN(128 + 64, 64);self.d1b = Conv2dGN(64, 64)
        self.d0a = Conv2dGN(64 + 32, 32);self.d0b = Conv2dGN(32, 32)
        self.meta = nn.Sequential(nn.Linear(metadata_dim, 32), nn.ReLU(inplace=False))
        self.head1 = nn.Linear(128, 128);self.head2 = nn.Linear(128, self.future_frames * 3)
        self._init_weights();self.last_sources_executed = 0

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None: nn.init.zeros_(module.bias)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02);nn.init.zeros_(self.head2.weight);nn.init.zeros_(self.head2.bias)

    @staticmethod
    def _frame_apply(module, x):
        M,T,C,H,W=x.shape;y=module(x.reshape(M*T,C,H,W));return y.reshape(M,T,y.shape[1],y.shape[2],y.shape[3])

    def _features(self, semantics, valid, source_mask, relative_times, metadata, mirror_flags):
        if semantics.ndim != 5:raise ValueError('semantics must be [M,T,X,Y,Z]')
        M,T,X,Y,Z=semantics.shape
        if (X,Y,Z)!=(64,64,16) or T!=6:raise ValueError(f'SPEC-2 expects [M,6,64,64,16], got {tuple(semantics.shape)}')
        sem=semantics.long();val=valid.bool();sm=source_mask.bool();meta=metadata.float();flags=mirror_flags.bool()
        if bool(flags.any()):
            sem=torch.where(flags[:,None,None,None,None],sem.flip(3),sem);val=torch.where(flags[:,None,None,None,None],val.flip(3),val);sm=torch.where(flags[:,None,None],sm.flip(2),sm);meta=meta.clone();meta[:,1]=torch.where(flags,-meta[:,1],meta[:,1]);meta[:,3]=torch.where(flags,-meta[:,3],meta[:,3])
        emb=self.embedding(sem.clamp(0,self.class_tokens-1)).permute(0,1,4,5,2,3).contiguous().reshape(M,T,Z*self.embedding_dim,X,Y)
        v=val.permute(0,1,4,2,3).float();src=sm[:,None,None].expand(M,T,1,X,Y).float();xs=(torch.arange(X,device=sem.device,dtype=torch.float32)+.5)/X*2-1;ys=(torch.arange(Y,device=sem.device,dtype=torch.float32)+.5)/Y*2-1;gx,gy=torch.meshgrid(xs,ys,indexing='ij');xy=torch.stack([gx,gy],0)[None,None].expand(M,T,2,X,Y);tau=(relative_times.float()/2.5)[:,:,None,None,None].expand(M,T,1,X,Y);feat=torch.cat([emb,v,src,xy,tau],2)
        if feat.shape[2]!=84:raise AssertionError(feat.shape)
        return feat,sm,meta,flags

    def _forward_impl(self,semantics,valid,source_mask,relative_times,metadata,mirror_flags):
        x,sm,meta,flags=self._features(semantics,valid,source_mask,relative_times,metadata,mirror_flags);stem=self._frame_apply(self.stem2,self._frame_apply(self.stem1,x));s1=self._frame_apply(self.s1b,self._frame_apply(self.s1a,stem));t1=self.t1(s1.permute(0,2,1,3,4)).permute(0,2,1,3,4);s2=self._frame_apply(self.s2b,self._frame_apply(self.s2a,t1));t2=self.t2(s2.permute(0,2,1,3,4)).permute(0,2,1,3,4);s3=self.s3b(self.s3a(t2.mean(1)));d2=self.d2b(self.d2a(torch.cat([F.interpolate(s3,size=(16,16),mode='bilinear',align_corners=False),t2.mean(1)],1)));d1=self.d1b(self.d1a(torch.cat([F.interpolate(d2,size=(32,32),mode='bilinear',align_corners=False),t1.mean(1)],1)));d0=self.d0b(self.d0a(torch.cat([F.interpolate(d1,size=(64,64),mode='bilinear',align_corners=False),stem[:,-1]],1)));mask=sm[:,None].to(d0.dtype);mean=(d0*mask).sum((2,3))/mask.sum((2,3)).clamp_min(1);mx=d0.masked_fill(~sm[:,None],float('-inf')).amax((2,3))
        if not torch.isfinite(mx).all():raise RuntimeError('source BEV mask is empty')
        pooled=torch.cat([mean,mx,d0.mean((2,3))],1);h=F.relu(self.head1(torch.cat([pooled,self.meta(meta)],1)),inplace=False);out=self.head2(h).view(-1,self.future_frames,3)
        if bool(flags.any()):
            out=out.clone();sign=torch.where(flags,-torch.ones_like(flags,dtype=out.dtype),torch.ones_like(flags,dtype=out.dtype));out[:,:,1]*=sign[:,None];out[:,:,2]*=sign[:,None]
        return out

    def forward(self,batch:CropBatch,source_microbatch:int=16):
        M=batch.num_sources;self.last_sources_executed=int(M)
        if M==0:
            zero=sum((p.reshape(-1)[0]*0.0) for p in self.parameters() if p.numel());return torch.zeros((0,self.future_frames,3),device=batch.metadata.device),zero
        outs=[];step=max(1,int(source_microbatch))
        for lo in range(0,M,step):
            args=(batch.semantics[lo:lo+step],batch.valid[lo:lo+step],batch.source_mask[lo:lo+step],batch.relative_times[lo:lo+step],batch.metadata[lo:lo+step],batch.mirror_flags[lo:lo+step]);y=checkpoint(self._forward_impl,*args,use_reentrant=False) if self.activation_checkpointing and self.training else self._forward_impl(*args);outs.append(y)
        out=torch.cat(outs);return out,out.sum()*0.0
