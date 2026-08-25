import torch
import torch.nn as nn
from einops import rearrange

class RealMotionWindowCFM(nn.Module):
    """CFM trainer/sampler for windowed moving latent prediction."""
    def __init__(self, transition: nn.Module, rescale_factor=10.0,
                 sample_steps=10, alpha_shift=3.0):
        super().__init__()
        self.transition=transition
        self.rescale_factor=float(rescale_factor)
        self.sample_steps=int(sample_steps)
        self.alpha_shift=float(alpha_shift)
        self.time_scalar=1000.0

    @staticmethod
    def _sample_t(bs,device,dtype):
        return torch.sigmoid(torch.randn(bs,1,1,1,1,device=device,dtype=dtype))

    def flow_loss(self,history,future,prior,active_mask,trajectory=None,window_origins=None):
        """Inputs [B,F,C,H,W], active_mask [B,F,1,H,W]."""
        history=history*self.rescale_factor
        future=future*self.rescale_factor
        bs=future.shape[0]
        t=self._sample_t(bs,future.device,future.dtype)
        z0=torch.randn_like(future)
        noised=t*future+(1-t)*z0
        seq=torch.cat([history,noised],dim=1)
        # Prior covers full history+future sequence. If only future is supplied,
        # prepend zeros for history to preserve temporal alignment.
        if prior.shape[1]==future.shape[1]:
            prior=torch.cat([torch.zeros(
                prior.shape[0],history.shape[1],*prior.shape[2:],
                device=prior.device,dtype=prior.dtype),prior],dim=1)
        batch={
            "noised_sequence":seq,
            "timesteps":(t[:,0,0,0,0]*self.time_scalar),
            "trajectory":trajectory,
            "prior_condition":prior,
            "window_origins":window_origins,
        }
        pred=self.transition(batch)["predicted_latent"][:,history.shape[1]:]
        target=future-z0
        mask=active_mask.to(pred.dtype)
        if mask.shape[2]==1 and pred.shape[2]!=1:
            mask=mask.expand(-1,-1,pred.shape[2],-1,-1)
        denom=mask.sum().clamp_min(1.0)
        loss=((pred-target).square()*mask).sum()/denom
        return loss,{"pred":pred,"target":target,"loss_mask_fraction":active_mask.float().mean().detach()}

    @torch.no_grad()
    def sample(self,history,future_shape,prior,trajectory=None,window_origins=None):
        """Single-pass-per-step sampler: no CFG by design."""
        hist=history*self.rescale_factor
        future=torch.randn(future_shape,device=hist.device,dtype=hist.dtype)
        timesteps=torch.linspace(0,1,self.sample_steps+1,device=hist.device,dtype=hist.dtype)
        shifted=1-(self.alpha_shift*timesteps)/(1+(self.alpha_shift-1)*timesteps)
        shifted=shifted.flip(0)
        if prior.shape[1]==future.shape[1]:
            prior=torch.cat([torch.zeros(
                prior.shape[0],hist.shape[1],*prior.shape[2:],
                device=prior.device,dtype=prior.dtype),prior],dim=1)
        for tc,tp in zip(shifted[:-1],shifted[1:]):
            seq=torch.cat([hist,future],dim=1)
            batch={
                "noised_sequence":seq,
                "timesteps":torch.full((hist.shape[0],),float(tc*self.time_scalar),device=hist.device,dtype=hist.dtype),
                "trajectory":trajectory,
                "prior_condition":prior,
                "window_origins":window_origins,
            }
            v=self.transition(batch)["predicted_latent"][:,hist.shape[1]:]
            future=future+(tp-tc)*v
        return future/self.rescale_factor
