import importlib.util, pathlib, torch
spec=importlib.util.spec_from_file_location('cfm_mod',pathlib.Path(__file__).parents[1]/'real_motion/models/cfm.py')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
RealMotionWindowCFM=mod.RealMotionWindowCFM

class ZeroTransition(torch.nn.Module):
    def forward(self,b):
        b['predicted_latent']=torch.zeros_like(b['noised_sequence']); return b

def test_sampler_clamps_outside_support_and_accepts_global_noise():
    m=RealMotionWindowCFM(ZeroTransition(),rescale_factor=10,sample_steps=2)
    hist=torch.zeros(1,2,1,3,3); prior=torch.zeros(1,1,1,3,3)
    active=torch.zeros(1,1,1,3,3,dtype=torch.bool); active[...,1,1]=1
    known=torch.full((1,1,1,3,3),2.0); noise=torch.ones_like(known)
    out=m.sample(hist,known.shape,prior,active,known,initial_noise=noise)
    outside=~active.expand_as(out); assert torch.allclose(out[outside],known[outside])

class CaptureTransition(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.last=None
    def forward(self,b):
        self.last={k:v.clone() if torch.is_tensor(v) else v for k,v in b.items()}
        b['predicted_latent']=torch.zeros_like(b['noised_sequence']); return b


def test_future_prior_is_prefixed_and_rescaled_like_occfm_latents():
    tr=CaptureTransition(); m=RealMotionWindowCFM(tr,rescale_factor=10,sample_steps=1)
    hist=torch.zeros(1,2,1,2,2)
    future=torch.zeros(1,1,1,2,2)
    prior=torch.full((1,1,2,2,2),0.25)
    active=torch.ones(1,1,1,2,2,dtype=torch.bool)
    known=torch.zeros_like(future)
    m.flow_loss(hist,future,prior,active,known)
    got=tr.last['prior_condition']
    assert got.shape[:2]==(1,3)
    assert torch.equal(got[:,:2],torch.zeros_like(got[:,:2]))
    assert torch.allclose(got[:,2:],torch.full_like(got[:,2:],2.5))


def test_prior_temporal_length_is_not_silently_guessed():
    m=RealMotionWindowCFM(ZeroTransition(),rescale_factor=10,sample_steps=1)
    hist=torch.zeros(1,2,1,2,2)
    future=torch.zeros(1,1,1,2,2)
    bad_prior=torch.zeros(1,2,2,2,2)  # neither F=1 nor H+F=3
    active=torch.ones(1,1,1,2,2,dtype=torch.bool)
    known=torch.zeros_like(future)
    import pytest
    with pytest.raises(ValueError):
        m.flow_loss(hist,future,bad_prior,active,known)
