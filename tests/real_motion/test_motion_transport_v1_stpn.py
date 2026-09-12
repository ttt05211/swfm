import copy,numpy as np,torch
from real_motion.motion_transport_v1.contracts import CropBatch
from real_motion.motion_transport_v1.stpn import STPNMotionNetwork
def batch(M=3,mirror=False):
    gen=torch.Generator().manual_seed(7);sem=torch.randint(0,19,(M,6,64,64,16),generator=gen);valid=torch.ones_like(sem,dtype=torch.bool);mask=torch.zeros((M,64,64),dtype=torch.bool);mask[:,28:36,29:35]=True;times=torch.tensor([-2.5,-2,-1.5,-1,-.5,0.]).repeat(M,1);meta=torch.randn((M,19),generator=gen);return CropBatch(np.arange(M),sem,valid,mask,times,meta,torch.full((M,),mirror,dtype=torch.bool))
def test_shape_and_zero_identity():
    m=STPNMotionNetwork();m.eval();o,_=m(batch(2),source_microbatch=1);assert o.shape==(2,6,3) and torch.equal(o,torch.zeros_like(o)) and m.last_sources_executed==2
def test_chunk_unchunk_outputs_and_gradients_match():
    a=STPNMotionNetwork();b=copy.deepcopy(a)
    with torch.no_grad():a.head2.weight.normal_(0,.01);b.head2.weight.copy_(a.head2.weight)
    x=batch(3);ya,_=a(x,source_microbatch=3);yb,_=b(x,source_microbatch=1);assert torch.allclose(ya,yb,atol=2e-5,rtol=2e-5);ya.square().mean().backward();yb.square().mean().backward()
    for (na,pa),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
        if pa.grad is not None or pb.grad is not None:assert pa.grad is not None and pb.grad is not None and torch.allclose(pa.grad,pb.grad,atol=3e-5,rtol=3e-4),na
def test_checkpoint_embedding_and_empty_and_mirror():
    m=STPNMotionNetwork(activation_checkpointing=True);m.train()
    with torch.no_grad():m.head2.weight.normal_(0,.02)
    y,_=m(batch(1),source_microbatch=1);y.square().mean().backward();assert m.embedding.weight.grad is not None and float(m.embedding.weight.grad.abs().sum())>0
    e=STPNMotionNetwork();y,z=e(batch(0));assert y.shape==(0,6,3);z.backward();assert any(p.grad is not None for p in e.parameters())
    m=STPNMotionNetwork();m.eval()
    with torch.no_grad():m.head2.weight.normal_(0,.02);m.head2.bias.normal_(0,.01)
    x=batch(1,True);y,_=m(x);man=CropBatch(x.source_ids,x.semantics.flip(3),x.valid.flip(3),x.source_mask.flip(2),x.relative_times,x.metadata.clone(),torch.zeros_like(x.mirror_flags));man.metadata[:,1]*=-1;man.metadata[:,3]*=-1;yp,_=m(man);assert torch.allclose(y[:,:,0],yp[:,:,0],atol=1e-6) and torch.allclose(y[:,:,1],-yp[:,:,1],atol=1e-6) and torch.allclose(y[:,:,2],-yp[:,:,2],atol=1e-6)
