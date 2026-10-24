"""Global aligned-memory amplitude / one XY average filter; no spatial rules."""
import torch
from torch.nn import functional as F

BASELINE=('original',1.,0.)
CANDIDATES=(BASELINE,)+tuple((mode,strength,bias) for mode,strength in
    (('original',1.),('gain',.5),('gain',0.),('smooth',.5)) for bias in (-.25,0.,.25)
    if (mode,strength,bias)!=BASELINE)


def name(c):return f'{c[0]}_memory{c[1]:g}_free{c[2]:+g}'


def process(value,mode,strength):
    if mode=='gain':return value*strength
    if mode=='smooth':
        filtered=F.avg_pool3d(value.float(),(3,3,1),stride=1,padding=(1,1,0),count_include_pad=False)
        return value+(filtered-value)*strength
    raise ValueError('explicit gain/smooth mode required')


@torch.no_grad()
def predict(body,provider,row,fw):
    result={}
    def decode(mode,strength,z):
        v=z[[1,3,5]];score,classes=v[:,:17].max(1)
        for bias in (-.25,0.,.25):
            pred=torch.where(score>=v[:,17]+bias,classes,17)
            result[name((mode,strength,bias))]=pred.reshape(3,*fw['shape']).byte().cpu().numpy()
    decode('original',1.,fw['logits']);del fw['logits']
    for mode,strength in (('gain',.5),('gain',0.),('smooth',.5)):
        mem=process(fw['memory'],mode,strength)
        z=body.columns(mem,fw['prior'],range(6),fw['shape']);decode(mode,strength,z)
        del mem,z
    return result
