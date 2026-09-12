from __future__ import annotations
import torch
class WarmupEMA:
    def __init__(self,model,half_life_optimizer_steps=100):
        self.model=model;self.half_life=int(half_life_optimizer_steps);self.started=False;self.num_updates=0;self.shadow={}
    def start(self):
        self.shadow={k:v.detach().float().clone() for k,v in self.model.state_dict().items() if torch.is_floating_point(v)};self.started=True;self.num_updates=0
    @torch.no_grad()
    def update(self):
        if not self.started: raise RuntimeError('EMA must start after warmup')
        self.num_updates+=1;n=self.num_updates;beta=min(2.0**(-1.0/self.half_life),(1.0+n)/(10.0+n))
        state=self.model.state_dict()
        for k,s in self.shadow.items(): s.mul_(beta).add_(state[k].detach().float(),alpha=1.0-beta)
        return beta
    @torch.no_grad()
    def copy_to(self,model):
        st=model.state_dict()
        for k,v in self.shadow.items():
            if k in st: st[k].copy_(v.to(dtype=st[k].dtype,device=st[k].device))
    def state_dict(self): return {'started':self.started,'num_updates':self.num_updates,'half_life':self.half_life,'shadow':self.shadow}
    def load_state_dict(self,s):
        self.started=bool(s['started']);self.num_updates=int(s['num_updates']);self.half_life=int(s.get('half_life',100));self.shadow={k:v.detach().float().clone() for k,v in s.get('shadow',{}).items()}
