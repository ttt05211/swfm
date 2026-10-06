"""Height-aware causal field readout: encode four native grids ONCE.

Two experimental architectures share the exact old legal proposal population:
temporal_gate (centre evidence only), shared_field (shared spatial context).
Neither constructs sorted global sparse evidence/halos or per-query patch CNNs.
This is a NEW predictor, not a byte-exact execution backend for the old model.
"""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from .causal_column_completion import UNKNOWN, KEEP, ADD, GENERATE, ColumnPlan
from .causal_column_sampling import ColumnHistoryIndex
from .column_gpu_sampling import pack_column_window


def concatenate_plans(rows):
    if not rows: raise ValueError('nonempty horizon list required')
    return ColumnPlan(**{k:np.concatenate([getattr(p,k) for _,p,_,_ in rows])
                         for k in ColumnPlan.__dataclass_fields__})


@dataclass
class CentreEvidence:
    labels: torch.Tensor             # Q,T,Z, actual native height per query
    observed: torch.Tensor
    owned: torch.Tensor
    inside: torch.Tensor
    native_height: torch.Tensor
    field_address: torch.Tensor      # Q,T,Z, native XY address with frame offset
    actors: torch.Tensor
    horizons: torch.Tensor
    base: torch.Tensor
    fallback: torch.Tensor
    context: torch.Tensor
    kind: torch.Tensor
    classes: torch.Tensor
    legal: torch.Tensor
    boundary_cpu_rows: int = 0
    aligned_density: torch.Tensor | None = None

    def subset(self, ids):
        return CentreEvidence(**{k:(v[ids] if isinstance(v,torch.Tensor) else v) for k,v in vars(self).items()})

    def to(self, device):
        return CentreEvidence(**{k:(v.to(device) if isinstance(v,torch.Tensor) else v) for k,v in vars(self).items()})


@torch.no_grad()
def gather_centres(prepared, rows, grid, config, device, motion_factory):
    """Full SE(3), all Z, ownership, UNKNOWN. No future GT enters this path.

    Same source inverse matrices as original sampler, but only column centres,
    not 49 spatial points at every height. FP64 floor-boundary rows use CPU
    reference arithmetic BEFORE addresses/labels/features are gathered.
    """
    actors={int(a) for _,p,_,_ in rows for a in np.unique(p.actor) if a>=0}
    index=ColumnHistoryIndex(prepared,grid,actors=actors)
    packed=pack_column_window(prepared,rows,grid,config,motion_factory,history_index=index)
    shape=tuple(grid.shape_hwd); frames=len(prepared.raw['history_occ']); n=sum(packed.sizes)
    if frames!=4: raise ValueError('strict four-history field')
    history=torch.as_tensor(np.ascontiguousarray(prepared.raw['history_occ']),device=device).flatten()
    observed=torch.as_tensor(np.ascontiguousarray(prepared.raw['history_observed']),device=device).flatten()
    origin=torch.tensor([grid.x_min,grid.y_min,grid.z_min],device=device,dtype=torch.float64)
    step=torch.tensor(grid.voxel_size,device=device,dtype=torch.float64)
    centres=torch.as_tensor(packed.evidence_xy,device=device)
    zz=torch.arange(shape[2],device=device)
    coords=torch.cat((centres[:,None,:].expand(n,shape[2],2),zz[None,:,None].expand(n,shape[2],1)),dim=-1)
    xyz=origin+(coords.to(torch.float64)+.5)*step
    matrix=torch.as_tensor(packed.transforms,device=device)
    terms=[matrix[:,:,:3,i,None].transpose(-1,-2)*xyz[:,None,:,i,None] for i in range(3)]
    mapped=terms[0]+terms[1]+terms[2]+matrix[:,:,None,:3,3]
    uvw=(mapped-origin)/step
    valid_frames=torch.as_tensor(packed.valid_frames,device=device)
    scale=(sum(t.abs() for t in terms)+matrix[:,:,None,:3,3].abs()+origin.abs())/step.abs()
    error=128*np.finfo(np.float64).eps*(scale+uvw.abs()+1)
    near=(((uvw-uvw.round()).abs()<=error).any(-1)&valid_frames[:,:,None]).any(dim=(1,2))
    uncertain=near.cpu().numpy()
    if uncertain.any():
        points=xyz.cpu().numpy()[uncertain]
        # ORIGINAL transform_points multiplication order on whole Z rows.
        from .source_evidence_audit import transform_points
        cpu=[]
        for row_id,p in zip(np.flatnonzero(uncertain),points):
            cpu.append(np.stack([(transform_points(p,m)-origin.cpu().numpy())/step.cpu().numpy()
                                 for m in packed.transforms[row_id]]))
        uvw[torch.as_tensor(uncertain,device=device)]=torch.as_tensor(np.asarray(cpu),device=device)
    cells=int(np.prod(shape)); limits=torch.tensor(shape,device=device)
    ids=uvw.floor().long()
    inside=((ids>=0)&(ids<limits)).all(-1)&valid_frames[:,:,None]
    flat=(ids[...,0]*shape[1]+ids[...,1])*shape[2]+ids[...,2]
    safe=flat.clamp(0,cells-1); frame=torch.arange(frames,device=device)[None,:,None]
    labels=history[frame*cells+safe]; visibility=observed[frame*cells+safe]
    owned=torch.zeros_like(inside); members=torch.as_tensor(packed.membership,device=device)
    if len(members):
        slot=torch.as_tensor(packed.slots,device=device)[:,None,None]
        key=(slot*frames+frame)*cells+safe
        at=torch.searchsorted(members,key)
        owned=(at<len(members))&(members[at.clamp(max=len(members)-1)]==key)
    cls=torch.as_tensor(packed.classes,device=device)
    dynamic=torch.as_tensor(packed.dynamic,device=device)[:,None,None]
    owned=torch.where(dynamic,owned,labels==cls[:,None,None])&inside
    labels=torch.where(inside,labels,UNKNOWN)
    address=frame*shape[0]*shape[1]+ids[...,0].clamp(0,shape[0]-1)*shape[1]+ids[...,1].clamp(0,shape[1]-1)
    plan=concatenate_plans(rows)
    tensor=lambda x:torch.as_tensor(np.ascontiguousarray(x),device=device)
    return CentreEvidence(labels,visibility&inside,owned,inside,
        ids[...,2].clamp(0,shape[2]-1).float()/max(shape[2]-1,1),address,
        tensor(plan.actor),tensor(np.concatenate([np.full(len(p),h,np.int64) for h,p,_,_ in rows])),
        tensor(plan.base),tensor(plan.fallback),tensor(plan.context),tensor(plan.kind),tensor(plan.classes),
        tensor(plan.legal),int(uncertain.sum()))


class HeightCausalField(nn.Module):
    """Native ordered-height encoding + centre readout + source future query.

    Shared-field encodes each historical map once, not once per spatial query.
    Direct sampled semantics/visibility/ownership retain actual per-height
    evidence alongside the contextual BEV feature; no height max pooling.
    """
    def __init__(self, mode='shared_field', *, z_bins=16, source_dim=128, width=32, semantic_dim=4):
        super().__init__()
        if mode not in ('temporal_gate','shared_field','forward_field'):raise ValueError('unknown experimental architecture')
        self.mode=mode;self.z_bins=z_bins;self.source_dim=source_dim;self.width=width
        e=semantic_dim;self.semantic=nn.Embedding(19,e);field_dim=16
        self.column=nn.Linear(z_bins*(e+1),field_dim)
        self.spatial=nn.Sequential(*[nn.Sequential(nn.Conv2d(field_dim,field_dim,3,padding=1,groups=field_dim),
            nn.SiLU(),nn.Conv2d(field_dim,field_dim,1)) for _ in range(3)])
        sample_width=e+5+(field_dim if mode=='shared_field' else 0)+(1 if mode=='forward_field' else 0)
        self.history=nn.Linear(4*sample_width,width)
        self.query=nn.Linear(12+2*e,width)
        self.source=nn.Linear(2*source_dim,width,bias=False)
        self.existence=nn.Linear(1,width,bias=False)
        self.class_embedding=nn.Embedding(17,width);self.kind_embedding=nn.Embedding(2,width)
        self.height=nn.Embedding(z_bins,width)
        self.readout=nn.Sequential(nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.generation=nn.Linear(width,1);self.refinement=nn.Linear(width,3)
        nn.init.normal_(self.generation.weight,std=.01);nn.init.constant_(self.generation.bias,-2.)
        nn.init.normal_(self.refinement.weight,std=.01);nn.init.zeros_(self.refinement.bias)
        with torch.no_grad():self.refinement.bias[KEEP]=2.
        self.register_buffer('generation_pos_weight',torch.ones(()))
        self.register_buffer('refine_class_weights',torch.ones(3))

    def encode_history(self, history, observed):
        if history.ndim!=4 or history.shape[0]!=4 or history.shape[-1]!=self.z_bins or observed.shape!=history.shape:
            raise ValueError('four complete native grids and visibility required')
        if self.mode!='shared_field':return None
        valid=history!=UNKNOWN
        emb=self.semantic(history.long())*valid[...,None]
        x=torch.cat((emb,(observed&valid)[...,None].to(emb.dtype)),dim=-1).flatten(-2)
        x=self.column(x)*valid.any(-1)[...,None]
        x=self.spatial(x.permute(0,3,1,2)).permute(0,2,3,1)
        return x.reshape(-1,x.shape[-1])

    def forward(self, field, samples, output):
        n,t,z=samples.labels.shape
        if (t,z)!=(4,self.z_bins):raise ValueError('sample history/height contract mismatch')
        valid=samples.inside&(samples.labels!=UNKNOWN)
        raw=self.semantic(samples.labels.long())*valid[...,None]
        bits=torch.stack((samples.observed,samples.owned,valid,
            (samples.labels==samples.classes[:,None,None])&valid,
            samples.native_height*valid),dim=-1).to(raw.dtype)
        parts=[raw,bits]
        if self.mode=='forward_field':
            if samples.aligned_density is None:raise ValueError('source-aligned density required')
            parts.append(samples.aligned_density[...,None].to(raw.dtype))
        if self.mode=='shared_field':parts.append(field[samples.field_address]*valid[...,None])
        memory=torch.cat(parts,dim=-1).permute(0,2,1,3).flatten(-2)
        q=self.query(torch.cat((samples.context[:,None,:].expand(n,z,12),
            self.semantic(samples.base.long()),self.semantic(samples.fallback.long())),dim=-1))
        q=q+self.class_embedding(samples.classes.long())[:,None]+self.kind_embedding(samples.kind.long())[:,None]
        live=output['history_source_context'];future=output['future_transport_queries']
        if live.shape[0]:
            safe=samples.actors.clamp(min=0).long()
            source=torch.cat((live[safe],future[safe,samples.horizons.long()]),dim=-1)
            extra=self.source(source)*((samples.actors>=0)[:,None])
            if 'existence_logits' in output:
                prob=output['existence_logits'][safe,samples.horizons.long()].sigmoid()[:,None]
                extra=extra+self.existence(prob)*((samples.actors>=0)[:,None])
            q=q+extra[:,None]
        q=q+self.history(memory)+self.height(torch.arange(z,device=q.device))[None]
        x=self.readout(q)
        return self.generation(x)[...,0],self.refinement(x)

    def probabilities(self, generation, refinement, samples):
        r=(refinement.float()-self.refine_class_weights.log()).masked_fill(~samples.legal,-torch.inf).softmax(-1)
        g=(generation.float()-self.generation_pos_weight.log()).sigmoid()
        add=g*samples.legal[...,ADD]
        gp=torch.stack((1-add,add,torch.zeros_like(add)),dim=-1)
        return torch.where((samples.kind==GENERATE)[:,None,None],gp,r)


@torch.no_grad()
def gather_forward_fields(prepared,rows,grid,device):
    """Direct temporal evidence transport, not inverse nearest-cell lookup.

    Each causal occupied voxel is rasterized forward with its frame provenance.
    Static maps are class-conditional. Dynamic maps remain source-owned. Only
    integer 1D source membership is used, never a global (actor,class,XYZ) sort.
    Absence of evidence is UNKNOWN, not fabricated observed free space.
    """
    from .source_evidence_audit import planar_move,transform_points,raster_flat
    from .rigid_transport import rigid_source_points_world
    shape=tuple(grid.shape_hwd);z=shape[-1];origin=(grid.x_min,grid.y_min,grid.z_min)
    frames=4;plan=concatenate_plans(rows);n=len(plan)
    support=np.zeros((n,frames,z),bool);density=np.zeros_like(support,dtype=np.float32)
    static={}
    for f in range(frames):
        sem=np.asarray(prepared.raw['history_occ'][f]);visible=np.asarray(prepared.raw['history_observed'][f])
        for cls in (11,13):
            xyz=np.argwhere((sem==cls)&visible)
            static[f,cls]=rigid_source_points_world(xyz,prepared.raw['history_poses'][f],grid=grid)
    offsets=np.array([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
    cursor=0
    for h,p,_,_ in rows:
      for actor in np.unique(p.actor):
        group=np.flatnonzero(p.actor==actor)
        for cls in np.unique(p.classes[group]):
            take=group[p.classes[group]==cls];xy=p.evidence_xy[take]
            centres=np.concatenate((np.broadcast_to(xy[:,None,:],(len(take),z,2)),
                np.broadcast_to(np.arange(z)[None,:,None],(len(take),z,1))),axis=-1)
            neighbours=centres[:,:,None,:]+offsets
            valid=((neighbours>=0)&(neighbours<np.asarray(shape))).all(-1)
            query=(neighbours[...,0]*shape[1]+neighbours[...,1])*z+neighbours[...,2]
            for f in range(frames):
                if actor<0:points=static[f,int(cls)]
                else:
                    reg=prepared.registrations[int(actor)][f]
                    if reg is None:continue
                    if getattr(prepared,'aligned_history_points',None) is not None:
                        aligned=prepared.aligned_history_points[int(actor)][f]
                    else:
                        aligned=transform_points(rigid_source_points_world(reg[1],prepared.raw['history_poses'][f],grid=grid),reg[0])
                    points=planar_move(aligned,prepared.state['current'][int(actor)]['centroid_world'],
                        prepared.targets[h][int(actor)],prepared.yaws[h][int(actor)])
                ids,_=raster_flat(points,prepared.state['world_to_future'][h],origin,grid.voxel_size,shape)
                if not len(ids):continue
                at=np.searchsorted(ids,query)
                hit=valid&(at<len(ids))&(ids[np.minimum(at,len(ids)-1)]==query)
                support[cursor+take,f]=hit[...,0]
                density[cursor+take,f]=hit.mean(-1)
      cursor+=len(p)
    tensor=lambda x:torch.as_tensor(np.ascontiguousarray(x),device=device)
    labels=np.where(support,plan.classes[:,None,None],UNKNOWN).astype(np.uint8)
    height=np.broadcast_to(np.arange(z)[None,None,:]/max(z-1,1),(n,4,z)).astype(np.float32)
    return CentreEvidence(tensor(labels),tensor(support),tensor(support),tensor(support),tensor(height),
        torch.zeros((n,4,z),dtype=torch.int64,device=device),tensor(plan.actor),
        tensor(np.concatenate([np.full(len(p),h,np.int64) for h,p,_,_ in rows])),
        tensor(plan.base),tensor(plan.fallback),tensor(plan.context),tensor(plan.kind),tensor(plan.classes),tensor(plan.legal),
        aligned_density=tensor(density))
