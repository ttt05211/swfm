"""Opt-in live tensor geometry, candidates, inverse sampling and composition.

Fixed history extraction/association/frontier and the existing Strong prior are
explicit input preparation, NOT falsely claimed as GPU kernels. No learned
poses are copied to CPU; future labels enter only targets()/sample_training().
Collision/source ordering and REMOVE fallback follow the frozen compositor.
This is experimental until CPU/device integer/count gates pass on real data.
"""
from dataclasses import dataclass, fields
import numpy as np
import torch
from .causal_column_completion import ColumnPlan, FREE, UNKNOWN, GENERATE, REFINE, KEEP, ADD, REMOVE
from .local_st_world_model_v18_se2 import YAW_ENABLED_CLASS_IDS


@dataclass
class DevicePlan:
    xy: torch.Tensor
    kind: torch.Tensor
    actor: torch.Tensor
    classes: torch.Tensor
    flat: torch.Tensor
    base: torch.Tensor
    fallback: torch.Tensor
    legal: torch.Tensor
    context: torch.Tensor
    evidence_xy: torch.Tensor
    def __len__(self): return len(self.kind)
    def subset(self, ids): return DevicePlan(**{f.name:getattr(self,f.name)[ids] for f in fields(self)})
    def cpu(self): return ColumnPlan(**{f.name:getattr(self,f.name).detach().cpu().numpy() for f in fields(self)})
    @classmethod
    def from_cpu(cls, plan, device):
        integer=('xy','kind','actor','classes','flat','evidence_xy')
        return cls(**{f.name:torch.as_tensor(getattr(plan,f.name),device=device,
            dtype=torch.long if f.name in integer else None) for f in fields(cls)})


def transform(points, matrix):
    return points@matrix[:3,:3].T+matrix[:3,3]


class DeviceColumnWindow:
    def __init__(self, prepared, grid, config, device):
        self.device=torch.device(device);self.grid,self.config=grid,config
        self.shape=tuple(grid.shape_hwd);self.size=int(np.prod(self.shape));self.sources=prepared.state['current']
        if self.shape[2] != config.z_bins:raise ValueError('Z contract mismatch')
        if config.patch!=7 or config.boundary_padding_cells!=1:
            raise ValueError('device pilot requires frozen seven-cell patch / one-cell cross support')
        if len(prepared.raw['history_occ']) != 4:raise ValueError('device pilot requires four histories')
        self.origin=self.tensor((grid.x_min,grid.y_min,grid.z_min),torch.float64)
        self.step=self.tensor(grid.voxel_size,torch.float64)
        self.limit=self.tensor(self.shape,torch.int64)
        self.labels=self.tensor(prepared.raw['history_occ'],torch.uint8)
        self.visibility=self.tensor(prepared.raw['history_observed'],torch.bool)
        self.history_inverse=self.tensor(np.linalg.inv(np.asarray(prepared.raw['history_poses'])),torch.float64)
        self.future_poses=self.tensor(prepared.raw['future_poses'],torch.float64)
        self.world_to_future=self.tensor(prepared.state['world_to_future'],torch.float64)
        self.current_pose=self.tensor(prepared.state['current_pose'],torch.float64)
        self.centers=self.tensor(np.asarray([s['centroid_world'] for s in self.sources]).reshape(-1,3),torch.float64)
        self.classes=self.tensor([s['class_id'] for s in self.sources],torch.int64)
        points=prepared.state.get('source_world_points')
        if points is None:
            points=[transform((self.tensor(s['voxel_indices'],torch.float64)+.5)*self.step+self.origin,
                              self.current_pose).cpu().numpy() for s in self.sources]
        self.points=[self.tensor(p,torch.float64) for p in points]
        self.all_points=(torch.cat(self.points) if self.points else
            torch.empty((0,3),device=self.device,dtype=torch.float64))
        self.point_actor=torch.repeat_interleave(torch.arange(len(self.points),device=self.device),
            torch.tensor([len(p) for p in self.points],device=self.device,dtype=torch.long))
        fixed=prepared.state.get('column_backgrounds')
        if fixed is None:raise RuntimeError('fixed Strong/A1 backgrounds must be prepared explicitly')
        self.backgrounds=self.tensor(np.stack(fixed),torch.uint8)
        z0=prepared.state.get('source_z_t0')
        if z0 is None:z0=transform(self.centers,torch.linalg.inv(self.current_pose))[:,2].cpu().numpy()
        self.z0=self.tensor(z0,torch.float64)
        self.registrations=prepared.registrations
        self.registration_inverse=[];self.members=[];self.aligned=[];self.ages=[]
        for i,regs in enumerate(self.registrations):
            inv=[];members=[];aligned=[]
            for f,reg in enumerate(regs):
                inv.append(np.eye(4) if reg is None else np.linalg.inv(reg[0]))
                if reg is None:
                    members.append(torch.empty(0,device=self.device,dtype=torch.long));continue
                xyz=np.asarray(reg[1]);ids=np.ravel_multi_index(xyz.T,self.shape)
                members.append(self.tensor(np.unique(ids),torch.int64))
                if f < 3:
                    a=prepared.aligned_history_points[i][f] if prepared.aligned_history_points is not None else None
                    if a is None:
                        ego=self.origin+(self.tensor(xyz,torch.float64)+.5)*self.step
                        world=transform(ego,self.tensor(prepared.raw['history_poses'][f],torch.float64))
                        a=transform(world,self.tensor(reg[0],torch.float64))
                    aligned.append(torch.as_tensor(a,device=self.device,dtype=torch.float64))
            self.registration_inverse.append(self.tensor(inv,torch.float64));self.members.append(members)
            self.aligned.append(aligned);self.ages.append(.5*(3-min(f for f,r in enumerate(regs) if r is not None)))
        packed=[ids+(i*4+f)*self.size for i,row in enumerate(self.members) for f,ids in enumerate(row) if len(ids)]
        self.membership_keys=(torch.unique(torch.cat(packed)) if packed else
            torch.empty(0,device=self.device,dtype=torch.long))
        self.available=self.tensor([[True]*4]+[[r is not None for r in row] for row in self.registrations],torch.bool)
        from tools.real_motion.causal_column_common import fixed_candidate_geometry
        fixed=prepared.fixed_candidate_geometry
        if fixed is None:fixed=fixed_candidate_geometry(prepared.memory,prepared.footprints,grid,config)
        self.fixed=[]
        for h,g in enumerate(fixed):
            xy=g.get('generation_xy',np.argwhere(g['frontier'].causal_by_width[config.entry_radius_m]))
            ax=g['frontier'].nearest_x[tuple(xy.T)];ay=g['frontier'].nearest_y[tuple(xy.T)]
            self.fixed.append(dict(gen=self.tensor(xy,torch.int64),anchors=self.tensor(np.column_stack((ax,ay)),torch.int64),
                dominant=self.tensor(g['dominant'],torch.int64),historical=self.tensor(g['historical'],torch.bool),
                allowed=self.tensor(g['static_allowed'],torch.bool),footprint=self.tensor(prepared.footprints[h],torch.bool),
                memory=self.tensor(prepared.memory[h],torch.uint8)))
        self.rounding_boundary_points=torch.zeros((),device=self.device,dtype=torch.long)
        self.targets=self.yaws=None
        self.lookup_matrices={}

    def tensor(self,x,dtype=None):
        # Do not store a lambda closing over self on this GPU-heavy object:
        # window -> lambda -> window delays ALL tensor frees until cyclic GC.
        # A class method has no such per-instance ownership cycle.
        return torch.as_tensor(np.asarray(x),device=self.device,dtype=dtype)

    def _lookup_tables(self):
        static=self.history_inverse[None]@self.future_poses[:,None]
        ns=len(self.sources)
        if not ns:return {h:static[h][None] for h in range(6)}
        motion=torch.eye(4,device=self.device,dtype=torch.float64).expand(ns,6,4,4).clone()
        c=self.yaws.cos();s=self.yaws.sin()
        motion[...,0,0]=c;motion[...,0,1]=-s;motion[...,1,0]=s;motion[...,1,1]=c
        motion[...,:2,3]=self.targets[...,:2]-torch.einsum('shij,sj->shi',motion[...,:2,:2],self.centers[:,:2])
        inverse=torch.linalg.inv(motion)
        # Preserve the reference's LEFT-associated FP64 product, but batch
        # source/horizon/frame dimensions instead of launching per-source jobs.
        prefix=self.history_inverse[None]@torch.stack(self.registration_inverse)
        matrices=(prefix[:,None]@inverse[:,:,None])@self.future_poses[None,:,None]
        return {h:torch.cat((static[h][None],matrices[:,h])) for h in range(6)}

    def raster(self, points, h):
        xyz=(transform(points,self.world_to_future[h])-self.origin)/self.step
        # Count near-floor boundaries for review, NOT silently round indices.
        self.rounding_boundary_points+=((xyz-xyz.round()).abs()<1e-8).any(1).sum()
        ijk=xyz.floor().long();valid=((ijk>=0)&(ijk<self.limit)).all(1)
        ijk=ijk[valid]
        return torch.unique((ijk[:,0]*self.shape[1]+ijk[:,1])*self.shape[2]+ijk[:,2])

    def render(self, anchors, residual, yaw):
        # Keep old FLOAT32 anchor+residual addition before float64 transforms.
        xy=(torch.as_tensor(anchors,device=self.device).float()+residual.detach().float()).double()
        p=torch.cat((xy,self.z0[:,None,None].expand(-1,6,1),torch.ones_like(xy[...,:1])),2)
        self.targets=(p@self.current_pose.T)[...,:3]
        enabled=torch.isin(self.classes,self.classes.new_tensor(YAW_ENABLED_CLASS_IDS))
        self.yaws=torch.where(enabled[:,None],yaw.detach().double(),0.)
        self.lookup_matrices.clear()
        # ALL sources/horizons at once. Integer amax chooses the last source
        # and its next lower DISTINCT source: duplicate source points must not
        # turn REMOVE fallback into the visible source itself.
        ns=len(self.sources);actor=self.point_actor
        rel=self.all_points[:,:2]-self.centers[actor,:2]
        c=self.yaws[actor].T.cos();s=self.yaws[actor].T.sin()
        moved=self.all_points[None].expand(6,-1,-1).clone()
        moved[...,0]=self.targets[actor,:,0].T+c*rel[None,:,0]-s*rel[None,:,1]
        moved[...,1]=self.targets[actor,:,1].T+s*rel[None,:,0]+c*rel[None,:,1]
        mapped=moved@self.world_to_future[:,:3,:3].transpose(1,2)+self.world_to_future[:,None,:3,3]
        xyz=(mapped-self.origin)/self.step
        self.rounding_boundary_points+=((xyz-xyz.round()).abs()<1e-8).any(-1).sum()
        ijk=xyz.floor().long();valid=((ijk>=0)&(ijk<self.limit)).all(-1)
        flat=(ijk[...,0]*self.shape[1]+ijk[...,1])*self.shape[2]+ijk[...,2]
        horizon=torch.arange(6,device=self.device)[:,None].expand_as(flat)
        aa=actor[None].expand_as(flat)
        packed=torch.unique(((horizon*max(ns,1)+aa)*self.size+flat)[valid])
        slots=packed//self.size;ids=packed%self.size;hi=slots//max(ns,1);src=slots%max(ns,1)
        destination=hi*self.size+ids
        owner=torch.full((6*self.size,),-1,device=self.device,dtype=torch.long)
        owner.scatter_reduce_(0,destination,src,reduce='amax',include_self=True)
        lower=torch.where(src<owner[destination],src,-1)
        previous=torch.full_like(owner,-1)
        previous.scatter_reduce_(0,destination,lower,reduce='amax',include_self=True)
        bg=self.backgrounds.flatten();classes=torch.cat((self.classes.new_tensor([FREE]),self.classes)).byte()
        base=torch.where(owner>=0,classes[owner+1],bg)
        fall=torch.where(previous>=0,classes[previous+1],bg)
        self.baseline=list(base.reshape(6,*self.shape).unbind())
        self.owners=list(owner.reshape(6,*self.shape).unbind())
        self.fallback=list(fall.reshape(6,*self.shape).unbind())
        # ONE bounded metadata readback, never predicted poses/labels/features.
        counts=torch.bincount(slots,minlength=6*ns).cpu().tolist()
        pieces=ids.split(counts)
        self.component_ids=[list(pieces[h*ns:(h+1)*ns]) for h in range(6)]
        return self

    def moved_history(self, points, i, h):
        rel=points[:,:2]-self.centers[i,:2];c=self.yaws[i,h].cos();s=self.yaws[i,h].sin()
        # Reference history path uses matrix multiply then add (not fused formula).
        r=torch.stack((torch.stack((c,-s)),torch.stack((s,c))))
        moved=points.clone();moved[:,:2]=rel@r.T+self.targets[i,h,:2]
        return moved

    def candidates(self,h):
        b=self.baseline[h];owner=self.owners[h];fallback=self.fallback[h];g=self.fixed[h];z=self.shape[2]
        rows=[];relative=torch.linalg.inv(self.current_pose)@self.future_poses[h]
        def append(xy,kind,actor,classes,allowed,anchor,age):
            if not len(xy):return
            flat=(xy[:,0:1]*self.shape[1]+xy[:,1:2])*z+torch.arange(z,device=self.device)
            base=b.flatten()[flat];fall=base.clone()
            legal=torch.zeros((*base.shape,3),device=self.device,dtype=torch.bool);legal[...,KEEP]=True
            legal[...,ADD]=(base==FREE)&allowed
            if kind==REFINE:
                own=owner.flatten()[flat]==actor if actor>=0 else base==classes[:,None]
                restored=fallback.flatten()[flat] if actor>=0 else torch.full_like(base,FREE)
                fall=torch.where(own,restored,fall)
                legal[...,REMOVE]=own&(base==classes[:,None])&(fall!=base)
            active=legal[...,1:].any(2).any(1)
            xy,flat,base,fall,legal,classes,anchor=(a[active] for a in (xy,flat,base,fall,legal,classes,anchor))
            context=torch.zeros((len(xy),12),device=self.device,dtype=torch.float32)
            context[:,:2]=((xy.double()+.5)/xy.new_tensor(self.shape[:2])*2-1).float()
            context[:,2]=.5*(h+1)/3;context[:,3]=relative[0,3]/40;context[:,4]=relative[1,3]/40
            context[:,5]=torch.atan2(relative[1,0],relative[0,0])/np.pi
            context[:,6:8]=((xy-anchor).double()*self.step[:2]/self.config.entry_radius_m).float()
            context[:,8]=torch.linalg.vector_norm(context[:,6:8],dim=1);context[:,9]=age/2.5
            context[:,10]=legal[...,ADD].float().mean(1);context[:,11]=actor>=0
            rows.append(DevicePlan(xy,torch.full((len(xy),),kind,device=self.device,dtype=torch.long),
                torch.full((len(xy),),actor,device=self.device,dtype=torch.long),classes,flat,base,fall,legal,context,
                anchor.long() if kind==GENERATE else xy))
        xy=g['gen'];anchor=g['anchors'];classes=g['dominant'][anchor[:,0],anchor[:,1]]
        append(xy,GENERATE,-3,classes,torch.ones((len(xy),z),device=self.device,dtype=torch.bool),anchor.double(),0.)
        static=g['footprint']&g['historical'].any(2)&(g['historical']&(g['memory']!=b)).any(2)
        xy=static.nonzero();append(xy,REFINE,-2,g['dominant'][xy[:,0],xy[:,1]],g['allowed'][xy[:,0],xy[:,1]],xy.double(),0.)
        for i,points in enumerate(self.aligned):
            if not points:continue
            ids=torch.cat([self.component_ids[h][i],*[self.raster(self.moved_history(p,i,h),h) for p in points]])
            if not len(ids):continue
            zz=ids%z;flatxy=ids//z
            mask=torch.zeros(self.shape[:2],device=self.device,dtype=torch.bool);mask.flatten()[flatxy]=True
            grown=mask.clone();grown[1:]|=mask[:-1];grown[:-1]|=mask[1:];grown[:,1:]|=mask[:,:-1];grown[:,:-1]|=mask[:,1:]
            xy=grown.nonzero();depth=torch.arange(z,device=self.device)
            allowed=((depth>=zz.min()-1)&(depth<=zz.max()+1)).expand(len(xy),z)
            center=xy.double().mean(0).expand(len(xy),2)
            append(xy,REFINE,i,self.classes[i].expand(len(xy)),allowed,center,self.ages[i])
        if rows:return DevicePlan(**{f.name:torch.cat([getattr(r,f.name) for r in rows]) for f in fields(DevicePlan)})
        empty=torch.empty(0,device=self.device,dtype=torch.long)
        return DevicePlan(empty.reshape(0,2),empty,empty,empty,empty.reshape(0,z),empty.reshape(0,z).byte(),
            empty.reshape(0,z).byte(),torch.empty((0,z,3),device=self.device,dtype=torch.bool),
            torch.empty((0,12),device=self.device),empty.reshape(0,2))

    def lookup(self,h,plan,*,sparse=True):
        axis=torch.tensor([-3,0,3] if sparse else list(range(-3,4)),device=self.device)
        xx,yy=torch.meshgrid(axis,axis,indexing='ij');k=len(axis)**2;z=self.shape[2];n=len(plan)
        offsets=torch.stack((xx.flatten(),yy.flatten()),1)
        ij=plan.evidence_xy[:,None,:]+offsets[None]
        xyz=torch.cat((ij[:,:,None,:].expand(n,k,z,2),torch.arange(z,device=self.device).view(1,1,z,1).expand(n,k,z,1)),3)
        points=self.origin+(xyz.double()+.5)*self.step
        if h not in self.lookup_matrices:
            self.lookup_matrices=self._lookup_tables()
        group=torch.where(plan.actor>=0,plan.actor+1,0)
        matrix=self.lookup_matrices[h][group]
        mapped=torch.einsum('ntij,nkzj->ntkzi',matrix[:,:,:3,:3],points)+matrix[:,:,None,None,:3,3]
        ijk=((mapped-self.origin)/self.step).floor().long()
        valid=((ijk>=0)&(ijk<self.limit)).all(-1)
        valid=valid&self.available[group,:,None,None]
        safe=ijk.clamp_min(0).minimum(self.limit-1)
        ff=torch.arange(4,device=self.device).view(1,4,1,1)
        labels=self.labels[ff,safe[...,0],safe[...,1],safe[...,2]]
        labels=torch.where(valid,labels,UNKNOWN)
        flags=self.visibility[ff,safe[...,0],safe[...,1],safe[...,2]].byte()*valid
        static_members=(plan.actor<0)[:,None,None,None]&(labels==plan.classes[:,None,None,None])&valid
        flags|=static_members.byte()*2
        flat=(safe[...,0]*self.shape[1]+safe[...,1])*z+safe[...,2]
        if len(self.membership_keys):
            key=(plan.actor[:,None,None,None].clamp_min(0)*4+ff)*self.size+flat
            at=torch.searchsorted(self.membership_keys,key.contiguous()).clamp_max(len(self.membership_keys)-1)
            member=(self.membership_keys[at]==key)&valid&(plan.actor>=0)[:,None,None,None]
            flags|=member.byte()*2
        # Mark absent/out-of-bounds lookup for shared session, not a clamped voxel.
        ijk=torch.where(valid[...,None],ijk,-1)
        return ijk,labels,flags

    def source_features(self,h,plan,output):
        q=output['future_transport_queries'];active=plan.actor>=0
        padded=torch.cat((q.new_zeros((1,6,q.shape[-1])),q))
        return padded[torch.where(active,plan.actor+1,0),h]


def targets(plan,gt):
    # Dedicated supervision call; candidate/lookup APIs cannot consume GT.
    labels=gt.flatten()[plan.flat];out=torch.zeros_like(plan.flat)
    out[plan.legal[...,ADD]&(labels==plan.classes[:,None])]=ADD
    out[plan.legal[...,REMOVE]&(labels==plan.fallback)&(labels!=plan.base)]=REMOVE
    return out


def sample_training(plan,labels,budget,generator):
    selected=[];weights=[];positive=(labels!=KEEP).any(1)
    strata=((plan.kind==GENERATE,budget),((plan.kind==REFINE)&(plan.actor<0),max(2,budget//2)),
            ((plan.kind==REFINE)&(plan.actor>=0),max(2,budget//2)))
    for population,count in strata:
        for sign in (True,False):
            bucket=(population&(positive if sign else ~positive)).nonzero().flatten()
            take=min(len(bucket),max(1,count//2))
            if not take:continue
            choice=bucket[torch.randperm(len(bucket),device=bucket.device,generator=generator)[:take]]
            selected.append(choice);weights.append(torch.full((take,),len(bucket)/take,device=bucket.device))
    if not selected:return plan.flat.new_empty(0),plan.context.new_empty(0)
    return torch.cat(selected),torch.cat(weights)


def actions(plan,probability,gates):
    p=probability.float();gate=[float('inf') if v is None else v for v in gates]
    addgate=torch.where(plan.kind==GENERATE,p.new_tensor(gate[0]),p.new_tensor(gate[1]))[:,None]
    add=plan.legal[...,ADD]&(p[...,ADD]>=addgate)&(p[...,ADD]>p[...,KEEP])
    remove=plan.legal[...,REMOVE]&(p[...,REMOVE]>=gate[2])&(p[...,REMOVE]>p[...,KEEP])
    result=torch.zeros_like(plan.flat);result[add]=ADD;result[remove]=REMOVE
    return result


def compose(baseline,plan,action,*,generation=True,refine=True):
    out=baseline.flatten().clone()
    if refine:
        remove=(plan.kind[:,None]==REFINE)&(action==REMOVE)
        out[plan.flat[remove]]=plan.fallback[remove]
        add=(plan.kind[:,None]==REFINE)&(action==ADD)
        ids=plan.flat[add];actor=plan.actor[:,None].expand_as(action)[add]
        # Deterministic winner, unlike duplicate-index assignment/scatter race.
        winners=torch.full((out.numel(),),-4,device=out.device,dtype=torch.long)
        winners.scatter_reduce_(0,ids,actor,reduce='amax',include_self=True)
        take=add&(winners[plan.flat]==plan.actor[:,None])
        out[plan.flat[take]]=plan.classes[:,None].expand_as(action)[take].to(out.dtype)
    if generation:
        add=(plan.kind[:,None]==GENERATE)&(action==ADD)&(out[plan.flat]==FREE)
        out[plan.flat[add]]=plan.classes[:,None].expand_as(action)[add].to(out.dtype)
    return out.reshape(baseline.shape)
