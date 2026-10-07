"""Canonical evidence ONCE, source/time-conditioned repair for six futures.

Experimental CCR, not an execution backend for the old column checkpoint.
History-only entity-local integer lattices avoid global five-key sorting and
per-future patch readers. Metric samples (especially t0) are never snapped.
Future labels enter ONLY repair_targets; the live renderer determines ownership.
"""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from .source_evidence_audit import transform_points, planar_move

FREE, STATIC = 17, -2
FACE = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], np.int64)
PROTOCOL = 'canonical_causal_repair_temporal_v1'
FEATURE_DIM = 27  # xyz3, presence4, observed4, inside4, neighbours6, age1, density4, t0-owned1


@dataclass
class CanonicalEvidence:
    features: np.ndarray
    labels: np.ndarray
    actor: np.ndarray
    classes: np.ndarray
    world: np.ndarray
    presence: np.ndarray
    audit: dict
    layouts: list | None = None  # TRAIN-only ephemeral integer lattices, no learned features

    def __len__(self): return len(self.actor)


@dataclass
class CompactCanonicalSupport:
    """TRAIN-only exact canonical support before point materialization.

    Integer lattices retain the complete candidate population and exact global
    ordering, but omit O(N) world/presence/feature arrays until GT-independent
    Monte-Carlo IDs are known.
    """
    layouts: list
    audit: dict
    points: int

    def __len__(self): return int(self.points)


@dataclass
class RepairPlan:
    flat: np.ndarray             # [N,6], out-of-query = -1
    base: np.ndarray
    fallback: np.ndarray
    legal: np.ndarray            # [N,6,2] ADD,REMOVE (disjoint)
    context: np.ndarray          # [N,6,8], causal future position/time/ego


def grid_arrays(grid):
    return np.array([grid.x_min,grid.y_min,grid.z_min]),np.asarray(grid.voxel_size),np.asarray(grid.shape_hwd)


def _entity(actor, cls, frame_points, frame_cells, prepared, grid, *, halo, max_lattice_cells,
            materialize_features=True, kernels=None, compact_only=False):
    """Bounded per-entity lattice. Large sparse extents use scalar-key lookup.

    No source/grid crop or resolution change. The temporary dense budget is a
    representation choice, not a candidate cap; sparse fallback retains ALL.
    """
    origin,step,_ = grid_arrays(grid)
    lengths=[len(x) for x in frame_points]
    if not sum(lengths): return None
    points=np.concatenate(frame_points); cells=np.concatenate(frame_cells)
    times=np.repeat(np.arange(4),lengths)
    lo=cells.min(0)-int(halo); shape=cells.max(0)-lo+1+int(halo)
    volume=int(np.prod(shape.astype(object)))
    if volume >= np.iinfo(np.int64).max: raise ValueError('canonical lattice extent overflow')
    strides=np.array([int(shape[1])*int(shape[2]),int(shape[2]),1],np.int64)
    packed=(cells-lo)@strides
    dense=volume<=max_lattice_cells
    native_last=None
    if dense and kernels is not None:
        bits,keys,native_last=kernels.ccr_lattice(packed,times,shape,halo)
        flags=bits[keys]
        at=np.stack(np.unravel_index(keys,tuple(shape)),axis=1)+lo
    elif dense:
        bits=np.zeros(volume,np.uint8)
        np.bitwise_or.at(bits,packed,(1<<times).astype(np.uint8))
        view=bits.reshape(tuple(shape)); candidates=view!=0
        if halo:
            grown=candidates.copy()
            for axis in range(3):
                a=[slice(None)]*3; b=a.copy();a[axis]=slice(1,None);b[axis]=slice(None,-1)
                grown[tuple(a)] |= candidates[tuple(b)]; grown[tuple(b)] |= candidates[tuple(a)]
            candidates=grown
        keys=np.flatnonzero(candidates.ravel()); flags=bits[keys]
        lookup=np.full(volume,-1,np.int32);lookup[keys]=np.arange(len(keys))
        point_ids=lookup[packed]
        at=np.stack(np.unravel_index(keys,tuple(shape)),axis=1)+lo
    else:
        keys=np.unique(packed)
        if halo:
            # One-dimensional local keys, NEVER global (actor,class,xyz) rows.
            keys=np.unique(np.concatenate([keys,*[packed+int(d@strides) for d in FACE]]))
        point_ids=np.searchsorted(keys,packed)
        flags=np.zeros(len(keys),np.uint8)
        np.bitwise_or.at(flags,point_ids,(1<<times).astype(np.uint8))
        at=np.stack(np.unravel_index(keys,tuple(shape)),axis=1)+lo
    presence=((flags[:,None]>>np.arange(4))&1).astype(bool)
    # Frame concatenation is chronological; latest metric sample wins. t0 is
    # last, so observed t0 source coordinates are always preserved exactly.
    if native_last is not None:last=native_last
    else:
        last=np.full(len(keys),-1,np.int64);np.maximum.at(last,point_ids,np.arange(len(points)))
    if compact_only:
        layout=dict(keys=keys,flags=flags,bits=bits if dense else None,at=at,lo=lo,shape=shape,
                    dense=dense,volume=volume,last=last,points=points)
        return None,None,None,None,dense,volume,layout
    world=transform_points(origin+(at+.5)*step,prepared.state['current_pose'])
    real=last>=0;world[real]=points[last[real]]
    if not materialize_features:
        layout=dict(keys=keys,flags=flags,bits=bits if dense else None,at=at,lo=lo,shape=shape,dense=dense,volume=volume)
        return None,None,world,presence,dense,volume,layout
    observed=np.zeros((len(keys),4),bool);inside=observed.copy()
    labels=np.full((len(keys),4),18,np.uint8)
    for f in range(4):
        registration=np.eye(4) if actor==STATIC else prepared.registrations[actor][f]
        if registration is None: continue
        reg=np.eye(4) if actor==STATIC else registration[0]
        inverse=np.linalg.inv(prepared.raw['history_poses'][f])@np.linalg.inv(reg)
        ijk=np.floor((transform_points(world,inverse)-origin)/step).astype(np.int64)
        if kernels is not None:
            kernels.ccr_history(ijk,prepared.raw['history_occ'][f],prepared.raw['history_observed'][f],
                                f,labels,observed,inside)
            continue
        valid=((ijk>=0)&(ijk<np.asarray(grid.shape_hwd))).all(1)
        inside[:,f]=valid
        observed[valid,f]=np.asarray(prepared.raw['history_observed'][f],bool)[tuple(ijk[valid].T)]
        labels[valid,f]=np.asarray(prepared.raw['history_occ'][f])[tuple(ijk[valid].T)]
    if actor>=0:
        center=transform_points(np.asarray(prepared.state['current'][actor]['centroid_world'])[None],
                                np.linalg.inv(prepared.state['current_pose']))[0]
        relative=(origin+(at+.5)*step-center)/8
    else:relative=(origin+(at+.5)*step)/40
    if kernels is not None:
        features=kernels.ccr_features(keys,flags,bits if dense else None,shape,relative,presence,observed,inside)
        return features,labels,world,presence,dense,volume,None
    neighbours=np.zeros((len(keys),6),np.float32)
    density=np.zeros((len(keys),4),np.float32)
    for d,delta in enumerate(FACE):
        offset=int(delta@strides); neighbour=keys+offset
        valid=((at+delta-lo>=0)&(at+delta-lo<shape)).all(1)
        if dense:
            neighbour_flags=bits[neighbour.clip(0,volume-1)]
        else:
            loc=np.searchsorted(keys,neighbour);found=loc<len(keys)
            found[found]&=keys[loc[found]]==neighbour[found]
            neighbour_flags=np.zeros(len(keys),np.uint8);neighbour_flags[found]=flags[loc[found]]
        neighbour_flags=np.where(valid,neighbour_flags,0)
        neighbours[:,d]=(neighbour_flags!=0)
        density+=((neighbour_flags[:,None]>>np.arange(4))&1).astype(np.float32)/6
    age=np.argmax(presence[:,::-1],axis=1).astype(np.float32)/3
    age[~real]=1.
    features=np.concatenate([relative,presence,observed,inside,neighbours,age[:,None],density,
                              presence[:,-1,None]],axis=1).astype(np.float32)
    return features,labels,world,presence,dense,volume,None


def build_canonical_evidence(prepared, grid, *, halo=True, max_lattice_cells=4_000_000, materialize_features=True, kernels=None, executor=None):
    """Causal union of existing source geometry and observed road/sidewalk.

    Every t0 source is included, even without an accepted earlier association.
    Past source evidence is visibility filtered; no GT box/trajectory/motion is
    read. Historical points outside the t0 query grid are retained.
    """
    raw,state=prepared.raw,prepared.state
    if any(len(raw[k])!=4 for k in ('history_occ','history_observed','history_poses')):
        raise ValueError('CCR requires exactly four historical frames')
    if max_lattice_cells<1: raise ValueError('positive temporary lattice budget required')
    origin,step,_=grid_arrays(grid);pose=np.asarray(state['current_pose']);inverse=np.linalg.inv(pose)
    groups=[];actors=[];classes=[];counts={'dense_entities':0,'sparse_entities':0,'max_lattice_cells':0}
    def entity(actor,cls,pts,cells):
        options=dict(halo=halo,max_lattice_cells=max_lattice_cells,materialize_features=materialize_features,kernels=kernels)
        return (_entity(actor,cls,pts,cells,prepared,grid,**options) if executor is None else
                executor.submit(_entity,actor,cls,pts,cells,prepared,grid,**options))
    for actor,comp in enumerate(state['current']):
        pts=[];cells=[]
        for f,reg in enumerate(prepared.registrations[actor]):
            if reg is None: pts.append(np.empty((0,3)));cells.append(np.empty((0,3),np.int64));continue
            ijk=np.asarray(reg[1],np.int64)
            aligned=transform_points(transform_points(origin+(ijk+.5)*step,raw['history_poses'][f]),reg[0])
            if f<3:
                visible=np.asarray(raw['history_observed'][f],bool)[tuple(ijk.T)]
                aligned=aligned[visible]
            # t0 integer source cells are exact: don't recover them via floor
            # after a floating-point roundtrip through the world transform.
            cell=ijk.copy() if f==3 else np.floor((transform_points(aligned,inverse)-origin)/step).astype(np.int64)
            pts.append(aligned);cells.append(cell)
        groups.append(entity(actor,int(comp['class_id']),pts,cells))
        actors.append(actor);classes.append(int(comp['class_id']))
    static_points={11:[],13:[]};static_cells={11:[],13:[]}
    for f in range(4):
        occ=np.asarray(raw['history_occ'][f]);vis=np.asarray(raw['history_observed'][f],bool)
        mask=vis&((occ==11)|(occ==13));ijk=np.argwhere(mask)
        labels=occ[tuple(ijk.T)] if len(ijk) else np.empty(0,np.uint8)
        world=transform_points(origin+(ijk+.5)*step,raw['history_poses'][f])
        cell=ijk if f==3 else np.floor((transform_points(world,inverse)-origin)/step).astype(np.int64)
        for cls in (11,13):
            take=labels==cls
            static_points[cls].append(world[take]);static_cells[cls].append(cell[take])
    for cls in (11,13):
        groups.append(entity(STATIC,cls,static_points[cls],static_cells[cls]))
        actors.append(STATIC);classes.append(cls)
    data=[];labels=[];worlds=[];presence=[];aa=[];cc=[];layouts=[];cursor=0
    for group,actor,cls in zip(groups,actors,classes):
        if executor is not None:group=group.result()
        if group is None:continue
        feature,lab,world,pres,dense,volume,layout=group
        if materialize_features:data.append(feature);labels.append(lab)
        else:layouts.append(dict(start=cursor,stop=cursor+len(world),actor=actor,**layout))
        cursor+=len(world);worlds.append(world);presence.append(pres)
        aa.append(np.full(len(world),actor,np.int32));cc.append(np.full(len(world),cls,np.uint8))
        counts['dense_entities' if dense else 'sparse_entities']+=1
        counts['max_lattice_cells']=max(counts['max_lattice_cells'],volume)
    cat=lambda xs,shape,dtype:np.concatenate(xs) if xs else np.empty(shape,dtype)
    actor=cat(aa,(0,),np.int32);cls=cat(cc,(0,),np.uint8);pres=cat(presence,(0,4),bool)
    return CanonicalEvidence(cat(data,(0,FEATURE_DIM),np.float32) if materialize_features else None,
        cat(labels,(0,4),np.uint8) if materialize_features else None,actor,cls,cat(worlds,(0,3),np.float64),pres,
        {**counts,'points':len(actor),'dynamic_points':int((actor>=0).sum()),'static_points':int((actor==STATIC).sum()),
         'halo_points':int((~pres.any(1)).sum()),'future_GT_used':False,'metric_observations_preserved':True,
         'support':'all t0 sources + visible registered source history + observed road/sidewalk + one face halo',
         'features_materialized':materialize_features},None if materialize_features else layouts)


def build_compact_canonical_support(prepared, grid, *, halo=True, max_lattice_cells=4_000_000, kernels=None, executor=None):
    """Complete canonical population with no full-population world/features.

    Global point ordering is identical to build_canonical_evidence().  Dynamic
    entities are followed by static road11 and sidewalk13, and each entity uses
    the same sorted lattice keys.  This is a training execution optimization,
    never a support approximation.
    """
    raw,state=prepared.raw,prepared.state
    if any(len(raw[k])!=4 for k in ('history_occ','history_observed','history_poses')):
        raise ValueError('CCR requires exactly four historical frames')
    if max_lattice_cells<1: raise ValueError('positive temporary lattice budget required')
    origin,step,_=grid_arrays(grid);inverse=np.linalg.inv(np.asarray(state['current_pose']))
    groups=[];actors=[];classes=[];counts={'dense_entities':0,'sparse_entities':0,'max_lattice_cells':0}
    def entity(actor,cls,pts,cells):
        options=dict(halo=halo,max_lattice_cells=max_lattice_cells,materialize_features=False,
                     kernels=kernels,compact_only=True)
        return (_entity(actor,cls,pts,cells,prepared,grid,**options) if executor is None else
                executor.submit(_entity,actor,cls,pts,cells,prepared,grid,**options))
    for actor,comp in enumerate(state['current']):
        pts=[];cells=[]
        for f,reg in enumerate(prepared.registrations[actor]):
            if reg is None:
                pts.append(np.empty((0,3)));cells.append(np.empty((0,3),np.int64));continue
            ijk=np.asarray(reg[1],np.int64)
            aligned=transform_points(transform_points(origin+(ijk+.5)*step,raw['history_poses'][f]),reg[0])
            if f<3:
                visible=np.asarray(raw['history_observed'][f],bool)[tuple(ijk.T)]
                aligned=aligned[visible]
            cell=ijk.copy() if f==3 else np.floor((transform_points(aligned,inverse)-origin)/step).astype(np.int64)
            pts.append(aligned);cells.append(cell)
        groups.append(entity(actor,int(comp['class_id']),pts,cells));actors.append(actor);classes.append(int(comp['class_id']))
    static_points={11:[],13:[]};static_cells={11:[],13:[]}
    for f in range(4):
        occ=np.asarray(raw['history_occ'][f]);vis=np.asarray(raw['history_observed'][f],bool)
        mask=vis&((occ==11)|(occ==13));ijk=np.argwhere(mask)
        labels=occ[tuple(ijk.T)] if len(ijk) else np.empty(0,np.uint8)
        world=transform_points(origin+(ijk+.5)*step,raw['history_poses'][f])
        cell=ijk if f==3 else np.floor((transform_points(world,inverse)-origin)/step).astype(np.int64)
        for cls in (11,13):
            take=labels==cls;static_points[cls].append(world[take]);static_cells[cls].append(cell[take])
    for cls in (11,13):
        groups.append(entity(STATIC,cls,static_points[cls],static_cells[cls]));actors.append(STATIC);classes.append(cls)
    layouts=[];cursor=0;dynamic_points=static_points_count=halo_points=0
    for group,actor,cls in zip(groups,actors,classes):
        if executor is not None:group=group.result()
        if group is None:continue
        _,_,_,_,dense,volume,layout=group;n=len(layout['keys'])
        layout=dict(start=cursor,stop=cursor+n,actor=actor,cls=cls,**layout)
        layouts.append(layout);cursor+=n
        counts['dense_entities' if dense else 'sparse_entities']+=1
        counts['max_lattice_cells']=max(counts['max_lattice_cells'],volume)
        if actor>=0:dynamic_points+=n
        else:static_points_count+=n
        halo_points+=int((layout['flags']==0).sum())
    audit={**counts,'points':cursor,'dynamic_points':dynamic_points,'static_points':static_points_count,
           'halo_points':halo_points,'future_GT_used':False,'metric_observations_preserved':True,
           'support':'all t0 sources + visible registered source history + observed road/sidewalk + one face halo',
           'features_materialized':False,'compact_sampled_only':True}
    return CompactCanonicalSupport(layouts,audit,cursor)


def materialize_canonical_features(evidence, prepared, grid, indices):
    """EXACT feature rows for TRAIN's sampled points, after complete GT scan.

    Full support/projection/labels precede TRAIN sampling. This avoids computing
    four inverse visibility/semantic lookups and neighbourhood features for the
    tens of thousands of points never read by a sampled training update.
    No population/importance/RNG changes; inference still evaluates every point.
    """
    ids=np.asarray(indices)
    if ids.ndim!=1 or ids.dtype.kind not in 'iu' or np.any(ids>=len(evidence)) or np.any(ids<0):
        raise ValueError('invalid canonical TRAIN point indices')
    if evidence.layouts is None:
        return CanonicalEvidence(evidence.features[ids],evidence.labels[ids],evidence.actor[ids],evidence.classes[ids],
                                 evidence.world[ids],evidence.presence[ids],evidence.audit)
    features=np.empty((len(ids),FEATURE_DIM),np.float32);labels=np.full((len(ids),4),18,np.uint8)
    origin,step,shape=grid_arrays(grid)
    inverse=np.linalg.inv(prepared.state['current_pose'])
    for layout in evidence.layouts:
        selected=np.flatnonzero((ids>=layout['start'])&(ids<layout['stop']))
        if not len(selected):continue
        local=ids[selected]-layout['start'];actor=layout['actor']
        world=evidence.world[ids[selected]];pres=evidence.presence[ids[selected]]
        at=layout['at'][local];keys=layout['keys'][local]
        inside=np.zeros((len(local),4),bool);observed=inside.copy()
        for f in range(4):
            registration=np.eye(4) if actor==STATIC else prepared.registrations[actor][f]
            if registration is None:continue
            reg=np.eye(4) if actor==STATIC else registration[0]
            matrix=np.linalg.inv(prepared.raw['history_poses'][f])@np.linalg.inv(reg)
            ijk=np.floor((transform_points(world,matrix)-origin)/step).astype(np.int64)
            valid=((ijk>=0)&(ijk<shape)).all(1);inside[:,f]=valid
            observed[valid,f]=np.asarray(prepared.raw['history_observed'][f],bool)[tuple(ijk[valid].T)]
            labels[selected[valid],f]=np.asarray(prepared.raw['history_occ'][f])[tuple(ijk[valid].T)]
        density=np.zeros((len(local),4),np.float32);neighbours=np.zeros((len(local),6),np.float32)
        strides=np.array([int(layout['shape'][1])*int(layout['shape'][2]),int(layout['shape'][2]),1],np.int64)
        for d,delta in enumerate(FACE):
            neighbour=keys+int(delta@strides)
            valid=((at+delta-layout['lo']>=0)&(at+delta-layout['lo']<layout['shape'])).all(1)
            if layout['dense']:flags=layout['bits'][neighbour.clip(0,layout['volume']-1)]
            else:
                loc=np.searchsorted(layout['keys'],neighbour);found=loc<len(layout['keys'])
                found[found]&=layout['keys'][loc[found]]==neighbour[found]
                flags=np.zeros(len(local),np.uint8);flags[found]=layout['flags'][loc[found]]
            flags=np.where(valid,flags,0);neighbours[:,d]=flags!=0
            density+=((flags[:,None]>>np.arange(4))&1).astype(np.float32)/6
        if actor>=0:
            center=transform_points(np.asarray(prepared.state['current'][actor]['centroid_world'])[None],inverse)[0]
            relative=(origin+(at+.5)*step-center)/8
        else:relative=(origin+(at+.5)*step)/40
        age=np.argmax(pres[:,::-1],axis=1).astype(np.float32)/3;age[~pres.any(1)]=1.
        features[selected]=np.concatenate([relative,pres,observed,inside,neighbours,age[:,None],density,
                                          pres[:,-1,None]],axis=1).astype(np.float32)
    return CanonicalEvidence(features,labels,evidence.actor[ids],evidence.classes[ids],evidence.world[ids],evidence.presence[ids],evidence.audit)


def map_canonical_reference(evidence, prepared, grid):
    """Six cheap projections; visible source owner and restored fallback exact.

    ADD cannot overwrite original occupied voxels. REMOVE cannot delete another
    owner, including a dynamic owner of a road-labelled voxel. Static class
    conflicts fail closed. Zero edits return baseline exactly.
    """
    origin,step,shape=grid_arrays(grid);n=len(evidence)
    flat=np.full((n,6),-1,np.int64);base=np.full((n,6),FREE,np.uint8);fallback=base.copy()
    legal=np.zeros((n,6,2),bool);context=np.zeros((n,6,8),np.float32)
    actor_rows=[(a,np.flatnonzero(evidence.actor==a)) for a in np.unique(evidence.actor)]
    for h in range(6):
        rel=np.linalg.inv(prepared.state['current_pose'])@prepared.raw['future_poses'][h]
        context[:,h,2]=.5*(h+1)/3
        context[:,h,3:6]=(rel[0,3]/40,rel[1,3]/40,np.arctan2(rel[1,0],rel[0,0])/np.pi)
        for actor,ids in actor_rows:
            points=evidence.world[ids]
            if actor>=0:
                points=planar_move(points,prepared.state['current'][actor]['centroid_world'],
                                   prepared.targets[h][actor],prepared.yaws[h][actor])
                context[ids,h,6]=prepared.yaws[h][actor]/np.pi
            mapped=transform_points(points,prepared.state['world_to_future'][h])
            ijk=np.floor((mapped-origin)/step).astype(np.int64)
            good=((ijk>=0)&(ijk<shape)).all(1);selected=ids[good]
            at=(ijk[good,0]*shape[1]+ijk[good,1])*shape[2]+ijk[good,2]
            flat[selected,h]=at;base[selected,h]=np.asarray(prepared.baseline[h]).ravel()[at]
            fallback[selected,h]=base[selected,h]
            context[ids,h,:2]=mapped[:,:2]/40
            owner=np.asarray(prepared.owners[h]).ravel()[at]
            legal[selected,h,0]=base[selected,h]==FREE
            owns=(owner==actor) if actor>=0 else ((owner<0)&(base[selected,h]==evidence.classes[selected]))
            restored=(np.asarray(prepared.fallbacks[h]).ravel()[at] if actor>=0 else np.full(len(at),FREE,np.uint8))
            fallback[selected[owns],h]=restored[owns]
            legal[selected,h,1]=owns&(base[selected,h]==evidence.classes[selected])&(restored!=base[selected,h])
            context[selected,h,7]=(owner==actor) if actor>=0 else (owner<0)
        static=np.flatnonzero((evidence.actor==STATIC)&(flat[:,h]>=0))
        if len(static):
            # Scalar destination IDs, only TWO static semantic populations.
            left=np.unique(flat[static[evidence.classes[static]==11],h])
            right=np.unique(flat[static[evidence.classes[static]==13],h])
            conflicts=np.intersect1d(left,right,assume_unique=True)
            legal[static[np.isin(flat[static,h],conflicts)],h,0]=False
    if np.any(legal[...,0]&legal[...,1]):raise RuntimeError('CCR actions are not exclusive')
    return RepairPlan(flat,base,fallback,legal,context)


def map_canonical_evidence(evidence, prepared, grid, *, kernels=None, executor=None):
    """Vectorized whole-population ego projection and ownership gather.

    Source-local planar_move arithmetic remains reference float64. Only the
    final common ego projection/legality is batched, not six actor-row loops.
    Reference implementation is retained for real-window equivalence tests.
    """
    origin,step,shape=grid_arrays(grid);n=len(evidence)
    flat=np.full((n,6),-1,np.int64);base=np.full((n,6),FREE,np.uint8);fallback=base.copy()
    legal=np.zeros((n,6,2),bool);context=np.zeros((n,6,8),np.float32)
    actors=evidence.actor;dynamic=actors>=0
    rows=[(a,np.flatnonzero(actors==a)) for a in np.unique(actors[dynamic])]
    inverse=np.linalg.inv(prepared.state['current_pose'])
    def horizon(h):
        points=evidence.world.copy()
        for actor,ids in rows:
            points[ids]=planar_move(points[ids],prepared.state['current'][actor]['centroid_world'],
                                   prepared.targets[h][actor],prepared.yaws[h][actor])
            context[ids,h,6]=prepared.yaws[h][actor]/np.pi
        mapped=transform_points(points,prepared.state['world_to_future'][h])
        ijk=np.floor((mapped-origin)/step).astype(np.int64)
        if kernels is not None:
            rel=inverse@prepared.raw['future_poses'][h]
            context[:,h,:2]=mapped[:,:2]/40;context[:,h,2]=.5*(h+1)/3
            context[:,h,3:6]=(rel[0,3]/40,rel[1,3]/40,np.arctan2(rel[1,0],rel[0,0])/np.pi)
            kernels.ccr_plan(ijk,actors,evidence.classes,prepared.baseline[h],prepared.owners[h],
                             prepared.fallbacks[h],h,RepairPlan(flat,base,fallback,legal,context),context)
            return
        good=((ijk>=0)&(ijk<shape)).all(1);ids=np.flatnonzero(good)
        at=(ijk[good,0]*shape[1]+ijk[good,1])*shape[2]+ijk[good,2]
        flat[ids,h]=at;labels=np.asarray(prepared.baseline[h]).ravel()[at]
        owner=np.asarray(prepared.owners[h]).ravel()[at]
        cls=evidence.classes[ids];actor=actors[ids];dyn=dynamic[ids]
        owns=np.where(dyn,owner==actor,(owner<0)&(labels==cls))
        restored=np.where(dyn,np.asarray(prepared.fallbacks[h]).ravel()[at],FREE)
        base[ids,h]=labels;fallback[ids,h]=np.where(owns,restored,labels)
        legal[ids,h,0]=labels==FREE
        legal[ids,h,1]=owns&(labels==cls)&(restored!=labels)
        rel=inverse@prepared.raw['future_poses'][h]
        context[:,h,:2]=mapped[:,:2]/40;context[:,h,2]=.5*(h+1)/3
        context[:,h,3:6]=(rel[0,3]/40,rel[1,3]/40,np.arctan2(rel[1,0],rel[0,0])/np.pi)
        context[ids,h,7]=np.where(dyn,owner==actor,owner<0)
        static=ids[~dyn]
        if len(static):
            left=np.unique(flat[static[evidence.classes[static]==11],h])
            right=np.unique(flat[static[evidence.classes[static]==13],h])
            conflicts=np.intersect1d(left,right,assume_unique=True)
            legal[static[np.isin(flat[static,h],conflicts)],h,0]=False
    if executor is None or n<16384:
        for h in range(6):horizon(h)
    else:list(executor.map(horizon,range(6)))
    return RepairPlan(flat,base,fallback,legal,context)


def repair_targets(evidence, plan, future_gt):
    """GT labels ACTUAL edits, not merely absence of a source class.

    Future GT never modifies support/features/legality. Unknown target labels
    are ignored. Predicted-motion projections are deliberately used so the
    repair objective matches deployment, rather than GT-pose-only utility.
    """
    gt=np.asarray(future_gt)
    if gt.ndim!=4 or gt.shape[0]!=6:raise ValueError('six future label grids required')
    target=np.zeros_like(plan.legal);valid=np.zeros_like(plan.legal)
    for h in range(6):
        ids=np.flatnonzero(plan.flat[:,h]>=0);g=gt[h].ravel()[plan.flat[ids,h]]
        known=(g>=0)&(g<=FREE)
        valid[ids,h]=plan.legal[ids,h]&known[:,None]
        target[ids,h,0]=(g==evidence.classes[ids])&valid[ids,h,0]
        target[ids,h,1]=(g==plan.fallback[ids,h])&(g!=plan.base[ids,h])&valid[ids,h,1]
    return target,valid


def compose_canonical(baseline, evidence, plan, add, remove, *, thresholds=(.5,.95), role='all'):
    """Visible-owner removal first, source-order additions on originally free.

    Source-local removal exposes stored lower owner/background, never global
    deletion. All additions respect immutable V18 occupancy. Static proposals
    are applied before existing t0 sources (last-source-wins).
    """
    add=np.asarray(add);remove=np.asarray(remove)
    if add.shape!=plan.flat.shape or remove.shape!=plan.flat.shape:raise ValueError('CCR probability shape mismatch')
    if any(not np.isfinite(p).all() or np.any((p<0)|(p>1)) for p in (add,remove)):
        raise ValueError('invalid CCR probabilities')
    if role not in ('all','static','dynamic'):raise ValueError('unknown entity role')
    if any(t is not None and (not np.isfinite(t) or not .5<=t<=1) for t in thresholds):raise ValueError('unsafe threshold')
    ta,tr=[np.inf if t is None else t for t in thresholds]
    roles=np.ones(len(evidence),bool) if role=='all' else ((evidence.actor<0) if role=='static' else (evidence.actor>=0))
    dense=[]
    for h in range(6):
        out=np.array(baseline[h],copy=True);dest=out.ravel()
        rm=roles&plan.legal[:,h,1]&(remove[:,h]>=tr)
        flat=plan.flat[rm,h];restored=plan.fallback[rm,h]
        if len(flat):
            order=np.argsort(flat,kind='stable');f=flat[order];v=restored[order]
            if np.any((f[1:]==f[:-1])&(v[1:]!=v[:-1])):raise RuntimeError('conflicting visible-owner fallback')
            dest[flat]=restored
        active=roles&plan.legal[:,h,0]&(add[:,h]>=ta)
        for actor in np.unique(evidence.actor[active]):
            ids=np.flatnonzero(active&(evidence.actor==actor));dest[plan.flat[ids,h]]=evidence.classes[ids]
        dense.append(out)
    return dense


class CanonicalRepairHead(nn.Module):
    """Shared geometry encoder ONCE + factorized six-time binary action heads.

    Space/time fusion is pointwise, not six full patch encoders. Inherited
    semantics, source context, local neighbourhood and real height are retained.
    """
    def __init__(self, source_dim=128, width=64):
        super().__init__();self.source_dim=source_dim;self.width=width
        self.semantic=nn.Embedding(19,8);self.role=nn.Embedding(2,8)
        self.encoder=nn.Sequential(nn.Linear(FEATURE_DIM+48,128),nn.SiLU(),nn.Linear(128,width),nn.SiLU())
        self.source=nn.Linear(source_dim,width,bias=False)
        self.future=nn.Linear(source_dim,width,bias=False)
        self.context=nn.Linear(8+2*8+2,width)
        self.readout=nn.Sequential(nn.SiLU(),nn.Linear(width,32),nn.SiLU(),nn.Linear(32,2))
        nn.init.normal_(self.readout[-1].weight,std=.01);nn.init.constant_(self.readout[-1].bias,-2.)
        self.register_buffer('positive_weight',torch.ones(2,2)) # role(static/dynamic),action

    def project_sources(self,output):
        # Per-actor/time projections, not recomputed for every voxel/chunk.
        # Lifetime is ONE forward/backward, never a learned-feature cache.
        return {**output,'_ccr_source':self.source(output['history_source_context']),
                '_ccr_future':self.future(output['future_transport_queries'])}

    def encode(self,features,labels,actors,classes,output):
        dyn=actors>=0
        history=self.semantic(labels.long()).flatten(-2)
        x=self.encoder(torch.cat([features,history,self.semantic(classes.long()),self.role(dyn.long())],-1))
        context=output['history_source_context']
        if len(context):
            projected=output['_ccr_source'] if '_ccr_source' in output else self.source(context)
            x=x+projected[actors.clamp_min(0).long()]*dyn[:,None]
        return x

    def decode(self,encoded,actors,context,base,fallback,legal,output):
        dyn=actors>=0
        f=output['future_transport_queries']
        x=encoded[:,None]
        if len(f):
            projected=output['_ccr_future'] if '_ccr_future' in output else self.future(f)
            x=x+projected[actors.clamp_min(0).long()]*dyn[:,None,None]
        extra=torch.cat([context,self.semantic(base.long()),self.semantic(fallback.long()),legal.to(context.dtype)],-1)
        return self.readout(x+self.context(extra))

    def probabilities(self,logits,actors):
        return (logits.float()-self.positive_weight[(actors>=0).long(),None,:].log()).sigmoid()


def sampled_tasks(evidence, target, valid, rng, *, per_group=768):
    """TRAIN-only positive/negative stratification with inverse probabilities.

    The four role/action populations retain their natural priors independently.
    Union scalar point IDs so the geometry encoder never repeats sampled points.
    """
    choices=[]
    for role in range(2):
        for action in range(2):
            alive=((evidence.actor>=0)==bool(role))&valid[...,action].any(1)
            positive=(target[...,action]&valid[...,action]).any(1)
            for bucket in (np.flatnonzero(alive&positive),np.flatnonzero(alive&~positive)):
                if not len(bucket):continue
                count=min(len(bucket),max(1,per_group//2));ids=rng.choice(bucket,count,replace=False)
                choices.append((ids,action,len(bucket)/count))
    all_ids=np.unique(np.concatenate([x[0] for x in choices])) if choices else np.empty(0,np.int64)
    weight=np.zeros((len(all_ids),6,2),np.float32)
    for ids,action,factor in choices:
        at=np.searchsorted(all_ids,ids);weight[at,:,action]=valid[ids,:,action]*factor
    return all_ids,weight


def repair_loss(head,logits,actors,target,weight,*,remove_weight=.25):
    if not np.isfinite(remove_weight) or remove_weight<0:raise ValueError('invalid removal loss weight')
    loss=logits.sum()*0;roles=actors>=0
    positive=head.positive_weight[roles.long(),None,:]
    bce=torch.nn.functional.binary_cross_entropy_with_logits(logits.float(),target.float(),
                                                             pos_weight=positive,reduction='none')
    for role in (False,True):
        for action in range(2):
            w=weight[...,action]*(roles==role)[:,None]
            scale=remove_weight if action else 1.
            loss=loss+scale*(bce[...,action]*w).sum()/w.sum().clamp_min(1)
    return loss
