"""TRAIN-only two-pass complete candidates, never a deployment/GT feature path.

Scan all legal candidates for exact positive/negative strata, then materialize
only the ORIGINAL RNG-selected rows. No cached GT, stale poses or truncation.
"""
import numpy as np

from real_motion.causal_column_completion import ColumnPlan, GENERATE, REFINE, _valid_semantics
from real_motion.native_column_cpu import get_native
from tools.real_motion import causal_column_common as col


def can_compact(grid, config):
    config.validate()
    if not 0 < config.z_bins <= 64: return False
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        extent=np.asarray(grid.shape_hwd[:2])*np.asarray(grid.voxel_size[:2])/config.entry_radius_m
    return bool(np.isfinite(extent).all() and np.all(np.abs(extent) <= np.sqrt(np.finfo(np.float32).max/4)))


def pack_allowed(allowed):
    allowed=np.asarray(allowed)
    if allowed.ndim != 2 or allowed.shape[1] > 64 or allowed.dtype != bool:
        raise ValueError('compact allowed mask requires N,Z bool, Z<=64')
    packed=np.packbits(allowed,axis=1,bitorder='little')
    buffer=np.zeros((len(allowed),8),np.uint8); buffer[:,:packed.shape[1]]=packed
    return buffer.view('<u8').reshape(-1).astype(np.uint64,copy=False)


class CompactColumns:
    """Only scalar-per-column descriptors before selection, no N*Z arrays."""
    def __init__(self, prep, h, grid, config, parts, *, count_prior=False):
        self.prep,self.h,self.grid,self.config=prep,h,grid,config
        self.native=get_native(); self.parts=parts
        ends=np.cumsum([len(p[0]) for p in parts],dtype=np.int64)
        self.ends=ends
        def join(index,dtype,shape):
            return np.concatenate([p[index] for p in parts]).astype(dtype,copy=False) if parts else np.empty(shape,dtype)
        xy=join(0,np.int32,(0,2)); cls=join(3,np.uint8,(0,)); masks=join(4,np.uint64,(0,))
        kinds=np.concatenate([np.full(len(p[0]),p[1],np.uint8) for p in parts]) if parts else np.empty(0,np.uint8)
        actors=np.concatenate([np.full(len(p[0]),p[2],np.int32) for p in parts]) if parts else np.empty(0,np.int32)
        gt=np.asarray(prep.raw['future_gt_occ'][h])
        if not _valid_semantics(gt): raise ValueError('invalid future supervision grid')
        scan=self.native.compact(xy,kinds,actors,cls,masks,prep.baseline[h],prep.owners[h],prep.fallbacks[h],gt,prior_counts=count_prior)
        active,positive=scan[:2]
        self.prior_counts=scan[2] if count_prior else None
        start=0
        for p,end in zip(parts,ends):
            if p[1] == GENERATE:
                live=active[start:end]
                if not np.isfinite(p[5][live]).all() or not np.isfinite(p[6][live]).all() or np.any(p[5][live] < 0) or np.any(p[6][live] < 0):
                    raise ValueError('invalid column plan evidence')
            start=end
        self.raw_rows=np.flatnonzero(active)
        self.xy,self.kind,self.actor,self.classes,self.masks=(a[active] for a in (xy,kinds,actors,cls,masks))
        self.positive_rows=positive[active]
        # Full-population identity check, NOT only the sampled rows.
        if len(self):
            area=int(grid.shape_hwd[0])*int(grid.shape_hwd[1])
            keys=(self.actor.astype(np.int64)+3)*area+self.xy[:,0].astype(np.int64)*int(grid.shape_hwd[1])+self.xy[:,1]
            if len(np.unique(keys)) != len(keys): raise ValueError('duplicate actor-column query')
            self.rel=np.linalg.inv(prep.state['current_pose'])@prep.raw['future_poses'][h]
            if (not np.isfinite(self.rel).all()
                    or np.any(np.abs(self.rel[:2,3]) > float(np.finfo(np.float32).max)*40)):
                raise ValueError('invalid TRAIN context pose')
        else: self.rel=None
        self.materialized_rows=0

    def __len__(self): return len(self.kind)

    def __getattr__(self,name):
        if name in ('flat','base','fallback','legal','context'):
            raise RuntimeError('compact TRAIN voxel fields must be materialized after selection')
        raise AttributeError(name)

    def _indices(self,indices):
        ids=np.arange(len(self))[indices] if isinstance(indices,slice) else np.asarray(indices)
        if ids.ndim != 1: raise ValueError('TRAIN subset indices must be one dimensional')
        if ids.dtype == bool:
            if len(ids) != len(self): raise ValueError('TRAIN boolean subset population mismatch')
            ids=np.flatnonzero(ids)
        if ids.size and ids.dtype.kind not in 'iu': raise IndexError('TRAIN subset requires integer indices')
        if ids.dtype.kind == 'u' and np.any(ids > np.iinfo(np.int64).max): raise IndexError('TRAIN subset index out of bounds')
        ids=ids.astype(np.int64,copy=False); ids=np.where(ids < 0,ids+len(self),ids)
        if np.any((ids < 0)|(ids >= len(self))): raise IndexError('TRAIN subset index out of bounds')
        return ids

    def materialize(self,indices):
        ids=self._indices(indices); h=self.h; prep=self.prep
        xy,kinds,actors,cls,masks=(a[ids] for a in (self.xy,self.kind,self.actor,self.classes,self.masks))
        active,positive,flat,base,fall,legal,labels=self.native.compact(xy,kinds,actors,cls,masks,
            prep.baseline[h],prep.owners[h],prep.fallbacks[h],np.asarray(prep.raw['future_gt_occ'][h]),materialize=True)
        if not active.all() or not np.array_equal(positive,self.positive_rows[ids]):
            raise RuntimeError('TRAIN candidate/supervision changed between scan and materialization')
        ax,ay,age=(np.empty(len(ids),np.float64) for _ in range(3))
        raw=self.raw_rows[ids]; groups=np.searchsorted(self.ends,raw,side='right')
        for group in np.unique(groups):
            p=self.parts[group]; start=0 if group == 0 else self.ends[group-1]
            take=np.flatnonzero(groups == group); local=raw[take]-start
            ax[take]=p[5][local] if np.ndim(p[5]) else p[5]
            ay[take]=p[6][local] if np.ndim(p[6]) else p[6]
            age[take]=p[7]
        context=col._column_context(xy,legal,actors,ax,ay,age,h,self.rel,self.grid,self.config) if len(ids) else np.empty((0,col.CONTEXT_DIM),np.float32)
        evidence=xy.copy(); gen=kinds == GENERATE
        evidence[gen]=np.column_stack((ax[gen],ay[gen])).astype(np.int32)
        plan=ColumnPlan(xy,kinds,actors,cls,flat,base,fall,legal,context,evidence)
        self.materialized_rows+=len(ids)
        return plan,labels

    def subset(self,indices): return self.materialize(indices)[0]

    def audit(self):
        return dict(population=len(self),materialized_rows=self.materialized_rows,
            compact_bytes=sum(a.nbytes for a in (self.raw_rows,self.xy,self.kind,self.actor,self.classes,self.masks,self.positive_rows)))


def dynamic_evidence(prep,grid):
    """Immutable aligned histories reused across six horizons, not predicted poses."""
    tasks=[]
    for actor,comp in enumerate(prep.state['current']):
        regs=prep.registrations[actor]
        if not any(r is not None for r in regs[:-1]): continue
        points=[]
        for f,reg in enumerate(regs[:-1]):
            if reg is None: continue
            if getattr(prep,'aligned_history_points',None) is not None:
                aligned=prep.aligned_history_points[actor][f]
            else:
                world=col.rigid_source_points_world(reg[1],prep.raw['history_poses'][f],grid=grid)
                aligned=col.transform_points(world,reg[0])
            points.append(aligned)
        age=.5*(len(regs)-1-min(f for f,r in enumerate(regs) if r is not None))
        tasks.append((actor,comp,points,age))
    return tasks


def build_compact_candidates(prep,grid,config,*,count_prior=False,horizons=range(6),tasks=None):
    config.validate(); shape=tuple(grid.shape_hwd); z=shape[2]
    if z != config.z_bins: raise RuntimeError('Z lattice/checkpoint mismatch')
    if not np.isclose(grid.voxel_size[0],grid.voxel_size[1],rtol=0,atol=1e-10):
        raise RuntimeError('frontier distance contract requires an isotropic XY lattice')
    native=get_native(); tasks=dynamic_evidence(prep,grid) if tasks is None else tasks; results=[]
    for h in horizons:
        if not 0 <= h < 6: raise ValueError('future horizon must be 0..5')
        b,m,footprint=prep.baseline[h],prep.memory[h],prep.footprints[h]
        fixed=getattr(prep,'fixed_candidate_geometry',None)
        geometry=fixed[h] if fixed is not None and fixed[h]['radius'] == config.entry_radius_m else col.fixed_candidate_geometry([m],[footprint],grid,config)[0]
        frontier,dominant=geometry['frontier'],geometry['dominant']; parts=[]
        def append(xy,kind,actor,classes,masks,ax,ay,age):
            if len(xy): parts.append((xy.astype(np.int32,copy=False),kind,actor,classes,masks,ax,ay,age))
        xy=geometry.get('generation_xy')
        if xy is None: xy=np.argwhere(frontier.causal_by_width[config.entry_radius_m])
        ax,ay=frontier.nearest_x[tuple(xy.T)],frontier.nearest_y[tuple(xy.T)]
        append(xy,GENERATE,-3,dominant[ax,ay],np.full(len(xy),(1 << z)-1,np.uint64),ax,ay,0.)
        xy=native.static(footprint,geometry['historical'],m,b)
        append(xy,REFINE,-2,dominant[tuple(xy.T)],pack_allowed(geometry['static_allowed'][tuple(xy.T)]),xy[:,0],xy[:,1],0.)
        point_groups=[]
        for actor,comp,points,age in tasks:
            all_flat=[np.ravel_multi_index(prep.components[h][actor].voxel_indices.T,shape)]
            for aligned in points:
                moved=col.planar_move(aligned,comp['centroid_world'],prep.targets[h][actor],prep.yaws[h][actor])
                ids,_=col.raster_flat(moved,prep.state['world_to_future'][h],
                    (grid.x_min,grid.y_min,grid.z_min),grid.voxel_size,shape,deduplicate=False,optimize=True)
                all_flat.append(ids)
            point_groups.append(np.concatenate(all_flat))
        if config.boundary_padding_cells == 1:
            packed,offsets,bounds=native.support_many(point_groups,shape)
            supports=[(packed[offsets[i]:offsets[i+1]],*bounds[i]) for i in range(len(tasks))]
        else:
            supports=[]
            for ids in point_groups:
                if not len(ids): supports.append((np.empty((0,2),np.int64),z,-1)); continue
                xyz=np.column_stack(np.unravel_index(ids,shape))
                supports.append((col.padded_support_xy(xyz[:,:2],shape[:2],config.boundary_padding_cells,optimize=True),xyz[:,2].min(),xyz[:,2].max()))
        for (actor,comp,_,age),(xy,zlo,zhi) in zip(tasks,supports):
            if not len(xy): continue
            lo,hi=max(0,int(zlo)-1),min(z-1,int(zhi)+1)
            mask=((1 << (hi-lo+1))-1) << lo
            center=xy.mean(0)  # original FULL support mean, never sampled-subset mean
            append(xy,REFINE,actor,np.full(len(xy),int(comp['class_id']),np.uint8),
                np.full(len(xy),mask,np.uint64),center[0],center[1],age)
        results.append((h,CompactColumns(prep,h,grid,config,parts,count_prior=count_prior),None))
    return results
