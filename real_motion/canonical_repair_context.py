"""Shared spatial CCR and bounded cache of FIXED, history-only descriptors.

The sparse neighbourhood is encoded once, not once per future. Cache values
contain neither learned activations nor future labels, predicted motion or
ownership. Live projection and edit supervision remain outside this module.
"""
from collections import OrderedDict
import hashlib
from pathlib import Path
from threading import RLock
import time

import numpy as np
import torch
from torch import nn

from .canonical_causal_repair import (CanonicalRepairHead, CanonicalEvidence, FACE, STATIC, FEATURE_DIM,
    build_canonical_evidence, materialize_canonical_features, grid_arrays,
    map_canonical_evidence)
from .source_evidence_audit import transform_points
from .causal_geometry_cache import CausalGeometryCache

PROTOCOL = 'canonical_causal_spatial_repair_v2'
OFFSETS = np.concatenate((FACE, FACE * 2))


def canonical_neighbors(evidence):
    """Two face radii, SAME entity; missing neighbours are -1, never wrap.

    Only integer lookups. No GT, future pose, learned output or candidate
    truncation. Dense and scalar-key sparse layouts have the same ordering.
    """
    if evidence.layouts is None:
        raise ValueError('canonical integer layout required for shared context')
    result = np.full((len(evidence), len(OFFSETS)), -1, np.int32)
    for layout in evidence.layouts:
        keys, at, shape = layout['keys'], layout['at'], layout['shape']
        strides = np.array([int(shape[1])*int(shape[2]), int(shape[2]), 1], np.int64)
        for d, delta in enumerate(OFFSETS):
            neighbor = keys + int(delta @ strides)
            inside = ((at+delta-layout['lo'] >= 0) & (at+delta-layout['lo'] < shape)).all(1)
            loc = np.searchsorted(keys, neighbor)
            good = inside & (loc < len(keys))
            good[good] &= keys[loc[good]] == neighbor[good]
            result[layout['start']:layout['stop'], d][good] = loc[good] + layout['start']
    return result


def _hash_array(h, value):
    a = np.ascontiguousarray(value)
    if a.dtype.hasobject:
        raise ValueError('object arrays cannot identify causal geometry')
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode())
    if a.size: h.update(memoryview(a).cast('B'))


def fixed_history_digest(prepared, grid):
    """Content keyed: motion/GT mutation does not affect fixed history cache.

    Hashing is charged on EVERY lookup. Keys/scene names alone are insufficient
    because population, registration or cache versions may change on resume.
    """
    h = hashlib.sha256(b'CCR_FIXED_FOUR_HISTORY_v2')
    for a in grid_arrays(grid): _hash_array(h, a)
    for name in ('history_occ', 'history_observed', 'history_poses'):
        _hash_array(h, prepared.raw[name])
    _hash_array(h, prepared.state['current_pose'])
    for comp, rows in zip(prepared.state['current'], prepared.registrations):
        h.update(str(int(comp['class_id'])).encode())
        _hash_array(h, comp['centroid_world'])
        for reg in rows:
            h.update(b'none' if reg is None else b'reg')
            if reg is not None:
                _hash_array(h, reg[0]); _hash_array(h, reg[1])
    if len(prepared.state['current']) != len(prepared.registrations):
        raise ValueError('source/registration population mismatch')
    return h.hexdigest()


def build_fixed_canonical(prepared, grid, *, neighbors=True, kernels=None, executor=None, lazy_sampled=False):
    if lazy_sampled and neighbors:
        raise ValueError('lazy sampled CCR cache is point-head only; neighbours require full features')
    if lazy_sampled:
        lazy=build_canonical_evidence(prepared,grid,materialize_features=False,
                                      kernels=kernels,executor=executor)
        lazy.causal_strata=build_causal_strata(lazy)
        lazy.audit={**lazy.audit,'shared_neighbors':False,'fixed_raw_history_descriptors':True,
                    'learned_features_cached':False,'features_materialized':False,
                    'sampled_feature_materialization':True}
        return lazy,None
    if kernels is not None and not neighbors:
        full=build_canonical_evidence(prepared,grid,kernels=kernels,executor=executor)
        full.causal_strata=build_causal_strata(full)
        full.audit={**full.audit,'shared_neighbors':False,'fixed_raw_history_descriptors':True,
                    'learned_features_cached':False,'features_materialized':True}
        return full,None
    lazy = build_canonical_evidence(prepared, grid, materialize_features=False)
    full = materialize_canonical_features(lazy, prepared, grid, np.arange(len(lazy)))
    # Lookup layouts are needed only while constructing the immutable graph.
    graph = canonical_neighbors(lazy) if neighbors else None
    full.audit = {**full.audit, 'shared_neighbors': bool(neighbors),
                  'fixed_raw_history_descriptors': True, 'learned_features_cached': False,
                  'features_materialized': True}
    full.causal_strata = build_causal_strata(full)
    return full, graph


def fixed_bytes(value):
    evidence, graph = value
    arrays = [evidence.features, evidence.labels, evidence.actor, evidence.classes,
              evidence.world, evidence.presence, graph]
    arrays += [a for pair in getattr(evidence,'causal_strata',()) for a in pair]
    for layout in getattr(evidence,'layouts',()) or ():
        arrays += [v for v in layout.values() if isinstance(v,np.ndarray)]
    return sum(a.nbytes for a in arrays if a is not None)


class _CanonicalDiskStore(CausalGeometryCache):
    """Reuse the existing bounded/integrity-checked local artifact store.

    Address is the verified four-history/registration content digest; known
    future ego conflict sets are separately ephemeral, not part of this file.
    Like existing geometry/checkpoint caches, these are trusted LOCAL pickle
    artifacts, not files to import from an untrusted party.
    """
    def _address(self, key, raw):
        if len(key)!=1 or len(key[0])!=64 or any(c not in '0123456789abcdef' for c in key[0]):
            raise ValueError('verified canonical content digest required')
        return key[0], key[0]


class FixedCanonicalCache:
    """Bounded CPU LRU, zero disk writes, no labels/activations across steps.

    Historical occupancy labels and deterministic visibility/geometry features
    are inputs, not future supervision. They may be reused. Entries are keyed
    by their actual causal contents plus neighbour configuration.
    """
    def __init__(self, max_mib=256, *, neighbors=True, disk_root=None, max_disk_mib=0, async_writes=False, kernels=None, executor=None,
                 lazy_sampled=False):
        if not np.isfinite(max_mib) or max_mib < 0: raise ValueError('invalid cache budget')
        self.limit = int(max_mib*2**20); self.neighbors = bool(neighbors)
        self.lazy_sampled=bool(lazy_sampled)
        if self.lazy_sampled and self.neighbors:
            raise ValueError('lazy sampled CCR cache requires neighbors=False')
        self.kernels=kernels
        self.executor=executor
        self.values = OrderedDict(); self.bytes = 0; self.lock = RLock()
        self.hits = self.misses = self.evictions = 0
        self.hash_seconds = self.build_seconds = 0.
        self.static_queries = OrderedDict()
        self.static_query_bytes = 0; self.static_query_limit = 8*2**20
        self.disk = None; self.disk_io_seconds = 0.; self.async_writes = bool(async_writes)
        if disk_root is not None:
            sources = (Path(__file__),Path(__file__).with_name('canonical_causal_repair.py'))
            if kernels is not None:
                sources += (Path(__file__).with_name('native_column_cpu.py'),
                            Path(__file__).parent/'native'/'column_cpu.cpp')
            code = hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest()
            self.disk = _CanonicalDiskStore(disk_root,PROTOCOL+str(self.neighbors)+code,
                max_bytes=int(max_disk_mib*2**20),ram_bytes=0,reserve_bytes=128*2**20)

    def get(self, prepared, grid):
        tick = time.perf_counter(); digest = fixed_history_digest(prepared, grid)
        with self.lock:
            self.hash_seconds += time.perf_counter()-tick
            if digest in self.values:
                self.hits += 1; value = self.values.pop(digest)
                self.values[digest] = value
                return value
            self.misses += 1
        def builder():
            tick = time.perf_counter(); value = build_fixed_canonical(
                prepared, grid, neighbors=self.neighbors, kernels=self.kernels,
                executor=self.executor, lazy_sampled=self.lazy_sampled)
            value[0].fixed_history_sha256 = digest
            with self.lock: self.build_seconds += time.perf_counter()-tick
            return value
        if self.disk is not None:
            tick = time.perf_counter()
            value,hit = self.disk.get_or_build((digest,),None,builder,defer_write=self.async_writes)
            if not hit and self.async_writes:
                self.disk.store((digest,),None,value,asynchronous=True)
            with self.lock: self.disk_io_seconds += time.perf_counter()-tick
        else: value = builder()
        value[0].fixed_history_sha256 = digest
        size = fixed_bytes(value)
        with self.lock:
            if size <= self.limit:
                while self.bytes+size > self.limit and self.values:
                    _, old = self.values.popitem(last=False)
                    self.bytes -= fixed_bytes(old); self.evictions += 1
                # Safe also if multiple workers built the same immutable input.
                if digest in self.values: self.bytes -= fixed_bytes(self.values.pop(digest))
                self.values[digest] = value; self.bytes += size
        return value

    def static_conflicts(self, evidence, prepared, grid):
        """Known ego query/static geometry only; NEVER predicted source poses.

        Cached to preserve FULL-population class conflicts when TRAIN samples
        before projection. Changed future ego transforms invalidate this cache.
        """
        h = hashlib.sha256()
        h.update(evidence.fixed_history_sha256.encode())
        _hash_array(h, prepared.state['world_to_future'])
        key = h.hexdigest()
        with self.lock:
            if key in self.static_queries:
                value = self.static_queries.pop(key); self.static_queries[key] = value
                return value
        value = full_static_conflicts(evidence, prepared, grid)
        size = sum(a.nbytes for a in value)
        with self.lock:
            if size <= self.static_query_limit:
                if key in self.static_queries:
                    self.static_query_bytes -= sum(a.nbytes for a in self.static_queries.pop(key))
                while self.static_queries and (len(self.static_queries)>=64 or self.static_query_bytes+size>self.static_query_limit):
                    _, old = self.static_queries.popitem(last=False)
                    self.static_query_bytes -= sum(a.nbytes for a in old)
                self.static_queries[key] = value; self.static_query_bytes += size
        return value

    def stats(self):
        with self.lock:
            return dict(hits=self.hits, misses=self.misses, evictions=self.evictions,
                        entries=len(self.values), mib=self.bytes/2**20, limit_mib=self.limit/2**20,
                        hash_seconds=self.hash_seconds, build_seconds=self.build_seconds,
                        disk_writes=self.disk.writes if self.disk is not None else 0,
                        disk_io_seconds=self.disk_io_seconds, asynchronous_disk_writes=self.async_writes,
                        disk=self.disk.stats() if self.disk is not None else None,
                        static_query_mib=self.static_query_bytes/2**20,
                        static_query_limit_mib=self.static_query_limit/2**20,
                        lazy_sampled_features=self.lazy_sampled,
                        future_supervision_cached=False, learned_features_cached=False)

    def close(self):
        if self.disk is not None:self.disk.close()
        with self.lock:
            self.values.clear();self.bytes=0
            self.static_queries.clear();self.static_query_bytes=0


class SpatialCanonicalRepairHead(CanonicalRepairHead):
    """One canonical sparse spatial layer, then six independent readouts.

    Inner/outer neighbour summaries preserve metric height and inherited
    semantics. Training encodes the unique sampled dependency closure; full
    inference encodes ALL points once. No future-specific spatial encoder.
    """
    def __init__(self, source_dim=128, width=64, *, normalized=True, zero_residual=False):
        super().__init__(source_dim, width)
        self.spatial = nn.Sequential(nn.Linear(width*4, width), nn.SiLU(),
                                     nn.Linear(width, width))
        self.spatial_norm = nn.LayerNorm(width) if normalized else nn.Identity()
        if zero_residual:
            nn.init.zeros_(self.spatial[-1].weight); nn.init.zeros_(self.spatial[-1].bias)

    def encode_queries(self, evidence, graph, query_ids, output, device):
        ids = np.asarray(query_ids, np.int64)
        if graph.shape != (len(evidence), 12): raise ValueError('CCR neighbour population mismatch')
        adjacent = graph[ids]
        support = np.unique(np.concatenate((ids, adjacent[adjacent >= 0])))
        def upload(value): return torch.as_tensor(np.ascontiguousarray(value), device=device)
        actors = upload(evidence.actor[support])
        encoded = self.encode(upload(evidence.features[support]), upload(evidence.labels[support]),
                              actors, upload(evidence.classes[support]), output)
        row = upload(np.searchsorted(support, ids)).long()
        local = upload(np.searchsorted(support, adjacent.clip(0))).long()
        valid = upload(adjacent >= 0)
        # With an empty actor's support, zero padding remains a legal feature.
        neighbors = encoded[local.clamp_max(max(len(encoded)-1, 0))] * valid[..., None]
        inner, outer = neighbors[:, :6], neighbors[:, 6:]
        mean1 = inner.sum(1)/valid[:, :6].sum(1).clamp_min(1)[:, None]
        mean2 = outer.sum(1)/valid[:, 6:].sum(1).clamp_min(1)[:, None]
        maximum = inner.masked_fill(~valid[:, :6, None], -torch.inf).max(1).values
        max1 = torch.where(valid[:, :6].any(1)[:, None], maximum, torch.zeros_like(maximum))
        center = encoded[row]
        mixed = self.spatial(torch.cat((center, mean1, mean2, max1), -1))
        return self.spatial_norm(center+mixed)


def attach_neighbors(evidence, graph):
    # Ephemeral experiment attribute; not a learned-state/persistent format.
    evidence.neighbor_graph = graph
    return evidence


def full_static_conflicts(evidence, prepared, grid):
    origin, step, shape = grid_arrays(grid); static = evidence.actor < 0
    world, classes = evidence.world[static], evidence.classes[static]
    result = []
    for h in range(6):
        mapped = transform_points(world, prepared.state['world_to_future'][h])
        ijk = np.floor((mapped-origin)/step).astype(np.int64)
        good = ((ijk >= 0) & (ijk < shape)).all(1)
        flat = (ijk[good,0]*shape[1]+ijk[good,1])*shape[2]+ijk[good,2]
        cls = classes[good]
        result.append(np.intersect1d(flat[cls==11], flat[cls==13]))
    return result


def build_causal_strata(evidence):
    """Stable integer-only causal sampling index, reusable across updates."""
    status = np.where(evidence.presence[:,-1], 0, np.where(evidence.presence.any(1), 1, 2))
    code = (evidence.actor.astype(np.int64)+2)*57 + evidence.classes.astype(np.int64)*3 + status
    result=[]
    for dynamic in (False, True):
        rows=np.flatnonzero((evidence.actor>=0)==dynamic)
        _,inverse,counts=np.unique(code[rows],return_inverse=True,return_counts=True)
        order=np.argsort(inverse,kind='stable')
        result.append((rows[order].astype(np.int32),counts.astype(np.int32)))
    return result


def sample_compact_causal_points(support, rng, *, per_role=1024):
    """Exact sample_causal_points() result without O(N) evidence arrays."""
    if per_role < 1: raise ValueError('positive causal sampling budget required')
    population=getattr(support,'causal_strata',None)
    if population is None:
        # Backward-compatible reconstruction for older compact artifacts.
        population=[]
        for dynamic in (False,True):
            buckets={}
            for layout in support.layouts:
                if (int(layout['actor'])>=0)!=dynamic:continue
                flags=np.asarray(layout['flags'],np.uint8)
                status=np.where((flags&8)!=0,0,np.where((flags&7)!=0,1,2))
                base=np.arange(int(layout['start']),int(layout['stop']),dtype=np.int64)
                for st in (0,1,2):
                    ids=base[status==st]
                    if not len(ids):continue
                    code=(int(layout['actor'])+2)*57+int(layout['cls'])*3+st
                    buckets.setdefault(code,[]).append(ids)
            rows=[];counts=[]
            for code in sorted(buckets):
                parts=buckets[code];bucket=np.concatenate(parts) if len(parts)>1 else parts[0]
                rows.append(bucket);counts.append(len(bucket))
            population.append((
                np.concatenate(rows).astype(np.int32,copy=False) if rows else np.empty(0,np.int32),
                np.asarray(counts,np.int32)))
    selected=[];weights=[]
    for rows,stored_counts in population:
        if not len(rows):continue
        counts=stored_counts.astype(np.int64)
        budget=min(len(rows),max(per_role,len(counts)))
        allocation=np.ones(len(counts),np.int64)
        while allocation.sum()<budget:
            room=counts-allocation;active=room>0;share=np.sqrt(counts)*active
            extra=np.minimum(room,np.floor((budget-allocation.sum())*share/share.sum()).astype(np.int64))
            if not extra.sum():
                rank=np.argsort(-share/(allocation+1),kind='stable')
                extra[rank[active[rank]][:min(int(budget-allocation.sum()),int(active.sum()))]]=1
            allocation+=extra
        cursor=0
        for n,k in zip(counts,allocation):
            bucket=rows[cursor:cursor+n].astype(np.int64);cursor+=int(n)
            chosen=rng.choice(bucket,int(k),replace=False)
            selected.append(chosen);weights.append(np.full(len(chosen),n/k,np.float32))
    ids=np.concatenate(selected) if selected else np.empty(0,np.int64)
    importance=np.concatenate(weights) if weights else np.empty(0,np.float32)
    order=np.argsort(ids)
    return ids[order],importance[order]


def _compact_layout_world(layout, prepared, grid, local=None):
    local=(np.arange(len(layout['keys']),dtype=np.int64) if local is None else np.asarray(local,np.int64))
    if layout.get('static_world') is not None:
        return np.asarray(layout['static_world'],np.float64)[local]
    origin,step,_=grid_arrays(grid)
    at=np.asarray(layout['at'])[local]
    world=transform_points(origin+(at+.5)*step,prepared.state['current_pose'])
    last=np.asarray(layout['last'],np.int64)[local];real=last>=0
    if real.any():
        world[real]=np.asarray(layout['points'],np.float64)[last[real]]
    return world


def materialize_compact_canonical_features(support, prepared, grid, indices):
    """Materialize ONLY sampled canonical rows from exact compact lattices."""
    ids=np.asarray(indices)
    if ids.ndim!=1 or ids.dtype.kind not in 'iu' or np.any(ids>=len(support)) or np.any(ids<0):
        raise ValueError('invalid compact canonical TRAIN point indices')
    n=len(ids);features=np.empty((n,FEATURE_DIM),np.float32);labels=np.full((n,4),18,np.uint8)
    actors=np.empty(n,np.int32);classes=np.empty(n,np.uint8);worlds=np.empty((n,3),np.float64)
    presence=np.empty((n,4),bool)
    origin,step,shape=grid_arrays(grid);inverse=np.linalg.inv(prepared.state['current_pose'])
    for layout in support.layouts:
        selected=np.flatnonzero((ids>=layout['start'])&(ids<layout['stop']))
        if not len(selected):continue
        local=(ids[selected]-layout['start']).astype(np.int64,copy=False)
        actor=int(layout['actor']);cls=int(layout['cls'])
        at=np.asarray(layout['at'])[local];keys=np.asarray(layout['keys'])[local]
        flags=np.asarray(layout['flags'],np.uint8)[local]
        pres=((flags[:,None]>>np.arange(4))&1).astype(bool)
        world=_compact_layout_world(layout,prepared,grid,local)
        actors[selected]=actor;classes[selected]=cls;worlds[selected]=world;presence[selected]=pres
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
        neighbours=np.zeros((len(local),6),np.float32);density=np.zeros((len(local),4),np.float32)
        strides=np.array([int(layout['shape'][1])*int(layout['shape'][2]),int(layout['shape'][2]),1],np.int64)
        all_keys=np.asarray(layout['keys'])
        for d,delta in enumerate(FACE):
            neighbour=keys+int(delta@strides)
            valid=((at+delta-layout['lo']>=0)&(at+delta-layout['lo']<layout['shape'])).all(1)
            if layout['dense']:
                neighbour_flags=np.asarray(layout['bits'])[neighbour.clip(0,layout['volume']-1)]
            else:
                loc=np.searchsorted(all_keys,neighbour);found=loc<len(all_keys)
                found[found]&=all_keys[loc[found]]==neighbour[found]
                neighbour_flags=np.zeros(len(local),np.uint8)
                neighbour_flags[found]=np.asarray(layout['flags'])[loc[found]]
            neighbour_flags=np.where(valid,neighbour_flags,0)
            neighbours[:,d]=neighbour_flags!=0
            density+=((neighbour_flags[:,None]>>np.arange(4))&1).astype(np.float32)/6
        if actor>=0:
            center=transform_points(np.asarray(prepared.state['current'][actor]['centroid_world'])[None],inverse)[0]
            relative=(origin+(at+.5)*step-center)/8
        else:
            relative=(origin+(at+.5)*step)/40
        real=np.asarray(layout['last'],np.int64)[local]>=0
        age=np.argmax(pres[:,::-1],axis=1).astype(np.float32)/3;age[~real]=1.
        features[selected]=np.concatenate([relative,pres,observed,inside,neighbours,age[:,None],density,
                                          pres[:,-1,None]],axis=1).astype(np.float32)
    return CanonicalEvidence(features,labels,actors,classes,worlds,presence,support.audit)


def compact_static_conflicts(support, prepared, grid):
    """Exact full static conflict destinations without full CanonicalEvidence."""
    origin,step,shape=grid_arrays(grid)
    static={}
    for layout in support.layouts:
        if int(layout['actor'])!=STATIC:continue
        cls=int(layout['cls']);world=_compact_layout_world(layout,prepared,grid)
        static.setdefault(cls,[]).append(world)
    for cls in tuple(static):
        static[cls]=np.concatenate(static[cls]) if len(static[cls])>1 else static[cls][0]
    result=[]
    for h in range(6):
        flats={}
        for cls in (11,13):
            world=static.get(cls)
            if world is None or not len(world):
                flats[cls]=np.empty(0,np.int64);continue
            mapped=transform_points(world,prepared.state['world_to_future'][h])
            ijk=np.floor((mapped-origin)/step).astype(np.int64)
            good=((ijk>=0)&(ijk<shape)).all(1);q=ijk[good]
            flats[cls]=np.unique((q[:,0]*shape[1]+q[:,1])*shape[2]+q[:,2])
        result.append(np.intersect1d(flats[11],flats[13],assume_unique=True))
    return result


def map_sampled_compact_canonical(support, ids, prepared, grid, *, kernels=None, static_conflicts=None):
    """Sampled-only exact CCR plan with full-population static conflict guard."""
    sampled=materialize_compact_canonical_features(support,prepared,grid,ids)
    plan=map_canonical_evidence(sampled,prepared,grid,kernels=kernels)
    conflicts=(compact_static_conflicts(support,prepared,grid)
               if static_conflicts is None else static_conflicts)
    static=sampled.actor<0
    for h in range(6):
        plan.legal[static&np.isin(plan.flat[:,h],conflicts[h]),h,0]=False
    return sampled,plan


def sample_causal_points(evidence, rng, *, per_role=1024):
    """GT-independent stratified Monte Carlo in the COMPLETE candidate domain.

    Each source/class and t0/past-only/halo stratum receives >=1 sample. Larger
    strata receive sqrt(size)-proportional extra samples. N/k weights restore
    full-population SUMS in expectation; sampled normalized means have the
    usual ratio-estimator variance/bias, not exact finite-sample equivalence.
    No future validity, GT labels or model scores enter sample selection.
    """
    if per_role < 1: raise ValueError('positive causal sampling budget required')
    selected, weights = [], []
    population=getattr(evidence,'causal_strata',None)
    if population is None:population=build_causal_strata(evidence)
    for rows,stored_counts in population:
        if not len(rows): continue
        counts=stored_counts.astype(np.int64)
        budget = min(len(rows), max(per_role, len(counts)))
        allocation = np.ones(len(counts), np.int64)
        while allocation.sum() < budget:
            room = counts-allocation; active = room > 0
            share = np.sqrt(counts)*active
            extra = np.minimum(room, np.floor((budget-allocation.sum())*share/share.sum()).astype(np.int64))
            if not extra.sum():
                rank = np.argsort(-share/(allocation+1), kind='stable')
                extra[rank[active[rank]][:min(int(budget-allocation.sum()), int(active.sum()))]] = 1
            allocation += extra
        cursor=0
        for n,k in zip(counts,allocation):
            bucket=rows[cursor:cursor+n].astype(np.int64);cursor+=int(n)
            chosen = rng.choice(bucket, int(k), replace=False)
            selected.append(chosen); weights.append(np.full(len(chosen), n/k, np.float32))
    ids = np.concatenate(selected) if selected else np.empty(0, np.int64)
    importance = np.concatenate(weights) if weights else np.empty(0, np.float32)
    order = np.argsort(ids)
    return ids[order], importance[order]


def map_sampled_canonical(evidence, ids, prepared, grid, static_conflicts, *, kernels=None):
    """Live selected-point projection, with FULL static conflict protection."""
    sampled = materialize_canonical_features(evidence, prepared, grid, ids)
    plan = map_canonical_evidence(sampled, prepared, grid,kernels=kernels)
    static = sampled.actor < 0
    for h in range(6):
        plan.legal[static & np.isin(plan.flat[:,h], static_conflicts[h]), h, 0] = False
    return sampled, plan
