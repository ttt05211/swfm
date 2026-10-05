"""Real-data, history-only sparse evidence adapter (experimental ADD-only head).

Exact registered world points are retained: quantized keys are for neighborhood
lookup, NEVER a substitute for the frozen renderer's metric coordinates.
No annotations, future occupancy, learned poses or features are cached here.
"""
from dataclasses import dataclass
import numpy as np
from .sparse_evidence_repair import EvidenceMemory, FREE, STATIC, _locate
from .source_evidence_audit import transform_points, planar_move

PROTOCOL = 'p0_f9_sparse_source_repair_pilot_v1'
FACE = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], np.int64)


@dataclass
class RepairEvidence:
    memory: EvidenceMemory
    world_points: np.ndarray
    audit: dict


def _grid(grid):
    return np.array([grid.x_min, grid.y_min, grid.z_min]), np.asarray(grid.voxel_size), tuple(grid.shape_hwd)


def _cells(points, inverse, origin, step):
    points = transform_points(points, inverse)
    if not np.isfinite(points).all(): raise ValueError('nonfinite causal evidence')
    return np.floor((points-origin)/step).astype(np.int64)


def build_evidence(prepared, grid, *, expand=True):
    """Four histories, same causal source association/order as Local.

One face-neighbor halo is explicit, not claimed to be observed. Static support
    is restricted to observed historical road/sidewalk (including outside t0's grid). Dynamic
support is restricted to existing t0 sources with a registered past observation.
No dormant/new dynamic source is invented. Visibility is inverse-registered
per source so an unobserved cell is not mislabeled as observed free.
"""
    raw, state = prepared.raw, prepared.state
    if len(raw['history_occ']) != 4 or len(raw['history_poses']) != 4 or len(raw['history_observed']) != 4:
        raise ValueError('strict four histories required')
    origin, step, shape = _grid(grid)
    current_pose = np.asarray(state['current_pose']); inverse = np.linalg.inv(current_pose)
    poses = np.asarray(raw['history_poses'])
    rows, worlds, times = [], [], []
    centers = np.asarray([c['centroid_world'] for c in state['current']]).reshape(-1,3)
    center_cells = _cells(centers, inverse, origin, step)
    for actor, comp in enumerate(state['current']):
        registration = prepared.registrations[actor]
        if len(registration) != 4: raise ValueError('registration history mismatch')
        if not any(r is not None for r in registration[:-1]): continue
        for t, reg in enumerate(registration):
            if reg is None: continue
            ids = np.asarray(reg[1], np.int64)
            points = transform_points(origin+(ids+.5)*step, poses[t])
            points = transform_points(points, reg[0])
            if t != 3:
                visible = np.asarray(raw['history_observed'][t], bool)[tuple(ids.T)]
                points = points[visible]
            cells = _cells(points, inverse, origin, step)-center_cells[actor]
            rows.append(np.column_stack((np.full(len(cells),actor), np.full(len(cells),int(comp['class_id'])),cells)))
            worlds.append(points); times.append(np.full(len(cells),t))
    for t, (occ, observed, pose) in enumerate(zip(raw['history_occ'],raw['history_observed'],poses)):
        ids = np.argwhere(np.isin(occ,(11,13)) & np.asarray(observed,bool))
        points = transform_points(origin+(ids+.5)*step, pose)
        cells = _cells(points,inverse,origin,step)
        # Past visible road can be outside the t0 grid yet enter a future query.
        # Do not crop away causal history before future ego projection.
        classes = np.asarray(occ)[tuple(ids.T)]
        rows.append(np.column_stack((np.full(len(cells),STATIC),classes,cells)))
        worlds.append(points); times.append(np.full(len(cells),t))
    keys0 = np.concatenate(rows).astype(np.int64) if rows else np.empty((0,5),np.int64)
    points0 = np.concatenate(worlds) if worlds else np.empty((0,3))
    t0 = np.concatenate(times).astype(np.int64) if times else np.empty(0,np.int64)
    keys, inverse_key = np.unique(keys0,axis=0,return_inverse=True)
    presence = np.zeros((len(keys),4),bool); presence[inverse_key,t0] = True
    # Latest registered metric sample wins WITHIN a canonical cell. Quantization
    # does not round-trip the world point; the source/semantic identities stay.
    last = np.full(len(keys),-1,np.int64)
    order = np.argsort(t0,kind='stable')
    np.maximum.at(last,inverse_key[order],np.arange(len(order)))
    points = points0[order[last]] if len(keys) else points0
    if expand and len(keys):
        halo = np.repeat(keys,6,axis=0); halo[:,2:] += np.tile(FACE,(len(keys),1))
        all_keys = np.unique(np.concatenate((keys,halo)),axis=0)
        old, valid = _locate(keys,all_keys)
        new_points = np.empty((len(all_keys),3)); new_presence = np.zeros((len(all_keys),4),bool)
        new_points[valid] = points[old[valid]]; new_presence[valid] = presence[old[valid]]
        # Halo coordinates use t0 voxel centres; observed keys keep exact points.
        absolute = all_keys[:,2:].copy(); dynamic = all_keys[:,0]>=0
        absolute[dynamic] += center_cells[all_keys[dynamic,0]]
        new_points[~valid] = transform_points(origin+(absolute[~valid]+.5)*step,current_pose)
        keys,points,presence = all_keys,new_points,new_presence
    _, group, count = np.unique(keys[:,[0,2,3,4]],axis=0,return_inverse=True,return_counts=True)
    keep = count[group]==1
    ambiguous = int((count>1).sum())
    keys,points,presence = keys[keep],points[keep],presence[keep]
    visibility = presence.copy()
    for actor in np.unique(keys[:,0]):
        ids = np.flatnonzero(keys[:,0]==actor)
        for t in range(4):
            reg = np.eye(4) if actor==STATIC else prepared.registrations[actor][t]
            if reg is None: continue
            matrix = np.eye(4) if actor==STATIC else reg[0]
            cells = _cells(points[ids],np.linalg.inv(poses[t])@np.linalg.inv(matrix),origin,step)
            valid = ((cells>=0)&(cells<shape)).all(1)
            visibility[ids[valid],t] |= np.asarray(raw['history_observed'][t],bool)[tuple(cells[valid].T)]
    memory = EvidenceMemory(keys,presence,visibility,ambiguous)
    return RepairEvidence(memory,points,dict(points=len(keys),observed_points=int(presence.any(1).sum()),
        halo_points=int((~presence.any(1)).sum()),static_points=int((keys[:,0]==STATIC).sum()),
        dynamic_points=int((keys[:,0]>=0).sum()),ambiguous_cells=ambiguous,
        support='registered historical source union + observed road/sidewalk + one face-neighbor halo',
        canonical_world_roundtrip=False,future_GT_used=False))


def map_evidence(evidence, prepared, grid):
    """Frozen planar XY/yaw, full future ego SE(3), original-free legality.

Return point-aligned flat IDs (invalid=-1). Static semantic conflicts after
rasterization fail closed. Current dynamic observations belong to V18, not ADD.
"""
    origin,step,shape = _grid(grid); keys = evidence.memory.keys
    mapped = np.full((len(keys),6),-1,np.int64)
    for h in range(6):
        for actor in np.unique(keys[:,0]):
            ids = np.flatnonzero(keys[:,0]==actor)
            if actor>=0: ids = ids[~evidence.memory.presence[ids,-1]]
            points = evidence.world_points[ids]
            if actor>=0:
                points = planar_move(points,prepared.state['current'][actor]['centroid_world'],prepared.targets[h][actor],prepared.yaws[h][actor])
            cells = _cells(points,prepared.state['world_to_future'][h],origin,step)
            valid = ((cells>=0)&(cells<shape)).all(1)
            flat = np.ravel_multi_index(cells[valid].T,shape)
            selected = ids[valid]
            legal = prepared.baseline[h].reshape(-1)[flat]==FREE
            if actor==STATIC and len(flat):
                pairs = np.unique(np.column_stack((flat,keys[selected,1])),axis=0)
                locations,counts = np.unique(pairs[:,0],return_counts=True)
                legal &= ~np.isin(flat,locations[counts>1])
            mapped[selected[legal],h] = flat[legal]
    return mapped


def compose_repair(baseline, evidence, mapped, probability, h, *, threshold=.5, actors='all'):
    if (mapped.shape!=(len(evidence.memory),6) or np.shape(probability)!=(len(evidence.memory),6)
            or not np.isfinite(probability).all() or np.any((probability<0)|(probability>1))):
        raise ValueError('invalid sparse repair probability or population')
    output = np.array(baseline,copy=True); keys=evidence.memory.keys
    for actor in np.unique(keys[:,0]):
        if actors=='static' and actor!=STATIC or actors=='dynamic' and actor<0: continue
        use = (keys[:,0]==actor)&(mapped[:,h]>=0)&(probability[:,h]>=threshold)
        flat = mapped[use,h]
        if np.any(baseline.reshape(-1)[flat]!=FREE): raise RuntimeError('ADD attempted to overwrite transport')
        output.reshape(-1)[flat] = keys[use,1].astype(output.dtype)
    return output


def oracle_probabilities(evidence, mapped, future_gt):
    """Explicit future-GT diagnostic ONLY. Never a model input or proposal gate."""
    target = np.zeros(mapped.shape,np.float32)
    for h in range(6):
        valid = mapped[:,h]>=0
        target[valid,h] = np.asarray(future_gt[h]).reshape(-1)[mapped[valid,h]]==evidence.memory.keys[valid,1]
    return target
