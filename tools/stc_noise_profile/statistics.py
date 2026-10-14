from functools import lru_cache
import numpy as np
from scipy import ndimage as ndi

SHAPE = (200, 200, 16)
ORIGIN = (-40., -40., -1.)
STEP = (.4, .4, .4)
DYNAMIC = (2, 3, 4, 5, 6, 7, 9, 10)
FAMILIES = ('false_occupancy', 'missed_occupancy', 'wrong_class')
DISTANCE = (0., 10., 20., 30., 40., 60., np.inf)
BOUNDARY = (0., .4, .8, 1.6, 3.2, 6.4, np.inf)
OFFSET = (-np.inf, -4., -2., -1., -.4, 0., .4, 1., 2., 4., np.inf)
JITTER = (0., .1, .2, .4, .8, 1.6, 3.2, np.inf)
RATIO = (0., .25, .5, .75, 1., 1.5, 2., 4., np.inf)


@lru_cache(maxsize=4)
def centers(shape=SHAPE, origin=ORIGIN, step=STEP):
    ids = np.stack(np.meshgrid(*(np.arange(n) for n in shape), indexing='ij'), -1)
    points = (ids+.5)*np.asarray(step)+np.asarray(origin)
    points.setflags(write=False)
    return points


def confusion(pred, gt):
    return np.bincount(18*gt.ravel().astype(np.int64)+pred.ravel(), minlength=324).reshape(18, 18)


def errors(pred, gt):
    return dict(false_occupancy=(pred != 17) & (gt == 17),
        missed_occupancy=(pred == 17) & (gt != 17),
        wrong_class=(pred != 17) & (gt != 17) & (pred != gt))


def hist(values, edges):
    # inf is a deliberate overflow category when there is no opposite surface.
    return np.bincount(np.clip(np.searchsorted(edges, np.asarray(values), side='right')-1,
                               0, len(edges)-2), minlength=len(edges)-1).astype(np.int64).tolist()


def accumulate(total, value):
    """All fitted sufficient statistics are nonnegative integer counters."""
    for key, v in value.items():
        if isinstance(v, dict): accumulate(total.setdefault(key, {}), v)
        elif isinstance(v, list):
            a = np.asarray(v)
            if a.dtype.kind not in 'iu' or (a < 0).any(): raise ValueError('integer counters required')
            if key not in total: total[key] = a.tolist()
            else:
                old = np.asarray(total[key])
                if old.shape != a.shape: raise ValueError('histogram shapes differ')
                total[key] = (old+a).tolist()
        else:
            if type(v) is not int or v < 0: raise ValueError('nonnegative integer counters required')
            total[key] = total.get(key, 0)+v
    return total


def frame_statistics(pred, gt, known, *, origin=ORIGIN, step=STEP):
    if (pred.shape != gt.shape or known.shape != gt.shape or known.dtype != bool
            or any(a.dtype.kind not in 'ui' or (a < 0).any() or (a > 17).any() for a in (pred, gt))):
        raise ValueError('paired integer labels 0..17 and bool historical known required')
    xyz = centers(tuple(gt.shape), tuple(origin), tuple(step))
    radial = np.linalg.norm(xyz[..., :2], axis=-1)
    dbin = np.clip(np.searchsorted(DISTANCE, radial, side='right')-1, 0, len(DISTANCE)-2)
    c = np.stack([confusion(pred[dbin == d], gt[dbin == d]) for d in range(len(DISTANCE)-1)])
    occupied = gt != 17
    outside = ndi.distance_transform_edt(~occupied, sampling=step) if occupied.any() else np.full(gt.shape, np.inf)
    inside = ndi.distance_transform_edt(occupied, sampling=step) if (~occupied).any() else np.full(gt.shape, np.inf)
    out = dict(frames=1, confusion=confusion(pred, gt).tolist(),
        historical_known_confusion=confusion(pred[known], gt[known]).tolist(),
        historical_unknown_confusion=confusion(pred[~known], gt[~known]).tolist(),
        distance_confusion=c.tolist(), families={})
    structure = ndi.generate_binary_structure(3, 1)  # fixed six-neighbor regions
    for family, mask in errors(pred, gt).items():
        labels = gt if family == 'missed_occupancy' else pred
        item = dict(voxels=int(mask.sum()), class_voxels=np.bincount(labels[mask], minlength=18).tolist(),
            boundary_distance_voxel_hist=hist((outside if family == 'false_occupancy' else inside)[mask], BOUNDARY),
            components=0, component_joint={}, component_sizes={})
        # Different semantic classes never merge into one region.
        for cid in np.unique(labels[mask]):
            at, n = ndi.label(mask & (labels == cid), structure=structure)
            if not n: continue
            points = np.argwhere(at > 0); ids = at[tuple(points.T)]-1
            sizes = np.bincount(ids, minlength=n)
            lo = np.full((n, 3), gt.size, np.int64); hi = np.zeros((n, 3), np.int64)
            sums = np.zeros((n, 3), np.float64)
            for axis in range(3):
                np.minimum.at(lo[:, axis], ids, points[:, axis]); np.maximum.at(hi[:, axis], ids, points[:, axis])
                np.add.at(sums[:, axis], ids, points[:, axis])
            center = (sums/sizes[:, None]+.5)*step+origin
            radius = np.linalg.norm(center[:, :2], axis=1)
            bins = np.clip(np.searchsorted(DISTANCE, radius, side='right')-1, 0, len(DISTANCE)-2)
            extents = hi-lo+1
            signatures, count = np.unique(np.column_stack((np.full(n, int(cid)), bins, extents, sizes)), axis=0, return_counts=True)
            for signature, frequency in zip(signatures, count):
                key = ','.join(str(int(x)) for x in signature)
                item['component_joint'][key] = item['component_joint'].get(key, 0)+int(frequency)
            values, frequencies = np.unique(sizes, return_counts=True)
            for v, f in zip(values, frequencies): item['component_sizes'][str(int(v))] = item['component_sizes'].get(str(int(v)), 0)+int(f)
            item['components'] += int(n)
        out['families'][family] = item
    return out


def resample_previous(previous, current_pose, *, origin=ORIGIN, step=STEP):
    """Nearest voxel-center sampling, with an explicit overlap mask."""
    gt, pred, pose = previous['gt'], previous['pred'], previous['pose']
    xyz = centers(tuple(gt.shape), tuple(origin), tuple(step))
    matrix = np.linalg.inv(pose) @ current_pose
    local = xyz @ matrix[:3, :3].T+matrix[:3, 3]
    idx = np.floor((local-origin)/step).astype(np.int64)
    valid = ((idx >= 0) & (idx < gt.shape)).all(-1)
    clipped = np.clip(idx, 0, np.asarray(gt.shape)-1)
    addresses = tuple(clipped[..., a] for a in range(3))
    return gt[addresses], pred[addresses], valid


def temporal_statistics(history, *, origin=ORIGIN, step=STEP):
    if len(history) != 4: raise ValueError('exact four historical frames required')
    latest = history[-1]; target = latest['gt']
    valid = ~np.isin(target, DYNAMIC)
    frames = []
    for row in history[:-1]:
        gt, pred, overlap = resample_previous(row, latest['pose'], origin=origin, step=step)
        valid &= overlap & (gt == target) & ~np.isin(gt, DYNAMIC)
        frames.append(errors(pred, gt))
    frames.append(errors(latest['pred'], target))
    result = dict(windows4=1, static_stable_overlap_voxels=int(valid.sum()), static_patterns={})
    for name in FAMILIES:
        bits = sum((f[name].astype(np.uint8) << i) for i, f in enumerate(frames))
        result['static_patterns'][name] = np.bincount(bits[valid], minlength=16).tolist()
    dynamic = dict(tracks_GT_supported_all4=0, missing_STC_patterns=[0]*16, residual_pair_tracks=0,
                   delta_offset_xy_hist=[0]*(len(JITTER)-1))
    common = set.intersection(*(set(f['objects']) for f in history))
    for token in common:
        objects = [f['objects'][token] for f in history]
        if any(o['gt_voxels'] == 0 for o in objects): continue
        dynamic['tracks_GT_supported_all4'] += 1
        bits = sum((o['pred_voxels'] == 0) << i for i, o in enumerate(objects))
        dynamic['missing_STC_patterns'][bits] += 1
    # Adjacent pair only, counted once per latest four-frame window. First
    # two scene pairs are intentionally outside this four-history statistic.
    prev = history[-2]['objects']; cur = latest['objects']
    offsets = []
    for token in set(prev) & set(cur):
        if prev[token]['offset'] is not None and cur[token]['offset'] is not None:
            offsets.append(np.linalg.norm(np.asarray(cur[token]['offset'])[:2]-np.asarray(prev[token]['offset'])[:2]))
    dynamic['residual_pair_tracks'] = len(offsets)
    dynamic['delta_offset_xy_hist'] = hist(offsets, JITTER)
    result['dynamic'] = dynamic
    return result


def object_statistics(gt, pred, pose, annotations, *, origin=ORIGIN, step=STEP):
    """Offline SAME-frame annotation crops; NOT predicted object identities.

    Compare STC and GT occupied shape in each annotated object's local frame,
    then use those residuals across history. Actual object motion is removed.
    """
    xyz = centers(tuple(gt.shape), tuple(origin), tuple(step))
    points = {}
    for cid in {a['class_id'] for a in annotations}:
        points[cid] = [(xyz[labels == cid] @ pose[:3, :3].T+pose[:3, 3]) for labels in (gt, pred)]
    objects, offsets, ratios = {}, [], []
    for ann in annotations:
        arrays = []
        for world in points[ann['class_id']]:
            local = (world-np.asarray(ann['translation'])) @ np.asarray(ann['rotation_matrix'])
            half = np.asarray(ann['size_lwh'])/2+np.asarray(step)  # one-voxel crop margin, NOT a scoring mask
            arrays.append(local[(np.abs(local) <= half).all(1)])
        a, b = arrays
        offset = (b.mean(0)-a.mean(0)).tolist() if len(a) and len(b) else None
        objects[ann['instance_token']] = dict(gt_voxels=len(a), pred_voxels=len(b), offset=offset)
        if len(a): ratios.append(len(b)/len(a))
        if offset is not None: offsets.append(offset)
    offsets = np.asarray(offsets).reshape(-1, 3)
    counts = dict(annotation_boxes=len(annotations), GT_supported_boxes=sum(o['gt_voxels'] > 0 for o in objects.values()),
        missing_STC_boxes=sum(o['gt_voxels'] > 0 and o['pred_voxels'] == 0 for o in objects.values()),
        jointly_supported_boxes=len(offsets), pred_to_GT_voxel_ratio_hist=hist(ratios, RATIO),
        offset_xyz_hist=[hist(offsets[:, a], OFFSET) for a in range(3)])
    return objects, counts


def weighted_quantiles(histogram, quantiles=(.5, .9)):
    if not histogram: return [None]*len(quantiles)
    items = sorted((int(v), n) for v, n in histogram.items())
    values, frequencies = map(np.asarray, zip(*items)); cumulative = frequencies.cumsum()
    return [int(values[min(np.searchsorted(cumulative, q*cumulative[-1], side='left'), len(values)-1)]) for q in quantiles]


def validate(value, volume):
    accumulate({}, value)
    c = np.asarray(value['confusion'])
    if c.shape != (18, 18) or c.sum() != value['frames']*volume: raise ValueError('invalid frame totals')
    if (not np.array_equal(np.asarray(value['historical_known_confusion'])+value['historical_unknown_confusion'], c)
            or not np.array_equal(np.asarray(value['distance_confusion']).sum(0), c)):
        raise ValueError('conditional confusion does not reproduce full frame counts')
    expected = dict(false_occupancy=int(c[17, :17].sum()), missed_occupancy=int(c[:17, 17].sum()),
        wrong_class=int(c[:17, :17].sum()-np.diag(c)[:17].sum()))
    for family, n in expected.items():
        row = value['families'][family]
        if (row['voxels'] != n or sum(row['class_voxels']) != n or sum(row['boundary_distance_voxel_hist']) != n
                or sum(row['component_sizes'].values()) != row['components']
                or sum(int(k)*v for k, v in row['component_sizes'].items()) != n
                or sum(row['component_joint'].values()) != row['components']
                or sum(int(k.split(',')[-1])*v for k, v in row['component_joint'].items()) != n):
            raise ValueError('component/error counters inconsistent')
    if 'objects' in value:
        o = value['objects']
        if (o['missing_STC_boxes']+o['jointly_supported_boxes'] != o['GT_supported_boxes']
                or o['GT_supported_boxes'] > o['annotation_boxes'] or sum(o['pred_to_GT_voxel_ratio_hist']) != o['GT_supported_boxes']
                or any(sum(h) != o['jointly_supported_boxes'] for h in o['offset_xyz_hist'])):
            raise ValueError('object counters inconsistent')
    if 'temporal' in value:
        t = value['temporal']; d = t['dynamic']
        if (t['static_stable_overlap_voxels'] > t['windows4']*volume
                or any(len(h) != 16 or sum(h) != t['static_stable_overlap_voxels'] for h in t['static_patterns'].values())
                or sum(d['missing_STC_patterns']) != d['tracks_GT_supported_all4']
                or sum(d['delta_offset_xy_hist']) != d['residual_pair_tracks']): raise ValueError('temporal counters inconsistent')


def noise_model(stats):
    """Empirical sufficient statistics for a later simulator, NOT a recipe."""
    c = np.asarray(stats['distance_confusion'], float)
    denom = c.sum(-1, keepdims=True)
    conditional = np.divide(c, denom, out=np.zeros_like(c), where=denom > 0)
    return dict(schema='STC_empirical_error_statistics_v1', simulator_implemented=False,
        conditional_pred_given_GT_and_distance=conditional.tolist(), condition_counts=denom[..., 0].astype(np.int64).tolist(),
        component_joint= {f: stats['families'][f]['component_joint'] for f in FAMILIES},
        component_class_policy=dict(false_occupancy='predicted', missed_occupancy='GT', wrong_class='predicted'),
        temporal_bit_order='bit0 oldest, bit3 latest; four historical keyframes only',
        temporal=stats.get('temporal', {}), object_geometry=stats.get('objects', {}),
        note='VAL error-calibrated development artifact. Zero support is missing evidence, not zero error probability. '
             'No coordinates, tokens, GT shapes or source frame contents are copied into this artifact.')
