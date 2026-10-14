"""History-only empirical copula noise; NOT an exact STCOcc simulator."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
from tools.stc_noise_profile.statistics import centers, DISTANCE, BOUNDARY, DYNAMIC
from tools.stc_noise_profile.data import fingerprint, file_sha

RECIPE = dict(protocol='empirical_correlated_history_noise_v1', selected_window_probability=.5,
    spatial_sigma_vox_bounds=[.25, 8.], spatial_extent_divisor=6.,
    component_sampling='voxel_mass_weighted_joint_extent; covariance approximation, not exact region copying',
    temporal_sampling='four-bit conditional pattern; actual marginal clipping audited',
    dynamic_sampling='history registration aligns per-voxel fields; ROI support pattern is NOT whole-object erasure',
    known_policy='dense_valid_BOTH_arms', future_labels_used=False, new_GT_or_ego_losses=False)


class NoiseModel:
    def __init__(self, value):
        if value.get('schema') != 'STC_empirical_error_statistics_v1' or value.get('simulator_implemented') is not False:
            raise ValueError('completed empirical STC profile required')
        split = value['fitting_split']; fit, remaining = set(split['fit_scenes']), set(split['remaining_scenes'])
        if not fit or not remaining or fit & remaining or len(fit | remaining) != 150:
            raise ValueError('declared disjoint original 150-scene development split required')
        self.value, self.fit, self.remaining = value, fit, remaining
        self.p = np.asarray(value['conditional_pred_given_GT_and_distance'], float)
        self.support = np.asarray(value['condition_counts'])
        if (self.p.shape != (6, 18, 18) or self.support.shape != (6, 18)
                or not np.isfinite(self.p).all() or (self.p < 0).any() or (self.p > 1).any()
                or self.support.dtype.kind not in 'iu' or (self.support < 0).any()
                or not np.allclose(self.p.sum(-1)[self.support > 0], 1., atol=1e-8)
                or np.any(self.p[self.support == 0] != 0)):
            raise ValueError('invalid conditional probabilities/support')
        self.patterns = {}
        for family in ('false_occupancy', 'missed_occupancy', 'wrong_class'):
            self.patterns[family] = self._pattern(value['temporal']['static_patterns'][family])
        self.dynamic_pattern = self._pattern(value['temporal']['dynamic']['missing_STC_patterns'])
        self.extents = {}
        for family, table in value['component_joint'].items():
            rows, mass = [], []
            for key, count in sorted(table.items()):
                cid, radius, x, y, z, size = map(int, key.split(','))
                if (not 0 <= cid < 17 or not 0 <= radius < 6 or min(x, y, z, size, count) <= 0
                        or x > 200 or y > 200 or z > 16 or size > x*y*z): raise ValueError('invalid component joint')
                rows.append((x, y, z)); mass.append(count*size)
            if not rows: raise ValueError('no supported component shapes: '+family)
            mass = np.asarray(mass, float); self.extents[family] = np.asarray(rows), mass/mass.sum()
        # Full profile is bound and supplies the marginal distance-to-surface
        # histogram. It contains aggregate statistics, never VAL voxel arrays.
        boundary = value.get('boundary_distance_voxel_hist')
        if boundary is None: raise ValueError('loader must supply bound profile boundary histogram')
        self.boundary = np.asarray(boundary, float)
        if self.boundary.shape != (6,) or (self.boundary < 0).any() or not self.boundary.sum():
            raise ValueError('invalid false-occupancy boundary distribution')
        self.boundary /= self.boundary.sum()
        self.fp_classes = self.p[:, 17].copy(); self.fp_classes[:, 17] = 0
        self.wrong_classes = self.p.copy(); self.wrong_classes[..., 17] = 0
        self.wrong_classes[:, np.arange(18), np.arange(18)] = 0
        self.wrong_mass = self.wrong_classes.sum(-1)

    @staticmethod
    def _pattern(hist):
        h = np.asarray(hist)
        if h.shape != (16,) or h.dtype.kind not in 'iu' or (h < 0).any(): raise ValueError('invalid history pattern')
        weights = h[1:].astype(float)
        if not weights.sum(): raise ValueError('unsupported temporal error family; no invented fallback')
        weights /= weights.sum(); mean = sum(w*int(i).bit_count()/4 for i, w in enumerate(weights, 1))
        return weights.cumsum(), mean


def load_model(path):
    path = Path(path).resolve(); profile_path = path.with_name('evaluation.json')
    value = json.loads(path.read_text(encoding='utf-8')); profile = json.loads(profile_path.read_text(encoding='utf-8'))
    contract = profile['contract']
    if (profile.get('status') != 'complete' or profile['protocol'] != 'STC_VAL_development_error_fit_four_history_v1'
            or fingerprint(contract) != value['contract_sha256']
            or {k: v for k, v in profile['split'].items() if k != 'seed_DEV64_keys'} != value['fitting_split']):
        raise ValueError('noise_model/profile source identity mismatch')
    value = dict(value, boundary_distance_voxel_hist=profile['statistics']['families']['false_occupancy']['boundary_distance_voxel_hist'])
    return NoiseModel(value), dict(noise_model_sha256=file_sha(path), profile_sha256=file_sha(profile_path),
                                 source_GT_sha256=contract['source_model_declared_sha256'])


def rng_for(record, seed):
    digest = hashlib.sha256(f'{seed}:{record["scene_name"]}:{record["t0_token"]}'.encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], 'little'))


def uniform_field(shape, sigma, rng):
    values = gaussian_filter(rng.standard_normal(shape).astype(np.float32), sigma=sigma, mode='reflect')
    order = np.argsort(values.ravel(), kind='stable'); ranks = np.empty(values.size, np.float32)
    ranks[order] = (np.arange(values.size)+.5)/values.size
    return ranks.reshape(shape)


def history_addresses(labels, poses, registrations, origin, step):
    shape = tuple(labels.shape[1:]); xyz = centers(shape, tuple(origin), tuple(step))
    inv = np.linalg.inv(poses[-1]); addresses, overlaps = [], []
    for frame in range(4):
        world = xyz @ poses[frame][:3, :3].T+poses[frame][:3, 3]
        at = (world @ inv[:3, :3].T+inv[:3, 3]-origin)/step
        # Pure historical source registrations map historical actor geometry
        # into current source coordinates. Never read future motion/labels.
        for track in registrations or ():
            reg = track[frame]
            if reg is None: continue
            matrix, cells = reg; cells = np.asarray(cells, np.int64)
            cells = cells[((cells >= 0) & (cells < shape)).all(1)]
            if not len(cells): continue
            w = world[tuple(cells.T)]; mapped = w @ np.asarray(matrix)[:3, :3].T+np.asarray(matrix)[:3, 3]
            at[tuple(cells.T)] = (mapped @ inv[:3, :3].T+inv[:3, 3]-origin)/step
        ids = np.floor(at).astype(np.int64); good = ((ids >= 0) & (ids < shape)).all(-1)
        ids = np.clip(ids, 0, np.asarray(shape)-1)
        addresses.append(np.ravel_multi_index(tuple(ids[..., a] for a in range(3)), shape)); overlaps.append(good)
    return addresses, overlaps


def corrupt(historical, record, grid, model, seed, *, registrations=(), force=False):
    labels = np.asarray(historical['history_occ']); poses = np.asarray(historical['history_poses'], float)
    if labels.ndim != 4 or len(labels) != 4 or labels.dtype.kind not in 'ui' or poses.shape != (4, 4, 4) or not np.isfinite(poses).all():
        raise ValueError('FOUR historical semantic grids/poses required')
    origin = np.array([grid.x_min, grid.y_min, grid.z_min]); step = np.asarray(grid.voxel_size)
    shape = tuple(grid.shape_hwd)
    if labels.shape[1:] != shape: raise ValueError('noise grid differs')
    valid = (labels >= 0) & (labels < 18)
    result = dict(history_occ=labels.copy(), history_observed=valid.copy(), history_poses=poses.copy())
    rng = rng_for(record, seed)
    audit = dict(selected=False, family='clean_dense', changed_voxels=[0]*4, clean_geometry_reusable=True,
                 probability_clipped_cells=0, spatial_sigma_vox={}, known_policy='dense_valid')
    if not force and rng.random() >= .5: return result, audit
    addresses, overlap = history_addresses(labels, poses, registrations, origin, step)
    fields, patterns = {}, {}
    for family in model.patterns:
        extents, weights = model.extents[family]
        extent = extents[rng.choice(len(extents), p=weights)]
        sigma = np.clip(extent/6., .25, 8.)
        audit['spatial_sigma_vox'][family] = sigma.tolist()
        fields[family] = uniform_field(shape, sigma, rng).ravel()
        patterns[family] = uniform_field(shape, sigma, rng).ravel()
    classes = uniform_field(shape, (1., 1., .25), rng).ravel()
    xyz = centers(shape, tuple(origin), tuple(step))
    radial = np.clip(np.searchsorted(DISTANCE, np.linalg.norm(xyz[..., :2], axis=-1), side='right')-1, 0, 5)
    counts = {f: [0]*4 for f in model.patterns}
    radial_counts = {f: np.zeros(6, np.int64) for f in model.patterns}
    opportunities = {f: np.zeros(6, np.int64) for f in ('GT_free', 'GT_occupied')}
    for frame in range(4):
        gt = labels[frame]; safe = np.clip(gt, 0, 17); good = valid[frame]
        def field(table):
            x = table[addresses[frame]].copy()
            x[~overlap[frame]] = rng.random(np.count_nonzero(~overlap[frame]))
            return x
        u = field(fields['missed_occupancy']); class_u = field(classes)
        def active(family):
            v = field(patterns[family]); cdf, mean = model.patterns[family]
            bits = np.searchsorted(cdf, v, side='left')+1
            density = np.full(shape, mean)
            if family != 'false_occupancy':
                mask = np.isin(gt, DYNAMIC); cdf, dmean = model.dynamic_pattern
                bits[mask] = np.searchsorted(cdf, v[mask], side='left')+1; density[mask] = dmean
            return (bits & (1 << frame)) != 0, density
        missed_active, missed_mean = active('missed_occupancy')
        wrong_active, wrong_mean = active('wrong_class')
        qmiss = model.p[radial, safe, 17]/missed_mean
        qwrong = model.wrong_mass[radial, safe]/wrong_mean
        total = qmiss+qwrong
        audit['probability_clipped_cells'] += int(((total > 1) & good & (gt != 17)).sum())
        divisor = np.maximum(total, 1.); qmiss /= divisor; qwrong /= divisor
        missed = good & (gt != 17) & (u < qmiss) & missed_active
        wrong = good & (gt != 17) & (u >= qmiss) & (u < qmiss+qwrong) & wrong_active
        occupied = good & (gt != 17)
        distance = distance_transform_edt(~occupied, sampling=step) if occupied.any() else np.full(shape, np.inf)
        boundary = np.clip(np.searchsorted(BOUNDARY, distance, side='right')-1, 0, 5)
        qfp = np.zeros(shape)
        fp_active, fp_mean = active('false_occupancy')
        for radius in range(6):
            free = good & (gt == 17) & (radial == radius)
            available = np.bincount(boundary[free], minlength=6)
            weights = model.boundary*(available > 0)
            if not weights.sum(): continue
            weights /= weights.sum()
            intended = free.sum()*model.fp_classes[radius].sum()
            probabilities = np.divide(intended*weights, available, out=np.zeros(6), where=available > 0)
            qfp[free] = probabilities[boundary[free]]/fp_mean[free]
        audit['probability_clipped_cells'] += int(((qfp > 1) & good & (gt == 17)).sum())
        false = good & (gt == 17) & (field(fields['false_occupancy']) < np.minimum(qfp, 1.)) & fp_active
        out = result['history_occ'][frame]; out[missed] = 17
        for mask, is_false in ((false, True), (wrong, False)):
            if not mask.any(): continue
            p = (model.fp_classes[radial[mask]] if is_false
                 else model.wrong_classes[radial[mask], safe[mask]])
            p /= p.sum(1, keepdims=True)
            cdf = p.cumsum(1); cdf[:, -1] = 1.
            out[mask] = (class_u[mask][:, None] > cdf).sum(1).astype(np.uint8)
        for family, mask in (('false_occupancy', false), ('missed_occupancy', missed), ('wrong_class', wrong)):
            counts[family][frame] = int(mask.sum())
            radial_counts[family] += np.bincount(radial[mask], minlength=6)
        opportunities['GT_free'] += np.bincount(radial[good & (gt == 17)], minlength=6)
        opportunities['GT_occupied'] += np.bincount(radial[good & (gt != 17)], minlength=6)
    changes = np.count_nonzero(result['history_occ'] != labels, axis=(1, 2, 3)).tolist()
    audit.update(selected=True, family='empirical_correlated', changed_voxels=changes,
                 clean_geometry_reusable=not any(changes), generated_errors=counts,
                 radial_errors={k: v.tolist() for k, v in radial_counts.items()},
                 radial_opportunities={k: v.tolist() for k, v in opportunities.items()})
    return result, audit
