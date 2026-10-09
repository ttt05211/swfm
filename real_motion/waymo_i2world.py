"""Pinned I2-World Occ3D-Waymo 2Hz protocol; no model/nuScenes cache changes.

Metadata is reduced to timestamps, identities and ego poses before prediction.
Future labels have a separate API, and no visibility/annotation mask is used.
"""
from collections import Counter, OrderedDict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

PROTOCOL = 'p0_f9_surface_ccr_i2world_waymo_2hz_v1'
UPSTREAM_COMMIT = '661d830f9b34ee03ce368db164a72753ab8764a3'
UPSTREAM_URL = 'https://github.com/lzzzzzm/II-World/tree/' + UPSTREAM_COMMIT
SHAPE = (200, 200, 16)
REPORT_INDICES = (1, 3, 5)
CLASS_NAMES = ('others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
               'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
               'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
               'vegetation', 'free')
LABEL_MAP = {0: 0, 1: 4, 2: 7, 3: 15, 4: 2, 5: 15, 6: 15, 7: 8,
             8: 2, 9: 6, 10: 15, 11: 16, 12: 16, 13: 11, 14: 13, 23: 17}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def remap_labels(value, *, raw_free_label=23, shape=SHAPE):
    """Literal author mapping; documented Occ3D free=15 requires explicit opt-in.

    Never let an unrecognised free/unknown label silently become manmade or a
    dynamic class. Predictions are already in nuScenes IDs and are NOT remapped.
    """
    value = np.asarray(value)
    if value.shape != tuple(shape) or value.dtype.kind not in 'ui':
        raise ValueError('Waymo voxel_label must be an integer XYZ grid: ' + str(tuple(shape)))
    if raw_free_label not in (15, 23):
        raise ValueError('raw free label must be explicitly 23 (I2-World) or 15 (Occ3D)')
    mapping = {**{k: v for k, v in LABEL_MAP.items() if k != 23}, raw_free_label: 17}
    unique = np.unique(value)
    unknown = [int(v) for v in unique if int(v) not in mapping]
    if unknown:
        raise ValueError(f'unmapped Waymo labels {unknown}; check --raw-free-label, do not infer from GT')
    lut = np.full(24, 255, np.uint8)
    for k, v in mapping.items():
        lut[k] = v
    return np.ascontiguousarray(lut[value])


@dataclass(frozen=True)
class Frame:
    sample: int
    scene: int
    frame: int
    timestamp_us: float
    pose: np.ndarray
    path: Path

    @property
    def token(self):
        return f'{self.sample:07d}'


@dataclass(frozen=True)
class WaymoWindow:
    anchor: int
    history: tuple
    future: tuple
    history_padding: int
    future_padding: int


class WaymoI2WorldSource:
    def __init__(self, infos, poses, root, *, raw_free_label=23, cache_mib=256, shape=SHAPE):
        self.root = Path(root).resolve()
        self.raw_free_label, self.shape = raw_free_label, tuple(shape)
        if raw_free_label not in (15, 23) or not 0 <= cache_mib <= 4096:
            raise ValueError('invalid label encoding / bounded frame-cache budget')
        if not isinstance(infos, list) or not infos:
            raise ValueError('I2-World waymo_infos_val.pkl must contain a nonempty LIST, not a nuScenes info dict')
        # Exact upstream order: GLOBAL sort, THEN stride 5, NOT per-scene stride.
        ordered = sorted(infos, key=lambda row: row['timestamp'])
        self.native_frames = len(ordered)
        self.frames = []
        seen = set()
        for row in ordered[::5]:
            sample = row['image']['image_idx']; timestamp = row['timestamp']
            if (isinstance(sample, (bool, np.bool_)) or not isinstance(sample, (int, np.integer))
                    or sample < 0 or not np.isfinite(timestamp)):
                raise ValueError('invalid Waymo image_idx/timestamp')
            sample = int(sample)
            scene, frame = sample % 1000000 // 1000, sample % 1000000 % 1000
            if (scene, frame) in seen:
                raise ValueError('duplicate sampled Waymo scene/frame')
            seen.add((scene, frame))
            pose = np.array(poses[scene][frame][0]['ego2global'], dtype=np.float64, copy=True)
            if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                    or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
                    or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4, rtol=0)
                    or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-4, rtol=0)):
                raise ValueError(f'non-rigid ego2global for {sample}')
            pose.setflags(write=False)
            self.frames.append(Frame(sample, scene, frame, float(timestamp), pose,
                                     self.root / 'validation' / f'{scene:03d}' / f'{frame:03d}_04.npz'))
        # Repeated scene segments are handled just like the official prev links.
        starts, ends = [], []
        begin = 0
        for i, row in enumerate(self.frames):
            if i and row.scene != self.frames[i-1].scene:
                ends.extend([i-1] * (i-begin)); begin = i
            starts.append(begin)
        ends.extend([len(self.frames)-1] * (len(self.frames)-begin))
        self.windows = []
        intervals = []; frame_steps = Counter(); timing_outliers = []
        for i, row in enumerate(self.frames):
            if i > starts[i]:
                previous = self.frames[i-1]
                dt = (row.timestamp_us - previous.timestamp_us) / 1e6
                frame_step = row.frame - previous.frame
                if dt <= 0 or frame_step <= 0:
                    raise ValueError(f'sampled Waymo frames are not ordered: scene={row.scene}, '
                                     f'frames={previous.frame}->{row.frame}, dt_s={dt}; '
                                     'do not silently reorder or retime the model')
                intervals.append(dt)
                frame_steps[frame_step] += 1
                if not .35 <= dt <= .65:
                    timing_outliers.append(dict(scene=row.scene,
                        frame_pair=[previous.frame, row.frame], dt_s=dt,
                        timestamp_pair_us=[previous.timestamp_us, row.timestamp_us],
                        frame_step=frame_step))
            hist = tuple(max(starts[i], i-j) for j in (3, 2, 1, 0))
            future = tuple(min(ends[i], i+j) for j in range(1, 7))
            self.windows.append(WaymoWindow(i, hist, future, max(0, 3-(i-starts[i])),
                                            max(0, 6-(ends[i]-i))))
        # The official loader uses GLOBAL timestamp sort + stride 5, not a
        # timestamp-uniform resampler. Real release files contain rare gaps
        # (e.g. five native frames can span 0.7--1.2s). Preserve those anchors
        # and their real poses/timestamps; audit, never interpolate/drop them.
        # Still reject a dataset with a different overall cadence or units.
        median_dt = float(np.median(intervals)) if intervals else None
        if median_dt is not None and not .35 <= median_dt <= .65:
            raise ValueError(f'sampled Waymo frames are not nominal 2Hz: median_dt_s={median_dt}; '
                             'check timestamp units/native frame rate, do not silently retime the model')
        self.metadata = dict(protocol=PROTOCOL, upstream_commit=UPSTREAM_COMMIT,
            native_frames=self.native_frames, sampled_frames=len(self.frames),
            scenes=len({row.scene for row in self.frames}), load_interval=5, sample_hz=2,
            global_sort_then_stride=True, history_frames_including_t0=4, future_frames=6,
            scene_boundary='repeat_nearest_valid_frame_including_future_targets',
            history_padded_windows=sum(w.history_padding > 0 for w in self.windows),
            future_padded_windows=sum(w.future_padding > 0 for w in self.windows),
            actual_adjacent_dt_s=(dict(min=min(intervals), max=max(intervals), mean=float(np.mean(intervals)),
                                      median=median_dt)
                                  if intervals else None), raw_free_label=raw_free_label,
            timestamp_gap_audit=dict(pair_count=len(intervals), nominal_dt_s=.5,
                typical_bounds_s=[.35, .65], outlier_count=len(timing_outliers),
                outlier_fraction=len(timing_outliers)/len(intervals) if intervals else 0.,
                outliers_by_scene=dict(Counter(str(r['scene']) for r in timing_outliers)),
                examples=timing_outliers[:20],
                frame_step_histogram={str(k): v for k, v in sorted(frame_steps.items())},
                validation='positive ordered links; median nominal 2Hz; rare gaps audited',
                resampled=False, dropped_windows=0),
            label_encoding='author_free23' if raw_free_label == 23 else 'explicit_Occ3D_free15_normalization',
            label_map={str(k): v for k, v in LABEL_MAP.items() if k != 23},
            history_visibility='dense_input_all_true; no extra sensor mask',
            metric_visibility='none; identical to upstream use_lidar_mask=False/use_image_mask=False')
        self.metadata['label_map'][str(raw_free_label)] = 17
        self.metadata['actual_report_dt_s_including_padded_targets'] = {}
        for h, seconds in zip(REPORT_INDICES, (1, 2, 3)):
            times = [(self.frames[w.future[h]].timestamp_us-self.frames[w.anchor].timestamp_us)/1e6
                     for w in self.windows]
            self.metadata['actual_report_dt_s_including_padded_targets'][str(seconds)] = dict(
                min=min(times), mean=float(np.mean(times)), max=max(times), zero_time_targets=sum(t == 0 for t in times))
        self.manifest_fingerprint = fingerprint([
            [r.sample, r.timestamp_us, r.pose.tolist(), list(w.history), list(w.future)]
            for r, w in zip(self.frames, self.windows)])
        self.cache = OrderedDict(); self.cache_bytes = 0; self.cache_limit = int(cache_mib * 1024**2)
        self.io_reads = self.cache_hits = 0
        self.inventory = {}

    @classmethod
    def from_files(cls, root, *, info_file=None, pose_file=None, **kwargs):
        root = Path(root)
        info_file = Path(info_file) if info_file else root / 'waymo_infos_val.pkl'
        pose_file = Path(pose_file) if pose_file else root / 'cam_infos_vali.pkl'
        # Pickles are executable: callers must supply trusted official files.
        before = [file_sha256(p) for p in (info_file, pose_file)]
        with info_file.open('rb') as handle:
            infos = pickle.load(handle)
        with pose_file.open('rb') as handle:
            poses = pickle.load(handle)
        value = cls(infos, poses, root, **kwargs)
        if before != [file_sha256(p) for p in (info_file, pose_file)]:
            raise RuntimeError('Waymo metadata changed during read')
        value.metadata['source_files'] = [dict(path=str(p.resolve()), sha256=s)
                                        for p, s in zip((info_file, pose_file), before)]
        return value

    def preflight(self, selected):
        indices = sorted({i for w in selected for i in (*w.history, *w.future)})
        inventory = []
        for i in indices:
            row = self.frames[i]
            try:
                stat = row.path.stat()
            except FileNotFoundError as error:
                raise FileNotFoundError(f'missing Occ3D-Waymo validation NPZ: {row.path}') from error
            if not row.path.is_file() or stat.st_size == 0:
                raise ValueError('invalid Waymo NPZ: ' + str(row.path))
            self.inventory[i] = (stat.st_size, stat.st_mtime_ns)
            inventory.append([row.token, stat.st_size, stat.st_mtime_ns])
        return dict(files=len(indices), total_bytes=sum(r[1] for r in inventory),
                    file_stat_fingerprint=fingerprint(inventory),
                    integrity_scope='size/mtime checked on every read; metadata+weights SHA256, not all NPZ SHA256')

    def occupancy(self, index):
        row = self.frames[index]
        if index in self.inventory:
            stat = row.path.stat()
            if (stat.st_size, stat.st_mtime_ns) != self.inventory[index]:
                raise RuntimeError('Waymo NPZ changed during evaluation: ' + str(row.path))
        if index in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(index)
            return self.cache[index]
        with np.load(row.path, allow_pickle=False) as archive:
            result = remap_labels(archive['voxel_label'], raw_free_label=self.raw_free_label, shape=self.shape)
        self.io_reads += 1
        result.setflags(write=False)
        if result.nbytes <= self.cache_limit:
            while self.cache_bytes + result.nbytes > self.cache_limit and self.cache:
                _, old = self.cache.popitem(last=False); self.cache_bytes -= old.nbytes
            self.cache[index] = result; self.cache_bytes += result.nbytes
        return result

    def prediction_inputs(self, window):
        """Read ONLY history occupancy and poses; future poses are ego conditioning."""
        hist = np.stack([self.occupancy(i) for i in window.history])
        raw = dict(history_occ=hist, history_observed=np.ones(hist.shape, bool),
                   history_poses=[self.frames[i].pose for i in window.history],
                   future_poses=[self.frames[i].pose for i in window.future], future_gt_occ=None)
        row = self.frames[window.anchor]
        record = dict(scene_name=f'waymo-validation-{row.scene:03d}', t0_token=row.token,
                      history_tokens=tuple(self.frames[i].token for i in window.history),
                      future_tokens=tuple(self.frames[i].token for i in window.future))
        return record, raw

    def metric_targets(self, window):
        # This method is called only AFTER all six forecasts have completed.
        return [self.occupancy(window.future[h]) for h in REPORT_INDICES]


class WaymoMetrics:
    """Nonmutating integer counts; publish literal I2 and conventional mIoU."""
    def __init__(self, counts=None, windows=0):
        if counts is not None and np.asarray(counts).dtype.kind not in 'ui':
            raise ValueError('Waymo counts must be integers, not cast float scores')
        self.counts = np.zeros((3, 18, 18), np.int64) if counts is None else np.array(counts, dtype=np.int64)
        if self.counts.shape != (3, 18, 18) or (self.counts < 0).any():
            raise ValueError('invalid saved Waymo integer counts')
        self.windows = windows

    def add(self, predictions, targets):
        if len(predictions) != 6 or len(targets) != 3:
            raise ValueError('six predictions / three reporting targets required')
        delta = np.zeros_like(self.counts)
        for j, h in enumerate(REPORT_INDICES):
            pred, gt = np.asarray(predictions[h]), np.asarray(targets[j])
            if pred.shape != gt.shape or pred.dtype.kind not in 'ui' or gt.dtype.kind not in 'ui':
                raise ValueError('prediction/GT integer shapes differ')
            if pred.size == 0 or pred.max() > 17 or gt.max() > 17 or pred.min() < 0 or gt.min() < 0:
                raise ValueError('Waymo metric requires mapped IDs 0..17, no ignore masks')
            delta[j] = np.bincount(18 * gt.ravel().astype(np.int64) + pred.ravel(), minlength=324).reshape(18, 18)
        self.counts += delta; self.windows += 1

    def report(self):
        horizons = {}
        for seconds, hist in zip((1, 2, 3), self.counts):
            tp = hist.diagonal(); union = hist.sum(0) + hist.sum(1) - tp
            score = np.divide(tp, union, out=np.full(18, np.nan), where=union > 0) * 100
            standard = score[:17][np.isfinite(score[:17])]
            official = standard[standard != 0]  # Literal upstream count_miou zero->None.
            occ_tp = int(hist[:17, :17].sum())
            occ_union = int(hist[:17].sum() + hist[:, :17].sum() - occ_tp)
            iou = 100 * occ_tp / occ_union if occ_union else None
            miou = float(official.mean()) if len(official) else None
            horizons[str(seconds)] = dict(IoU=iou, i2world_mIoU=miou,
                standard_mIoU=float(standard.mean()) if len(standard) else None,
                i2world_rounded_IoU=float(np.round(iou, 2)) if iou is not None else None,
                i2world_rounded_mIoU=float(np.round(miou, 2)) if miou is not None else None,
                i2world_zero_IoU_classes_excluded=[CLASS_NAMES[i] for i in range(17) if score[i] == 0],
                per_class_IoU={name: float(v) if np.isfinite(v) else None for name, v in zip(CLASS_NAMES, score)})
        average = {}
        for key in ('IoU', 'i2world_mIoU', 'standard_mIoU', 'i2world_rounded_IoU', 'i2world_rounded_mIoU'):
            values = [r[key] for r in horizons.values()]
            average[key] = float(np.mean(values)) if all(v is not None for v in values) else None
        return dict(windows=self.windows, average=average, horizons=horizons,
                    aggregation='pooled integer counts per horizon, then arithmetic mean of 1/2/3s scores')
