"""Literal I2-World native-frame protocol, isolated from the running 2Hz ABI.

Official load_interval=1 / eval_metric='miou' / eval_time=1,3,5 uses
ZERO-based future indices. Native 10Hz targets are +2/+4/+6 frames, NOT
physical 1/2/3 seconds. Reuse unchanged label/I-O and integer metric code.
"""
from collections import Counter, OrderedDict
from copy import deepcopy
from pathlib import Path

import numpy as np

from .waymo_i2world import (Frame, LABEL_MAP, REPORT_INDICES, SHAPE, UPSTREAM_COMMIT,
                            WaymoI2WorldSource, WaymoWindow, fingerprint)

PROTOCOL = 'p0_f9_surface_ccr_i2world_waymo_10hz_index_v1'
REPORT_KEYS = tuple(f'eval_time_{h}' for h in REPORT_INDICES)
REPORT_SECONDS = (.2, .4, .6)


class WaymoI2World10HzSource(WaymoI2WorldSource):
    """All globally timestamp-sorted anchors; no stride5, interpolation or GT input.

    Kept separate to leave the existing 2Hz implementation fingerprints intact:
    a concurrently running 2Hz evaluation can still resume after this addition.
    """
    def __init__(self, infos, poses, root, *, raw_free_label=23, cache_mib=256, shape=SHAPE):
        self.root = Path(root).resolve()
        self.raw_free_label, self.shape = raw_free_label, tuple(shape)
        if raw_free_label not in (15, 23) or not 0 <= cache_mib <= 4096:
            raise ValueError('invalid label encoding / bounded frame-cache budget')
        if not isinstance(infos, list) or not infos:
            raise ValueError('I2-World waymo_infos_val.pkl must contain a nonempty LIST')
        ordered = sorted(infos, key=lambda row: row['timestamp'])  # literal [::1]
        self.native_frames = len(ordered); self.frames = []; seen = set()
        for row in ordered:
            sample, timestamp = row['image']['image_idx'], row['timestamp']
            if (isinstance(sample, (bool, np.bool_)) or not isinstance(sample, (int, np.integer))
                    or sample < 0 or not np.isfinite(timestamp)):
                raise ValueError('invalid Waymo image_idx/timestamp')
            sample = int(sample); scene, frame = sample % 1000000 // 1000, sample % 1000
            if (scene, frame) in seen:
                raise ValueError('duplicate native Waymo scene/frame')
            seen.add((scene, frame))
            pose = np.array(poses[scene][frame][0]['ego2global'], dtype=np.float64, copy=True)
            if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                    or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
                    or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4, rtol=0)
                    or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-4, rtol=0)):
                raise ValueError(f'non-rigid ego2global for {sample}')
            pose.setflags(write=False)
            self.frames.append(Frame(sample, scene, frame, float(timestamp), pose,
                self.root/'validation'/f'{scene:03d}'/f'{frame:03d}_04.npz'))
        starts, ends, begin = [], [], 0
        for i, row in enumerate(self.frames):
            if i and row.scene != self.frames[i-1].scene:
                ends.extend([i-1]*(i-begin)); begin = i
            starts.append(begin)
        ends.extend([len(self.frames)-1]*(len(self.frames)-begin))
        intervals, outliers, steps = [], [], Counter(); self.windows = []
        bounds = (.07, .13)
        for i, row in enumerate(self.frames):
            if i > starts[i]:
                previous = self.frames[i-1]
                dt = (row.timestamp_us-previous.timestamp_us)/1e6
                step = row.frame-previous.frame
                if dt <= 0 or step <= 0:
                    raise ValueError(f'native Waymo frames are not ordered: scene={row.scene}, '
                                     f'frames={previous.frame}->{row.frame}, dt_s={dt}')
                intervals.append(dt); steps[step] += 1
                if not bounds[0] <= dt <= bounds[1]:
                    outliers.append(dict(scene=row.scene, frame_pair=[previous.frame, row.frame],
                        timestamp_pair_us=[previous.timestamp_us, row.timestamp_us], dt_s=dt, frame_step=step))
            self.windows.append(WaymoWindow(i,
                tuple(max(starts[i], i-j) for j in (3, 2, 1, 0)),
                tuple(min(ends[i], i+j) for j in range(1, 7)),
                max(0, 3-(i-starts[i])), max(0, 6-(ends[i]-i))))
        median = float(np.median(intervals)) if intervals else None
        if median is not None and not bounds[0] <= median <= bounds[1]:
            raise ValueError(f'Waymo cadence is not nominal 10Hz: median_dt_s={median}; check timestamp units')
        self.metadata = dict(protocol=PROTOCOL, upstream_commit=UPSTREAM_COMMIT,
            native_frames=self.native_frames, sampled_frames=len(self.frames),
            scenes=len({r.scene for r in self.frames}), load_interval=1, sample_hz=10,
            global_sort_then_stride=True, history_frames_including_t0=4, future_frames=6,
            upstream_eval_metric='miou', upstream_eval_times=list(REPORT_INDICES),
            report_zero_based_indices=list(REPORT_INDICES), report_future_native_steps=[2, 4, 6],
            report_nominal_seconds=list(REPORT_SECONDS),
            upstream_comment_labels=['1s', '2s', '3s'],
            timing_note='official eval_time is a zero-based frame index; comments are NOT physical seconds at 10Hz',
            scene_boundary='repeat_nearest_valid_frame_including_future_targets',
            history_padded_windows=sum(w.history_padding > 0 for w in self.windows),
            future_padded_windows=sum(w.future_padding > 0 for w in self.windows),
            actual_adjacent_dt_s=(dict(min=min(intervals), max=max(intervals), median=median,
                                      mean=float(np.mean(intervals))) if intervals else None),
            timestamp_gap_audit=dict(pair_count=len(intervals), nominal_dt_s=.1,
                typical_bounds_s=list(bounds), outlier_count=len(outliers),
                outlier_fraction=len(outliers)/len(intervals) if intervals else 0.,
                outliers_by_scene=dict(Counter(str(r['scene']) for r in outliers)), examples=outliers[:20],
                frame_step_histogram={str(k): v for k, v in sorted(steps.items())},
                validation='positive ordered links; median nominal 10Hz; gaps audited',
                resampled=False, dropped_windows=0),
            raw_free_label=raw_free_label,
            label_encoding='author_free23' if raw_free_label == 23 else 'explicit_Occ3D_free15_normalization',
            label_map={str(k): v for k, v in LABEL_MAP.items() if k != 23},
            history_visibility='dense_input_all_true; no extra sensor mask',
            metric_visibility='none; identical to upstream use_lidar_mask=False/use_image_mask=False')
        self.metadata['label_map'][str(raw_free_label)] = 17
        spans = {}
        for key, h, seconds in zip(REPORT_KEYS, REPORT_INDICES, REPORT_SECONDS):
            times = [(self.frames[w.future[h]].timestamp_us-self.frames[w.anchor].timestamp_us)/1e6
                     for w in self.windows]
            spans[key] = dict(nominal_seconds=seconds, min=min(times), mean=float(np.mean(times)),
                             max=max(times), zero_time_targets=sum(t == 0 for t in times))
        self.metadata['actual_report_dt_s_including_padded_targets'] = spans
        self.manifest_fingerprint = fingerprint([
            [r.sample, r.timestamp_us, r.pose.tolist(), list(w.history), list(w.future)]
            for r, w in zip(self.frames, self.windows)])
        self.cache = OrderedDict(); self.cache_bytes = 0; self.cache_limit = int(cache_mib*1024**2)
        self.io_reads = self.cache_hits = 0; self.inventory = {}


def format_10hz_reports(reports):
    """Integer math is unchanged; remove misleading physical 1/2/3s labels."""
    formatted = deepcopy(reports)
    for report in formatted.values():
        rows = report.pop('horizons')
        report['horizons'] = {}
        for old, key, h, seconds in zip(('1', '2', '3'), REPORT_KEYS, REPORT_INDICES, REPORT_SECONDS):
            row = rows[old]
            row.update(eval_time=h, future_native_step=h+1, nominal_seconds=seconds)
            report['horizons'][key] = row
        report['aggregation'] = 'pooled integer counts at eval_time=1/3/5, then arithmetic mean; nominal 0.2/0.4/0.6s'
    return formatted
