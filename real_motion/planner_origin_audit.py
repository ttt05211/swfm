"""Read-only original BEV-Planner row audit; no model, semantics or alignment.

JSON rows have no sample tokens. Ordinal + measured current XY are independent
evidence of correspondence, NOT a token-level proof on stationary/repeated poses.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from real_motion.stc_camera_protocol import PLAN_SHA, PLAN_TAG, validate_pose

PROTOCOL = 'p0_f9_original_planner_t0_origin_audit_v1'
XY_TOLERANCE_M = .02
POSE_TOLERANCE = 1e-6


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def rebuild_poses(current, rows, *, cumulative=False):
    """COME convention: absolute global XY, independent t0-relative yaw."""
    current = validate_pose(current)
    rows = np.asarray(rows, dtype=np.float64)
    if rows.shape != (7, 3) or not np.isfinite(rows).all():
        raise ValueError('planner row must be finite (7,3)')
    result = []
    yaw = 0.
    for x, y, delta in rows[1:]:
        yaw = yaw + delta if cumulative else delta
        c, s = np.cos(yaw), np.sin(yaw)
        p = current.copy()
        p[:3, :3] = current[:3, :3] @ np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        p[:2, 3] = [x, y]
        result.append(p)
    return np.stack(result)


def load_cached_poses(source, window):
    """Open only approved identity / pose members, never legacy latent/GT fields."""
    path = source.plan_cache / 'records' / (window.key + '.npz')
    before = path.stat()
    with np.load(path, allow_pickle=False) as z:
        if (str(z['format'].item()) != 'stochocc_transport_cache_v2'
                or str(z['protocol_tag'].item()) != PLAN_TAG
                or str(z['scene'].item()) != window.scene
                or tuple(z['history_tokens'].tolist()) != window.history
                or tuple(z['future_tokens'].tolist()) != window.future):
            raise ValueError('cached record identity / ordering mismatch: ' + window.key)
        p = np.array(z['future_e2g'], dtype=np.float64, copy=True)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('planner cache record changed during read: ' + window.key)
    if p.shape != (6, 4, 4):
        raise ValueError('cached future_e2g must be (6,4,4)')
    for value in p:
        validate_pose(value)
    return p, (str(path), after.st_size, after.st_mtime_ns)


class PlannerRows:
    def __init__(self, catalog, path, *, expected_sha=PLAN_SHA):
        self.path = Path(path).resolve()
        self.digest = sha256(self.path)
        if self.digest != expected_sha:
            raise ValueError('original planner JSON SHA256 mismatch; no substitution allowed')
        payload = json.loads(self.path.read_text(encoding='utf-8'), object_pairs_hook=_unique_object)
        if sha256(self.path) != self.digest:
            raise RuntimeError('original planner JSON changed during read')
        rows = payload.get('trajs')
        if not isinstance(rows, dict) or not rows:
            raise ValueError('missing original planner trajs')
        scenes = {frame.scene for frame in catalog.values()}
        self.frames, self.rows, self.positions = {}, {}, {}
        grouped = {scene: [] for scene in scenes}
        for key, value in rows.items():
            match = re.fullmatch(r'(scene-\d+)-(\d+)', key)
            if not match:
                raise ValueError('invalid planner row key: ' + key)
            array = np.asarray(value, dtype=np.float64)
            if array.shape != (7, 3) or not np.isfinite(array).all():
                raise ValueError('invalid original planner row: ' + key)
            if match[1] in grouped:
                grouped[match[1]].append((int(match[2]), key, array))
        self.row_audit = []
        for scene in sorted(scenes):
            frames = sorted((f for f in catalog.values() if f.scene == scene), key=lambda f: f.timestamp)
            entries = sorted(grouped[scene], key=lambda item: item[0])
            if len(entries) != len(frames) or len({e[0] for e in entries}) != len(entries):
                raise ValueError('planner / complete scene count or suffix mismatch: ' + scene)
            for i, frame in enumerate(frames):
                if (frame.prev != (frames[i-1].token if i else '')
                        or frame.next != (frames[i+1].token if i+1 < len(frames) else '')
                        or (i and frame.timestamp <= frames[i-1].timestamp)):
                    raise ValueError('scene chronology / chain mismatch: ' + scene)
                self.positions[frame.token] = i
            self.frames[scene], self.rows[scene] = frames, entries
            for i, frame in enumerate(frames):
                self.row_audit.append(self.origin(frame, i))
        self.total_json_rows = len(rows)

    def origin(self, frame, ordinal):
        entries, frames = self.rows[frame.scene], self.frames[frame.scene]
        _, key, row = entries[ordinal]
        distances = np.linalg.norm(np.stack([f.pose[:2, 3] for f in frames]) - row[0, :2], axis=1)
        compatible = np.flatnonzero(distances <= XY_TOLERANCE_M).tolist()
        nearest = int(np.argmin(distances))
        alternatives = []
        for j in range(max(0, ordinal-2), min(len(entries), ordinal+3)):
            alternatives.append(dict(row_offset=j-ordinal, json_key=entries[j][1],
                row0_to_current_xy_m=float(np.linalg.norm(entries[j][2][0, :2] - frame.pose[:2, 3]))))
        return dict(scene=frame.scene, t0_token=frame.token, t0_ordinal=ordinal, json_key=key,
            current_xy_error_m=float(distances[ordinal]), origin_xy_compatible=ordinal in compatible,
            xy_identity_unique=compatible == [ordinal], compatible_frame_ordinals=compatible,
            nearest_frame_ordinal=nearest, nearest_frame_xy_error_m=float(distances[nearest]),
            nearest_frame_time_offset_s=(frames[nearest].timestamp-frame.timestamp)/1e6,
            adjacent_json_row_checks=alternatives)

    def window(self, source, window):
        current = source.catalog[window.t0]
        ordinal = self.positions[window.t0]
        result = self.origin(current, ordinal)
        entries = self.rows[window.scene]
        cached, stat = load_cached_poses(source, window)
        rebuilt = rebuild_poses(current.pose, entries[ordinal][2])
        error = float(np.max(np.abs(cached - rebuilt)))
        cumulative = rebuild_poses(current.pose, entries[ordinal][2], cumulative=True)
        # Neighbour checks are diagnostics only, never a row re-selection.
        for alt in result['adjacent_json_row_checks']:
            j = ordinal + alt['row_offset']
            alt['cached_pose_max_abs_error'] = float(np.max(np.abs(
                cached - rebuild_poses(current.pose, entries[j][2]))))
        result.update(cached_pose_max_abs_error=error,
            cached_pose_reconstruction_pass=error <= POSE_TOLERANCE,
            cumulative_yaw_pose_max_abs_error=float(np.max(np.abs(cached - cumulative))))
        result['status'] = ('mismatch' if not result['origin_xy_compatible'] or error > POSE_TOLERANCE else
                            'compatible_unique_xy' if result['xy_identity_unique'] else 'compatible_ambiguous_xy')
        return result, cached, stat

    def verify_unchanged(self):
        if sha256(self.path) != self.digest:
            raise RuntimeError('original planner JSON changed during audit')


def _errors(values):
    a = np.asarray(values, dtype=np.float64)
    return dict(median=float(np.median(a)), p90=float(np.quantile(a, .9)), maximum=float(a.max()))


def audit(source, original_json, *, expected_sha=PLAN_SHA, progress=None):
    manifest = source.plan_cache / 'manifest.json'
    manifest_sha = sha256(manifest)
    if manifest_sha != source.metadata['manifest_sha256']:
        raise RuntimeError('planner manifest changed since source construction')
    rows = PlannerRows(source.catalog, original_json, expected_sha=expected_sha)
    results = []; pose_digest = hashlib.sha256(); cache_stats = []
    for i, window in enumerate(source.windows, 1):
        result, poses, stat = rows.window(source, window)
        results.append(result)
        cache_stats.append(stat)
        pose_digest.update(window.key.encode() + b'\0' + poses.astype('<f8').tobytes())
        if progress and (i % 512 == 0 or i == len(source.windows)):
            progress(i, len(source.windows))
    if not results:
        raise ValueError('empty audit population')
    rows.verify_unchanged()
    for path, size, mtime in cache_stats:
        st = Path(path).stat()
        if (st.st_size, st.st_mtime_ns) != (size, mtime):
            raise RuntimeError('planner cache record changed during audit: ' + path)
    if sha256(manifest) != manifest_sha:
        raise RuntimeError('planner manifest changed during audit')
    statuses = dict(Counter(r['status'] for r in results))
    bad = [r for r in results if r['status'] == 'mismatch']
    full_origin = rows.row_audit
    return dict(protocol=PROTOCOL, windows=len(results), scenes=len(rows.frames),
        original_json=dict(path=str(rows.path), sha256=rows.digest, total_rows=rows.total_json_rows),
        cache_manifest=dict(path=str(manifest.resolve()), sha256=manifest_sha),
        cached_selected_pose_fingerprint=pose_digest.hexdigest(),
        tolerances=dict(current_xy_m=XY_TOLERANCE_M, cached_pose_max_abs=POSE_TOLERANCE),
        scope='read-only ALL planner-covered windows; no model/STC/GT occupancy/masks/score alignment',
        correspondence_evidence='numeric suffix sorted, then full-scene timestamp/chain ordinal; independently checked row0 global XY',
        identity_limitation='JSON contains no sample tokens. Stationary/revisited XY is not unique token identity proof.',
        statuses=statuses, origin_xy_errors_m=_errors([r['current_xy_error_m'] for r in results]),
        cached_pose_errors=_errors([r['cached_pose_max_abs_error'] for r in results]),
        full_scene_rows=dict(checked=len(full_origin),
            origin_xy_compatible=sum(r['origin_xy_compatible'] for r in full_origin),
            origin_xy_mismatch=sum(not r['origin_xy_compatible'] for r in full_origin),
            unique_xy=sum(r['xy_identity_unique'] for r in full_origin),
            origin_xy_errors_m=_errors([r['current_xy_error_m'] for r in full_origin])),
        cache_reconstruction_pass=sum(r['cached_pose_reconstruction_pass'] for r in results),
        mismatches=bad, window_audit=results,
        all_scene_origin_mismatches=[r for r in full_origin if not r['origin_xy_compatible']],
        decision=('correspondence_consistent_no_row_shift_evidence' if not bad and
                  all(r['origin_xy_compatible'] for r in full_origin) else 'inspect_mismatches_no_automatic_correction'))


def summary(result):
    lines = ['===== ORIGINAL PLANNER JSON / CURRENT t0 AUDIT =====',
        f"windows={result['windows']} scenes={result['scenes']} original_rows={result['original_json']['total_rows']}",
        'original_sha256=' + result['original_json']['sha256'],
        'statuses=' + json.dumps(result['statuses'], sort_keys=True),
        'current_xy_error_m=' + json.dumps(result['origin_xy_errors_m']),
        'cached_pose_max_abs_error=' + json.dumps(result['cached_pose_errors']),
        f"cache_reconstruction_pass={result['cache_reconstruction_pass']}/{result['windows']}",
        'full_scene_rows=' + json.dumps(result['full_scene_rows']),
        'decision=' + result['decision'],
        'Sorted suffix -> scene ordinal, NOT suffix == ordinal. JSON has no sample tokens;',
        'stationary/revisited XY can only establish compatibility, not unique identity.',
        'Read only: no model/GPU/occupancy/masks, no reindexing, pose alignment or score changes.']
    for r in result['mismatches'][:8]:
        lines.append('MISMATCH ' + json.dumps(r, sort_keys=True))
    return '\n'.join(lines) + '\n'
