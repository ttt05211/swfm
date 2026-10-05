"""Pinned PUBLIC CODE comparison contract, not a claimed paper reproduction.

Only sample identities/cadence are read from the official pickle. Its future
boxes, trajectories, masks and other annotations NEVER enter model inputs.
"""
import hashlib
import pickle
from pathlib import Path
import numpy as np

from real_motion.nuscenes_adapter import WindowTokens
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion.joint_column_full_common import EvaluationJointColumnProvider, prepare_causal_evidence, build_fixed_geometry
from tools.real_motion import joint_long_rollout_common as rollout

CODE_COMMIT = 'da48a529ffbe14136688e9b7a56f5d1061c366c5'
ARTIFACT_REVISION = '17e37acfff5b10517393a669ecf471f75f34d43f'
INFO_SHA256 = '0426072260a908260625c6dd91b9f06919726f5265c10849b87156d282547ded'
INFO_BYTES = 103781328
ORDERED_KEY_FINGERPRINT = '3bdfe0d56e8bccd6c499239550de7dc7827e969772d57bb570eb52bc34075604'
INFO_URL = f'https://huggingface.co/ANIYA673/GenieDrive/resolve/{ARTIFACT_REVISION}/world-nuscenes_infos_val.pkl'
POPULATION_PROTOCOL = 'geniedrive_public_code_strict4_plus20_metadata_v1'
METRIC_PROTOCOL = 'geniedrive_public_code_drop_exact_zero_semantic_iou_round2_v1'


def validate_grid(pcfg):
    grid = pcfg.grid
    if (tuple(grid.shape_hwd) != (200, 200, 16)
            or not np.allclose((grid.x_min, grid.y_min, grid.z_min), (-40, -40, -1), rtol=0, atol=1e-12)
            or not np.allclose(grid.voxel_size, (.4, .4, .4), rtol=0, atol=1e-12)):
        raise RuntimeError('GenieDrive comparison requires the full unmasked Occ3D 200x200x16 0.4m grid')


def verify_info(path):
    path = Path(path)
    if path.stat().st_size != INFO_BYTES:
        raise RuntimeError('official GenieDrive metadata size mismatch')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''): digest.update(chunk)
    if digest.hexdigest() != INFO_SHA256:
        raise RuntimeError('official GenieDrive metadata SHA256 mismatch')
    return digest.hexdigest()


def complete_start_indices(infos):
    """Literal equivalent of get_data_info padding + evaluate duplicate rejection.

    The dataset first sorts ALL samples globally by timestamp. Scene-local
    iteration is NOT substituted: previous-token checks on adjacent info rows
    define its population, including failures caused by interleaved scenes.
    """
    ordered = sorted(infos, key=lambda row: row['timestamp'])
    tokens = [str(row['token']) for row in ordered]
    if len(tokens) != len(set(tokens)): raise RuntimeError('duplicate official metadata token')
    starts = [i for i in range(3, len(ordered)-20)
              if all(str(ordered[j]['prev']) == tokens[j-1] for j in range(i-2, i+21))]
    return ordered, starts


def validate_start_manifest(keys):
    if (len(keys) != 2569 or len({k[0] for k in keys}) != 150
            or stable_json_fingerprint(keys) != ORDERED_KEY_FINGERPRINT):
        raise RuntimeError('official ordered start manifest does not match the independently audited pinned metadata')


def select_population(path, source, records):
    digest = verify_info(path)  # Authenticate BEFORE deserializing the trusted official artifact.
    with Path(path).open('rb') as stream: payload = pickle.load(stream)
    if payload.get('metadata', {}).get('version') != 'v1.0-trainval' or not isinstance(payload.get('infos'), list):
        raise RuntimeError('official GenieDrive metadata schema/version mismatch')
    ordered, starts = complete_start_indices(payload['infos'])
    samples = {}; scenes = {}
    for row in ordered:
        token = str(row['token']); sample = source.nusc.get('sample', token)
        scene = source.nusc.get('scene', sample['scene_token'])['name']
        if (str(row['prev']) != str(sample['prev']) or int(row['timestamp']) != int(sample['timestamp'])
                or str(row['scene_token']) != str(sample['scene_token'])
                or Path(str(row['occ_path'])).name != token
                or Path(str(row['occ_path'])).parent.name != scene):
            raise RuntimeError('official metadata and local nuScenes identity/order mismatch')
        samples[token], scenes[token] = sample, scene
    if source.allowed_scenes is None or set(scenes.values()) != set(source.allowed_scenes):
        raise RuntimeError('official/local validation scene populations differ')
    if set(samples) != set(source.info_by_token):
        raise RuntimeError('official/local validation sample populations differ')
    by = {(str(r['scene_name']), str(r['t0_token'])): r for r in records}
    if len(by) != len(records): raise RuntimeError('duplicate cache identities')
    selected = []; extended_tokens = []; missing = []
    for i in starts:
        chain = [str(row['token']) for row in ordered[i-3:i+21]]
        if len({scenes[t] for t in chain}) != 1:
            raise RuntimeError('official complete sequence crosses scene boundary')
        if any(str(samples[a]['next']) != b for a, b in zip(chain, chain[1:])):
            raise RuntimeError('official complete sequence is not contiguous locally')
        key = (scenes[chain[3]], chain[3])
        window = WindowTokens(key[0], tuple(chain[:4]), key[1], tuple(chain[4:16]))
        record = by.get(key)
        if record is None:
            missing.append(list(key))
            # Metadata-only seed; Strong/motion inputs will be built from FOUR
            # real histories, not extra GT fields copied from the official info.
            record = dict(scene_name=key[0], t0_token=key[1],
                          history_tokens=window.history_tokens, future_tokens=window.future_tokens[:6])
        elif (tuple(record['history_tokens'][-4:]) != window.history_tokens
                or tuple(record['future_tokens']) != window.future_tokens[:6]):
            raise RuntimeError('official/cache token identity or order mismatch')
        selected.append((window, record)); extended_tokens.append(chain)
    if not selected: raise RuntimeError('no complete official four-history/twenty-future windows')
    keys = [[w.scene_name, w.t0_token] for w, _ in selected]
    key_fingerprint = stable_json_fingerprint(keys)
    validate_start_manifest(keys)
    audit = dict(population='all', alignment=POPULATION_PROTOCOL, code_commit=CODE_COMMIT,
        official_info_sha256=digest, artifact_revision=ARTIFACT_REVISION,
        requested_parent_windows=len(ordered), eligible_windows=len(selected), selected_windows=len(selected),
        scenes=len({k[0] for k in keys}), selected_keys=keys, selected_key_fingerprint=key_fingerprint,
        sequence_4_plus20_fingerprint=stable_json_fingerprint(extended_tokens),
        selected_sequence_tokens_4_plus20=extended_tokens,
        missing_six_history_cache_keys=missing, cached_windows=len(selected)-len(missing),
        selection='global timestamp order; 24 distinct contiguous keys; NO cache intersection/padding',
        selection_future_frames=20, prediction_future_frames=12,
        paper_table_population_verified=False,
        caveat='public long config names II_World whereas current implementation exports EE_World; paper run manifest unavailable')
    return selected, audit


class AlignedColumnProvider(EvaluationJointColumnProvider):
    """No official trajectory ABI/older-history read, including the early starts."""
    def load_raw_columns(self, source, record, *, include_gt):
        if include_gt: raise RuntimeError('aligned prediction inputs must be GT-future-free')
        from real_motion.prepared import _load_history_semantics_and_observation
        tokens = tuple(record['history_tokens'][-4:])
        future = tuple(record['future_tokens'])
        if len(tokens) != 4 or len(future) != 6: raise RuntimeError('aligned first-block 4+6 input contract')
        hist, observed = _load_history_semantics_and_observation(source, record['scene_name'],
            tokens, self.pcfg.free_label, min(4, self.workers))
        raw = dict(history_occ=hist, history_observed=observed, future_gt_occ=None,
            history_poses=[source.pose(t) for t in tokens], future_poses=[source.pose(t) for t in future])
        if 'features' in record:
            raw['_column_causal_preparation'] = build_fixed_geometry(raw, record, self.pcfg,
                self.strong, min(3, self.workers), self.joint.columns.config)
        return raw

    def prepare_columns(self, source, record, *, include_gt, raw_window=None, outputs=None):
        if include_gt: raise RuntimeError('aligned prediction inputs must be GT-future-free')
        raw = self.load_raw_columns(source, record, include_gt=False) if raw_window is None else raw_window
        if 'features' not in record:
            state = rollout.build_four_history_state(raw['history_occ'], raw['history_poses'], raw['future_poses'],
                self.pcfg, self.strong, self.device)
            rebuilt = {**state['rec'], **record}
            state['rec'] = rebuilt
            evidence = prepare_causal_evidence(raw, self.pcfg, self.strong, min(3, self.workers),
                state=state, column_config=self.joint.columns.config)
            evidence['prepared_state'] = state; raw['_column_causal_preparation'] = evidence
            record = rebuilt
        return super().prepare_columns(source, record, include_gt=False, raw_window=raw, outputs=outputs)


def compatibility_metrics(raw):
    """Same counts, explicit unusual zero exclusion; NEVER change main metrics."""
    sem = np.full_like(raw['sem_inter'], np.nan, dtype=np.float64)
    np.divide(raw['sem_inter'], raw['sem_union'], out=sem, where=raw['sem_union'] > 0)
    occ = np.full(6, np.nan)
    np.divide(raw['occ_inter'], raw['occ_union'], out=occ, where=raw['occ_union'] > 0)
    rows = {}
    for hi, horizon in enumerate(rollout.REPORT_HORIZONS):
        valid = sem[hi][np.isfinite(sem[hi]) & (sem[hi] != 0)]
        # Preserve NumPy scalar round semantics used by the author's code;
        # casting BEFORE round can differ at binary half-cent boundaries.
        rows[str(horizon)] = dict(mIoU=float(round(valid.mean()*100, 2)) if len(valid) else None,
            IoU=float(round(occ[hi]*100, 2)) if np.isfinite(occ[hi]) else None,
            excluded_exact_zero_class_ids=np.flatnonzero(sem[hi] == 0).tolist(),
            absent_class_ids=np.flatnonzero(~np.isfinite(sem[hi])).tolist())
    def average(horizons):
        result = {}
        for metric in ('mIoU', 'IoU'):
            values = [rows[str(h)][metric] for h in horizons if rows[str(h)][metric] is not None]
            result[metric] = float(np.mean(values)) if values else None
        return result
    return dict(protocol=METRIC_PROTOCOL, code_commit=CODE_COMMIT, mask_lidar=False, mask_camera=False,
        per_horizon=rows, average_1s_2s_3s=average(rollout.REPORT_HORIZONS[:3]),
        average_4s_5s_6s=average(rollout.REPORT_HORIZONS[3:]))
