"""Causal provenance, source lifecycle and consistent motion-state rebuilding."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from real_motion.causal_rollout_handoff import PredictedSourceHandoff, handoff_from_prepared
from real_motion.geometry import OccupancyGrid
from real_motion.motion_transport import FEATURE_NAMES
from real_motion.prepared import PrepareConfig
from real_motion.strong_w2det import StrongW2DetConfig
from tools.real_motion import joint_long_rollout_common as common
from tools.real_motion import eval_p0_f9_joint_zero_shot_long_rollout as cli


def fixture():
    grid = OccupancyGrid(0, 0, 0, (.4, .4, .4), (24, 24, 4))
    history = np.full((4, *grid.shape_hwd), 17, np.uint8)
    history[:, :, :, 0] = 11
    history[:, 8:12, 8:12, 1:3] = 4
    # Only the final shape is extended. Physical source trajectory is stationary.
    history[-1, 12:14, 8:12, 1:3] = 4
    pcfg = PrepareConfig(grid=grid)
    state = common.build_four_history_state(history, [np.eye(4)]*4, [np.eye(4)]*6,
        pcfg, StrongW2DetConfig(), torch.device('cpu'))
    owners = np.full(grid.shape_hwd, -1, np.int32); owners[8:12, 8:12, 1:3] = 0
    centers = np.tile(np.array([4., 4., .8]), (1, 4, 1))
    handoff = PredictedSourceHandoff(owners, np.array([4]), centers)
    return history, pcfg, state, handoff


def test_asymmetric_refinement_does_not_become_velocity_and_all_inputs_rebuild():
    history, pcfg, original, handoff = fixture()
    assert original['velocities'][0][0] == pytest.approx(.8)
    rebuilt = common.build_four_history_state(history, [np.eye(4)]*4, [np.eye(4)]*6,
        pcfg, StrongW2DetConfig(), torch.device('cpu'), motion_handoff=handoff,
        component_frames=original['components_by_frame'])
    assert np.allclose(rebuilt['velocities'][0], 0)
    rec = rebuilt['rec']
    assert torch.count_nonzero(rec['kta_displacement_xy_m']) == 0
    for key in ('current_vx_norm', 'current_vy_norm', 'current_speed_norm'):
        # Feature names are frozen; verify every active historical segment too.
        names = [n for n in FEATURE_NAMES if n == key or n.startswith('hist_vel_')]
        for name in names: assert rec['features'][0, FEATURE_NAMES.index(name)] == 0
    assert rec['frame_motion_features'].isfinite().all()
    assert not rec['target_source_mask_tube'][:, :2].any()
    assert rebuilt['motion_handoff_audit']['matched_sources'] == 1
    assert rebuilt['current'] is original['current']  # shape/order remain authoritative
    assert np.array_equal(rebuilt['current_sem'], history[-1])
    assert not any(k in rec for k in ('existence', 'target_yaw_rad', 'supervised_source'))
    # No mutation of the reference state/cache: matched/displacement arrays copied.
    assert original['velocities'][0][0] == pytest.approx(.8)
    assert torch.count_nonzero(original['rec']['kta_displacement_xy_m']) > 0


def test_predicted_accelerating_trajectory_and_current_shape_origin_are_kept():
    _, _, state, handoff = fixture()
    centers = handoff.centers_world.copy(); centers[0, :, 0] = [1., 1.2, 1.7, 2.5]
    carry = PredictedSourceHandoff(handoff.owners, handoff.class_ids, centers)
    tracks = np.zeros((1, 6, 3)); valid = np.zeros((1, 6), bool)
    velocity, actual, mask, audit = carry.reconcile(state['current'], {}, tracks, valid)
    assert velocity[0][0] == pytest.approx(1.6)
    assert np.allclose(np.diff(actual[0, -4:, 0]), [.2, .5, .8])
    assert np.array_equal(actual[0, -1], state['current'][0]['centroid_world'])
    assert mask[0].tolist() == [False, False, True, True, True, True]
    assert audit['source_identity_pairs'] == [[0, 0]]
    assert not tracks.any() and not valid.any()


@pytest.mark.parametrize('kind', ['split', 'merge', 'unmatched', 'wrong_class', 'excess_speed', 'excess_offset'])
def test_ambiguous_or_unreliable_sources_fall_back_exactly(kind):
    _, _, state, handoff = fixture()
    current = [dict(state['current'][0])]; owners = handoff.owners.copy()
    classes = handoff.class_ids.copy(); centers = handoff.centers_world.copy()
    if kind == 'split':
        current = [dict(current[0], voxel_indices=current[0]['voxel_indices'][:16]),
                   dict(current[0], voxel_indices=current[0]['voxel_indices'][16:])]
    elif kind == 'merge':
        owners[10:12, 8:12, 1:3] = 1; classes = np.array([4, 4]); centers = np.repeat(centers, 2, axis=0)
    elif kind == 'unmatched': owners[:] = -1
    elif kind == 'wrong_class': classes[:] = 7
    elif kind == 'excess_speed': centers[0, -1, 0] += 30.
    elif kind == 'excess_offset': centers[:, :, 0] += 10.
    carry = PredictedSourceHandoff(owners, classes, centers)
    tracks = np.full((len(current), 6, 3), 2.); valid = np.ones((len(current), 6), bool)
    velocities = {i: np.array([.8, .1, 0]) for i in range(len(current))}
    actual, hist, mask, audit = carry.reconcile(current, velocities, tracks, valid)
    assert audit['matched_sources'] == 0 and audit['memory_only_sources_added'] == 0
    for i in actual: assert np.array_equal(actual[i], velocities[i])
    assert np.array_equal(hist, tracks) and np.array_equal(mask, valid)


def test_export_filters_changed_semantics_and_never_uses_labels():
    _, _, state, handoff = fixture()
    pred = np.full(handoff.owners.shape, 17, np.uint8); pred[handoff.owners >= 0] = 4
    pred[8, 8, 1] = 7
    class Poison:
        def __getattribute__(self, name): raise AssertionError('GT/raw access is forbidden')
    prep = SimpleNamespace(state={'current': state['current']}, owners=[handoff.owners]*4,
        targets=[list(handoff.centers_world[:, f].copy()) for f in range(4)], raw=Poison(), outputs=Poison())
    copied = handoff_from_prepared(prep, pred)
    assert copied.owners[8, 8, 1] == -1
    assert handoff.owners[8, 8, 1] == 0
    assert np.array_equal(copied.centers_world[:, :, :2], handoff.centers_world[:, :, :2])
    assert np.allclose(copied.centers_world[:, :, 2], handoff.centers_world[:, :, 2])
    assert not copied.owners.flags.writeable
    # Intermediate target Z can vary under ego pitch; rendering keeps world Z.
    prep.targets[0][0][2] += 1.
    stable_z = handoff_from_prepared(prep, pred)
    assert np.allclose(stable_z.centers_world[:, :, 2], state['current'][0]['centroid_world'][2])
    assert prep.targets[0][0][2] == pytest.approx(1.8)  # exporter never mutates caller


def test_missing_predicted_visibility_is_not_fabricated_as_history():
    _, _, state, handoff = fixture()
    visible = np.array([[False, True, True, True]])
    carry = PredictedSourceHandoff(handoff.owners, handoff.class_ids, handoff.centers_world, history_visible=visible)
    tracks = np.ones((1, 6, 3)); valid = np.ones((1, 6), bool)
    _, actual, mask, audit = carry.reconcile(state['current'], state['velocities'], tracks, valid)
    assert mask[0].tolist() == [False, False, False, True, True, True]
    assert not actual[0, :3].any() and audit['matched_sources'] == 1
    visible[0, -2] = False
    carry = PredictedSourceHandoff(handoff.owners, handoff.class_ids, handoff.centers_world, history_visible=visible)
    velocity, _, _, audit = carry.reconcile(state['current'], state['velocities'], tracks, valid)
    assert audit['rejected_visibility_sources'] == 1
    assert np.array_equal(velocity[0], state['velocities'][0])


def test_empty_population_and_invalid_inputs():
    empty = PredictedSourceHandoff(np.full((2, 2, 2), -1), np.array([], dtype=int), np.empty((0, 4, 3)))
    result = empty.reconcile([], {}, np.empty((0, 6, 3)), np.empty((0, 6), bool))
    assert result[-1]['matched_sources'] == 0
    _, _, _, good = fixture()
    with pytest.raises(ValueError, match='owner index'):
        PredictedSourceHandoff(np.full(good.owners.shape, 1), good.class_ids, good.centers_world)
    bad = good.centers_world.copy(); bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        PredictedSourceHandoff(good.owners, good.class_ids, bad)
    with pytest.raises(ValueError, match='cadence'):
        PredictedSourceHandoff(good.owners, good.class_ids, good.centers_world, dt_s=0)


def test_modes_require_fixed_original_route_and_reject_duplicates():
    assert cli.parse_handoff_modes('redetect,reconciled,transport_history') == cli.HANDOFF_MODES
    for value in ('reconciled', 'redetect,oracle', 'redetect,redetect'):
        with pytest.raises(cli.argparse.ArgumentTypeError): cli.parse_handoff_modes(value)


def test_matched_clean_reference_only_loads_two_older_inputs(monkeypatch):
    sem = np.full((2, 2, 2), 17, np.uint8); poses = [np.eye(4)]*12
    calls = []; loads = []
    tokens = ['old0', 'old1', 'h0', 'h1', 'h2', 't0']
    samples = {token: dict(scene_token='s', timestamp=i*500000,
        next=tokens[i+1] if i < 5 else 'future') for i, token in enumerate(tokens)}
    class Source:
        nusc = SimpleNamespace(get=lambda table, token: samples[token])
        def load_semantics(self, scene, token):
            assert token in ('old0', 'old1'); loads.append(token); return sem
        def pose(self, token): assert token in ('old0', 'old1'); return np.eye(4)
    def build(history, hp, fp, *args):
        assert len(history) == len(hp) == len(fp) == 6; calls.append(history)
        return {'history': history}
    monkeypatch.setattr(common.legacy, '_build_block_state', build)
    monkeypatch.setattr(cli.runtime, '_stage_gpu_inputs', lambda *a: None)
    monkeypatch.setattr(cli.runtime, '_release_gpu_inputs', lambda *a: None)
    pred1 = [sem.copy() for _ in range(6)]
    monkeypatch.setattr(cli.runtime, '_forecast_once', lambda *a: pred1)
    provider = SimpleNamespace(pcfg=None, strong=None, device='cpu', reference=None, clean_rollout_checked=True)
    raw = dict(history_occ=[sem]*4, history_poses=[np.eye(4)]*4, future_gt_occ=object())
    actual = cli.forecast_clean_reference(provider, Source(),
        dict(scene_name='s', history_tokens=['old0', 'old1', 'h0', 'h1', 'h2', 't0']), raw, poses)
    assert len(actual) == 12 and loads == ['old0', 'old1']
    assert calls[1] is pred1  # E14 uses its OWN predictions, not Joint or GT.


def test_actual_clean_reference_matches_cached_native_six_renderer():
    from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
    from real_motion.local_st_world_model_v18_se2 import LocalSpatialTemporalWorldModelV18SE2
    history4, pcfg, _, _ = fixture()
    history = [history4[0].copy(), history4[0].copy(), *list(history4)]
    tokens = ['old0', 'old1', 'h0', 'h1', 'h2', 't0']
    samples = {token: dict(scene_token='s', timestamp=i*500000,
        next=tokens[i+1] if i < 5 else 'future') for i, token in enumerate(tokens)}
    class Source:
        nusc = SimpleNamespace(get=lambda table, token: samples[token])
        def pose(self, token): assert token in tokens[:2]; return np.eye(4)
        def load_semantics(self, scene, token):
            assert token in tokens[:2]; return history[tokens.index(token)]
    state = common.legacy._build_block_state(history, [np.eye(4)]*6, [np.eye(4)]*6,
        pcfg, StrongW2DetConfig(), torch.device('cpu'))
    rec = state['rec']; rec.update(scene_name='s', t0_token='t0', history_tokens=tokens,
        future_tokens=[f'f{i}' for i in range(6)], sample_id='native6-reference')
    model = LocalSpatialTemporalWorldModelV18SE2(LocalSTWMV17Config(d_model=16, semantic_dim=4,
        blocks=1, decoder_blocks=1, history_frames=6)).eval().requires_grad_(False)
    provider = SimpleNamespace(pcfg=pcfg, strong=StrongW2DetConfig(), device=torch.device('cpu'), reference=model)
    raw = dict(history_occ=np.stack(history[-4:]), history_poses=[np.eye(4)]*4)
    actual = cli.forecast_clean_reference(provider, Source(), rec, raw, [np.eye(4)]*12)
    assert len(actual) == 12 and provider.clean_rollout_checked
    assert all(np.isin(x, np.arange(18)).all() for x in actual)
