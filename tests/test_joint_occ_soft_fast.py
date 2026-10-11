"""Real small soft rendering; CPU equality is NOT a claim of L40S speed."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.joint_occ_experiments.engine import forward_window
from tools.joint_occ_experiments.fast_autograd import compose_sparse
from tools.joint_occ_experiments.soft_inference import engine
from tools.joint_occ_experiments.soft_inference.fast import InferenceScene, compose_inference, predict
from test_joint_occ_ego import fixture


@pytest.mark.parametrize('group', (1, 2, 3))
@pytest.mark.parametrize('variant', ('transport', 'add_only', 'add_remove'))
@torch.no_grad()
def test_fast_probabilities_and_six_semantics_equal_reference(group, variant):
    model, _, provider, row = fixture()
    model.eval().requires_grad_(False)
    fw = forward_window(model, provider, row, 'gt', soft=True)
    # Distinct fractional coordinates, border exits and overlapping source
    # classes exercise interpolation/ordered overlays, not just identity warp.
    fw['poses'] = fw['poses'].clone()
    fw['poses'][:, 0, 3] = torch.tensor([0, .125, .5, .999, 3, -2])
    fw['poses'][:, 1, 3] = torch.tensor([.1, -.2, .5, 0, -.75, .5])
    fw['targets'] = torch.cat((fw['targets'], fw['targets']+.2, fw['targets']-.1))
    fw['yaws'] = torch.cat((fw['yaws'], fw['yaws']+.07, fw['yaws']-.03))
    current = fw['prep'].state['current'][0]
    fw['prep'].state['current'] += [dict(current, class_id=6), dict(current, class_id=4)]
    cloud = fw['prep'].state['source_world_points'][0]
    fw['prep'].state['source_world_points'] += [cloud.copy(), cloud.copy()]
    old = engine.GroupedScene(fw['prep'], provider.pcfg.grid, provider.device)
    fast = InferenceScene(fw, provider.pcfg.grid, provider.device, remove=variant == 'add_remove')
    actions = fw['actions']
    if variant == 'add_only': actions = torch.stack((actions[..., 0], torch.zeros_like(actions[..., 1])), -1)
    expected = []
    for h in range(6):
        p = old.render_group([h], poses=fw['poses'], targets=fw['targets'], yaws=fw['yaws'],
            mapped=fw['mapped'], evidence=fw['evidence'], plan=fw['plan'], actions=actions,
            edits=variant != 'transport')
        expected.append(p[0])
    for start in range(0, 6, group):
        hs = list(range(start, min(start+group, 6)))
        result = fast.render_group(hs, edits=variant != 'transport')
        for i, h in enumerate(hs):
            torch.testing.assert_close(result[i], expected[h], atol=2e-7, rtol=2e-6)
            assert torch.equal(result[i].argmax(0), expected[h].argmax(0))
    actual = predict(provider, fw, variant, range(6), group_size=group)
    for h, volume in enumerate(actual):
        np.testing.assert_array_equal(volume, expected[h].argmax(0).reshape(fast.shape).byte().numpy())
    assert all(p.grad is None for p in model.parameters())


@torch.no_grad()
def test_inference_composition_exact_order_with_overlaps_and_zero_edits():
    background = torch.arange(18*8).reshape(18, 8).float()/1000
    operations = [('overlay', torch.tensor([0, 3, 6]), 4, None),
                  ('overlay', torch.tensor([3, 4]), 6, None),
                  ('remove', torch.tensor([3, 6]), 6, torch.tensor([11, 17])),
                  ('overlay', torch.tensor([0, 4]), 13, None)]
    alphas = [torch.tensor([.2, .7, 1.]), torch.tensor([.5, .1]),
              torch.tensor([.3, .9]), torch.zeros(2)]
    expected = compose_sparse(background, operations, alphas)
    original = background.clone()
    scratch = background.clone()
    actual = compose_inference(scratch, [(*op, a) for op, a in zip(operations, alphas)])
    assert torch.equal(actual, expected)
    assert actual.data_ptr() == scratch.data_ptr()
    assert torch.equal(background, original)


def test_fast_rejects_training_and_builds_legal_indices_only_once():
    _, _, provider, row = fixture()
    with pytest.raises(RuntimeError, match='no_grad'): compose_inference(torch.zeros(18, 3), [])
    model, _, provider, row = fixture()
    with torch.no_grad():
        fw = forward_window(model, provider, row, 'gt')
        scene = InferenceScene(fw, provider.pcfg.grid, provider.device, remove=False)
        assert not scene.groups['static_remove'] and not scene.groups['dynamic_remove']
        identities = [id(r) for _, rows in scene.groups['add'] for r in rows]
        fw['plan'].legal[:] = False  # selects were already prepared this forward
        scene.render_group([0, 1, 2]); scene.render_group([3, 4, 5])
        assert identities == [id(r) for _, rows in scene.groups['add'] for r in rows]


@torch.no_grad()
def test_fast_empty_sources_and_future_labels_are_irrelevant():
    model, _, provider, row = fixture()
    fw = forward_window(model.eval(), provider, row, 'gt')
    fw['prep'].state['current'] = []
    fw['prep'].state['source_world_points'] = []
    fw['targets'], fw['yaws'] = fw['targets'][:0], fw['yaws'][:0]
    evidence = copy.copy(fw['evidence']); evidence.actor = np.full_like(evidence.actor, -1)
    fw['evidence'] = evidence
    expected = engine.soft_predictions(provider, fw, ('add_only', 'add_remove'), range(6))
    fw['prep'].raw['future_gt_occ'] = 'POISON_LABELS'
    for variant in expected:
        for a, b in zip(predict(provider, fw, variant, range(6)), expected[variant]):
            np.testing.assert_array_equal(a, b)


def test_comparison_times_all_five_routes_and_failure_never_promoted(monkeypatch):
    _, _, provider, row = fixture()
    volume = np.zeros((2, 2, 2), np.uint8)
    monkeypatch.setattr(engine, 'quality_predictions', lambda *a, **kw: {r: [volume]*6 for r in engine.ROUTES})
    monkeypatch.setattr(engine, 'prepare_speed_row', lambda *a: (row, .01))
    order = []
    def infer(*args, **kw): order.append(args[3]); return [volume.copy()]*6
    monkeypatch.setattr(engine, 'infer', infer)
    timing = engine.measure_window(None, provider, row, 0, compare_optimized=True)
    routes = list(engine.OPTIMIZED_SPEED_ROUTES)
    assert order == routes+routes+list(reversed(routes))+routes
    assert len(timing['samples']) == 10 and timing['six_frame_byte_parity']
    for sample in timing['samples']:
        sample['seconds'] = .1 if sample['route'].endswith('_fast') else .2
        sample['peak_reserved_mib'], sample['device_total_mib'] = 100., 1000.
    report = engine.speed_summary([timing], actual_cuda=True, routes=routes)
    assert report['optimization_comparison']['add_only']['pass_gate']
    timing['mismatched_voxels']['soft_add_only_fast'] = 1; timing['six_frame_byte_parity'] = False
    report = engine.speed_summary([timing], actual_cuda=True, routes=routes)
    assert not report['optimization_comparison']['add_only']['pass_gate']
    assert not report['optimization_comparison']['add_remove']['pass_gate']
    assert report['automatic_promotion'] is False


def test_live_prior_preserves_exact_legacy_upgrade_not_tolerance():
    history = SimpleNamespace(current=[dict(centroid_world=np.array([21.123456789, -7.123456789, .2]))],
        current_pose=np.eye(4), source_count=1, velocities={0: np.array([.3, .7, 0.])})
    kta, anchors = engine.live_prior({}, history, .5)
    features = torch.zeros(1, 20)
    features[0, :2] = torch.tensor([21.123456789/40, -7.123456789/40])
    source = features[:, :2].numpy().astype(np.float64)*40
    legacy = (anchors.astype(np.float64)-source[:, None]).astype(np.float32)
    assert not np.array_equal(kta, legacy)
    record = dict(features=features, anchors_xy_t0_m=torch.from_numpy(anchors.copy()),
                  kta_displacement_xy_m=torch.from_numpy(legacy.copy()))
    assert engine.recognize_prior_encoding(record, history, .5) == 'legacy_upgrade'
    rebuilt, rebuilt_anchors = engine.live_prior(record, history, .5)
    np.testing.assert_array_equal(rebuilt, legacy)
    np.testing.assert_array_equal(rebuilt_anchors, anchors)
    record['kta_displacement_xy_m'][0, 0, 0] += 1e-6
    history.kta_encoding = 'physical'
    with pytest.raises(RuntimeError, match='BOTH exact'):
        engine.recognize_prior_encoding(record, history, .5)
