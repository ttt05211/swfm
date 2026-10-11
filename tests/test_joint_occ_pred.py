"""Focused real CPU Pred gradients, causal readouts and checkpoint recovery."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from real_motion.strong_w2det import StrongW2DetConfig
from tools.joint_occ_experiments import run as legacy
from tools.joint_occ_experiments.engine import train_arm as reference_step
from tools.joint_occ_experiments.pred import engine, run
from tools.joint_occ_experiments.performance.probe import legacy_fingerprint
from test_joint_occ_ego import fixture
from test_height_field_screen import assert_nested_equal


def pred_fixture():
    model, optimizer, provider, row = fixture()
    provider.strong = StrongW2DetConfig(free_label=17)
    return model, optimizer, provider, row


def test_grouped_pred_objective_and_adam_match_original_and_link_ego():
    torch.manual_seed(12)
    model, optimizer, provider, row = pred_fixture()
    other, adam = copy.deepcopy((model, optimizer))
    for update in (1, 2):
        rng = legacy.rng_state()
        expected = reference_step(model, optimizer, provider, [row, row], 'pred',
                                  update=update, total_updates=8, probe=True)
        legacy.restore_rng(rng)
        actual = engine.train_arm(other, adam, provider, [row, row], 'pred',
                                  update=update, total_updates=8, probe=True)
        for key in ('CE', 'Lovasz', 'source_geometry', 'existence', 'ego_geometry', 'grad_norm'):
            assert actual[key] == pytest.approx(expected[key], rel=2e-4, abs=2e-5)
        for key, value in actual['gradient_links'].items():
            assert value == pytest.approx(expected['gradient_links'][key], rel=2e-4, abs=2e-5)
        assert actual['gradient_links']['occupancy_to_ego'] > 0
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, other.state_dict()[key], rtol=2e-4, atol=2e-5)
        for key in optimizer.state_dict()['state']:
            for field, value in optimizer.state_dict()['state'][key].items():
                torch.testing.assert_close(value, adam.state_dict()['state'][key][field], rtol=2e-4, atol=2e-5)


@torch.no_grad()
def test_fast_all_six_pred_outputs_ignore_future_gt_pose_background_and_labels():
    model, _, provider, row = pred_fixture()
    model.eval().requires_grad_(False)
    expected, _ = engine.readouts(model, provider, row, verify=True)
    raw = copy.deepcopy(row[1])
    raw['future_poses'] = 'POISON GT FUTURE POSES'
    raw['future_gt_occ'] = 'POISON GT OCCUPANCY'
    raw['_column_causal_preparation']['prepared_state']['column_backgrounds'] = 'POISON GT BACKGROUND'
    raw['_column_causal_preparation']['prepared_state']['world_to_future'] = 'POISON GT PROJECTION'
    history, commands, target = row[2]
    actual, _ = engine.readouts(model, provider, (row[0], raw, (history, commands, target+1000), row[3]))
    assert set(expected) == set(engine.ROUTES)
    for route in expected:
        assert len(expected[route]) == len(actual[route]) == 6
        for a, b in zip(expected[route], actual[route]): np.testing.assert_array_equal(a, b)


def test_full_soft_hard_evaluation_integer_resume_and_stop(monkeypatch):
    model, _, provider, row = pred_fixture()
    monkeypatch.setattr(engine, 'gt_moving_support_sequence', lambda *a, **k: [])
    monkeypatch.setattr(engine, 'moving_support_masks', lambda _, shape: np.ones((6, *shape), bool))
    source = SimpleNamespace(nusc=None)
    states = []
    full = engine.evaluate({'pred': model}, provider, source, [[row], [row]])
    engine.evaluate({'pred': model}, provider, source, [[row]], save=lambda s: states.append(copy.deepcopy(s)))
    state = json.loads(json.dumps(states[-1]))
    saved_before = copy.deepcopy(state)
    resumed = engine.evaluate({'pred': model}, provider, source, [[row], [row]], saved=state)
    assert state == saved_before
    assert full['windows'] == resumed['windows'] == 2
    assert full['fast_six_frame_byte_parity'] and resumed['fast_six_frame_byte_parity']
    assert json.dumps(full['variants'], sort_keys=True) == json.dumps(resumed['variants'], sort_keys=True)
    assert full['trajectory'] == resumed['trajectory']
    import threading
    stopped = threading.Event(); stopped.set()
    with pytest.raises(InterruptedError):
        engine.evaluate({'pred': model}, provider, source, [[row], [row]], saved=state,
                        stop_event=stopped, save=lambda s: states.append(s))
    assert states[-1]['windows'] == 1
    corrupt = copy.deepcopy(state); corrupt['metrics']['soft_add_only']['si'][0][0] = .5
    with pytest.raises(RuntimeError, match='integer counts'):
        engine.evaluate({'pred': model}, provider, source, [], saved=corrupt)


def test_pred_exact_optimizer_rng_recovery_after_complete_update():
    model, optimizer, provider, row = pred_fixture()
    c = dict(modes=['pred'], schedule='three_epochs')
    engine.train_arm(model, optimizer, provider, [row], 'pred', update=1, total_updates=3)
    saved = copy.deepcopy(legacy.checkpoint({'pred': model}, {'pred': optimizer}, c, 0, 1, 1, {}))
    engine.train_arm(model, optimizer, provider, [row], 'pred', update=2, total_updates=3)
    expected, adam = copy.deepcopy((model.state_dict(), optimizer.state_dict()))
    assert legacy.restore_checkpoint(saved, {'pred': model}, {'pred': optimizer}, c)[:3] == (0, 1, 1)
    engine.train_arm(model, optimizer, provider, [row], 'pred', update=2, total_updates=3)
    assert_nested_equal(expected, model.state_dict()); assert_nested_equal(adam, optimizer.state_dict())


def cli_args(tmp_path):
    args = []
    for name in ('config', 'train-cache', 'dev-cache', 'population-manifest', 'base-checkpoint',
                 'dataroot', 'train-info', 'dev-info', 'ccr-history-cache', 'ccr-val-history-cache'):
        args += ['--'+name, str(tmp_path/name)]
    return args+['--out-dir', str(tmp_path/'pred')]


def test_wrapper_defaults_hash_restore_and_preserves_original_gt(monkeypatch, tmp_path):
    root = Path(run.__file__).resolve().parents[3]
    before = legacy_fingerprint(root)
    functions = legacy.parser, legacy.train_arm, legacy.evaluate, legacy.summarize, legacy.stable_json_fingerprint
    def fake_main(event, argv):
        a = legacy.parser().parse_args(argv)
        assert (a.mode, a.epochs, a.train_fraction, a.window_batch_size, a.source_budget) == ('pred', 3, .2, 4, 128)
        assert legacy.train_arm is engine.train_arm and legacy.evaluate is engine.evaluate
        identity = legacy.stable_json_fingerprint({'tools/joint_occ_experiments/run.py': 'source'})
        assert identity != functions[-1]({'tools/joint_occ_experiments/run.py': 'source'})
        result = dict(status='complete', epoch=3, updates=3088, contract=dict(modes=['pred']))
        assert 'GT future pose is supervision ONLY' in legacy.summarize(result)
        return 0
    monkeypatch.setattr(legacy, 'main', fake_main)
    assert run.main(argv=cli_args(tmp_path)) == 0
    assert (legacy.parser, legacy.train_arm, legacy.evaluate, legacy.summarize, legacy.stable_json_fingerprint) == functions
    assert legacy_fingerprint(root) == before
    with pytest.raises(ValueError, match='keep INTERNAL'):
        run.main(argv=cli_args(tmp_path)+['--train-fraction', '1'])
    with pytest.raises(ValueError, match='no GT checkpoint import'):
        run.main(argv=cli_args(tmp_path)+['--resume', str(tmp_path/'gt'/'last.pt')])
