"""Exact deployed threshold semantics, immutable calibration and safe recovery."""
import copy
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from real_motion.native_column_cpu import backend_name, prepare_native
from tools.real_motion import causal_column_common as col
from tools.real_motion import joint_threshold_sweep as scan
from tools.real_motion.static_evidence_selector_common import finite_json
from test_causal_columns import plan_fixture, scene_fixture, moving_fixture
from test_joint_checkpoint_selection import _eval_fixture
from test_joint_causal_columns_full import full_cli_fixture

if backend_name() == 'native': prepare_native()


def probabilities(plan, seed):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.full(3, .15), size=plan.base.shape).astype(np.float32)
    p *= plan.legal
    p[..., col.KEEP] += 1e-7
    p /= p.sum(-1, keepdims=True)
    return p


@pytest.mark.parametrize('seed', range(8))
@pytest.mark.parametrize('fixture_name', ('overlap', 'scene', 'empty'))
def test_all_84_factorized_proposals_equal_original_compositor(seed, fixture_name):
    if fixture_name == 'overlap': plan = plan_fixture()
    else:
        prep, grid, cfg = scene_fixture(); plan = col.candidate_plan(prep, 5, grid, cfg)
        if fixture_name == 'empty': plan = plan.subset(np.empty(0, np.int64))
    p = probabilities(plan, seed)
    actual = list(scan.proposals(plan, p, verify=True))
    assert len(actual) == 84 and len({r[0] for r in actual}) == 84
    for (name, layout, after, _), (expected_name, gates, kind) in zip(actual, scan.specifications()):
        assert name == expected_name
        actions = col.actions_from_probabilities(plan, p, gates)
        expected = col.compose_sparse(plan, actions, layout=layout,
            enable_generation=kind != 'refine', enable_refine=kind != 'generation')[2]
        assert np.array_equal(after, expected)


def test_keep_wins_ties_and_refinement_precedes_generation_and_remove_restores_owner():
    plan = plan_fixture(); p = np.zeros((*plan.base.shape, 3), np.float32)
    p[..., col.KEEP] = 1
    p[:, 1] = (.5, .5, 0)  # threshold alone must not beat KEEP
    assert all(np.array_equal(after, layout[2]) for _, layout, after, _ in scan.proposals(plan, p))
    p[:, 1] = (.1, .9, 0); p[2, 0] = (.1, 0, .9)
    rows = {n: after for n, _, after, _ in scan.proposals(plan, p, verify=True)}
    assert rows['g0'].tolist() == [4, 11]
    assert rows['j0'].tolist() == [5, 4]  # restore class5, source1 ADD beats GEN
    assert rows['j3'].tolist() == [4, 4]  # REMOVE off really disables it
    assert rows['j63'].tolist() == [4, 17]
    p[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='probabilities'): list(scan.proposals(plan, p))


@pytest.mark.parametrize('gates', ((.5, .5, None), (.5, .5, .5), (.75, .5, .95), (None, .95, None), (None, None, None)))
def test_one_probability_pass_matches_original_metrics_quality_and_scene_deltas(gates):
    prep, grid, joint, records, source, owner = _eval_fixture(); records = records[:2]
    with torch.no_grad(): joint.columns.generation.bias.fill_(1.5)
    with patch.object(col, 'gt_moving_support_sequence', return_value=moving_fixture(prep)):
        expected = col.evaluate_columns(owner(joint), source, records, joint.columns, gates,
            batch_size=7, diagnostic_thresholds=None)['all']
        # Deliberately collide every cache hash: exact voxel equality, NOT a
        # probabilistic hash-only shortcut, must still protect all counts.
        with patch.object(scan, 'hash', return_value=0, create=True):
            report, perf = scan.sweep(owner(joint), source, records, joint.columns, batch_size=7)
    name = next(n for n, g, kind in scan.specifications() if kind == 'joint' and g == gates)
    actual = next(r for r in report['candidates'] if r['name'] == name)
    assert finite_json(actual['metrics']) == finite_json(expected['variants']['joint']['metrics'])
    assert finite_json(actual['quality']) == finite_json(expected['variants']['joint']['quality'])
    assert finite_json(report['baseline']) == finite_json(expected['baseline'])
    assert finite_json(report['reference_metrics']) == finite_json(expected['reference_metrics'])
    assert perf['probability_horizons'] == 6 and perf['composition_exactness_horizons'] == 3
    assert perf['metric_reuse_hits'] > 0
    assert perf['metric_reuse_hits']+perf['metric_distinct_predictions'] == 6*84


@pytest.mark.parametrize('stop_after_first', (False, True))
def test_interrupt_resume_and_finalize_without_replaying_completed_windows(stop_after_first):
    prep, grid, joint, records, source, owner = _eval_fixture(); records = records[:2]
    saved = {}; stop = Event()
    def save(wi, state, totals): saved.update(cursor=wi, state=json.loads(json.dumps(state)))
    def progress(row):
        if row['window'] == 1: stop.set()
    with patch.object(col, 'gt_moving_support_sequence', return_value=moving_fixture(prep)):
        expected, _ = scan.sweep(owner(joint), source, records, joint.columns)
        if not stop_after_first: stop.set()
        with pytest.raises(InterruptedError):
            scan.sweep(owner(joint), source, records, joint.columns, stop_event=stop,
                save_state=save, progress=progress, checkpoint_every=8)
        assert saved['cursor'] == int(stop_after_first)
        resumed, _ = scan.sweep(owner(joint), source, records, joint.columns,
            start_window=saved['cursor'], saved_state=saved['state'], save_state=save)
        final, perf = scan.sweep(owner(joint), source, records, joint.columns,
            start_window=saved['cursor'], saved_state=saved['state'])
    assert finite_json(resumed) == finite_json(expected) == finite_json(final)
    assert perf['probability_horizons'] == 0
    with pytest.raises(RuntimeError, match='cursor'):
        scan.sweep(owner(joint), source, records, joint.columns, saved_state=saved['state'])


def fake_metrics(miou=40., moving=30.):
    values = dict(mIoU=miou, IoU=50., MovingMacro=moving, MovingMicro=moving)
    return {**values, 'per_horizon': {str(h): {**values, 'semantic_per_class': {}, 'moving_per_class': {}}
                                   for h in (1., 2., 3.)}}


@pytest.mark.parametrize('gain', (True, False))
def test_guard_does_not_trade_moving_or_a_horizon_for_miou_and_keeps_fixed_on_ties(gain):
    state = scan.initial_state(); state['counts_windows']['all'] = 512
    reports = dict(baseline=fake_metrics(), variants={n: dict(metrics=fake_metrics(), quality={})
                                                    for n, _, _ in scan.specifications()})
    reports['variants']['j0']['metrics'] = fake_metrics(42., 29.)
    reports['variants']['j1']['metrics'] = fake_metrics(41., 30.2)
    reports['variants']['j1']['metrics']['per_horizon']['1.0']['MovingMicro'] = 29.9
    if gain: reports['variants']['j2']['metrics'] = fake_metrics(40.25, 30.1)
    with patch.object(col, 'report_states', return_value=reports): result = scan.summarize(state)
    assert result['overall_mIoU_best']['name'] == 'j0'
    assert result['selected']['name'] == ('j2' if gain else 'j3')
    assert result['selected']['eligible']
    assert not next(r for r in result['candidates'] if r['name'] == 'j1')['eligible']
    assert 'NOT independent' in result['note']


def test_real_cli_immutable_checkpoint_population_tamper_and_resume(tmp_path, full_cli_fixture):
    from tools.real_motion import calibrate_p0_f9_joint_thresholds as cli
    from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import sha256
    run, files, _ = full_cli_fixture
    prep, grid, dev, manifest, keys, factory = run.eval_data
    trained = tmp_path/'full1_history4'/'model'
    assert run(trained, 1, history_frames=4) in (None, 0)
    originals = {p: p.read_bytes() for p in trained.iterdir() if p.is_file()}
    checkpoint = trained/'epoch_0001.pt'
    def evaluate(destination, resume=False, event=None, keys_override=None):
        argv = ['calibrate', '--config', str(Path(__file__).resolve().parents[1]/'configs/real_motion_occfm.yaml'),
            '--checkpoint', str(checkpoint), '--dev-cache', str(files['dev-cache']),
            '--population-manifest', str(files['population-manifest']), '--base-checkpoint', str(files['base-checkpoint']),
            '--dev-info', str(files['dev-info']), '--dataroot', str(tmp_path), '--out-dir', str(destination),
            '--device', 'cpu', '--cpu-workers', '1']
        if resume: argv.append('--resume')
        # Reduce only the number of toy evaluation records, not production's
        # frozen512 checks, actual rendering/probabilities, or checkpoint load.
        with patch('sys.argv', argv), patch.object(cli, 'CALIBRATION_WINDOWS', 2), \
                patch.object(cli, 'CLEAN_SHA256', 'a'*64), \
                patch.object(cli, 'make_prepare_config', return_value=SimpleNamespace(grid=grid)), \
                patch.object(cli, 'load_manifest', return_value=(json.loads(json.dumps(manifest)), keys, None)), \
                patch.object(cli, 'load_cache', return_value=({}, list(reversed(dev)))), \
                patch.object(cli, 'evaluation_keys', return_value=tuple(keys[:2]) if keys_override is None else keys_override), \
                patch.object(cli, 'sha256', side_effect=lambda p: 'a'*64 if Path(p) == files['base-checkpoint'] else sha256(p)), \
                patch.object(cli, 'EvaluationJointColumnProvider', side_effect=factory), \
                patch.object(cli, 'NuScenesWindowSource', return_value=SimpleNamespace(nusc=None)), \
                patch.object(col, 'gt_moving_support_sequence', return_value=moving_fixture(prep)):
            return cli.main(event)
    uninterrupted = tmp_path/'calibration'; assert evaluate(uninterrupted) == 0
    cancelled = tmp_path/'cancelled'; stop = Event(); stop.set()
    assert evaluate(cancelled, event=stop) == 130
    assert not (cancelled/'calibration.json').exists() and not (cancelled/'summary.txt').exists()
    state_path = cancelled/'threshold_state.json'; original = state_path.read_bytes()
    broken = json.loads(original); broken['completed_windows'] = 1; state_path.write_text(json.dumps(broken))
    with pytest.raises(RuntimeError, match='resume provenance/state'): evaluate(cancelled, resume=True)
    state_path.write_bytes(original)
    assert evaluate(cancelled, resume=True) == 0
    a = json.loads((uninterrupted/'calibration.json').read_text()); b = json.loads((cancelled/'calibration.json').read_text())
    assert a['report'] == b['report'] and a['execution'] == b['execution']
    assert a['report']['windows'] == 2 and a['execution']['fixed_thresholds'] == [.5, .5, None]
    assert a['execution']['population'] == 'dev512_ALREADY_used_for_checkpoint_selection'
    assert all(p.read_bytes() == value for p, value in originals.items())
    assert (uninterrupted/'checkpoint_snapshot.pt').read_bytes() == checkpoint.read_bytes()
    with pytest.raises(SystemExit): evaluate(cancelled, resume=True)
    with pytest.raises(RuntimeError, match='512 unique'): evaluate(tmp_path/'bad_duplicates', keys_override=(keys[0], keys[0]))
