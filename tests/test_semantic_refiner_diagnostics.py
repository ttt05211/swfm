import copy
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import numpy as np
import pytest
import torch
from test_shared_semantic_refiner import fixture
from tools.shared_history_experiments.refine import engine as quality
from tools.shared_history_experiments.refiner_diagnostics import engine, run
from tools.shared_history_experiments.refine.model import PROTOCOL


def test_edit_partition_correct_damage_relabel_ignore_and_binary():
    old = np.array([17, 17, 4, 4, 4, 4, 11, 11, 17])
    new = np.array([4, 4, 17, 17, 11, 11, 4, 11, 4])
    gt = np.array([4, 17, 17, 4, 11, 4, 5, 11, 255])
    result = engine.edit_counts(old, new, gt)
    assert result['changed'] == 7 and result['valid_voxels'] == 8
    assert sum(result[k]['changed'] for k in ('added', 'removed', 'relabeled')) == 7
    assert result['added']['corrected'] == result['added']['damaged'] == 1
    assert result['removed']['corrected'] == result['removed']['damaged'] == 1
    assert result['relabeled']['corrected'] == result['relabeled']['damaged'] == result['relabeled']['wrong_to_wrong'] == 1
    assert result['relabeled']['occupancy_corrected'] == result['relabeled']['occupancy_damaged'] == 0
    masked = engine.edit_counts(old, new, gt, gt == 4)
    assert masked['valid_voxels'] == 3 and masked['changed'] == 3
    with pytest.raises(ValueError): engine.edit_counts(old, new[:-1], gt)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_real_model_fresh_parity_all6_no_gt_cached_strong_no_ego_readonly(monkeypatch, device):
    if device == 'cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()): pytest.skip('CUDA/BF16')
    model, _, p, row = fixture(device); model.eval(); p.strong.max_match_speed_mps = 30.
    state = row[3][0]['state']; velocities = {0: np.array([.2, 0., 0.])}
    causal = SimpleNamespace(current=state['current'], current_pose=state['current_pose'],
        source_count=len(state['current']), velocities=velocities)
    kta, anchors = engine.live_prior(row[0], causal, p.pcfg.frame_dt_s)
    row = ({**row[0], 'kta_displacement_xy_m': torch.from_numpy(kta),
        'anchors_xy_t0_m': torch.from_numpy(anchors)}, *row[1:])
    with torch.no_grad(): model.columns.readout[-1].weight.normal_(std=.01)
    fw = quality.forward_window(model, p, row)
    expected = {n: v.argmax(1).reshape(6, *fw['shape']).byte().cpu().numpy() for n, v in
        (('transport', fw['prior']), ('semantic_refine', fw['logits']))}
    del fw
    background = row[1]['_column_causal_preparation']['prepared_state']['column_backgrounds']
    calls, encoder_calls = [], []
    monkeypatch.setattr(engine.runtime, 'extract_instances_cropped_exact', lambda *a, **k: [])
    monkeypatch.setattr(engine.runtime, 'match_instances', lambda *a, **k: velocities)
    def strong(*args, **kwargs):
        calls.append(1)
        return background, [None]*6
    monkeypatch.setattr(engine.runtime, '_strong_all_horizons', strong)
    monkeypatch.setattr(engine.runtime, 'baseline_clear_flat_indices', lambda *a, **k: [])
    monkeypatch.setattr(engine, 'compose_component_replacements_fast_exact', lambda a, *b, **k: a.copy())
    monkeypatch.setattr(model.ego, 'forward', lambda *a: pytest.fail('GT auxiliary ego inside timing'))
    hook = model.transport.history_encoder.register_forward_hook(lambda *a: encoder_calls.append(1))
    raw = {**row[1], 'future_gt_occ': 'POISON', 'future_annotations': 'POISON',
        '_column_causal_preparation': 'POISON CACHED STRONG'}
    before = copy.deepcopy(model.state_dict())
    timing = engine.measure(model, p, (row[0], raw, row[2], row[3]), expected, repeats=2)
    hook.remove()
    assert timing['six_frame_byte_parity'] and len(timing['seconds']) == 2
    assert len(calls) == len(encoder_calls) == 4  # warm + TWO timed calls + untimed serialized profile
    for k, v in before.items(): torch.testing.assert_close(v, model.state_dict()[k], atol=0, rtol=0)
    failed = engine.measure(model, p, (row[0], raw, row[2], row[3]),
        {**expected, 'semantic_refine': np.full_like(expected['semantic_refine'], 16)}, repeats=2)
    assert not failed['six_frame_byte_parity'] and not failed['seconds']


def test_full_window_integer_resume_and_distinct_dynamic_moving_populations(monkeypatch):
    model, _, p, row = fixture(); source = SimpleNamespace(nusc=None)
    monkeypatch.setattr(engine, 'gt_moving_support_sequence', lambda *a, **k: [])
    monkeypatch.setattr(engine, 'moving_support_masks', lambda _, shape: np.zeros((6, *shape), bool))
    monkeypatch.setattr(engine, 'measure', lambda *a: dict(six_frame_byte_parity=True, seconds=[.1]*3, serialized_profile_seconds={}))
    keys = {(row[0]['scene_name'], row[0]['t0_token'])}
    states = []
    engine.evaluate({'gt': model}, p, source, [[row]], speed_keys=keys, save=lambda s: states.append(copy.deepcopy(s)))
    saved = copy.deepcopy(states[-1]); original = copy.deepcopy(saved)
    resumed, direct = [], []
    engine.evaluate({'gt': model}, p, source, [[row], [row]], speed_keys=keys, saved=saved,
        save=lambda s: resumed.append(copy.deepcopy(s)))
    engine.evaluate({'gt': model}, p, source, [[row], [row]], speed_keys=keys, save=lambda s: direct.append(copy.deepcopy(s)))
    assert resumed[-1] == direct[-1] and saved == original
    for groups in saved['edits'].values():
        assert groups['original_t0_Moving_support']['valid_voxels'] == 0
        assert groups['GT_dynamic_semantics']['valid_voxels'] > 0
    with pytest.raises(RuntimeError): engine.validate_saved({**saved, 'windows': 513})
    bad = copy.deepcopy(saved); bad['edits']['1.0']['full_grid']['changed'] = .5
    with pytest.raises(RuntimeError): engine.validate_saved(bad)


def test_timing_invalid_never_publishes_fps():
    row = dict(six_frame_byte_parity=True, seconds=[.1]*3, serialized_profile_seconds={'fresh_Strong_CLEAR': .02})
    report = engine.speed_summary({'a': row}, 1, 3, True)
    assert report['valid_same_quality_route_timing'] and report['output_frame_fps'] == pytest.approx(60.)
    assert report['window_fps'] == pytest.approx(10.)
    assert not engine.speed_summary({'a': row}, 2, 3, True)['valid_same_quality_route_timing']
    assert engine.speed_summary({'a': {**row, 'six_frame_byte_parity': False}}, 1, 3, True)['output_frame_fps'] is None
    assert engine.speed_summary({'a': row}, 1, 3, False)['output_frame_fps'] is None


def test_cli_snapshot_readonly_epoch2_and_resume_identity(tmp_path, monkeypatch):
    source = tmp_path/'train'; source.mkdir(); checkpoint = source/'epoch_0002.pt'
    contract = dict(modes=['gt'], total_updates=2060, eval_keys=[['DEV', str(i)] for i in range(512)])
    torch.save(dict(protocol=PROTOCOL, epoch=2, batch=0, updates=2060, contract=contract), checkpoint)
    untouched = checkpoint.read_bytes()
    paths = {n: tmp_path/n for n in ('config', 'train-cache', 'dev-cache', 'population-manifest',
        'base-checkpoint', 'train-info', 'dev-info')}
    for p in paths.values(): p.write_text('fixture')
    out = tmp_path/'diagnostic'
    args = [v for n, p in paths.items() for v in ('--'+n, str(p))]
    args += ['--out-dir', str(out), '--resume', str(checkpoint), '--dataroot', str(tmp_path),
        '--ccr-history-cache', str(tmp_path), '--ccr-val-history-cache', str(tmp_path)]
    calls = []
    def fake(event, argv):
        calls.append(argv)
        assert '--evaluate-only' in argv and '--max-updates' in argv and argv[argv.index('--max-updates')+1] == '0'
        assert Path(argv[argv.index('--resume')+1]) == out/'frozen/epoch2.pt'
        (out/'results.json').write_text(json.dumps(dict(status='stopped_at_complete_eval_window')))
        return 0
    monkeypatch.setattr(run.training, 'main', fake)
    prior = run.original_engine.evaluate
    assert run.main(threading.Event(), args) == 0
    assert checkpoint.read_bytes() == untouched and run.original_engine.evaluate is prior
    assert run.main(threading.Event(), args+['--diagnostic-resume']) == 0
    assert len(calls) == 2 and checkpoint.read_bytes() == untouched
    with pytest.raises(RuntimeError, match='settings differ'):
        run.main(threading.Event(), args+['--diagnostic-resume', '--speed-windows', '16'])
    assert not list(out.glob('*.pt'))


def test_report_quality_reproduction_gate(tmp_path):
    metrics = {n: dict(mIoU=40., IoU=52., MovingMicro=28.) for n in ('transport', 'semantic_refine')}
    zero = engine.edit_counts(np.array([17]), np.array([17]), np.array([17]))
    result = dict(status='complete', contract=dict(execution=dict(device_type='cuda')),
        evaluation=dict(windows=512, variants={'gt': metrics},
            edits={h: {'full_grid': zero} for h in ('1.0', '2.0', '3.0')},
            speed={'a': dict(six_frame_byte_parity=True, seconds=[.1]*3, serialized_profile_seconds={})}))
    reference = copy.deepcopy(result)
    run.report(tmp_path, result, 1, 3, reference)
    assert json.loads((tmp_path/'speed.json').read_text())['valid_same_quality_route_timing']
    reference['evaluation']['variants']['gt']['semantic_refine']['mIoU'] += .1
    run.report(tmp_path, result, 1, 3, reference)
    speed = json.loads((tmp_path/'speed.json').read_text())
    assert not speed['valid_same_quality_route_timing'] and speed['output_frame_fps'] is None
