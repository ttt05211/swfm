"""Matched training, cache immutability, live readout and recovery, small grids."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.unified_source_training import data, train, evaluation, workflow
from tools.joint_occ_experiments import run as legacy
from tools.shared_history_experiments.full_gt import engine as gt
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from test_unified_source_tracks import real_fixture


def training_row(device='cpu'):
    runner, rec, raw = real_fixture(device)
    with torch.no_grad():
        row = runner.prepare(rec, raw, times=np.arange(4)*.5)
        fw = runner.forward(row, poses=np.tile(np.eye(4), (6, 1, 1)))
    row[1]['_column_causal_preparation']['prepared_state']['column_backgrounds'] = fw['state']['column_backgrounds']
    row[1].update(future_poses=np.tile(np.eye(4), (6, 1, 1)),
        future_gt_occ=np.tile(raw['history_occ'][-1], (6, 1, 1, 1)), _track_history_seconds=np.arange(4)*.5)
    return runner, row


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_cached_fit_rebuild_and_one_real_occ_update_uses_new_anchor_without_old_cache_write(device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.set_num_threads(1); runner, row = training_row(device)
    before_kta = copy.deepcopy(row[0]); before_causal = copy.deepcopy(row[1]['_column_causal_preparation'])
    reader = data.history_inputs('unified_fit')(runner.provider, ram_mib=1)
    # Actual legacy compact cache lacks previous/velocity; reconstruct from history.
    for key in ('previous', 'velocities'): row[1]['_column_causal_preparation']['prepared_state'].pop(key, None)
    try:
        history = reader.get(row[0], row[1]); changed = data.publish_row((*row[:3], history), 'unified_fit')
        assert changed[3][0]['registrations'] is changed[1]['_column_causal_preparation']['registrations']
        np.testing.assert_allclose(changed[0]['kta_displacement_xy_m'][0, :, 0], np.arange(1, 7))
        np.testing.assert_array_equal(row[0]['kta_displacement_xy_m'], before_kta['kta_displacement_xy_m'])
        assert all(a is b for a, b in zip(changed[1]['_column_causal_preparation']['prepared_state']['column_backgrounds'],
                                         row[1]['_column_causal_preparation']['prepared_state']['column_backgrounds']))
        for a, b in zip(row[1]['_column_causal_preparation']['registrations'][0], before_causal['registrations'][0]):
            if a is not None: np.testing.assert_array_equal(a[0], b[0])
        assert changed[1]['future_gt_occ'] is row[1]['future_gt_occ']
        # Published KTA must be updated even on a bounded RAM cache hit.
        again = data.publish_row((*row[:3], reader.get(row[0], row[1])), 'unified_fit')
        np.testing.assert_array_equal(again[0]['anchors_xy_t0_m'], changed[0]['anchors_xy_t0_m'])
        baseline = data.live_variant(row, runner.provider, runner.history, np.arange(4)*.5, 'baseline')
        assert baseline is row
        model = runner.body.requires_grad_(True); model.train()
        state = tensor_state_fingerprint(model.state_dict()); adam = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for group in adam.param_groups: group['initial_lr'] = 1e-3
        result = gt.train_arm(model, adam, runner.provider, [changed], 'gt', update=1, total_updates=3, probe=True)
        assert result['gradient_links']['occupancy_to_xy'] > 0 and not result['auxiliary_computed']
        assert state != tensor_state_fingerprint(model.state_dict())
    finally: runner.close()


def test_track_builder_does_not_access_future_truth_fields():
    runner, row = training_row()
    try:
        original = data.track_from_cached(row[1], runner.provider)
        poisoned = dict(row[1], future_poses=object(), future_gt_occ=object(), future_annotations=object())
        actual = data.track_from_cached(poisoned, runner.provider)
        np.testing.assert_array_equal(actual.links, original.links)
        np.testing.assert_array_equal(actual.centers, original.centers)
    finally: runner.close()


def metadata_args(tmp_path):
    paths = {n: tmp_path/(n+'.bin') for n in ('config', 'train-cache', 'dev-cache', 'population-manifest',
        'base-checkpoint', 'train-info', 'dev-info')}
    for p in paths.values(): p.write_text('metadata fixture')
    args = [v for n, p in paths.items() for v in ('--'+n, str(p))]
    return paths, args+['--dataroot', str(tmp_path), '--ccr-history-cache', str(tmp_path),
                       '--ccr-val-history-cache', str(tmp_path), '--stc-root', str(tmp_path)]


def test_workflow_same_one_epoch_smoke_resume_completed_skip_and_initializer_guard(tmp_path, monkeypatch):
    _, args = metadata_args(tmp_path); out = tmp_path/'matched'; args += ['--out-dir', str(out)]
    monkeypatch.setattr(workflow, 'stc_inventory', lambda _: [])
    calls = []
    def launch(command, check):
        calls.append(command); arm = command[command.index('--track-arm')+1]
        directory = Path(command[command.index('--out-dir')+1]); directory.mkdir(exist_ok=True)
        if '--max-updates' in command:
            (directory/'last.pt').write_bytes(b'safe partial cursor')
            result = dict(status='stopped_at_complete_update')
        else:
            if (directory/'last.pt').exists(): assert '--resume' in command
            c = dict(protocol=train.PROTOCOL, epochs=1, total_updates=1030, implementation=arm, model='identical')
            torch.save(dict(contract=c, epoch=1, batch=0, updates=1030, track_arm=arm,
                            random_initializer='same_random'), directory/'last.pt')
            m = dict(mIoU=20., IoU=30., MovingMicro=10.)
            result = dict(status='complete', epoch=1, updates=1030, contract=c,
                evaluation=dict(windows=64, track_arm=arm, variants={'gt': {n: m for n in evaluation.ROUTES}}))
        (directory/'results.json').write_text(json.dumps(result)); return SimpleNamespace(returncode=0)
    assert workflow.main(argv=args+['--max-updates', '8'], launch=launch) == 0 and len(calls) == 2
    assert workflow.main(argv=args+['--resume-workflow'], launch=launch) == 0 and len(calls) == 4
    assert workflow.main(argv=args+['--resume-workflow'], launch=launch) == 0 and len(calls) == 4
    assert json.loads((out/'comparison.json').read_text())['no_automatic_promotion']
    saved = torch.load(out/'unified_fit/last.pt', weights_only=False); saved['random_initializer'] = 'wrong'
    torch.save(saved, out/'unified_fit/last.pt')
    with pytest.raises(RuntimeError, match='initializers'): workflow.comparison(out)
    with pytest.raises(ValueError, match='fixed'): workflow.main(argv=args+['--epochs', '2'], launch=launch)


@pytest.mark.parametrize('arm', data.ARMS)
def test_actual_training_cli_two_updates_resume_original_bindings_restore_and_source_unchanged(tmp_path, monkeypatch, arm):
    runner, row = training_row(); p = runner.provider; torch.set_num_threads(1)
    paths, args = metadata_args(tmp_path)
    torch.save({'model_config': asdict(runner.body.transport.config)}, paths['base-checkpoint'])
    original_source = paths['base-checkpoint'].read_bytes()
    records = [{**row[0], 'scene_name': 'TRAIN', 't0_token': f'train{i}', 'features': torch.zeros(1, 46)} for i in range(20430)]
    dev = [{**row[0], 'scene_name': 'DEV', 't0_token': f'dev{i}', 'features': torch.zeros(1, 46)} for i in range(4369)]
    keys = [('DEV', r['t0_token']) for r in dev[:512]]
    monkeypatch.setattr(train, 'stc_inventory', lambda _: [])
    monkeypatch.setattr(legacy, 'require_cuda', lambda _: torch.device('cpu'))
    monkeypatch.setattr(legacy, 'load_cache', lambda path: ({}, records if path == str(paths['train-cache']) else dev))
    monkeypatch.setattr(legacy, 'load_manifest', lambda _: ({'parent_keys': keys}, keys[:64], None))
    monkeypatch.setattr(legacy, 'load_runtime_config', lambda *a: None)
    monkeypatch.setattr(legacy, 'make_prepare_config', lambda _: p.pcfg)
    def provider(*args): p.joint = args[-2]; return p
    monkeypatch.setattr(legacy, 'PilotProvider', provider)
    monkeypatch.setattr(legacy, 'NuScenesWindowSource', lambda root, *, info_pkl: SimpleNamespace(nusc=None))
    class Disposable:
        kernels = pool = None
        def __init__(self, *a): pass
        def close(self): pass
    monkeypatch.setattr('real_motion.canonical_repair_execution.CanonicalCpuExecution', Disposable)
    monkeypatch.setattr('real_motion.strong_majority_execution.ParallelNativeMajority', Disposable)
    def caches(provider, *args):
        provider.ccr_history_cache = provider.ccr_val_history_cache = SimpleNamespace(stats=lambda: {}, close=lambda: None)
        return {'train': {'namespace': 'train'}, 'val': {'namespace': 'val'}}
    monkeypatch.setattr(legacy, 'attach_caches', caches)
    def groups(records, *a):
        assert len(records) == 4086
        return [[records[0]], [records[1]]]
    monkeypatch.setattr(legacy, 'epoch_groups', groups)
    def prefetch(provider, source, groups, descriptors, **kw):
        for group in groups:
            yield [(rec, row[1], row[2], descriptors.get(rec, row[1])) for rec in group]
    monkeypatch.setattr(legacy, 'prefetched_batches', prefetch)
    body = train.GTBody
    monkeypatch.setattr(train, 'GTBody', lambda cfg: body(cfg, feature_width=16, z_bins=4, memory_width=4, width=8, chunk_size=32))
    monkeypatch.setattr(train, 'evaluator', lambda *a, **kw: lambda *a, **kw: dict(windows=64, track_arm=arm,
        variants={'gt': {n: dict(mIoU=20., IoU=30., MovingMicro=10.) for n in evaluation.ROUTES}}, trajectory={}))
    out = tmp_path/'train'; args += ['--track-arm', arm, '--out-dir', str(out)]
    original_parser, original_ego = legacy.parser, train.old_data.ego_inputs
    try:
        assert train.main(argv=args+['--max-updates', '1']) == 0
        partial = torch.load(out/'last.pt', weights_only=False)
        assert partial['updates'] == 1 and partial['epoch'] == 0 and partial['batch'] == 1
        assert train.main(argv=args+['--resume', str(out/'last.pt')]) == 0
        final = torch.load(out/'last.pt', weights_only=False)
        assert final['updates'] == 2 and final['epoch'] == 1 and final['batch'] == 0
        assert final['track_arm'] == arm and final['random_initializer'] == partial['random_initializer']
        assert (out/'epoch_0001.pt').is_file() and paths['base-checkpoint'].read_bytes() == original_source
        assert legacy.parser is original_parser and train.old_data.ego_inputs is original_ego
        wrong = [data.ARMS[1] if v == 'baseline' else 'baseline' if v == 'unified_fit' else v for v in args]
        with pytest.raises(RuntimeError, match='differs'): train.main(argv=wrong+['--resume', str(out/'last.pt')])
        assert legacy.parser is original_parser and train.old_data.ego_inputs is original_ego
    finally: runner.close()


def test_real_dev_evaluator_uses_trained_convention_on_both_histories_and_complete_window_resume(tmp_path, monkeypatch):
    runner, row = training_row(); p = runner.provider; before = tensor_state_fingerprint(runner.body.state_dict())
    frames = []
    rows = [(dict(row[0], scene_name='VAL'+str(i)), *row[1:]) for i in range(64)]
    for record, *_ in rows:
        for token, sem in zip(row[0]['history_tokens'], row[1]['history_occ']):
            path = tmp_path/record['scene_name']/token/'labels.npz'; path.parent.mkdir(parents=True)
            np.savez_compressed(path, semantics=sem); st = path.stat(); frames.append([str(path.resolve()), st.st_size, st.st_mtime_ns])
    timestamps = {t: i*500000 for i, t in enumerate(row[0]['history_tokens'])}
    source = SimpleNamespace(nusc=SimpleNamespace(get=lambda _, t: dict(timestamp=timestamps[t])))
    monkeypatch.setattr(gt, 'gt_moving_support_sequence', lambda *a, **kw: None)
    monkeypatch.setattr(gt, 'moving_support_masks', lambda *a: [np.ones(p.pcfg.grid.shape_hwd, bool)]*6)
    event = threading.Event(); snapshots = []
    def save(v): snapshots.append(copy.deepcopy(v)); event.set()
    # One row published in finally even before the periodic save interval.
    evaluate = evaluation.evaluator(str(tmp_path), 'unified_fit', inventory=frames)
    try:
        with pytest.raises(InterruptedError): evaluate({'gt': runner.body}, p, source, iter([[r] for r in rows]), stop_event=event, save=save)
        assert snapshots[-1]['windows'] == 8
        previous = copy.deepcopy(snapshots[-1])
        result = evaluate({'gt': runner.body}, p, source, iter([[r] for r in rows]), saved=previous)
        assert result['windows'] == 64 and result['track_arm'] == 'unified_fit'
        assert snapshots[-1] == previous and before == tensor_state_fingerprint(runner.body.state_dict())
        bad = copy.deepcopy(previous); bad['counts']['occ_gt_transport']['confusion'][0][0][0] += 1
        with pytest.raises(RuntimeError, match='counts'): evaluate({'gt': runner.body}, p, source, iter([]), saved=bad)
    finally: runner.close()
