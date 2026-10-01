"""Full-epoch coverage, source batching, exact causal prefetch and resume."""
import copy
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from threading import get_ident
from threading import Event
import numpy as np
import pytest
import torch
from test_joint_causal_columns import fixture, optimizers, provider_for
from test_causal_columns import moving_fixture
from real_motion.joint_causal_columns import FULL_PROTOCOL, FULL_CONTRACT
from tools.real_motion import joint_column_full_common as full
from tools.real_motion import joint_column_common as common
from tools.real_motion import causal_column_common as columns
from tools.real_motion import train_p0_f9_joint_causal_columns_full as trainer
from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint


def test_all_epochs_cover_every_window_once_and_source_budgets_no_truncation():
    counts = [0, 3, 90, 8, 300, 1, 0, 45, 65]
    plans = trainer.training_plans(counts, 20, 43, 4, 128)
    for groups in plans:
        assert sorted(i for g in groups for i in g) == list(range(len(counts)))
        for group in groups:
            assert len(group) <= 4
            assert sum(counts[i] for i in group) <= 128 or len(group) == 1
    assert plans[0] != plans[1] and plans == trainer.training_plans(counts, 20, 43, 4, 128)
    assert plans[:15] == trainer.training_plans(counts, 15, 43, 4, 128)
    assert all(4 not in g or g == (4,) for g in plans[0])  # oversized source window kept whole


def test_prefetch_batch_order_cpu_worker_partial_and_error():
    rows = [{'features': np.zeros((n, 1)), 'id': i} for i, n in enumerate([2, 3, 1, 20, 0, 4])]
    threads = []
    def load(source, row, *, include_gt):
        assert include_gt; threads.append(get_ident()); return {'id': row['id']}
    provider = SimpleNamespace(load_raw_columns=load)
    batches = list(full.prefetch_column_batches(provider, None, rows, 3, 5))
    assert [[r['id'] for r, raw in b] for b in batches] == [[0, 1], [2], [3], [4, 5]]
    assert all(r['id'] == raw['id'] for b in batches for r, raw in b)
    assert all(t != get_ident() for t in threads)
    def fail(*args, **kwargs): raise RuntimeError('causal prefetch failed')
    with pytest.raises(RuntimeError, match='prefetch failed'):
        list(full.prefetch_column_batches(SimpleNamespace(load_raw_columns=fail), None, rows, 3))
    with pytest.raises(ValueError): list(full.prefetch_column_batches(provider, None, rows, 0))


def test_single_window_full_batch_matches_original_actual_updates():
    prep, grid, joint, control, rec = fixture(); other = copy.deepcopy(joint)
    opt, co = optimizers(joint, control); opt2, _ = optimizers(other, copy.deepcopy(control))
    provider = provider_for(prep, grid, joint); provider2 = provider_for(prep, grid, other)
    rng = np.random.default_rng(4); rng2 = np.random.default_rng(4)
    for update in (1, 2):
        common.train_window(joint, control, opt, co, provider, None, rec, None, rng, update, 10, probe=True)
        stats = full.train_full_batch(other, opt2, provider2, None, [(rec, None)], rng2, update, 10, probe=True)
        assert stats['windows'] == 1 and stats['paired_control_motion_loss'] is None
        assert all(torch.equal(v, other.state_dict()[k]) for k, v in joint.state_dict().items())
    assert stats['source_query_gradient_norm'] > 0


def test_multiwindow_source_identity_label_weighting_and_joint_link():
    prep, grid, joint, control, rec = fixture()
    second = copy.deepcopy(rec); second['features'] += .1; second['target_source_residual_xy_m'] += .2
    second['target_source_displacement_xy_m'] += .2
    opt, _ = optimizers(joint, control); provider = provider_for(prep, grid, joint)
    torch.nn.init.normal_(joint.columns.refinement.weight, std=.01)
    seen = []
    original_prepare = provider.prepare_columns
    def prepare(source, record, *, include_gt, raw_window, outputs):
        assert outputs['future_transport_queries'].shape == (1, 6, 8)
        seen.append(outputs['future_transport_queries'])
        return original_prepare(source, record, include_gt=include_gt, raw_window=raw_window, outputs=outputs)
    provider.prepare_columns = prepare
    stats = full.train_full_batch(joint, opt, provider, None, [(rec, None), (second, None)], np.random.default_rng(8), 1, 10, probe=True)
    assert stats['windows'] == 2 and stats['sources'] == 2 and stats['sampled_columns'] <= 512
    assert stats['source_query_gradient_norm'] > 0 and not torch.equal(seen[0], seen[1])
    assert torch.isfinite(joint.transport.residual_head.weight).all()


def test_unsupervised_no_query_window_does_not_decay_parameters():
    prep, grid, joint, control, rec = fixture(); rec['supervised_source'][:] = False
    opt, _ = optimizers(joint, control); before = copy.deepcopy(joint.state_dict())
    provider = provider_for(prep, grid, joint)
    with patch.object(full, 'select_online_columns', return_value=[]):
        stats = full.train_full_batch(joint, opt, provider, None, [(rec, None)], np.random.default_rng(2), 1, 10)
    assert not stats['optimizer_updated']
    assert all(torch.equal(v, joint.state_dict()[k]) for k, v in before.items())


def test_parallel_sampling_keeps_rng_order_updates_and_gpu_work_on_caller():
    prep, grid, joint, control, rec = fixture(); serial = copy.deepcopy(joint)
    opt, _ = optimizers(joint, control); opt2, _ = optimizers(serial, control)
    p, q = provider_for(prep, grid, joint), provider_for(prep, grid, serial)
    p.workers, q.workers = 4, 1
    r1, r2 = np.random.default_rng(71), np.random.default_rng(71)
    second = copy.deepcopy(rec); second['features'] += .2
    caller = get_ident(); seen = []; actual = full.sample_online_column
    actual_candidates = full.build_online_column_candidates
    def planned(*args):
        assert get_ident() != caller
        return actual_candidates(*args)
    def mapped(*args):
        assert get_ident() != caller
        seen.append(get_ident()); return actual(*args)
    actual_gather = joint.columns.source_features_for
    def gather(*args):
        assert get_ident() == caller; return actual_gather(*args)
    joint.columns.source_features_for = gather
    for update in (1, 2):
        with patch.object(full, 'sample_online_column', side_effect=mapped), \
             patch.object(full, 'build_online_column_candidates', side_effect=planned):
            full.train_full_batch(joint, opt, p, None, [(rec, None), (second, None)], r1, update, 12)
            full.train_full_batch(serial, opt2, q, None, [(rec, None), (second, None)], r2, update, 12)
        assert all(torch.equal(v, serial.state_dict()[k]) for k, v in joint.state_dict().items())
        assert r1.bit_generator.state == r2.bit_generator.state
    assert seen


def test_whole_run_cosine_depends_on_declared_total_steps_and_has_no_tail():
    prep, grid, joint, control, rec = fixture(); opt, _ = optimizers(joint, control)
    for step, factor in ((0, 1.), (75, .55), (150, .1)):
        common.set_lr(opt, step, 150)
        assert all(np.isclose(g['lr'], g['initial_lr']*factor) for g in opt.param_groups)
    common.set_lr(opt, 100, 150)
    assert opt.param_groups[0]['lr'] > opt.param_groups[0]['initial_lr']*.1
    a = opt.param_groups[0]['lr']; common.set_lr(opt, 100, 200)
    assert opt.param_groups[0]['lr'] > a


def test_causal_evidence_prefetch_matches_main_and_does_not_read_future_gt():
    from real_motion.strong_w2det import StrongW2DetConfig
    prep, grid, joint, control, rec = fixture(); raw = copy.deepcopy(prep.raw)
    # Frozen threshold intentionally excludes this one-voxel toy source.
    strong = StrongW2DetConfig()
    pcfg = SimpleNamespace(grid=grid, free_label=17, frame_dt_s=.5)
    evidence = full.prepare_causal_evidence(raw, pcfg, strong, 1)
    altered = {**raw, 'future_gt_occ': 'must not be read'}
    other = full.prepare_causal_evidence(altered, pcfg, strong, 1)
    assert np.array_equal(evidence['memory'], other['memory']) and np.array_equal(evidence['footprints'], other['footprints'])
    assert evidence['audit'] == other['audit']
    assert columns.runtime.component_lists_equal(evidence['current'], other['current'])
    # Compare actual prepare_columns with/without worker evidence on empty
    # Strong toy state. This exercises the full history/static/renderer path.
    state = {**prep.state, 'current': [], 'velocities': {}, 'source_world_points': [], 'source_rel_xy': [],
        'source_z_t0': np.empty(0), 'anchors': [raw['history_occ'][-1].copy() for _ in range(6)],
        'baseline_by_hi': [[]]*6, 'baseline_clear_flat_by_hi': [np.empty(0, np.int64)]*6}
    rec = {'source_centroid_xy_t0_m': torch.empty(0, 2), 'anchors_xy_t0_m': torch.empty(0, 6, 2)}
    output = {'residual_xy_m': torch.empty(0, 6, 2), 'yaw_delta_rad': torch.empty(0, 6)}
    provider = columns.FrozenColumns.__new__(columns.FrozenColumns)
    provider.pcfg, provider.device, provider.workers, provider.strong = pcfg, torch.device('cpu'), 1, strong
    provider.columns_checked = True
    with patch.object(columns, 'window_from_record', return_value=prep.window), patch.object(columns.runtime, '_prepare_record', return_value=state):
        a = provider.prepare_columns(None, rec, include_gt=True, raw_window=raw, outputs=output)
        b = provider.prepare_columns(None, rec, include_gt=True, raw_window={**raw, '_column_causal_preparation': evidence}, outputs=output)
    assert all(np.array_equal(x, y) for x, y in zip(a.baseline, b.baseline))
    assert np.array_equal(a.memory, b.memory) and np.array_equal(a.footprints, b.footprints)
    assert a.source_audit == b.source_audit and a.registrations == b.registrations


def test_nonempty_prefetch_preserves_registration_features_labels_and_live_geometry():
    from real_motion.strong_w2det import StrongW2DetConfig
    prep, grid, joint, control, rec = fixture(); raw = copy.deepcopy(prep.raw)
    prep.window.history_tokens = tuple(f'h{i}' for i in range(6))
    hist = raw['history_occ']; hist[hist == 4] = 17
    for f in range(6): hist[f, 1+f:4+f, 5:8, 1] = 4
    strong = StrongW2DetConfig(); pcfg = SimpleNamespace(grid=grid, free_label=17, frame_dt_s=.5)
    evidence = full.prepare_causal_evidence(raw, pcfg, strong, 1)
    assert len(evidence['current']) == 1 and sum(r is not None for r in evidence['registrations'][0]) > 1
    center = torch.tensor(np.asarray([c['centroid_world'][:2] for c in evidence['current']]), dtype=torch.float32)
    rec['source_centroid_xy_t0_m'] = center
    rec['anchors_xy_t0_m'] = center[:, None, :].repeat(1, 6, 1)+rec['kta_displacement_xy_m']
    provider = columns.FrozenColumns.__new__(columns.FrozenColumns)
    provider.pcfg, provider.device, provider.workers, provider.strong = pcfg, torch.device('cpu'), 1, strong
    provider.columns_checked = True
    joint.eval(); output = joint.motion(rec, torch.device('cpu'))
    # Only the record-to-window adapter is mocked: extraction, Strong renderer,
    # registration, candidate generation, feature sampling and GT labels are real.
    with patch.object(columns, 'window_from_record', return_value=prep.window), \
         patch.object(columns.runtime, 'window_from_record', return_value=prep.window):
        a = provider.prepare_columns(None, rec, include_gt=True, raw_window=raw, outputs=output)
        cached = {**raw, '_column_causal_preparation': evidence}
        b = provider.prepare_columns(None, rec, include_gt=True, raw_window=cached, outputs=output)
        for key in ('baseline', 'owners', 'fallbacks', 'targets', 'yaws', 'footprints', 'memory'):
            assert all(np.array_equal(x, y) for x, y in zip(getattr(a, key), getattr(b, key)))
        assert a.source_audit == b.source_audit
        for ra, rb in zip(a.registrations[0], b.registrations[0]):
            assert (ra is None) == (rb is None)
            if ra is not None: assert all(np.array_equal(x, y) for x, y in zip(ra, rb))
        for h in range(6):
            pa = columns.candidate_plan(a, h, grid, joint.columns.config)
            pb = columns.candidate_plan(b, h, grid, joint.columns.config)
            assert all(np.array_equal(v, getattr(pb, k)) for k, v in vars(pa).items())
            fa = columns.sample_column_features(a, h, pa, grid, joint.columns.config)
            fb = columns.sample_column_features(b, h, pb, grid, joint.columns.config)
            assert all(np.array_equal(v, fb[k]) for k, v in fa.items())
            gt = raw['future_gt_occ'][h].reshape(-1)
            assert np.array_equal(columns.action_targets(pa, gt), columns.action_targets(pb, gt))
        moved = {**output, 'residual_xy_m': output['residual_xy_m']+1.}
        c = provider.prepare_columns(None, rec, include_gt=True, raw_window=cached, outputs=moved)
        assert any(not np.array_equal(x, y) for x, y in zip(b.baseline, c.baseline))
        bad = copy.deepcopy(evidence); bad['current'][0]['class_id'] = 5
        with pytest.raises(RuntimeError, match='source identity mismatch'):
            provider.prepare_columns(None, rec, include_gt=True,
                raw_window={**raw, '_column_causal_preparation': bad}, outputs=output)


def test_full_cli_epochs_exact_resume_and_reject_schedule_extension(tmp_path):
    prep, grid, sample, _, template = fixture()
    train = [{**copy.deepcopy(template), 'scene_name': f'train{s}', 't0_token': f'{s}:{i}'} for s in range(10) for i in range(4)]
    dev = [{**copy.deepcopy(template), 'scene_name': 'dev', 't0_token': f'd{i}'} for i in range(512)]
    keys = [(r['scene_name'], r['t0_token']) for r in dev[:64]]
    manifest = dict(parent_keys=[(r['scene_name'], r['t0_token']) for r in dev], selected_key_fingerprint=trainer.DEV64_FP, manifest_fingerprint='c'*64)
    files = {k: tmp_path/k for k in ('train-cache', 'dev-cache', 'population-manifest', 'base-checkpoint', 'train-info', 'dev-info')}
    for f in files.values(): f.write_bytes(b'original')
    torch.save({'model_config': asdict(sample.transport.config), 'yaw_weight': 19.}, files['base-checkpoint'])
    originals = {k: f.read_bytes() for k, f in files.items()}
    pcfg = SimpleNamespace(grid=grid)
    def make_provider(checkpoint, expected_sha, cfg, device, workers, joint, control):
        result = provider_for(prep, grid, joint); result.workers = workers; result.reference_enabled = False
        result.joint, result.control, result.model = joint, control, joint.transport
        def load(source, record, *, include_gt): return copy.deepcopy(prep.raw)
        def prepare(source, record, *, include_gt, raw_window=None, outputs=None):
            row = copy.deepcopy(prep); row.window.scene_name = record['scene_name']; row.window.t0_token = record['t0_token']
            row.outputs = outputs if outputs is not None else result.joint.motion(record, device)
            return row
        result.load_raw_columns = load; result.prepare_columns = prepare
        result.reference_predictions = lambda p, r: {'frozen_E14': p.baseline} if result.reference_enabled else {}
        return result
    def run(out, epochs, resume=None, fail_update=None, stop_update=None):
        argv = ['train', '--config', str(Path(__file__).resolve().parents[1]/'configs/real_motion_occfm.yaml'),
            '--dataroot', str(tmp_path), '--out-dir', str(out), '--epochs', str(epochs), '--device', 'cpu',
            '--window-batch-size', '4', '--source-budget', '128', '--cpu-workers', '1', '--checkpoint-every', '1']
        for k, f in files.items(): argv += ['--'+k, str(f)]
        if resume: argv += ['--resume', str(resume)]
        # Reduce only evaluator population/width for this CPU orchestration test.
        actual_eval = columns.evaluate_columns
        def small_eval(provider, source, records, model, gates, **kwargs):
            chosen = records[:2]; kwargs['dev64_keys'] = keys[:2] if 'dev64_keys' in kwargs else None
            return actual_eval(provider, source, chosen, model, gates, **kwargs)
        actual_step = trainer.train_full_batch
        stop = Event()
        def step(*args, **kwargs):
            if fail_update is not None and args[6] == fail_update: raise RuntimeError('simulated interruption')
            result = actual_step(*args, **kwargs)
            if stop_update is not None and args[6] == stop_update: stop.set()
            return result
        with patch('sys.argv', argv), patch.object(trainer, 'TRAIN_WINDOWS', 40), patch.object(trainer, 'PRIOR_WINDOWS', 4), \
            patch.object(trainer, 'CALIBRATION_WINDOWS', 2), patch.object(trainer, 'make_prepare_config', return_value=pcfg), \
            patch.object(trainer, 'load_manifest', return_value=(manifest, keys, None)), \
            patch.object(trainer, 'load_cache', side_effect=[({}, train), ({}, dev)]), \
            patch.object(trainer, 'JointCausalColumns', side_effect=lambda mc, cfg: type(sample)(mc, sample.columns.config)), \
            patch.object(trainer, 'FullJointColumnProvider', side_effect=make_provider), \
            patch.object(trainer, 'validate_clean_e14_checkpoint', return_value='a'*64), \
            patch.object(trainer, 'NuScenesWindowSource', return_value=SimpleNamespace(nusc=None)), \
            patch.object(columns, 'gt_moving_support_sequence', return_value=moving_fixture(prep)), \
            patch.object(trainer, 'evaluate_columns', side_effect=small_eval), patch.object(trainer, 'train_full_batch', side_effect=step):
            return trainer.main(stop if stop_update is not None else None)
    out = tmp_path/'run'; run(out, 2)
    summary = json.loads((out/'summary.json').read_text(encoding='utf-8'))
    assert summary['epochs_completed'] == 2 and summary['executed_windows'] == 80
    assert summary['successful_updates'] == 20 and not summary['paired_control'] and summary['gradient_link_observed']
    ck = torch.load(out/'last.pt', weights_only=False)
    assert ck['protocol'] == FULL_PROTOCOL and ck['training_contract'] == FULL_CONTRACT and ck['cursor_epoch'] == 2 and ck['cursor_batch'] == 0
    assert not 'control_state_dict' in ck and len(ck['epoch_history']) == 2
    assert len(ck['train_keys']) == 40 and set(map(tuple, ck['calibration_keys'])).issubset(set(map(tuple, ck['train_keys'])))
    assert json.loads((out/'TRAIN_calibration.json').read_text(encoding='utf-8'))['held_out'] is False
    assert {k: f.read_bytes() for k, f in files.items()} == originals
    resumed = tmp_path/'resume'; run(resumed, 2, out/'last.pt')
    other = json.loads((resumed/'summary.json').read_text(encoding='utf-8'))
    assert summary['evaluation'] == other['evaluation'] and summary['thresholds'] == other['thresholds']
    # Mid-epoch interruption restores exact optimizer, RNG, cursor and sample order.
    stopped = tmp_path/'stop'
    with pytest.raises(RuntimeError, match='simulated interruption'): run(stopped, 2, fail_update=4)
    recovered = tmp_path/'recovered'; run(recovered, 2, stopped/'last.pt')
    a, b = (torch.load(p/'last.pt', weights_only=False) for p in (out, recovered))
    assert all(torch.equal(v, b['state_dict'][k]) for k, v in a['state_dict'].items())
    assert a['sampling_rng_state'] == b['sampling_rng_state'] and a['executed_windows'] == b['executed_windows']
    graceful = tmp_path/'graceful'
    assert run(graceful, 2, stop_update=4) == 130
    assert not (graceful/'summary.json').exists()
    stopped_ck = torch.load(graceful/'last.pt', weights_only=False)
    assert stopped_ck['attempted_updates'] == 4 and stopped_ck['cursor_batch'] == 4
    continued = tmp_path/'continued'; run(continued, 2, graceful/'last.pt')
    continued_ck = torch.load(continued/'last.pt', weights_only=False)
    assert all(torch.equal(v, continued_ck['state_dict'][k]) for k, v in a['state_dict'].items())
    assert a['sampling_rng_state'] == continued_ck['sampling_rng_state']
    with pytest.raises(RuntimeError, match='resume contract'): run(tmp_path/'extended', 3, out/'last.pt')
    with pytest.raises(RuntimeError, match='resume contract'): run(tmp_path/'short', 1, out/'last.pt')
    with pytest.raises(RuntimeError, match='resume contract'): run(tmp_path/'bad', 2, out/'candidate.pt')
