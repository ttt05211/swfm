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
    caller = get_ident(); seen = []; actual = full.sample_online_columns
    actual_candidates = full.build_online_column_candidates
    def planned(*args, **kwargs):
        assert get_ident() != caller
        return actual_candidates(*args, **kwargs)
    def mapped(*args):
        assert get_ident() != caller
        seen.append(get_ident()); return actual(*args)
    actual_gather = joint.columns.source_features_for
    def gather(*args):
        assert get_ident() == caller; return actual_gather(*args)
    joint.columns.source_features_for = gather
    for update in (1, 2):
        with patch.object(full, 'sample_online_columns', side_effect=mapped), \
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


@pytest.mark.parametrize('persistent', (False, True))
def test_nonempty_prefetch_preserves_registration_features_labels_and_live_geometry(tmp_path, persistent):
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
    provider.model = joint.transport; provider.columns_checked = False
    torch.nn.init.normal_(joint.columns.refinement.weight, std=.01)
    joint.train(); output = joint.motion(rec, torch.device('cpu'))
    assert output['residual_xy_m'].requires_grad and output['yaw_delta_rad'].requires_grad
    # Only the record-to-window adapter is mocked: extraction, Strong renderer,
    # registration, candidate generation, feature sampling and GT labels are real.
    with patch.object(columns, 'window_from_record', return_value=prep.window), \
         patch.object(columns.runtime, 'window_from_record', return_value=prep.window):
        if persistent:
            from real_motion.causal_geometry_cache import CausalGeometryCache
            cache = CausalGeometryCache(tmp_path, 'fixture', ram_bytes=0, reserve_bytes=0)
            def build():
                return full.build_fixed_geometry({**raw, 'future_gt_occ': 'forbidden'}, rec,
                    pcfg, strong, 1, joint.columns.config)
            fixed, hit = cache.get_or_build(('scene', 't0'), raw, build)
            assert not hit
            fresh = CausalGeometryCache(tmp_path, 'fixture', ram_bytes=0, reserve_bytes=0)
            fixed, hit = fresh.get_or_build(('scene', 't0'), raw, lambda: pytest.fail('must load disk'))
            assert hit and not any(k in fixed['prepared_state'] for k in ('rec', 'window', 'gpu', 'outputs'))
            def no_tensors(v):
                assert not isinstance(v, torch.Tensor)
                if isinstance(v, dict):
                    assert 'future_gt_occ' not in v
                    for item in v.values(): no_tensors(item)
                elif isinstance(v, (list, tuple)):
                    for item in v: no_tensors(item)
                elif hasattr(v, '__dict__'): no_tensors(vars(v))
            no_tensors(fixed)
            # A validly serialized but mathematically wrong background must
            # fail the formal first-use gate, not merely pass content hashes.
            corrupt = copy.deepcopy(fixed)
            corrupt['prepared_state']['anchors'][0].flat[0] = 0
            bad_provider = copy.copy(provider)
            with pytest.raises(RuntimeError, match='prepared Strong anchor mismatch'):
                bad_provider.prepare_columns(None, rec, include_gt=True,
                    raw_window={**raw, '_column_causal_preparation': corrupt}, outputs=output)
            # Strong's vectorized scatter may contain duplicate/unordered
            # destination rows after ego rotation: compare occupied sets, not
            # raw scatter order, just as the frozen CLEAR contract does.
            equivalent = copy.deepcopy(fixed)
            for rows in equivalent['prepared_state']['baseline_by_hi']:
                for i, comp in enumerate(rows):
                    rows[i] = type(comp)(comp.class_id,
                        np.concatenate((comp.voxel_indices[::-1], comp.voxel_indices[:1])), comp.source_voxel_count)
            equivalent_provider = copy.copy(provider)
            equivalent_provider.prepare_columns(None, rec, include_gt=True,
                raw_window={**raw, '_column_causal_preparation': equivalent}, outputs=output)
            # Cold resume: first exactness check now uses the persistent CPU
            # state but must preserve the original LIVE joint training graph.
            first_raw = {**raw, '_column_causal_preparation': fixed}
        else: first_raw = raw
        a = provider.prepare_columns(None, rec, include_gt=True, raw_window=first_raw, outputs=output)
        assert provider.columns_checked and a.outputs is output
        assert a.outputs['future_transport_queries'].requires_grad
        # The REAL first-call exactness/forecast NumPy path must not sever the
        # joint graph. Backpropagate after it, without any mocked renderer/check.
        batch = common.online_columns(a, joint.columns, grid, np.random.default_rng(91), torch.device('cpu'))
        g, r = joint.columns(**{k: batch[k] for k in (*common.FEATURE_KEYS, 'source_features')})
        loss, _ = full.column_loss(joint.columns, g, r, batch['kind'], batch['legal'], batch['target'], batch['weight'])
        grad = torch.autograd.grad(loss, output['future_transport_queries'], retain_graph=True)[0]
        assert grad.norm() > 0
        loss.backward(retain_graph=True)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in joint.transport.parameters())
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
        if persistent:
            d = provider.prepare_columns(None, rec, include_gt=True,
                raw_window={**raw, '_column_causal_preparation': fixed}, outputs=moved)
            assert all(np.array_equal(x, y) for x, y in zip(c.baseline, d.baseline))
            assert d.outputs['future_transport_queries'].requires_grad
        bad = copy.deepcopy(evidence); bad['current'][0]['class_id'] = 5
        with pytest.raises(RuntimeError, match='source identity mismatch'):
            provider.prepare_columns(None, rec, include_gt=True,
                raw_window={**raw, '_column_causal_preparation': bad}, outputs=output)


def test_cached_window_prefetch_order_errors_and_early_close():
    rows = [{'features': torch.zeros(1, 2), 'id': i} for i in range(9)]
    def load(source, record, **kwargs): return {'id': record['id']}
    provider = SimpleNamespace(load_raw_columns=load, causal_geometry_cache=object())
    batches = list(full.prefetch_column_batches(provider, None, rows, 4))
    assert [r['id'] for b in batches for r, raw in b] == list(range(9))
    assert all(r['id'] == raw['id'] for b in batches for r, raw in b)
    for _ in range(10):
        iterator = full.prefetch_column_batches(provider, None, rows, 4)
        assert len(next(iterator)) == 4
        iterator.close()  # pending outer load may still be about to call io.map
    def fail(*args, **kwargs): raise RuntimeError('cached geometry build failed')
    provider.load_raw_columns = fail
    with pytest.raises(RuntimeError, match='geometry build failed'):
        list(full.prefetch_column_batches(provider, None, rows, 4))


def test_live_device_cold_cache_then_exact_persistent_hit_preserves_gradients(tmp_path):
    from real_motion.causal_geometry_cache import CausalGeometryCache
    from real_motion.strong_w2det import StrongW2DetConfig
    prep, grid, joint, _, rec = fixture(); raw = copy.deepcopy(prep.raw)
    prep.window.history_tokens = tuple(f'h{i}' for i in range(6))
    raw['history_occ'][raw['history_occ'] == 4] = 17
    for f in range(6): raw['history_occ'][f, 1+f:4+f, 5:8, 1] = 4
    rec.update(scene_name='scene', t0_token='t0')
    strong = StrongW2DetConfig(); pcfg = SimpleNamespace(grid=grid, free_label=17, frame_dt_s=.5)
    evidence = full.prepare_causal_evidence(raw, pcfg, strong, 1)
    center = torch.tensor(np.asarray([c['centroid_world'][:2] for c in evidence['current']]), dtype=torch.float32)
    rec['source_centroid_xy_t0_m'] = center
    rec['anchors_xy_t0_m'] = center[:, None, :].repeat(1, 6, 1)+rec['kta_displacement_xy_m']
    provider = full.FullJointColumnProvider.__new__(full.FullJointColumnProvider)
    provider.pcfg, provider.strong, provider.device, provider.workers = pcfg, strong, torch.device('cpu'), 1
    provider.joint, provider.model, provider.columns_checked = joint, joint.transport, False
    provider.causal_geometry_cache = CausalGeometryCache(tmp_path, 'p', ram_bytes=0, reserve_bytes=0)
    caller = get_ident(); actual = columns.runtime._prepare_record; calls = []
    def main_device(*args, **kwargs):
        assert get_ident() == caller and args[4] == provider.device
        calls.append(args[4]); return actual(*args, **kwargs)
    torch.nn.init.normal_(joint.columns.refinement.weight, std=.01)
    output = joint.motion(rec, provider.device)
    with patch.object(columns, 'window_from_record', return_value=prep.window), \
         patch.object(columns.runtime, 'window_from_record', return_value=prep.window), \
         patch.object(columns, 'load_nuscenes_window_raw', side_effect=lambda *args, **kwargs: copy.deepcopy(raw)), \
         patch.object(full, 'build_fixed_geometry', side_effect=AssertionError('CPU Strong builder must not run')), \
         patch.object(columns.runtime, '_prepare_record', side_effect=main_device):
        cold_raw = provider.load_raw_columns(None, rec, include_gt=True)
        assert not calls and 'prepared_state' not in cold_raw['_column_causal_preparation']
        assert not list(provider.causal_geometry_cache.root.glob('*.cgc'))
        cold = provider.prepare_columns(None, rec, include_gt=True, raw_window=cold_raw, outputs=output)
        assert len(calls) == 1 and cold.outputs is output
        provider.causal_geometry_cache.close()
        provider.causal_geometry_cache = CausalGeometryCache(tmp_path, 'p', ram_bytes=0, reserve_bytes=0)
        warm_raw = provider.load_raw_columns(None, rec, include_gt=True)
        assert warm_raw['_causal_geometry_cache_hit'] and not warm_raw['_causal_cache_deferred']
        warm = provider.prepare_columns(None, rec, include_gt=True, raw_window=warm_raw, outputs=output)
        assert len(calls) == 1  # no Strong recomputation on hits
        for h in range(6):
            assert np.array_equal(cold.baseline[h], warm.baseline[h])
            a, b = (columns.candidate_plan(p, h, grid, joint.columns.config) for p in (cold, warm))
            assert all(np.array_equal(v, getattr(b, k)) for k, v in vars(a).items())
            x, y = (columns.sample_column_features(p, h, plan, grid, joint.columns.config) for p, plan in ((cold, a), (warm, b)))
            assert all(np.array_equal(v, y[k]) for k, v in x.items())
        batch = common.online_columns(warm, joint.columns, grid, np.random.default_rng(91), provider.device)
        g, r = joint.columns(**{k: batch[k] for k in (*common.FEATURE_KEYS, 'source_features')})
        loss, _ = full.column_loss(joint.columns, g, r, batch['kind'], batch['legal'], batch['target'], batch['weight'])
        assert torch.autograd.grad(loss, output['future_transport_queries'], retain_graph=True)[0].norm() > 0
        moved = provider.prepare_columns(None, rec, include_gt=True, raw_window=warm_raw,
            outputs={**output, 'residual_xy_m': output['residual_xy_m']+1})
        assert any(not np.array_equal(x, y) for x, y in zip(warm.baseline, moved.baseline))
        provider.causal_geometry_cache.close()


@pytest.fixture
def full_cli_fixture(tmp_path):
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
        from real_motion.strong_w2det import StrongW2DetConfig
        result = provider_for(prep, grid, joint); result.workers = workers; result.reference_enabled = False
        result.joint, result.control, result.model = joint, control, joint.transport
        result.strong = StrongW2DetConfig()
        def load(source, record, *, include_gt): return copy.deepcopy(prep.raw)
        def prepare(source, record, *, include_gt, raw_window=None, outputs=None):
            row = copy.deepcopy(prep); row.window.scene_name = record['scene_name']; row.window.t0_token = record['t0_token']
            n = joint.transport.config.history_frames
            for key in ('history_occ','history_observed','history_poses'): row.raw[key] = row.raw[key][-n:]
            row.registrations = [r[-n:] for r in row.registrations]
            row.outputs = outputs if outputs is not None else result.joint.motion(record, device)
            return row
        result.load_raw_columns = load; result.prepare_columns = prepare
        result.reference_predictions = lambda p, r: {'frozen_E14': p.baseline} if result.reference_enabled else {}
        return result
    def run(out, epochs, resume=None, fail_update=None, stop_update=None, stop_prior=None,
            stop_monitor=None, stop_calibration=False, stop_final=False, history_frames=6,
            causal_cache=None, stop_before_prior=False, extend=False, distributed=False, profile_every=0):
        argv = ['train', '--config', str(Path(__file__).resolve().parents[1]/'configs/real_motion_occfm.yaml'),
            '--dataroot', str(tmp_path), '--out-dir', str(out), '--epochs', str(epochs), '--device', 'cpu',
            '--window-batch-size', '4', '--source-budget', '128', '--cpu-workers', '1', '--checkpoint-every', '1', '--history-frames', str(history_frames)]
        for k, f in files.items(): argv += ['--'+k, str(f)]
        if resume: argv += ['--resume', str(resume)]
        if extend: argv += ['--extend-completed-run']
        if distributed: argv += ['--distributed']
        if profile_every: argv += ['--profile-every', str(profile_every)]
        if causal_cache: argv += ['--causal-geometry-cache', str(causal_cache)]
        # Reduce only evaluator population/width for this CPU orchestration test.
        actual_eval = columns.evaluate_columns; eval_calls = []
        stop = Event()
        if stop_before_prior: stop.set()
        def small_eval(provider, source, records, model, gates, **kwargs):
            eval_calls.append(1)
            original_progress = kwargs.get('progress')
            def progress(row):
                if original_progress: original_progress(row)
                if row['event'] == 'evaluation' and ((stop_monitor == len(eval_calls))
                        or stop_final and len(records) == 512): stop.set()
            kwargs['progress'] = progress
            chosen = records[:2]; kwargs['dev64_keys'] = keys[:2] if 'dev64_keys' in kwargs else None
            return actual_eval(provider, source, chosen, model, gates, **kwargs)
        actual_step = trainer.train_full_batch
        def step(*args, **kwargs):
            if fail_update is not None and args[6] == fail_update: raise RuntimeError('simulated interruption')
            result = actual_step(*args, **kwargs)
            if stop_update is not None and args[6] == stop_update: stop.set()
            return result
        actual_count = trainer.count_proposals; counts_calls = []
        def count(*args, **kwargs):
            result = actual_count(*args, **kwargs); counts_calls.append(1)
            if stop_prior == len(counts_calls): stop.set()
            return result
        actual_calibrate = trainer.calibrate_columns
        def calibrate(*args, **kwargs):
            original_progress = kwargs.get('progress')
            def progress(row):
                if original_progress: original_progress(row)
                if stop_calibration and row['event'] == 'TRAIN_calibration': stop.set()
            kwargs['progress'] = progress
            return actual_calibrate(*args, **kwargs)
        with patch('sys.argv', argv), patch.object(trainer, 'TRAIN_WINDOWS', 40), patch.object(trainer, 'PRIOR_WINDOWS', 4), \
            patch.object(trainer, 'CALIBRATION_WINDOWS', 2), patch.object(trainer, 'make_prepare_config', return_value=pcfg), \
            patch.object(trainer, 'load_manifest', return_value=(manifest, keys, None)), \
            patch.object(trainer, 'load_cache', side_effect=[({}, train), ({}, dev)]), \
            patch.object(trainer, 'JointCausalColumns', side_effect=lambda mc, cfg: type(sample)(mc, sample.columns.config)), \
            patch.object(trainer, 'FullJointColumnProvider', side_effect=make_provider), \
            patch.object(trainer, 'validate_clean_e14_checkpoint', return_value='a'*64), \
            patch.object(trainer, 'NuScenesWindowSource', return_value=SimpleNamespace(nusc=None)), \
            patch.object(columns, 'gt_moving_support_sequence', return_value=moving_fixture(prep)), \
            patch.object(trainer, 'evaluate_columns', side_effect=small_eval), patch.object(trainer, 'train_full_batch', side_effect=step), \
            patch.object(trainer, 'count_proposals', side_effect=count), patch.object(trainer, 'calibrate_columns', side_effect=calibrate):
            return trainer.main(stop)
    run.eval_data = (prep, grid, dev, manifest, keys, make_provider)
    return run, files, originals


def test_full_cli_real_cache_startup_log_and_resume_before_prior(tmp_path, full_cli_fixture, capsys):
    run, _, _ = full_cli_fixture
    cache = tmp_path/'geometry'; first = tmp_path/'first'; second = tmp_path/'second'
    for destination, resume in ((first, None), (second, first/'last.pt')):
        assert run(destination, 1, resume=resume, history_frames=4,
                   causal_cache=cache, stop_before_prior=True) == 130
        output = capsys.readouterr().out
        rows = [json.loads(line.split(': ', 1)[1]) for line in output.splitlines()
                if line.startswith('CAUSAL GEOMETRY CACHE: ')]
        assert len(rows) == 1
        row = rows[0]
        assert Path(row['directory']).is_dir() and Path(row['directory']).parent == cache
        assert row['namespace'] and row['cold_strong_device'] == 'cpu'
        assert row['disk_limit_mib'] == 48*1024 and row['hits'] == row['writes'] == 0
        ck = torch.load(destination/'last.pt', weights_only=False)
        assert ck['attempted_updates'] == 0 and ck['prior_cursor'] == 0 and not ck['prior_completed']
    a, b = (torch.load(p/'last.pt', weights_only=False) for p in (first, second))
    assert all(torch.equal(v, b['state_dict'][k]) for k, v in a['state_dict'].items())


def test_full_cli_epochs_exact_resume_and_reject_schedule_extension(tmp_path,full_cli_fixture):
    run,files,originals = full_cli_fixture
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


def test_explicit_completed_extension_preserves_parent_and_resumes_exactly(tmp_path, full_cli_fixture):
    run, _, _ = full_cli_fixture
    parent = tmp_path/'parent'; run(parent, 1, history_frames=4)
    original = {p.name: p.read_bytes() for p in parent.iterdir() if p.is_file()}
    baseline = tmp_path/'extended'; run(baseline, 2, parent/'last.pt', history_frames=4, extend=True)
    a = torch.load(baseline/'last.pt', weights_only=False)
    assert a['continuation']['original_epochs'] == 1 and a['cursor_epoch'] == 2
    assert a['executed_windows'] == 80 and a['attempted_updates'] == 20
    assert a['protocol'].endswith('history4_completed_extension_v1')
    before = torch.load(parent/'last.pt', weights_only=False)
    log = [json.loads(line) for line in (baseline/'progress.jsonl').read_text().splitlines()]
    updates = [row for row in log if row['event'] == 'train_full']
    assert updates[0]['update'] == 11 and updates[0]['epoch'] == 2
    assert updates[0]['learning_rates'] == [g['lr'] for g in before['optimizer']['param_groups']]
    assert np.allclose(updates[-1]['learning_rates'], np.array(updates[0]['learning_rates'])*.1)
    paused = tmp_path/'paused'; assert run(paused, 2, parent/'last.pt', history_frames=4, extend=True, stop_update=14) == 130
    resumed = tmp_path/'resumed'; run(resumed, 2, paused/'last.pt', history_frames=4)
    b = torch.load(resumed/'last.pt', weights_only=False)
    assert a['sampling_rng_state'] == b['sampling_rng_state'] and torch.equal(a['torch_rng_state'], b['torch_rng_state'])
    assert a['optimizer']['param_groups'] == b['optimizer']['param_groups']
    assert all(torch.equal(v, b['state_dict'][k]) for k, v in a['state_dict'].items())
    for k, value in a['optimizer']['state'].items():
        assert all(torch.equal(v, b['optimizer']['state'][k][n]) for n, v in value.items())
    assert original == {p.name: p.read_bytes() for p in parent.iterdir() if p.is_file()}
    with pytest.raises(RuntimeError, match='completed original'): run(tmp_path/'again', 3, baseline/'last.pt', history_frames=4, extend=True)
    with pytest.raises(RuntimeError, match='completed original'): run(tmp_path/'candidate', 2, parent/'candidate.pt', history_frames=4, extend=True)
    unfinished = tmp_path/'unfinished'; assert run(unfinished, 1, history_frames=4, stop_update=2) == 130
    with pytest.raises(RuntimeError, match='fully completed'): run(tmp_path/'bad_extension', 2, unfinished/'last.pt', history_frames=4, extend=True)


@pytest.mark.parametrize('where',['prior','last_batch','monitor','calibration','final_eval'])
def test_full4_interruption_boundaries_restore_optimizer_rng_prior_and_epoch_means(tmp_path,full_cli_fixture,where):
    run,_,_ = full_cli_fixture
    # Use ONE epoch for these CPU orchestration cases; scientific full is 15/20.
    baseline=tmp_path/'baseline';run(baseline,1,history_frames=4)
    options={'prior':{'stop_prior':2},'last_batch':{'stop_update':10},'monitor':{'stop_monitor':1},
        'calibration':{'stop_calibration':True},'final_eval':{'stop_final':True}}[where]
    paused=tmp_path/'paused';assert run(paused,1,history_frames=4,**options) == 130
    ck=torch.load(paused/'last.pt',weights_only=False)
    assert json.loads((paused/'runtime_status.json').read_text())['phase'] == 'stopped'
    if where == 'prior':assert ck['prior_cursor'] == 2 and not ck['prior_completed'] and ck['attempted_updates'] == 0
    elif where in ('last_batch','monitor'):assert ck['cursor_batch'] == 10 and ck['cursor_epoch'] == 0
    else:assert ck['cursor_epoch'] == 1 and ck['cursor_batch'] == 0
    resumed=tmp_path/'resumed';run(resumed,1,paused/'last.pt',history_frames=4)
    a,b=(torch.load(p/'last.pt',weights_only=False) for p in (baseline,resumed))
    assert a['attempted_updates'] == b['attempted_updates'] == 10 and a['executed_windows'] == b['executed_windows'] == 40
    assert all(torch.equal(v,b['state_dict'][k]) for k,v in a['state_dict'].items())
    assert a['optimizer']['param_groups'] == b['optimizer']['param_groups']
    for k,state in a['optimizer']['state'].items():
        assert all(torch.equal(v,b['optimizer']['state'][k][n]) for n,v in state.items())
    assert torch.equal(a['torch_rng_state'],b['torch_rng_state']) and a['sampling_rng_state'] == b['sampling_rng_state']
    assert a['TRAIN_weights'] == b['TRAIN_weights']
    assert len(b['epoch_history']) == 1 and b['epoch_history'][0]['training_statistics_complete_epoch']
    assert a['epoch_history'][0]['training_means_accumulated'] == b['epoch_history'][0]['training_means_accumulated']
    sa,sb=(json.loads((p/'summary.json').read_text()) for p in (baseline,resumed))
    assert sa['evaluation'] == sb['evaluation'] and sa['thresholds'] == sb['thresholds']


def test_interim_full_evaluation_uses_joint_snapshot_fixed_gates_and_frozen_population(tmp_path, full_cli_fixture):
    from tools.real_motion import eval_p0_f9_joint_causal_columns as interim
    run, files, _ = full_cli_fixture
    prep, grid, dev, manifest, keys, make_provider = run.eval_data
    # A real JSON manifest contains lists, not the fixture's Python tuples.
    manifest = json.loads(json.dumps(manifest))
    trained = tmp_path/'trained'; assert run(trained, 1, stop_update=2, history_frames=4) == 130
    checkpoint = trained/'last.pt'; original = checkpoint.read_bytes()
    out = tmp_path/'interim64'; actual_eval = columns.evaluate_columns; calls = []
    def small_eval(provider, source, records, model, gates, **kwargs):
        calls.append((len(records), gates, provider.joint.transport.config.history_frames))
        assert model is provider.joint.columns and provider.reference_enabled
        assert not hasattr(provider, 'causal_geometry_cache')
        return actual_eval(provider, source, records[:2], model, gates, **kwargs)
    def evaluate(destination, population='dev64', evaluator=small_eval, event=None, raw_workers=1, speed=False, column_suite=False):
        argv = ['eval', '--config', str(Path(__file__).resolve().parents[1]/'configs/real_motion_occfm.yaml'),
            '--checkpoint', str(checkpoint), '--dev-cache', str(files['dev-cache']),
            '--population-manifest', str(files['population-manifest']), '--base-checkpoint', str(files['base-checkpoint']),
            '--dev-info', str(files['dev-info']), '--dataroot', str(tmp_path), '--out-dir', str(destination),
            '--population', population, '--device', 'cpu', '--cpu-workers', str(max(1, raw_workers)),
            '--raw-prefetch-workers', str(raw_workers), '--raw-prefetch-depth', str(raw_workers)]
        if speed:argv+=['--speed-benchmark','--speed-windows','18','--speed-repeats','1']
        if column_suite:argv+=['--speed-column-probability']
        with patch('sys.argv', argv), patch.object(interim, 'CLEAN_SHA256', 'a'*64), \
            patch.object(interim, 'make_prepare_config', return_value=SimpleNamespace(grid=grid)), \
            patch.object(interim, 'load_manifest', return_value=(manifest, keys, None)), \
            patch.object(interim, 'load_cache', return_value=({}, list(reversed(dev)))), \
            patch.object(interim, 'sha256', side_effect=lambda p: 'a'*64 if Path(p) == files['base-checkpoint'] else trainer.sha256(p)), \
            patch.object(interim, 'FullJointColumnProvider', side_effect=make_provider), \
            patch.object(interim, 'NuScenesWindowSource', return_value=SimpleNamespace(nusc=None)), \
            patch.object(columns, 'gt_moving_support_sequence', return_value=moving_fixture(prep)), \
            patch.object(interim, 'evaluate_columns', side_effect=evaluator):
            return interim.main(event)
    assert evaluate(out) == 0
    assert checkpoint.read_bytes() == original and calls == [(64, (.5,.5,None), 4)]
    result = json.loads((out/'evaluation.json').read_text())
    assert result['attempted_updates'] == 2 and result['threshold_source'] == 'fixed_monitor_0.5_0.5_REMOVE_off'
    assert result['snapshot_sha256'] == trainer.sha256(checkpoint)
    assert (out/'checkpoint_snapshot.pt').read_bytes() == original
    report = result['reports']
    parallel = tmp_path/'parallel64'
    assert evaluate(parallel, raw_workers=4) == 0
    parallel_result = json.loads((parallel/'evaluation.json').read_text())
    assert parallel_result['raw_prefetch_workers'] == parallel_result['raw_prefetch_depth'] == 4
    assert parallel_result['reports'] == report
    text = (parallel/'summary.txt').read_text()
    assert 'joint: IoU=' in text and 'MovingMacro=' in text and '3.0s joint: IoU=' in text
    assert checkpoint.read_bytes() == original
    speed_out=tmp_path/'speed_only'
    assert evaluate(speed_out,speed=True,raw_workers=2) == 0
    speed_result=json.loads((speed_out/'speed.json').read_text())
    assert speed_result['integer_counts_exact'] and not speed_result['actual_cuda']
    assert len(speed_result['trials']) == 3 and speed_result['windows'] == 18
    assert not (speed_out/'evaluation.json').exists() and checkpoint.read_bytes() == original
    column_out=tmp_path/'column_speed_only'
    assert evaluate(column_out,speed=True,raw_workers=2,column_suite=True) == 0
    column_result=json.loads((column_out/'speed.json').read_text())
    assert column_result['column_probability_suite'] and column_result['integer_counts_exact']
    assert [t['name'] for t in column_result['trials']] == ['parallel_raw','parallel_columns','parallel_columns_prefetch']
    assert column_result['no_automatic_backend_promotion'] and checkpoint.read_bytes() == original
    stopped_speed=tmp_path/'stopped_speed';event=Event();event.set()
    assert evaluate(stopped_speed,event=event,speed=True,raw_workers=2) == 130
    assert json.loads((stopped_speed/'speed_status.json').read_text())['status'] == 'interrupted'
    assert not (stopped_speed/'speed.json').exists() and not (stopped_speed/'summary.txt').exists()
    def rotate_and_report(provider, source, records, model, gates, **kwargs):
        assert len(records) == 512 and kwargs['dev64_keys'] == keys
        assert trainer.record_keys(records) == tuple(map(tuple, manifest['parent_keys']))
        assert all(isinstance(key, list) for key in manifest['parent_keys'])
        replacement = tmp_path/'replacement.pt'; replacement.write_bytes(b'new last published by running trainer')
        replacement.replace(checkpoint)
        return report
    second = tmp_path/'interim512'
    assert evaluate(second, 'dev512', rotate_and_report) == 0
    assert (second/'checkpoint_snapshot.pt').read_bytes() == original
    checkpoint.write_bytes(original)
    def cancel(*args, **kwargs): raise InterruptedError('window boundary')
    cancelled = tmp_path/'cancelled'
    assert evaluate(cancelled, evaluator=cancel) == 130
    assert not (cancelled/'summary.txt').exists() and not (cancelled/'evaluation.json').exists()
    assert json.loads((cancelled/'evaluation_status.json').read_text())['status'] == 'interrupted'
    ck = torch.load(checkpoint, weights_only=False)
    ck.update(checkpoint_role='calibrated_candidate', thresholds=(.75,.9,None)); torch.save(ck, checkpoint)
    def calibrated(provider, source, records, model, gates, **kwargs):
        assert gates == (.75,.9,None); return report
    assert evaluate(tmp_path/'calibrated', evaluator=calibrated) == 0
    ck['prior_completed'] = False; torch.save(ck, checkpoint)
    with pytest.raises(RuntimeError, match='prior is incomplete'): evaluate(tmp_path/'bad_prior')
    ck['prior_completed'] = True; ck['info_fingerprints']['dev'] = 'changed'; torch.save(ck, checkpoint)
    with pytest.raises(RuntimeError, match='provenance'): evaluate(tmp_path/'bad_info')
    assert not (tmp_path/'bad_info'/'summary.txt').exists()
