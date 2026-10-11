"""Focused diagnosis tests; no synthetic result claimed as nuScenes quality."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.joint_occ_experiments.diagnostics import audit
from tools.joint_occ_experiments.diagnostics.run import discover_checkpoint, model_digest, metric_state, restore_metrics
from tools.joint_occ_experiments.engine import train_arm, forward_window, hard_predictions
from tools.joint_occ_experiments.model import PROTOCOL
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import Metrics
from test_joint_occ_ego import fixture


def test_gradient_direction_zero_and_actual_weighted_norms():
    named = [('residual_head.weight', torch.nn.Parameter(torch.zeros(2))),
             ('existence_head.bias', torch.nn.Parameter(torch.zeros(1)))]
    result = audit.compare_gradients(named, [torch.tensor([2., 0.]), None],
                                    [torch.tensor([-1., 0.]), torch.tensor([3.])])
    assert result['XY_head']['cosine'] == -1
    assert result['XY_head']['occupancy_to_motion_norm_ratio'] == 2
    assert result['existence_head']['cosine'] is None
    assert result['WM_all']['cosine'] == pytest.approx(-1/np.sqrt(10))


def test_nonfinite_gradients_are_errors_not_reported_as_missing_zero_cosine():
    named = [('residual_head.bias', torch.nn.Parameter(torch.zeros(1)))]
    with pytest.raises(RuntimeError, match='nonfinite'):
        audit.compare_gradients(named, [torch.tensor([float('nan')])], [None])


def test_actual_six_horizon_separated_gradients_sum_to_training_wm_gradient(monkeypatch):
    model, optimizer, provider, row = fixture()
    before, rng = copy.deepcopy(model.state_dict()), torch.get_rng_state()
    captured = {}
    original = audit.compare_gradients
    def capture(named, occ, motion):
        captured['sum'] = {name: (torch.zeros_like(p) if a is None else a)+(torch.zeros_like(p) if b is None else b)
                           for (name, p), a, b in zip(named, occ, motion)}
        return original(named, occ, motion)
    monkeypatch.setattr(audit, 'compare_gradients', capture)
    result = audit.gradient_batch(model, provider, [row, row], object_weight=.7, lovasz_weight=.8)
    assert all(p.grad is None for p in model.parameters())
    for name, value in model.state_dict().items(): torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    original_clip = torch.nn.utils.clip_grad_norm_
    def clip(parameters, *args, **kwargs):
        captured['train'] = {name: torch.zeros_like(p) if p.grad is None else p.grad.clone()
                             for name, p in model.transport.named_parameters()}
        return original_clip(parameters, *args, **kwargs)
    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', clip)
    monkeypatch.setattr(optimizer, 'step', lambda: None)
    torch.set_rng_state(rng)
    stats = train_arm(model, optimizer, provider, [row, row], 'gt', update=1, total_updates=3,
                      object_weight=.7, lovasz_weight=.8)
    for name in captured['sum']:
        torch.testing.assert_close(captured['sum'][name], captured['train'][name], rtol=1e-4, atol=2e-6, msg=name)
    assert result['losses']['CE'] == pytest.approx(stats['CE'], rel=1e-5)
    assert result['groups']['WM_all']['occupancy_norm'] > 0
    assert result['groups']['WM_all']['motion_norm'] > 0


def test_remove_funnel_unique_voxels_duplicates_and_new_add_protection():
    # Duplicate REMOVE rows must not inflate the number of removed voxels.
    baseline = np.array([17, 4, 4, 11], np.uint8)
    added = np.array([11, 4, 4, 11], np.uint8)
    both = np.array([11, 17, 4, 11], np.uint8)
    target = np.array([17, 17, 4, 13], np.uint8)
    legal = np.zeros((4, 6, 2), bool); legal[0, :, 0] = True; legal[1:, :, 1] = True
    flat = np.tile(np.array([0, 1, 1, 2])[:, None], (1, 6))
    fallback = np.full((4, 6), 17)
    actions = torch.zeros(4, 6, 2); actions[0, :, 0] = .8
    actions[1, :, 1] = .96; actions[2, :, 1] = .7; actions[3, :, 1] = .4
    fw = dict(evidence=SimpleNamespace(actor=np.array([-2, 0, 0, 1]), classes=np.array([11, 4, 4, 4])),
              plan=SimpleNamespace(flat=flat, legal=legal, fallback=fallback), actions=actions)
    outputs = {k: [v]*6 for k, v in dict(transport=baseline, add_only=added, add_remove=both).items()}
    result = audit.action_funnel(fw, outputs, target, 1)
    rm = result['actions']['REMOVE']['all']
    assert rm['rows'] == 3 and rm['unique_voxels'] == 2 and rm['correctable_unique_voxels'] == 1
    assert rm['accepted_rows'] == 1 and rm['accepted_unique_voxels'] == 1
    assert sum(rm['histogram']['correcting']) == 2
    assert sum(rm['histogram']['damaging']) == 1
    assert result['add_errors_not_undoable_by_remove'] == 1
    assert result['remove_changes_to_new_add_voxels'] == 0
    assert result['remove_incremental_quality']['corrected'] == 1
    assert result['add_quality']['damaged'] == 1
    merged = audit.merge_counts({}, result)
    audit.merge_counts(merged, result)
    assert merged['actions']['REMOVE']['all']['unique_voxels'] == 4  # per-horizon/window sum


def test_actual_soft_hard_routes_do_not_change_hard_forward_or_model():
    model, _, provider, row = fixture()
    model.eval(); digest = model_digest(model)
    with torch.no_grad(): expected = hard_predictions(forward_window(model, provider, row, 'gt', soft=False))
    predictions, funnels, gap = audit.evaluation_window(model, provider, row)
    assert set(predictions) == set(audit.ROUTES) and len(funnels) == 3
    for name, values in expected.items():
        for i, h in enumerate((1, 3, 5)): np.testing.assert_array_equal(values[h], predictions['hard_'+name][i])
    assert model_digest(model) == digest and all(p.grad is None for p in model.parameters())
    assert gap['elements'] > 0


def test_metric_integer_resume_and_shape_validation():
    metrics = {name: Metrics() for name in audit.ROUTES}
    prediction = np.array([17, 4, 11, 4]); target = np.array([17, 4, 17, 4])
    for m in metrics.values():
        for i in range(3): m.update(i, prediction, target, np.ones(4, bool))
    values = metric_state(metrics); resumed = restore_metrics(values)
    assert metric_state(resumed) == values
    values['hard_transport']['si'] = [1]
    with pytest.raises(RuntimeError, match='counts'): restore_metrics(values)


def test_discovery_selects_completed_gt_not_newer_partial_or_paired(tmp_path):
    def put(name, epoch, updates, modes=['gt']):
        directory = tmp_path/name; directory.mkdir()
        torch.save(dict(protocol=PROTOCOL, contract=dict(modes=modes, epochs=3, total_updates=100),
                        epoch=epoch, batch=0, updates=updates), directory/'last.pt')
    put('gt3_complete', 3, 100); put('gt3_partial', 1, 99); put('gt3_paired', 3, 100, ['gt', 'pred'])
    assert discover_checkpoint(tmp_path).parent.name == 'gt3_complete'


def test_real_cli_stop_resume_source_read_only_and_real_eval_session(monkeypatch, tmp_path):
    import json
    import threading
    from tools.joint_occ_experiments.diagnostics import run
    from tools.joint_occ_experiments.performance.probe import legacy_fingerprint
    from pathlib import Path
    model, _, provider, row = fixture()
    initializer = copy.deepcopy(model)
    paths = {name: tmp_path/(name+'.data') for name in ('config', 'train_cache', 'dev_cache',
        'population_manifest', 'base_checkpoint', 'train_info', 'dev_info')}
    for path in paths.values(): path.write_bytes(b'immutable source')
    train = [dict(row[0], t0_token='train'+str(i)) for i in range(4)]
    dev = [dict(row[0], scene_name='DEV', t0_token='dev'+str(i)) for i in range(64)]
    root = Path(run.__file__).resolve().parents[3]
    c = dict(modes=['gt'], training_backend='fused', model=model.configs(),
        implementation=legacy_fingerprint(root), seed=27, epochs=3, window_batch=4, source_budget=128,
        weights=[1., 1., 1.], thresholds=[.5, .95], total_updates=3,
        train_keys=[['TRAIN', r['t0_token']] for r in train], eval_keys=[['DEV', r['t0_token']] for r in dev],
        data={name: run.legacy.sha256(path) for name, path in paths.items() if name not in ('config', 'population_manifest')},
        cache_namespace=dict(train='TRAIN_NAMESPACE', val='VAL_NAMESPACE'),
        execution=dict(torch_version=str(torch.__version__), cuda_version=torch.version.cuda))
    source = tmp_path/'gt3_complete'; source.mkdir()
    torch.save(dict(protocol=PROTOCOL, contract=c, epoch=3, batch=0, updates=3,
                    models={'gt': model.state_dict()}, optimizers={'gt': 'UNTOUCHED'}, rng='UNTOUCHED'), source/'last.pt')
    original = (source/'last.pt').read_bytes()
    monkeypatch.setattr(run.legacy, 'require_cuda', lambda _: torch.device('cpu'))
    monkeypatch.setattr(run, 'JointOccEgo', lambda config, width: copy.deepcopy(initializer))
    monkeypatch.setattr(run.legacy, 'PilotProvider', lambda *args: provider)
    monkeypatch.setattr(run.legacy, 'load_runtime_config', lambda *args: None)
    monkeypatch.setattr(run.legacy, 'make_prepare_config', lambda _: provider.pcfg)
    monkeypatch.setattr(run.legacy, 'CachedColumnSource', lambda *args, **kwargs: SimpleNamespace(nusc=None))
    def source_reader(dataroot, *, info_pkl):
        assert info_pkl in (str(paths['train_info']), str(paths['dev_info']))
    monkeypatch.setattr(run.legacy, 'NuScenesWindowSource', source_reader)
    monkeypatch.setattr(run.legacy, 'load_cache', lambda path: ({}, train if path == str(paths['train_cache']) else dev))
    monkeypatch.setattr(run.legacy, 'load_manifest', lambda _: ({}, c['eval_keys'], None))
    monkeypatch.setattr(run, 'attach_caches', lambda *args: dict(train={'namespace': 'TRAIN_NAMESPACE'}, val={'namespace': 'VAL_NAMESPACE'}))
    monkeypatch.setattr(run, 'prefetched_batches', lambda p, s, jobs, d, **kw:
        ([ (r, *row[1:]) for r in group ] for group in jobs))
    class Execution:
        def __init__(self, *args): pass
        def close(self): pass
        def map(self, evidence, prep, grid):
            from real_motion.canonical_causal_repair import map_canonical_evidence
            return map_canonical_evidence(evidence, prep, grid)
    import real_motion.canonical_repair_execution as canonical
    import real_motion.surface_projection_execution as surface
    import real_motion.strong_majority_execution as majority
    monkeypatch.setattr(canonical, 'CanonicalCpuExecution', Execution)
    monkeypatch.setattr(surface, 'SurfaceMapExecution', Execution)
    monkeypatch.setattr(majority, 'ParallelNativeMajority', Execution)
    monkeypatch.setattr(run, 'gt_moving_support_sequence', lambda *args, **kwargs: None)
    monkeypatch.setattr(run, 'moving_support_masks', lambda *_: np.ones_like(row[1]['future_gt_occ'], bool))
    stopped = threading.Event()
    original_evaluate = run.evaluation_window
    def evaluate_and_stop(*args, **kwargs):
        result = original_evaluate(*args, **kwargs)
        stopped.set()
        return result
    monkeypatch.setattr(run, 'evaluation_window', evaluate_and_stop)
    out = tmp_path/'diagnostic'
    args = [item for name, path in paths.items() for item in ('--'+name.replace('_', '-'), str(path))]
    args += ['--dataroot', str(tmp_path), '--ccr-history-cache', str(tmp_path/'cache_train'),
        '--ccr-val-history-cache', str(tmp_path/'cache_val'), '--resume', str(source/'last.pt'),
        '--out-dir', str(out), '--mode', 'gt', '--eval-population', 'dev64', '--train-windows', '4']
    assert run.main(args, stopped) == 0
    state = json.loads((out/'state.json').read_text())
    assert state['dev_windows'] == 1 and len(state['gradients']) >= 1
    first_counts = state['metric_counts']
    monkeypatch.setattr(run, 'evaluation_window', original_evaluate)
    assert run.main(args+['--resume-audit']) == 0
    result = json.loads((out/'evaluation.json').read_text())
    assert result['status'] == 'complete' and result['dev_windows'] == 64 and result['scientific_updates'] == 0
    final_counts = json.loads((out/'state.json').read_text())['metric_counts']
    for name in audit.ROUTES:
        for key in first_counts[name]:
            np.testing.assert_array_equal(final_counts[name][key], np.asarray(first_counts[name][key])*64)
    assert (source/'last.pt').read_bytes() == original == (out/'source_snapshot.pt').read_bytes()
    assert not (out/'last.pt').exists()
