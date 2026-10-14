"""CPU/CUDA small replay and scoped workflow tests, not real STC quality."""
import copy
import json
import threading
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.stc_noise_adaptation import augmentation as aug, data, run, workflow
from tools.stc_noise_profile.data import fingerprint, file_sha
from tools.joint_occ_experiments import run as legacy
from tools.shared_history_experiments.full_gt import engine as gt
from tools.perception_robustness import evaluation
from test_perception_robustness import real_fixture, history_fixture


def empirical_value():
    p = np.repeat(np.eye(18)[None], 6, axis=0)
    for cid in range(17):
        p[:, cid, cid] = .65; p[:, cid, 17] = .3; p[:, cid, (cid+1)%17] = .05
    p[:, 17, 17] = .97; p[:, 17, 11] = .03
    pattern = [0]*16; pattern[15] = 1
    return dict(schema='STC_empirical_error_statistics_v1', simulator_implemented=False,
        fitting_split=dict(fit_scenes=['FIT'], remaining_scenes=['DEV']+[f'DEV{i}' for i in range(148)]),
        conditional_pred_given_GT_and_distance=p.tolist(), condition_counts=np.ones((6, 18), np.int64).tolist(),
        component_joint={f: {'11,0,2,2,1,4': 5} for f in ('false_occupancy', 'missed_occupancy', 'wrong_class')},
        temporal=dict(static_patterns={f: pattern for f in ('false_occupancy', 'missed_occupancy', 'wrong_class')},
                      dynamic=dict(missing_STC_patterns=pattern)),
        boundary_distance_voxel_hist=[0, 10, 2, 0, 0, 0])


def profile_files(root, digest):
    root.mkdir(parents=True)
    value = empirical_value(); split = value['fitting_split']
    contract = dict(source_model_declared_sha256=digest)
    value['contract_sha256'] = fingerprint(contract)
    profile = dict(status='complete', protocol='STC_VAL_development_error_fit_four_history_v1', contract=contract,
        split=dict(split, seed_DEV64_keys=[]), statistics={'families': {'false_occupancy':
            {'boundary_distance_voxel_hist': value.pop('boundary_distance_voxel_hist')}}})
    (root/'noise_model.json').write_text(json.dumps(value))
    (root/'evaluation.json').write_text(json.dumps(profile))
    return root/'noise_model.json'


def test_causal_deterministic_spatial_temporal_noise_and_global_rng():
    raw, record, grid = history_fixture(); model = aug.NoiseModel(empirical_value())
    before = copy.deepcopy(raw['history_occ']); rng = np.random.get_state()
    a, audit = aug.corrupt(raw, record, grid, model, 21, force=True)
    b, other = aug.corrupt(raw, record, grid, model, 21, force=True)
    assert audit == other and sum(audit['changed_voxels']) > 0
    assert set(a) == {'history_occ', 'history_poses', 'history_observed'}
    np.testing.assert_array_equal(a['history_occ'], b['history_occ'])
    np.testing.assert_array_equal(raw['history_occ'], before)
    assert a['history_observed'].all() and not raw['history_observed'].all()
    for frame in a['history_occ']: np.testing.assert_array_equal(frame, a['history_occ'][0])
    assert a['history_occ'].max() <= 17
    assert not np.shares_memory(a['history_occ'], before)
    now = np.random.get_state(); np.testing.assert_array_equal(now[1], rng[1]); assert now[2:] == rng[2:]
    assert audit['generated_errors']['false_occupancy'][0] > 0
    # Synthetic FP remains near historical occupied boundaries, not random far-air blobs.
    from scipy.ndimage import distance_transform_edt
    fp = (before[0] == 17) & (a['history_occ'][0] < 17)
    assert distance_transform_edt(before[0] == 17, sampling=grid.voxel_size)[fp].max() < 1.6


def test_missing_support_invalid_labels_and_pattern_validation():
    raw, record, grid = history_fixture(); v = empirical_value()
    v['conditional_pred_given_GT_and_distance'] = np.zeros((6, 18, 18)).tolist()
    v['condition_counts'] = np.zeros((6, 18), np.int64).tolist()
    raw['history_occ'][:, 0, 0, 0] = 255
    result, audit = aug.corrupt(raw, record, grid, aug.NoiseModel(v), 21, force=True)
    np.testing.assert_array_equal(result['history_occ'], raw['history_occ'])
    assert not result['history_observed'][:, 0, 0, 0].any()
    assert audit['clean_geometry_reusable']
    v = empirical_value(); v['temporal']['static_patterns']['wrong_class'] = [0]*16
    with pytest.raises(ValueError, match='unsupported'): aug.NoiseModel(v)


def test_history_actor_registration_moves_the_same_random_field():
    from real_motion.geometry import OccupancyGrid
    grid = OccupancyGrid(x_min=0, y_min=0, z_min=0, voxel_size=(1., 1., 1.), shape_hwd=(8, 8, 4))
    labels = np.full((4, 8, 8, 4), 17, np.uint8); poses = np.repeat(np.eye(4)[None], 4, axis=0)
    cells = np.array([[2, 3, 1]])
    matrix = np.eye(4); matrix[0, 3] = 2.
    regs = [[(matrix, cells), None, None, None]]
    address, overlap = aug.history_addresses(labels, poses, regs,
        np.array([0., 0., 0.]), np.array(grid.voxel_size))
    assert address[0][2, 3, 1] == address[3][4, 3, 1] and overlap[0][2, 3, 1]


def test_profile_binding_and_remaining_population(tmp_path, monkeypatch):
    path = profile_files(tmp_path/'fit', 'GTsha'); noise, identity = aug.load_model(path)
    assert identity['source_GT_sha256'] == 'GTsha'
    records = [dict(scene_name='FIT', t0_token='excluded', history_tokens=['a']*4)]
    records += [dict(scene_name='DEV', t0_token=str(i), history_tokens=['a']*4) for i in range(100)]
    stc = tmp_path/'stc/DEV/a'; stc.mkdir(parents=True); (stc/'labels.npz').write_bytes(b'inventory')
    monkeypatch.setattr(legacy, 'load_cache', lambda _: ({}, records))
    a = SimpleNamespace(dev_cache='metadata', stc_root=tmp_path/'stc')
    keys, inventory = run.selection(a, noise)
    assert len(keys) == 64 and len(inventory) == 256 and all(k[0] == 'DEV' for k in keys)
    profile = json.loads(path.with_name('evaluation.json').read_text()); profile['contract']['source_model_declared_sha256'] = 'other'
    path.with_name('evaluation.json').write_text(json.dumps(profile))
    with pytest.raises(ValueError, match='identity'): aug.load_model(path)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_warm_refiner_only_real_backward_and_rebuilt_geometry(monkeypatch, device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    old, _, provider, original = real_fixture(device); torch.set_num_threads(1)
    source = dict(contract={'model': old.configs()}, models={'gt': copy.deepcopy(old.state_dict())})
    model = run.AdaptBody(old.transport.config, source, 'sha').to(device); provider.joint = model
    model.requires_grad_(False).requires_grad_(True).train()
    assert not model.transport.training and not any(p.requires_grad for p in model.transport.parameters())
    before = copy.deepcopy(model.state_dict())
    hist = {k: copy.deepcopy(original[1][k]) for k in ('history_occ', 'history_observed', 'history_poses')}
    hist['history_occ'][:, 2:4, 2:4, 1:3] = 4; hist['history_observed'][:] = True
    row = data.historical_row(provider, original[0], hist, original[2][0])
    raw = dict(row[1], future_poses=original[1]['future_poses'], future_gt_occ=original[1]['future_gt_occ'])
    row = row[0], raw, row[2], row[3]
    assert len(row[0]['source_class_id']) > 0
    def drop(historical, *a, **kw):
        assert set(historical) == {'history_occ', 'history_observed', 'history_poses'}
        out = copy.deepcopy(historical); out['history_occ'][out['history_occ'] == 4] = 17
        return out, dict(selected=True, changed_voxels=[8]*4, clean_geometry_reusable=False)
    monkeypatch.setattr(data, 'corrupt', drop)
    changed = data.augment_row(provider, row, None, 21)
    assert not len(changed[0]['source_class_id']) and len(row[0]['source_class_id']) > 0
    assert changed[1]['future_gt_occ'] is original[1]['future_gt_occ']
    ready = data.finish_backgrounds(provider, changed)
    assert len(ready[1]['_column_causal_preparation']['prepared_state']['column_backgrounds']) == 6
    adam = torch.optim.AdamW([dict(params=model.columns.parameters(), lr=1e-3, initial_lr=1e-3)])
    for update in (1, 2):
        stats = gt.train_arm(model, adam, provider, [ready], 'gt', update=update, total_updates=500)
    assert stats['CE'] > 0 and not stats['auxiliary_computed']
    for k, value in before.items():
        if k.startswith('transport.'): torch.testing.assert_close(model.state_dict()[k], value, rtol=0, atol=0)
    assert not torch.equal(before['columns.readout.4.weight'], model.state_dict()['columns.readout.4.weight'])


def test_dense_train_original_occ_eval_no_pollution(monkeypatch):
    model, _, p, row = real_fixture()
    original = row[1]; original['history_observed'] = np.zeros_like(original['history_observed'], bool)
    train_record = dict(row[0], scene_name='TRAIN', t0_token='train')
    eval_record = dict(row[0], scene_name='DEV', t0_token='eval')
    descriptor = data.history_inputs([('DEV', 'eval')])(p, ram_mib=0)
    observed = []
    monkeypatch.setattr(descriptor, 'build', lambda raw: observed.append(raw['history_observed'].copy()) or row[3])
    descriptor.get(train_record, original); descriptor.get(eval_record, original)
    assert observed[0].all() and not observed[1].any() and not original['history_observed'].any()
    train_source = object(); p.ccr_history_cache_source = train_source
    wrapped = data.prefetch_adapter(lambda *a, **kw: iter([[row]]), 'clean', None, 21)
    training = list(wrapped(p, train_source, None, None))[0][0]
    evaluation_row = list(wrapped(p, object(), None, None))[0][0]
    assert training[1]['history_observed'].all() and not evaluation_row[1]['history_observed'].any()


def test_paired_workflow_smoke_resume_identical_source_and_keys(tmp_path, monkeypatch):
    gtpath = tmp_path/'source/GT.pt'; gtpath.parent.mkdir(); gtpath.write_bytes(b'source GT')
    noise = profile_files(tmp_path/'noise', file_sha(gtpath))
    paths = {n: tmp_path/n for n in ('config', 'train-cache', 'dev-cache', 'population-manifest',
                                    'base-checkpoint', 'train-info', 'dev-info')}
    for path in paths.values(): path.write_text('metadata')
    keys = [['DEV', str(i)] for i in range(64)]
    monkeypatch.setattr(workflow, 'selection', lambda *a: (keys, []))
    out = tmp_path/'experiment'
    args = [v for k, path in paths.items() for v in ('--'+k, str(path))]
    args += ['--dataroot', str(tmp_path), '--ccr-history-cache', str(tmp_path), '--ccr-val-history-cache', str(tmp_path),
             '--stc-root', str(tmp_path), '--out-dir', str(out), '--gt-checkpoint', str(gtpath), '--noise-model', str(noise)]
    calls = []
    def child(command, check):
        calls.append(command); d = Path(command[command.index('--out-dir')+1]); d.mkdir(exist_ok=True)
        arm = command[command.index('--augmentation')+1]
        if '--max-updates' in command:
            (d/'train').mkdir(); (d/'train/last.pt').write_bytes(b'cursor')
            result = dict(status='stopped_at_complete_update')
        else:
            assert '--resume' in command
            contract = dict(protocol=run.PROTOCOL, total_updates=500, augmentation=arm, eval_keys=keys)
            m = dict(mIoU=20., IoU=30., MovingMicro=10.)
            result = dict(status='complete', updates=500, contract=contract,
                evaluation=dict(windows=64, evaluation_keys=keys, variants={'gt': {n: m for n in evaluation.ROUTES}}))
            (d/'initialization_audit.json').write_text(json.dumps({'same_initializer': True}))
        (d/'results.json').write_text(json.dumps(result)); return SimpleNamespace(returncode=0)
    assert workflow.main(argv=args+['--max-updates', '8'], launch=child) == 0 and len(calls) == 2
    assert workflow.main(argv=args+['--resume-workflow'], launch=child) == 0 and len(calls) == 4
    assert workflow.main(argv=args+['--resume-workflow'], launch=child) == 0 and len(calls) == 4
    assert json.loads((out/'comparison.json').read_text())['no_automatic_promotion']
    with pytest.raises(RuntimeError, match='NEW output'): workflow.main(argv=args, launch=child)
    with pytest.raises(ValueError): workflow.main(argv=args+['--epochs', '2'], launch=child)


@pytest.mark.parametrize('arm', ['clean', 'empirical'])
def test_real_warm_cli_stop_resume_scoped_hooks_and_frozen_body(tmp_path, monkeypatch, arm):
    old, _, provider, row = real_fixture(); torch.set_num_threads(1)
    paths = {n: tmp_path/n for n in ('config', 'train-cache', 'dev-cache', 'population-manifest',
                                    'base-checkpoint', 'train-info', 'dev-info')}
    for path in paths.values(): path.write_text('metadata')
    torch.save({'model_config': asdict(old.transport.config)}, paths['base-checkpoint'])
    train = [{**row[0], 't0_token': f'train{i}'} for i in range(20430)]
    dev = [{**row[0], 'scene_name': 'DEV', 't0_token': f'dev{i}'} for i in range(4369)]
    parent = [('DEV', r['t0_token']) for r in dev[:512]]
    source_contract = dict(protocol='shared_semantic_refiner_full_gt_ce_lovasz_2epoch_v1', modes=['gt'], epochs=2,
        total_updates=10292, train_keys=[['TRAIN', str(i)] for i in range(20430)], eval_keys=parent,
        window_batch=4, source_budget=128, model=old.configs(),
        data={k.replace('-', '_'): file_sha(p) for k, p in paths.items()
              if k in ('train-cache', 'dev-cache', 'train-info', 'dev-info', 'base-checkpoint')})
    source = dict(protocol=source_contract['protocol'], contract=source_contract, epoch=2, batch=0,
                  updates=10292, models={'gt': copy.deepcopy(old.state_dict())})
    gtpath = tmp_path/'source/epoch2.pt'; gtpath.parent.mkdir(); torch.save(source, gtpath)
    source_bytes = gtpath.read_bytes(); noise = profile_files(tmp_path/'noise', file_sha(gtpath))
    stc = tmp_path/'stc'
    for token in row[0]['history_tokens']:
        path = stc/'DEV'/token/'labels.npz'; path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, semantics=row[1]['history_occ'][0])
    monkeypatch.setattr(legacy, 'require_cuda', lambda _: torch.device('cpu'))
    monkeypatch.setattr(legacy, 'load_cache', lambda path: ({}, train if path == str(paths['train-cache']) else dev))
    monkeypatch.setattr(legacy, 'load_manifest', lambda _: ({'parent_keys': parent}, parent[:64], None))
    monkeypatch.setattr(legacy, 'load_runtime_config', lambda *a: None)
    monkeypatch.setattr(legacy, 'make_prepare_config', lambda _: provider.pcfg)
    def provider_factory(*a): provider.joint = a[-2]; return provider
    monkeypatch.setattr(legacy, 'PilotProvider', provider_factory)
    monkeypatch.setattr(legacy, 'NuScenesWindowSource', lambda root, *, info_pkl: SimpleNamespace(nusc=None))
    class Disposable:
        kernels = pool = None
        def __init__(self, *a): pass
        def close(self): pass
    monkeypatch.setattr('real_motion.canonical_repair_execution.CanonicalCpuExecution', Disposable)
    monkeypatch.setattr('real_motion.strong_majority_execution.ParallelNativeMajority', Disposable)
    def caches(p, *a):
        p.ccr_history_cache = p.ccr_val_history_cache = SimpleNamespace(stats=lambda: {}, close=lambda: None)
        return {'train': {'namespace': 'train'}, 'val': {'namespace': 'val'}}
    monkeypatch.setattr(legacy, 'attach_caches', caches)
    def groups(records, *a):
        assert len(records) == 4086
        return [[records[i]] for i in range(600)]
    monkeypatch.setattr(legacy, 'epoch_groups', groups)
    def prefetch(p, source, groups, descriptors, **kw):
        for group in groups:
            yield [(rec, row[1], row[2], descriptors.get(rec, row[1])) for rec in group]
    monkeypatch.setattr(legacy, 'prefetched_batches', prefetch)
    # Generator/rebuild tested above. This test isolates original training driver/cursors.
    monkeypatch.setattr(data, 'augment_row', lambda p, row, model, seed: row)
    out = tmp_path/'experiment'
    args = [v for k, path in paths.items() for v in ('--'+k, str(path))]
    args += ['--dataroot', str(tmp_path), '--ccr-history-cache', str(tmp_path), '--ccr-val-history-cache', str(tmp_path),
        '--stc-root', str(stc), '--out-dir', str(out), '--gt-checkpoint', str(gtpath), '--noise-model', str(noise),
        '--augmentation', arm]
    originals = legacy.parser, legacy.load_manifest, legacy.JointOccEgo, legacy.prefetched_batches, run.joint_data.ego_inputs
    assert run.main(threading.Event(), args+['--max-updates', '1']) == 0
    saved = torch.load(out/'train/last.pt', weights_only=False)
    assert saved['updates'] == 1 and saved['batch'] == 1 and saved['epoch'] == 0
    assert saved['contract']['total_updates'] == 500 and saved['contract']['complete_dataset_epochs'] == 0
    assert saved['contract']['model']['objective'] == dict(ce=1., lovasz=1., motion=0., existence=0., ego=0.)
    assert saved['contract']['thresholds'] is None and saved['contract']['augmentation'] == arm
    for k, value in source['models']['gt'].items():
        if k.startswith('transport.'): torch.testing.assert_close(saved['models']['gt'][k], value, rtol=0, atol=0)
    assert run.main(threading.Event(), args+['--resume', str(out/'train/last.pt'), '--max-updates', '2']) == 0
    resumed = torch.load(out/'train/last.pt', weights_only=False)
    assert resumed['updates'] == 2 and resumed['batch'] == 2
    assert gtpath.read_bytes() == source_bytes
    assert json.loads((out/'freeze_audit.json').read_text())['Encoder_STWM_unchanged']
    assert originals == (legacy.parser, legacy.load_manifest, legacy.JointOccEgo, legacy.prefetched_batches, run.joint_data.ego_inputs)
