"""Small replay: no corruption, real full-body gradients, original-GT CLI resume."""
import copy
import json
import threading
from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.stc_clean_full import run
from tools.stc_noise_adaptation import data, run as frozen
from tools.joint_occ_experiments import run as legacy
from tools.shared_history_experiments.full_gt import engine as gt
from test_perception_robustness import real_fixture


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_clean_full_real_gradients(device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    old, _, provider, row = real_fixture(device); torch.set_num_threads(1)
    with torch.no_grad(): old.columns.readout[4].weight.normal_(0., .01)
    source = dict(contract={'model': old.configs()}, models={'gt': copy.deepcopy(old.state_dict())})
    model = run.CleanFullBody(old.transport.config, source, 'GTepoch2').to(device); provider.joint = model
    assert all(p.requires_grad for p in model.parameters())
    assert model.configs()['adaptation']['protocol'] == run.PROTOCOL
    before = copy.deepcopy(model.state_dict())
    adam = torch.optim.AdamW([dict(params=model.transport.parameters(), lr=1e-5, initial_lr=1e-5),
                             dict(params=model.columns.parameters(), lr=1e-4, initial_lr=1e-4)])
    stats = gt.train_arm(model, adam, provider, [row], 'gt', update=1, total_updates=500, probe=True)
    assert not stats['auxiliary_computed'] and stats['CE'] > 0
    assert stats['gradient_links']['occupancy_to_shared_history'] > 0
    assert stats['gradient_links']['occupancy_to_xy'] > 0
    after = model.state_dict()
    for prefix in ('transport.history_encoder.', 'transport.decoder.', 'columns.'):
        assert any(not torch.equal(before[k], after[k]) for k in before if k.startswith(prefix))
    for k, v in source['models']['gt'].items(): torch.testing.assert_close(v, old.state_dict()[k], atol=0, rtol=0)


def test_clean_dense_prefetch_never_calls_simulator(monkeypatch):
    _, _, provider, row = real_fixture(); provider.ccr_history_cache_source = object()
    raw = dict(row[1]); raw['history_observed'] = np.zeros_like(raw['history_occ'], dtype=bool)
    raw['history_occ'] = raw['history_occ'].copy(); raw['history_occ'][0, 0, 0, 0] = 255
    original = raw['history_occ'].copy()
    def forbidden(*a, **kw): raise AssertionError('clean experiment invoked simulator')
    monkeypatch.setattr(data, 'augment_row', forbidden); monkeypatch.setattr(data, 'corrupt', forbidden)
    def base(*a, **kw): yield [(row[0], raw, row[2], row[3])]
    batches = run.prefetch_adapter(base, 'clean', None, 21)
    result = list(batches(provider, provider.ccr_history_cache_source, [], None, workers=1))[0][0]
    np.testing.assert_array_equal(result[1]['history_occ'], original)
    np.testing.assert_array_equal(result[1]['history_observed'], original < 18)
    assert not raw['history_observed'].any() and '_empirical_noise' not in result[1]


def test_actual_cli_original_source_clean_full_and_resume(tmp_path, monkeypatch):
    model, _, provider, row = real_fixture(); torch.set_num_threads(1)
    with torch.no_grad(): model.columns.readout[4].weight.normal_(0., .01)
    paths = {k: tmp_path/k for k in ('config', 'train-cache', 'dev-cache', 'population-manifest',
                                   'base-checkpoint', 'train-info', 'dev-info')}
    for path in paths.values(): path.write_text('metadata')
    torch.save({'model_config': asdict(model.transport.config)}, paths['base-checkpoint'])
    train = [{**row[0], 't0_token': f'train{i}'} for i in range(20430)]
    dev = [{**row[0], 'scene_name': 'DEV', 't0_token': f'dev{i}'} for i in range(4369)]
    keys = [['DEV', r['t0_token']] for r in dev[:64]]
    parent_keys = [('DEV', r['t0_token']) for r in dev[:512]]
    monkeypatch.setattr(legacy, 'require_cuda', lambda _: torch.device('cpu'))
    monkeypatch.setattr(legacy, 'load_cache', lambda path: ({}, train if path == str(paths['train-cache']) else dev))
    monkeypatch.setattr(legacy, 'load_manifest', lambda _: ({'parent_keys': parent_keys}, parent_keys[:64], None))
    contract = dict(protocol='shared_semantic_refiner_full_gt_ce_lovasz_2epoch_v1', modes=['gt'], epochs=2,
        total_updates=10292, train_keys=[['TRAIN', str(i)] for i in range(20430)], eval_keys=parent_keys,
        window_batch=4, source_budget=128, model=model.configs(),
        data={k.replace('-', '_'): legacy.sha256(v) for k, v in paths.items()
              if k in ('train-cache', 'dev-cache', 'train-info', 'dev-info', 'base-checkpoint')})
    source = dict(protocol=contract['protocol'], contract=contract, epoch=2, batch=0,
                  updates=10292, models={'gt': copy.deepcopy(model.state_dict())})
    parent = tmp_path/'reference'; (parent/'clean/frozen').mkdir(parents=True)
    source_path = parent/'clean/frozen/GTepoch2.pt'; torch.save(source, source_path)
    # No profile/noise_model.json exists: only previous population/provenance is reused.
    stc = tmp_path/'stc'; inventory = []
    for token in row[0]['history_tokens']:
        path = stc/'DEV'/token/'labels.npz'; path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, semantics=row[1]['history_occ'][0]); stat = path.stat()
        inventory.append([str(path), stat.st_size, stat.st_mtime_ns])
    cli = [v for k, path in paths.items() for v in ('--'+k, str(path))]
    cli += ['--dataroot', str(tmp_path), '--ccr-history-cache', str(tmp_path), '--ccr-val-history-cache', str(tmp_path),
        '--gt-checkpoint', str(source_path), '--noise-model', 'DO_NOT_READ', '--stc-root', str(stc),
        '--out-dir', str(parent), '--augmentation', 'clean']
    fields = vars(frozen.parser().parse_args(cli)).copy()
    for key in ('augmentation', 'out_dir', 'max_updates'): fields.pop(key)
    (parent/'workflow.json').write_text(json.dumps(dict(args=fields, STC_inventory=inventory)))
    previous = dict(protocol=frozen.PROTOCOL, total_updates=500, eval_keys=keys,
                    source_GT_sha256=legacy.sha256(source_path))
    metrics = {f'{m}_{r}': dict(mIoU=20., IoU=30., MovingMicro=10.)
               for m in ('occ_gt', 'stc_gt') for r in ('transport', 'semantic_refine')}
    for arm in ('clean', 'empirical'):
        path = parent/arm/'results.json'; path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(dict(status='complete', updates=500, contract=dict(previous, augmentation=arm),
            evaluation=dict(windows=64, evaluation_keys=keys, variants={'gt': metrics}))))
    before = {str(x): x.read_bytes() for x in parent.rglob('*') if x.is_file()}
    monkeypatch.setattr(legacy, 'load_runtime_config', lambda *a: None)
    monkeypatch.setattr(legacy, 'make_prepare_config', lambda _: provider.pcfg)
    def make_provider(*args): provider.joint = args[-2]; return provider
    monkeypatch.setattr(legacy, 'PilotProvider', make_provider)
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
    monkeypatch.setattr(legacy, 'epoch_groups', lambda records, *a: [[records[i]] for i in range(600)])
    def prefetch(p, src, groups, descriptors, **kw):
        for group in groups: yield [(r, row[1], row[2], descriptors.get(r, row[1])) for r in group]
    monkeypatch.setattr(legacy, 'prefetched_batches', prefetch)
    monkeypatch.setattr(data, 'augment_row', lambda *a, **kw: pytest.fail('must not corrupt'))
    out = tmp_path/'new_clean_full'; args = ['--reference-workflow', str(parent), '--out-dir', str(out)]
    originals = legacy.JointOccEgo, legacy.epoch_groups, legacy.prefetched_batches, joint_data_hook()
    assert run.main(threading.Event(), args+['--max-updates', '1']) == 0
    checkpoint = torch.load(out/'train/last.pt', weights_only=False)
    assert checkpoint['contract']['augmentation'] == 'clean' and checkpoint['contract']['fully_unfrozen']
    assert checkpoint['contract']['total_updates'] == 500 and checkpoint['contract']['eval_keys'] == keys
    assert checkpoint['contract']['model']['objective'] == dict(ce=1., lovasz=1., motion=0., existence=0., ego=0.)
    for prefix in ('transport.history_encoder.', 'transport.decoder.', 'columns.'):
        assert any(not torch.equal(source['models']['gt'][k], checkpoint['models']['gt'][k])
                   for k in source['models']['gt'] if k.startswith(prefix))
    assert run.main(threading.Event(), args+['--resume-experiment', '--max-updates', '2']) == 0
    restored = torch.load(out/'train/last.pt', weights_only=False)
    assert restored['updates'] == 2 and restored['batch'] == 2
    assert originals == (legacy.JointOccEgo, legacy.epoch_groups, legacy.prefetched_batches, joint_data_hook())
    assert before == {str(x): x.read_bytes() for x in parent.rglob('*') if x.is_file()}
    text = run.summary(dict(status='complete', updates=500, evaluation={'variants': {'gt': metrics}}), run.references(parent)[2])
    assert 'full_clean_minus_clean' in text and 'vs clean tests trainable scope' in text
    with pytest.raises(RuntimeError, match='NEW directory'): run.main(argv=args)


def joint_data_hook():
    from tools.joint_occ_experiments import data as joint_data
    return joint_data.ego_inputs
