"""Full-unfreeze real gradients and single-arm original-GT checkpoint workflow."""
import copy
import json
import threading
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.stc_noise_full import run
from tools.stc_noise_adaptation import run as frozen, data
from tools.joint_occ_experiments import run as legacy
from tools.shared_history_experiments.full_gt import engine as gt
from test_perception_robustness import real_fixture
from test_stc_noise_adaptation import profile_files


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_loaded_ORIGINAL_GT_body_updates_encoder_STWM_and_refiner(device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    old, _, p, row = real_fixture(device); torch.set_num_threads(1)
    with torch.no_grad(): old.columns.readout[4].weight.normal_(0., .01)
    source = dict(contract={'model': old.configs()}, models={'gt': copy.deepcopy(old.state_dict())})
    model = run.FullBody(old.transport.config, source, 'originalGT').to(device); p.joint = model
    model.eval().requires_grad_(False).requires_grad_(True).train()
    assert model.transport.training and all(p.requires_grad for p in model.parameters())
    before = copy.deepcopy(model.state_dict())
    for k, v in before.items(): torch.testing.assert_close(v, source['models']['gt'][k], atol=0, rtol=0)
    adam = torch.optim.AdamW([dict(params=model.transport.parameters(), lr=1e-5, initial_lr=1e-5),
                             dict(params=model.columns.parameters(), lr=1e-4, initial_lr=1e-4)])
    rec = dict(row[0], future_displacement_xy_m=object(), future_yaw_delta_rad=object())
    row = rec, row[1], (row[2][0], object(), object()), row[3]
    stats = gt.train_arm(model, adam, p, [row], 'gt', update=1, total_updates=500, probe=True)
    assert stats['CE'] > 0 and not stats['auxiliary_computed']
    assert stats['gradient_links']['occupancy_to_shared_history'] > 0
    assert stats['gradient_links']['occupancy_to_xy'] > 0
    after = model.state_dict()
    for prefix in ('transport.history_encoder.', 'transport.decoder.', 'columns.'):
        assert any(not torch.equal(before[k], after[k]) for k in before if k.startswith(prefix)), prefix
    for k, v in source['models']['gt'].items(): torch.testing.assert_close(v, old.state_dict()[k], atol=0, rtol=0)


def test_actual_single_arm_cli_fresh_epoch2_same64_and_Adam_resume(tmp_path, monkeypatch):
    model, _, p, row = real_fixture(); torch.set_num_threads(1)
    with torch.no_grad(): model.columns.readout[4].weight.normal_(0., .01)
    paths = {k: tmp_path/k for k in ('config', 'train-cache', 'dev-cache', 'population-manifest',
                                   'base-checkpoint', 'train-info', 'dev-info')}
    for path in paths.values(): path.write_text('metadata')
    torch.save({'model_config': asdict(model.transport.config)}, paths['base-checkpoint'])
    train = [{**row[0], 't0_token': f'train{i}'} for i in range(20430)]
    dev = [{**row[0], 'scene_name': 'DEV', 't0_token': f'dev{i}'} for i in range(4369)]
    parent_keys = [('DEV', r['t0_token']) for r in dev[:512]]
    monkeypatch.setattr(legacy, 'require_cuda', lambda _: torch.device('cpu'))
    monkeypatch.setattr(legacy, 'load_cache', lambda path: ({}, train if path == str(paths['train-cache']) else dev))
    monkeypatch.setattr(legacy, 'load_manifest', lambda _: ({'parent_keys': parent_keys}, parent_keys[:64], None))
    source_contract = dict(protocol='shared_semantic_refiner_full_gt_ce_lovasz_2epoch_v1', modes=['gt'], epochs=2,
        total_updates=10292, train_keys=[['TRAIN', str(i)] for i in range(20430)], eval_keys=parent_keys,
        window_batch=4, source_budget=128, model=model.configs(),
        data={k.replace('-', '_'): legacy.sha256(v) for k, v in paths.items()
              if k in ('train-cache', 'dev-cache', 'train-info', 'dev-info', 'base-checkpoint')})
    source = dict(protocol=source_contract['protocol'], contract=source_contract, epoch=2, batch=0,
                  updates=10292, models={'gt': copy.deepcopy(model.state_dict())})
    parent = tmp_path/'reference'; (parent/'empirical/frozen').mkdir(parents=True)
    gtpath = parent/'empirical/frozen/GTepoch2.pt'; torch.save(source, gtpath)
    profile_files(parent/'profile', legacy.sha256(gtpath))
    noise, binding = run.load_model(parent/'profile/noise_model.json')
    stc = tmp_path/'stc'
    for token in row[0]['history_tokens']:
        path = stc/'DEV'/token/'labels.npz'; path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, semantics=row[1]['history_occ'][0])
    cli = [v for k, path in paths.items() for v in ('--'+k, str(path))]
    cli += ['--dataroot', str(tmp_path), '--ccr-history-cache', str(tmp_path), '--ccr-val-history-cache', str(tmp_path),
        '--gt-checkpoint', str(gtpath), '--noise-model', str(parent/'profile/noise_model.json'), '--stc-root', str(stc),
        '--out-dir', str(parent), '--augmentation', 'clean']
    fields = vars(frozen.parser().parse_args(cli)).copy()
    for key in ('augmentation', 'out_dir', 'max_updates'): fields.pop(key)
    keys, inventory = frozen.selection(SimpleNamespace(**fields), noise)
    workflow = dict(args=fields, STC_inventory=inventory)
    (parent/'workflow.json').write_text(json.dumps(workflow))
    contract = dict(protocol=frozen.PROTOCOL, total_updates=500, eval_keys=keys,
                    source_GT_sha256=legacy.sha256(gtpath), noise_identity=binding)
    metrics = {f'{m}_{r}': dict(mIoU=20., IoU=30., MovingMicro=10.)
               for m in ('occ_gt', 'stc_gt') for r in ('transport', 'semantic_refine')}
    for arm in ('clean', 'empirical'):
        path = parent/arm/'results.json'; path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(dict(status='complete', updates=500, contract=dict(contract, augmentation=arm),
            evaluation=dict(windows=64, evaluation_keys=keys, variants={'gt': metrics}))))
    before = {str(x): x.read_bytes() for x in parent.rglob('*') if x.is_file()}
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
    def caches(provider, *a):
        provider.ccr_history_cache = provider.ccr_val_history_cache = SimpleNamespace(stats=lambda: {}, close=lambda: None)
        return {'train': {'namespace': 'train'}, 'val': {'namespace': 'val'}}
    monkeypatch.setattr(legacy, 'attach_caches', caches)
    monkeypatch.setattr(legacy, 'epoch_groups', lambda records, *a: [[records[i]] for i in range(600)])
    def prefetch(provider, src, groups, descriptors, **kw):
        for group in groups: yield [(r, row[1], row[2], descriptors.get(r, row[1])) for r in group]
    monkeypatch.setattr(legacy, 'prefetched_batches', prefetch)
    monkeypatch.setattr(data, 'augment_row', lambda p, row, *a: row)
    out = tmp_path/'new_full'
    args = ['--reference-workflow', str(parent), '--out-dir', str(out)]
    originals = legacy.JointOccEgo, legacy.epoch_groups, legacy.parser, run.joint_data.ego_inputs
    assert run.main(threading.Event(), args+['--max-updates', '1']) == 0
    checkpoint = torch.load(out/'train/last.pt', weights_only=False)
    assert checkpoint['updates'] == 1 and checkpoint['contract']['total_updates'] == 500
    assert checkpoint['contract']['eval_keys'] == keys and checkpoint['contract']['fully_unfrozen']
    assert checkpoint['contract']['model']['objective'] == dict(ce=1., lovasz=1., motion=0., existence=0., ego=0.)
    for prefix in ('transport.history_encoder.', 'transport.decoder.', 'columns.'):
        assert any(not torch.equal(source['models']['gt'][k], checkpoint['models']['gt'][k])
                   for k in source['models']['gt'] if k.startswith(prefix))
    assert run.main(threading.Event(), args+['--resume-experiment', '--max-updates', '2']) == 0
    restored = torch.load(out/'train/last.pt', weights_only=False)
    assert restored['updates'] == 2 and restored['batch'] == 2
    assert originals == (legacy.JointOccEgo, legacy.epoch_groups, legacy.parser, run.joint_data.ego_inputs)
    assert before == {str(x): x.read_bytes() for x in parent.rglob('*') if x.is_file()}
    assert json.loads((out/'trainable_audit.json').read_text())['fully_trainable']
    assert json.loads((out/'initialization_audit.json').read_text())['imported_adapted_weights'] is False
    text = run.summary(dict(status='complete', updates=500, evaluation={'variants': {'gt': metrics}}), run.references(parent)[2])
    assert 'full_minus_empirical' in text and 'NOT a full-unfreeze clean control' in text
    with pytest.raises(RuntimeError, match='NEW directory'): run.main(argv=args)
    with pytest.raises(ValueError, match='OUTSIDE'): run.main(argv=['--reference-workflow', str(parent), '--out-dir', str(parent/'bad')])
