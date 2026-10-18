"""Latest Integrated TWO+TWO full plan, cadence, actual recovery and model isolation."""
import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.source_scene_integrated_full import run
from tools.source_scene_integrated_full.model import checkpoint_model


def test_full_plan_all_unique_windows_each_epoch_and_previous_keys(tmp_path):
    train = [dict(scene_name=f'T{i//50}', t0_token=f't{i}', features=np.empty((i % 5, 0))) for i in range(20430)]
    dev = [dict(scene_name=f'V{i%150}', t0_token=f'v{i}', features=[]) for i in range(4369)]
    keys = [[r['scene_name'], r['t0_token']] for r in dev[:64]]
    ref = tmp_path/'keys.json'; ref.write_text(json.dumps(dict(contract=dict(eval_keys=keys))))
    selected, evaluation, groups = run.plan(train, dev, ref, 20261011)
    assert len(selected) == len(set(map(tuple, selected))) == 20430 and len(groups) == 2
    for epoch in groups:
        flat = [(r['scene_name'], r['t0_token']) for g in epoch for r in g]
        assert len(flat) == 20430 and set(flat) == set(map(tuple, selected))
    assert list(run.base.record_keys(evaluation)) == list(map(tuple, keys))
    assert [r['t0_token'] for g in groups[0] for r in g] != [r['t0_token'] for g in groups[1] for r in g]


def test_cursor_second_epoch_schedule_change_and_checkpoint_cadence():
    groups = [[1, 2], [3, 4, 5]]; c = dict(world_size=2, schedule='whole_2_epoch_cosine_floor0.01')
    state = dict(protocol=run.PROTOCOL, contract=c, epoch=1, batch=1, updates=3, rank_rng=[{}, {}])
    assert run.validate_resume(state, c, groups) == (1, 1, 3)
    assert run.validate_resume(dict(state, epoch=2, batch=0, updates=5), c, groups) == (2, 0, 5)
    for bad in (dict(state, updates=4), dict(state, protocol='stage2_pilot'), dict(state, epoch=2)):
        with pytest.raises(RuntimeError): run.validate_resume(bad, c, groups)
    with pytest.raises(RuntimeError): run.validate_resume(state, dict(c, schedule='restarted_cosine'), groups)
    assert not run.evaluation_due(0, 0) and not run.evaluation_due(1999, 1)
    assert run.evaluation_due(2000, 1) and run.evaluation_due(4000, 1)
    assert run.evaluation_due(5145, 0) and run.evaluation_due(10292, 0)
    assert run.checkpoint_names(2000, 0, 20) == ['step_0002000.pt']
    assert run.checkpoint_names(5145, 1, 0) == ['epoch_0001.pt']
    assert run.checkpoint_names(4000, 1, 0) == ['step_0004000.pt', 'epoch_0001.pt']
    assert run.checkpoint_names(1999, 0, 10) == []


def test_actual_cuda_two_epoch_stop_resume_and_epoch_evaluations(tmp_path, monkeypatch, capsys):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): pytest.skip('BF16 CUDA')
    from dataclasses import asdict, replace
    from test_source_scene_stage2 import fixture
    from tools.source_scene_integrated_full.model import make_model
    model, runner, row = fixture()
    try:
        paths = {k: tmp_path/k for k in ('config', 'train_cache', 'dev_cache', 'base_checkpoint', 'train_info', 'dev_info', 'eval_reference', 'population_manifest')}
        for path in paths.values(): path.write_text('fixture')
        cfg = replace(model.transport.config, decoder_blocks=2)
        torch.save(dict(model_config=asdict(cfg)), paths['base_checkpoint'])
        stc = tmp_path/'stc'
        for token, sem in zip(row[0]['history_tokens'], row[1]['history_occ']):
            directory = stc/row[0]['scene_name']/str(token); directory.mkdir(parents=True)
            np.savez(directory/'labels.npz', semantics=sem)
        evaluation = [dict(row[0], t0_token='dev'+str(i)) for i in range(64)]
        class Provider:
            def __init__(self, *args):
                self.pcfg, self.strong = runner.provider.pcfg, runner.provider.strong
                self.device, self.workers, self.joint = args[3], 1, args[5]
            def attach_tracks(self, root, **kwargs): assert root.parent == tmp_path/'new'; return {}
            def close_tracks(self): pass
            def load_raw_columns(self, *args, **kwargs): return row[1]
        timestamps = {str(t): int(i*500000) for i, t in enumerate(row[0]['history_tokens'])}
        source = SimpleNamespace(nusc=SimpleNamespace(get=lambda _, t: {'timestamp': timestamps[str(t)]}))
        def caches(provider, *args):
            provider.ccr_history_cache = SimpleNamespace(close=lambda: None)
            provider.ccr_val_history_cache = SimpleNamespace(close=lambda: None)
            return {}
        monkeypatch.setattr(run, 'DirectProvider', Provider)
        monkeypatch.setattr(run, 'make_model', lambda cfg: make_model(cfg, feature_width=16, z_bins=4, memory_width=4, width=8, chunk_size=32))
        monkeypatch.setattr(run.base, 'attach_caches', caches)
        monkeypatch.setattr(run.base, 'load_cache', lambda *a: ({}, []))
        monkeypatch.setattr(run.base, 'load_runtime_config', lambda *a: {})
        monkeypatch.setattr(run.base, 'make_prepare_config', lambda *a: runner.provider.pcfg)
        monkeypatch.setattr(run.base, 'NuScenesWindowSource', lambda *a, **k: source)
        monkeypatch.setattr(run.base, 'CachedColumnSource', lambda *a, **k: source)
        monkeypatch.setattr(run, 'plan', lambda *a: ([['T', 't']], evaluation, [[[row[0]], [row[0]]], [[row[0]]]]))
        calls = []
        def evaluate(*args, **kwargs):
            calls.append(1)
            return dict(windows=64, keys=[list(k) for k in run.base.record_keys(evaluation)], variants={
                name: dict(mIoU=40., IoU=50., MovingMicro=20.) for name in
                ('occ_gt_transport', 'occ_gt_semantic_refine', 'stc_gt_transport', 'stc_gt_semantic_refine')})
        monkeypatch.setattr(run, 'evaluate', evaluate)
        out = tmp_path/'new'
        argv = ['--out-dir', str(out), '--stc-root', str(stc), '--dataroot', str(tmp_path),
                '--ccr-history-cache', str(tmp_path), '--ccr-val-history-cache', str(tmp_path),
                '--device', 'cuda', '--cpu-workers', '1', '--prefetch-workers', '1']
        for k, path in paths.items(): argv += ['--'+k.replace('_', '-'), str(path)]
        run.main(argv+['--max-updates', '2'])
        partial = torch.load(out/'last.pt', weights_only=False)
        assert partial['epoch'] == 1 and partial['updates'] == 2 and (out/'epoch_0001.pt').is_file()
        assert partial['contract']['schedule'] == 'whole_2_epoch_cosine_floor0.01'
        run.main(argv+['--resume', str(out/'last.pt')])
        final = torch.load(out/'last.pt', weights_only=False)
        assert final['contract']['model']['variant'] == 'source_scene_integrated_AdaGN'
        assert final['contract']['model']['encoder']['blocks'] == final['contract']['model']['motion']['decoder_blocks'] == 2
        assert final['contract']['model']['kta_conditioning'] is True
        assert final['contract']['known'] == 'TRAIN_OCC_STC_dense_valid'
        assert final['optimizers']['gt']['param_groups'][0]['lr'] == pytest.approx(.0005*(.01+.99*.5*(1+np.cos(np.pi*2/3))))
        assert final['epoch'] == 2 and final['updates'] == 3 and (out/'epoch_0002.pt').is_file()
        assert set(final['evaluations']) == {'2', '3'} and len(calls) == 2
        assert json.loads((out/'results.json').read_text())['status'] == 'complete'
        checkpoint_model(final)
        with pytest.raises(RuntimeError): checkpoint_model(dict(final, protocol='old_pilot'))
        before = copy.deepcopy(final['models'])
        run.main(argv+['--resume', str(out/'last.pt')])
        again = torch.load(out/'last.pt', weights_only=False)
        for k, v in before['gt'].items(): torch.testing.assert_close(again['models']['gt'][k], v, rtol=0, atol=0)
        assert len(calls) == 2
        printed = capsys.readouterr().out
        assert 'INTEGRATED_FULL EVAL update=2; windows=64' in printed and 'stc_gt_semantic_refine' in printed
    finally: runner.close()

def test_same_fresh_integrated_architecture_rng_and_loader_isolation():
    from dataclasses import replace
    from test_source_scene_stage2 import fixture
    from tools.source_scene_integrated.model import make_model as pilot
    from tools.source_scene_integrated_full.model import make_model
    original, runner, _ = fixture()
    try:
        cfg = replace(original.transport.config, decoder_blocks=2)
        kw = dict(feature_width=16, z_bins=4, memory_width=4, width=8, chunk_size=32)
        torch.manual_seed(21); expected = pilot(cfg, **kw); rng = torch.get_rng_state().clone()
        torch.manual_seed(21); actual = make_model(cfg, **kw)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for k, v in expected.state_dict().items(): torch.testing.assert_close(actual.state_dict()[k], v, rtol=0, atol=0)
        assert expected.configs() == actual.configs()
        saved = dict(protocol=run.PROTOCOL, contract=dict(model=actual.configs()), models={'gt': actual.state_dict()})
        restored = checkpoint_model(saved)
        for k, v in actual.state_dict().items(): torch.testing.assert_close(restored.state_dict()[k], v, rtol=0, atol=0)
        with pytest.raises(ValueError): make_model(replace(cfg, decoder_blocks=4), **kw)
        with pytest.raises(RuntimeError): checkpoint_model(dict(saved, protocol='scene_temporal_AdaGN_joint_RANDOM_TRAIN20x1_v1'))
        bad = copy.deepcopy(saved); bad['contract']['model']['kta_conditioning'] = False
        with pytest.raises(RuntimeError): checkpoint_model(bad)
    finally: runner.close()
