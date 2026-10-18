"""Original baseline adapter/counterfactual parity and matched read-only entry."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.source_motion_baseline_audit import engine, run, report


def fixture(device='cpu'):
    from test_full_gt_training import fixture as old
    from tools.shared_source_track.adapter import SharedTrackPredictor
    model, _, provider, original = old(); model = model.to(device).eval().requires_grad_(False)
    runner = SharedTrackPredictor(model, provider.pcfg, device, workers=1)
    raw = {k: original[1][k] for k in ('history_occ', 'history_observed', 'history_poses')}
    raw['history_occ'] = np.full_like(raw['history_occ'], 17)
    raw['history_occ'][:, 2:5, 2:5, 1] = 4
    raw['history_observed'] = np.ones_like(raw['history_occ'], bool)
    row = runner.prepare(original[0], raw, times=np.arange(4)*.5)
    assert len(row[0]['source_class_id']) > 0
    return model, runner, row, original[1]['future_poses']


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_original_forward_exact_and_only_future_motion_changes(device, monkeypatch):
    if device == 'cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()): pytest.skip('BF16 GPU')
    model, runner, row, poses = fixture(device)
    try:
        with torch.no_grad():
            model.transport.residual_head.bias.fill_(.2)
            model.transport.yaw_head.bias.fill_(.05)
            model.columns.readout[-1].weight.normal_(std=.1)
        original = copy.deepcopy(model.state_dict()); calls = []; memories = []
        direct, memory = engine.forward_conditioned, engine.aligned_memory
        def capture(*a, **kw):
            fw = direct(*a, **kw)
            calls.append({r: fw[k].argmax(1).reshape(6, *fw['shape']).byte().cpu().numpy()
                for r, k in (('transport', 'prior'), ('semantic_refine', 'logits'))})
            return fw
        def readout(columns, lifted, output, history, *a):
            memories.append((output['future_transport_queries'], history))
            return memory(columns, lifted, output, history, *a)
        monkeypatch.setattr(engine, 'forward_conditioned', capture)
        monkeypatch.setattr(engine, 'aligned_memory', readout)
        predictions, diagnostic = engine.predictions(model, runner.provider, row, poses)
        assert len(predictions) == 8 and len(calls) == 2 and len(memories) == 3
        assert all(h is row[3] and q is memories[0][0] for q, h in memories)
        assert hasattr(row[1]['_shared_source_track'], 'registrations')
        assert 'history_yaw_rad' not in diagnostic  # Do not fabricate a learned history yaw for old ICP.
        for r in ('transport', 'semantic_refine'):
            np.testing.assert_array_equal(predictions['learned/'+r], calls[0][r])
            assert diagnostic['reference_repeat_changed_voxels'][r] == np.count_nonzero(calls[0][r] != calls[1][r])
        assert np.linalg.norm(diagnostic['learned_world_displacement']-diagnostic['KTA_world_displacement']) > 0
        for name, value in original.items(): torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
        assert all(p.grad is None for p in model.parameters())
    finally: runner.close()


def reference(keys):
    metrics = {m+'/'+r: dict(mIoU=20., IoU=30., MovingMicro=10.)
        for m in engine.MODES for r in ('transport', 'semantic_refine')}
    return dict(status='complete', windows=64, contract=dict(protocol=run.STAGE2_PROTOCOL,
        keys=keys, modes=list(engine.MODES)), metrics={'occ_gt': metrics, 'stc_gt': metrics})


def test_population_protocol_route_checks_and_compact_comparison(tmp_path):
    keys = [['V', str(i)] for i in range(64)]; ref = reference(keys)
    run.validate_reference(ref, keys)
    for field, v in [('status', 'running'), ('windows', 63), ('metrics', {})]:
        bad = copy.deepcopy(ref); bad[field] = v
        with pytest.raises(RuntimeError): run.validate_reference(bad, keys)
    with pytest.raises(RuntimeError): run.validate_reference(ref, list(reversed(keys)))
    old = copy.deepcopy(ref); old['metrics']['occ_gt']['learned/transport']['mIoU'] += 2
    text = report.comparison(old, ref)
    assert '+2.0000' in text and 'OLD_KTA' in text and len(text.splitlines()) < 20
    assert '尚未完成' in report.comparison(dict(status='running'), ref)
    assert run.select_checkpoint(tmp_path/'x.pt', None, tmp_path) == (tmp_path/'x.pt').resolve()
    with pytest.raises(RuntimeError): run.select_checkpoint(None, None, tmp_path)
    for tag in ('one', 'two'):
        folder = tmp_path/('dense_valid_shared_full2_ddp_'+tag); folder.mkdir(); (folder/'epoch_0002.pt').touch()
    with pytest.raises(RuntimeError): run.select_checkpoint(None, None, tmp_path)


def test_cuda_entry_matched_inputs_stop_resume_and_no_optimizer(tmp_path, monkeypatch):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): pytest.skip('BF16 GPU')
    import threading
    from tools.dense_valid_ddp.run import PROTOCOL as ORIGINAL
    model, runner, row, poses = fixture()
    try:
        paths = {k: tmp_path/k for k in ('config', 'dev_cache', 'dev_info')}
        for path in paths.values(): path.write_text('fixture')
        md = tmp_path/'v1.0-trainval'; md.mkdir()
        names = ('sample.json', 'sample_annotation.json', 'instance.json', 'category.json', 'sample_data.json', 'ego_pose.json')
        for name in names: (md/name).write_text('[]')
        archive = tmp_path/'data.npz'; archive.write_bytes(b'fixture')
        records = [dict(row[0], t0_token='dev'+str(i)) for i in range(64)]
        keys = [list(k) for k in run.base.record_keys(records)]
        c = dict(protocol=ORIGINAL, total_updates=10292, epochs=2, train_keys=[['T', str(i)] for i in range(20430)],
            eval_keys=keys, training_known='dense_valid', source_policy='shared_pinned', global_batch=4, source_budget=128,
            objective=dict(CE=1., Lovasz=1., motion=0., existence=0., ego=0.), model=model.configs(),
            implementation={}, data={k: run.base.sha256(p) for k, p in paths.items()}, overrides=[])
        source = tmp_path/'train/epoch2.pt'; source.parent.mkdir()
        torch.save(dict(protocol=ORIGINAL, contract=c, epoch=2, batch=0, updates=10292,
            models={'gt': model.state_dict()}, rank_rng=[{}, {}]), source)
        ref = reference(keys); ref['contract'].update(
            files=[[str(archive), archive.stat().st_size, archive.stat().st_mtime_ns]],
            metadata={n: run.base.sha256(md/n) for n in names}, cpu_workers=4,
            runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)))
        refpath = tmp_path/'reference.json'; refpath.write_text(json.dumps(ref))
        shas = [run.base.sha256(p) for p in (source, refpath)]
        events = []; calls = []; event = threading.Event()
        raw = {k: row[1][k] for k in ('history_occ', 'history_observed', 'history_poses')}
        class Inputs:
            source = SimpleNamespace(nusc=None)
            def __init__(self, *a, **kw): pass
            def path(self, *a): return archive
            def stat(self, p): return p.stat().st_size, p.stat().st_mtime_ns
            def history(self, *a):
                events.append('history'); return row[0], dict(raw, history_observed=np.zeros_like(raw['history_observed'])), np.arange(4)*.5
            def poses(self, *a): return poses
            def targets(self, *a):
                assert events.count('forecast') == 2; events.clear(); events.append('labels')
                return np.repeat(raw['history_occ'][-1][None], 3, 0)
            def moving(self, *a):
                assert events == ['labels']; events.clear()
                return np.ones((3, *raw['history_occ'].shape[1:]), bool)
        class Predictor:
            def __init__(self, *a, **kw): self.provider = runner.provider
            def prepare(self, rec, historical, **kw):
                assert historical['history_observed'].all(); return row
            def close(self): pass
        def predict(*a):
            assert events[-1] == 'history'; events.append('forecast'); calls.append(1)
            if len(calls) == 2: event.set()
            p = np.repeat(raw['history_occ'][-1][None], 6, 0)
            return {m+'/'+r: p for m in engine.MODES for r in ('transport', 'semantic_refine')}, dict(
                reference_repeat_changed_voxels=dict(transport=0, semantic_refine=0))
        monkeypatch.setattr(run, 'NuscInputs', Inputs); monkeypatch.setattr(run, 'SharedTrackPredictor', Predictor)
        monkeypatch.setattr(run, 'predictions', predict); monkeypatch.setattr(run, 'box_motion_rows', lambda *a: [])
        monkeypatch.setattr(run.base, 'load_cache', lambda *a: ({}, records))
        monkeypatch.setattr(run.base, 'load_runtime_config', lambda *a: {})
        monkeypatch.setattr(run.base, 'make_prepare_config', lambda *a: runner.provider.pcfg)
        monkeypatch.setattr(torch.optim, 'AdamW', lambda *a, **kw: pytest.fail('no optimizer allowed'))
        out = tmp_path/'audit'
        argv = ['--checkpoint', str(source), '--stage2-report', str(refpath), '--out', str(out), '--dataroot', str(tmp_path), '--stc-root', str(tmp_path)]
        for k, path in paths.items(): argv += ['--'+k.replace('_', '-'), str(path)]
        run.main(argv, event=event)
        partial = json.loads((out/'results.json').read_text())
        assert partial['windows'] == 1 and partial['status'] == 'stopped_at_complete_window'
        assert '尚未完成' in (out/'comparison.txt').read_text(encoding='utf-8')
        run.main(argv+['--resume'])
        final = json.loads((out/'results.json').read_text())
        assert final['windows'] == 64 and final['status'] == 'complete' and len(calls) == 128
        assert final['history_yaw_abs_deg']['occ_gt'] == {'n': 0}
        run.main(argv+['--resume']); assert len(calls) == 128
        assert shas == [run.base.sha256(p) for p in (source, refpath)]
        ref['contract']['cpu_workers'] = 2; refpath.write_text(json.dumps(ref))
        with pytest.raises(RuntimeError, match='differs'): run.main(argv+['--resume'])
    finally: runner.close()
