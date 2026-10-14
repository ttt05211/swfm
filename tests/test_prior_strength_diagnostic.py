import copy
import json
import threading
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from test_final_semantic_diagnostics import scene
from tools.prior_strength import engine, run
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint


def paired_scene(device='cpu'):
    predictor, source, job, _ = scene(device)
    source.forecasts = 6
    targets = source.targets(job, engine.INDICES)
    source.forecasts = 0
    def labels(_, indices):
        assert source.forecasts == 2, 'both full forwards must precede future targets'
        assert indices == engine.INDICES
        return targets
    source.targets = labels
    source.commands = lambda _: (_ for _ in ()).throw(AssertionError('no head/navigation permitted'))
    return predictor, source, job


def test_exact_noop_and_fixed_half_formula_cannot_sweep():
    prior = torch.full((3, 18, 3), -9.2); prior[:, 4] = 0.
    residual = torch.zeros_like(prior); residual[:, 11, 0] = 6.
    logits = prior+residual
    original = logits.clone(); p0 = prior.clone()
    assert engine.reweight(logits, prior, 1.) is logits
    half = engine.reweight(logits, prior, .5)
    torch.testing.assert_close(half, prior*.5+residual, rtol=0, atol=0)
    assert logits[0, :, 0].argmax() == 4 and half[0, :, 0].argmax() == 11
    torch.testing.assert_close(logits, original, rtol=0, atol=0)
    torch.testing.assert_close(prior, p0, rtol=0, atol=0)
    with pytest.raises(ValueError, match='preregistered'): engine.reweight(logits, prior, .3)
    bad = prior.clone(); bad[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'): engine.reweight(logits, bad, .5)


def test_edits_correct_wrong_free_false_occupancy_class_and_damage():
    before = np.array([17, 4, 4, 11, 4], np.uint8)
    after = np.array([11, 17, 15, 17, 16], np.uint8)
    target = np.array([11, 17, 15, 11, 15], np.uint8)
    edits = engine.edit_counts(before, after, target)
    assert edits['all'] == dict(count=5, changed=5, corrected=3, damaged=1, wrong_to_other_wrong=1)
    assert edits['wrong_free']['corrected'] == 1
    assert edits['false_occupancy']['corrected'] == 1
    assert edits['wrong_class']['corrected'] == 1 and edits['wrong_class']['wrong_to_other_wrong'] == 1
    with pytest.raises(ValueError, match='integer'): engine.edit_counts(before.astype(float), after, target)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_real_two_forward_readout_same_residual_no_input_or_weight_changes(device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    predictor, source, job = paired_scene(device)
    before = tensor_state_fingerprint(predictor.body.state_dict())
    try:
        value = engine.row_report(predictor, source, job)
        assert source.forecasts == 2 and len(value['counts']) == 6
        assert value['edits']['occ_gt']['half_vs_original'][0]['all']['count'] == 256
        value['seconds'] = 1.
        engine.checked_rows(dict(contract={}, rows=[value]), [value['key']], {}, 256)
        report = engine.aggregate([value])
        assert report['scientific_updates'] == 0 and report['single_forward_per_modality']
        assert not report['automatic_promotion']
        assert set(report['half_minus_original']) == {'occ_gt', 'stc_gt'}
        assert before == tensor_state_fingerprint(predictor.body.state_dict())
        assert not any(p.grad is not None for p in predictor.body.parameters())
        assert 'half_vs_original' in run.summary(report)
    finally: predictor.close()


def test_complete_window_resume_reject_bad_counts_and_never_commit_half_window(monkeypatch):
    predictor, source, job = paired_scene()
    try:
        value = engine.row_report(predictor, source, job)
        keys = [value['key'], ['VAL', 'a1']]
        calls, states = [], []
        def row(*args):
            result = copy.deepcopy(value); result['key'] = list(keys[len(calls)])
            calls.append(result['key']); return result
        monkeypatch.setattr(engine, 'row_report', row)
        event = threading.Event()
        def save(s): states.append(copy.deepcopy(s)); event.set()
        with pytest.raises(InterruptedError):
            engine.evaluate(predictor, source, [job, job], keys, {}, save=save, stop_event=event)
        assert len(states[-1]['rows']) == 1
        resumed = engine.evaluate(predictor, source, [job, job], keys, {}, saved=states[-1])
        assert resumed['windows'] == 2 and calls == keys
        bad = copy.deepcopy(states[-1]); bad['rows'][0]['counts']['stc_gt/original_c1']['confusion'][0][0][0] += 1
        with pytest.raises(RuntimeError, match='integer'): engine.evaluate(predictor, source, [job], keys[:1], {}, saved=bad)
        bad = copy.deepcopy(states[-1]); bad['rows'][0]['edits']['stc_gt']['half_vs_original'][0]['all']['corrected'] = -1
        with pytest.raises(RuntimeError, match='edit counts'): engine.evaluate(predictor, source, [job], keys[:1], {}, saved=bad)
        bad = copy.deepcopy(states[-1]); bad['rows'][0]['key'][1] = 'wrong'
        with pytest.raises(RuntimeError, match='prefix'): engine.evaluate(predictor, source, [job], keys[:1], {}, saved=bad)
        monkeypatch.setattr(engine, 'row_report', lambda *a: (_ for _ in ()).throw(ValueError('incomplete')))
        failure = []
        with pytest.raises(ValueError, match='incomplete'):
            engine.evaluate(predictor, source, [job], keys[:1], {}, save=lambda s: failure.append(copy.deepcopy(s)))
        assert failure[-1]['rows'] == []
    finally: predictor.close()


def test_actual_cli_snapshot_resume_preserves_source_no_head_required(tmp_path, monkeypatch):
    from tools.shared_history_experiments.full_gt.model import PROTOCOL
    from real_motion.waymo_i2world import file_sha256
    predictor, source, job = paired_scene()
    try:
        value = engine.row_report(predictor, source, job); value['seconds'] = 1.
        config, info, cache = (tmp_path/n for n in ('config', 'info', 'cache'))
        for p in (config, info, cache): p.write_text('fixture')
        ckdir = tmp_path/'GT'; ckdir.mkdir(); ck = ckdir/'epoch2.pt'
        train_keys = [['TRAIN', str(i)] for i in range(20430)]
        eval_keys = [['VAL', str(i)] for i in range(512)]
        contract = dict(protocol=PROTOCOL, modes=['gt'], epochs=2, train_keys=train_keys, eval_keys=eval_keys,
            window_batch=4, source_budget=128, total_updates=4, model=predictor.body.configs(),
            data=dict(dev_info=file_sha256(info), dev_cache=file_sha256(cache)))
        torch.save(dict(protocol=PROTOCOL, contract=contract, epoch=2, updates=4, batch=0,
                        models=dict(gt=predictor.body.state_dict())), ck)
        before = file_sha256(ck)
        pcfg = predictor.provider.pcfg; pcfg.free_label = 17
        monkeypatch.setattr(run, 'make_prepare_config', lambda _: pcfg)
        monkeypatch.setattr(run, 'load_runtime_config', lambda *a: {})
        monkeypatch.setattr(run, 'validate_grid', lambda _: None)
        monkeypatch.setattr(run, 'NuscInputs', lambda *a, **kw: source)
        monkeypatch.setattr(run, 'load_cache', lambda _: ({}, [{}]))
        jobs = [SimpleNamespace(**{**vars(job), 't0_token': str(i)}) for i in range(64)]
        monkeypatch.setattr(run, 'nusc_jobs', lambda *a: (jobs, dict(windows=64)))
        source.preflight = lambda jobs, kind: [[kind, len(jobs)]]
        class Audit:
            sha = 'receipt'
            def __init__(self, *a): pass
            def close(self): pass
        monkeypatch.setattr(run, 'ArtifactAudit', Audit)
        monkeypatch.setattr(run, 'audit_gt_entrance', lambda *a: dict(pass_gate=True))
        calls = []
        def evaluate(*args, **kw):
            calls.append(kw['saved'])
            if len(calls) == 1:
                kw['save'](dict(contract=args[4], rows=[])); raise InterruptedError('boundary')
            assert kw['saved']['rows'] == []
            rows = [{**copy.deepcopy(value), 'key': key} for key in args[3]]
            kw['save'](dict(contract=args[4], rows=rows)); return engine.aggregate(rows)
        monkeypatch.setattr(run, 'evaluate', evaluate)
        out = tmp_path/'diagnostic'
        argv = ['--config', str(config), '--gt-checkpoint', str(ck), '--dataroot', str(tmp_path),
            '--dev-info', str(info), '--dev-cache', str(cache), '--stc-root', str(tmp_path/'compact'),
            '--out-dir', str(out), '--device', 'cpu']
        assert run.main(argv=argv) == 130
        assert run.main(argv=argv+['--resume']) == 0
        result = json.loads((out/'evaluation.json').read_text())
        assert result['windows'] == 64 and result['strengths'] == [1., .5]
        assert result['single_forward_per_modality'] and result['scientific_updates'] == 0
        assert file_sha256(ck) == before
        assert (out/'summary.txt').exists()
        with pytest.raises(FileExistsError): run.main(argv=argv)
        state_path = out/'state.json'; state_path.write_text('KEEP')
        info.write_text('metadata_changed')
        with pytest.raises(RuntimeError, match='metadata'): run.main(argv=argv+['--resume'])
        assert state_path.read_text() == 'KEEP'
    finally: predictor.close()
