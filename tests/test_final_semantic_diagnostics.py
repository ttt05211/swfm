import copy
import hashlib
import io
import json
import threading
import zipfile
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from test_full_gt_training import fixture
from tools.final_semantic_evaluation.predictor import Predictor
from tools.scene_ego_head.model import SceneEgoHead, SceneEgoConfig
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from tools.final_semantic_diagnostics import engine, statistics as stats, input_audit


def test_prior_competition_and_edits_are_exact_not_confidence_gate():
    p = np.full((3, 18, 2), -9.21, np.float32); p[:, 4] = 0.
    z = p.copy(); z[:, 11] = .01
    before = np.full((3, 1, 1, 2), 4, np.uint8); after = np.full_like(before, 11)
    snap = dict(transport=before, semantic_refine=after, prior_logits=p, logits=z)
    rows = stats.edits_and_margins(snap, after)
    assert rows[0]['corrected'] == 2 and rows[0]['damaged'] == 0
    assert rows[0]['groups']['wrong_prior']['overcame_old_winner'] == 2
    merged = stats.merge_edits([rows, rows])
    assert merged[0]['net_semantic_corrected'] == 4
    assert merged[0]['groups']['wrong_prior']['barrier']['mean'] == pytest.approx(9.21)
    assert merged[0]['groups']['wrong_prior']['final_correct_fraction'] == 1.
    snap['logits'][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='nonfinite'): stats.edits_and_margins(snap, after)


def test_axis_probe_records_all_directions_without_mutating_inputs():
    gt = np.arange(32, dtype=np.uint8).reshape(4, 4, 2) % 18
    camera = gt.transpose(1, 0, 2).copy(); before = camera.copy()
    result = input_audit.input_confusions(camera, gt, gt, gt)
    assert len(result) == 10
    assert stats.input_report(result['transposeXY'])['mIoU'] == 100.
    assert stats.input_report(result['identity'])['mIoU'] < 100.
    np.testing.assert_array_equal(camera, before)


def scene(device='cpu'):
    model, _, p, row = fixture(); model.to(device).eval().requires_grad_(False)
    head = SceneEgoHead(SceneEgoConfig(feature_width=16, width=16, heads=4, pool_side=2)).to(device).eval().requires_grad_(False)
    with torch.no_grad():
        model.columns.readout[-1].weight.normal_(std=.01); head.readout[-1].weight.normal_(std=.001)
    raw = {k: copy.deepcopy(row[1][k]) for k in ('history_occ', 'history_observed', 'history_poses')}
    raw['history_occ'][:, 2:4, 2:4, 1:3] = 4
    camera = copy.deepcopy(raw); camera['history_observed'][:] = True
    camera['history_occ'][:, 4, 4, 1] = 15
    record = dict(scene_name='VAL', t0_token='a0', history_tokens=('h0', 'h1', 'h2', 'a0'),
                  future_tokens=tuple('f'+str(i) for i in range(6)))
    tables = {}
    tokens = (*record['history_tokens'], *record['future_tokens'])
    for i, token in enumerate(tokens):
        tables['sample', token] = dict(scene_token='scene', timestamp=i*500000,
            next=tokens[i+1] if i+1 < len(tokens) else '', anns=[], data=dict(LIDAR_TOP=token))
        tables['sample_data', token] = dict(ego_pose_token=token)
        tables['ego_pose', token] = dict(translation=[0., 0., 0.], rotation=[1., 0., 0., 0.])
    tables['scene', 'scene'] = dict(name='VAL')
    source = SimpleNamespace(source=SimpleNamespace(nusc=SimpleNamespace(get=lambda t, k: tables[t, k])))
    source.history = lambda job, kind: (dict(record), copy.deepcopy(raw if kind == 'occ' else camera), np.arange(4)*.5)
    source.poses = lambda _: row[1]['future_poses']
    source.commands = lambda _: np.full(6, 2, np.int64)
    from tools.final_semantic_evaluation.sources import NuscInputs
    source.moving = lambda *args: NuscInputs.moving(source, *args)
    source.forecasts = 0
    def targets(job, ids):
        assert source.forecasts == 6, 'all six routes must finish before future labels'
        return [row[1]['future_gt_occ'][i] for i in ids]
    source.targets = targets
    class CountingPredictor(Predictor):
        def forward(self, *a, **kw):
            source.forecasts += 1
            return super().forward(*a, **kw)
    pred = CountingPredictor(model, p.pcfg, device, workers=1, head=head)
    job = SimpleNamespace(**record)
    return pred, source, job, raw


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_real_six_route_diagnostic_no_future_input_or_weight_changes(device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    pred, source, job, original = scene(device)
    before = (tensor_state_fingerprint(pred.body.state_dict()), tensor_state_fingerprint(pred.head.state_dict()))
    try:
        row = engine.row_report(pred, source, job)
        assert len(row['forecast_counts']) == 12 and len(row['edits']) == 6
        assert set(row['trajectory_errors']) == {'GT_history_head', 'GT_all_known_head_DIAGNOSTIC', 'STC_history_head', 'kinematic_prior'}
        assert row['geometry']['stc']['known_fraction'] == 1.
        report = engine.aggregate([row]); assert report['windows'] == 1 and report['diagnostic_only']
        assert set(report['isolated_differences']) == {'GT_known_to_allTrue', 'STC_GTscene_head_minus_STC_head'}
        assert before == (tensor_state_fingerprint(pred.body.state_dict()), tensor_state_fingerprint(pred.head.state_dict()))
        assert not any(p.grad is not None for p in pred.body.parameters())
    finally: pred.close()


def test_complete_window_resume_fail_before_commit_and_bad_prefix(monkeypatch):
    pred, source, job, _ = scene()
    try:
        value = engine.row_report(pred, source, job)
        value['seconds'] = 1.
        job2 = SimpleNamespace(**{**vars(job), 't0_token': 'a1'})
        calls = []
        def report(*args):
            calls.append(args[2].t0_token)
            out = copy.deepcopy(value); out['key'][1] = args[2].t0_token; return out
        monkeypatch.setattr(engine, 'row_report', report)
        snapshots = []; event = threading.Event()
        def save(s): snapshots.append(copy.deepcopy(s)); event.set()
        keys = [['VAL', 'a0'], ['VAL', 'a1']]
        with pytest.raises(InterruptedError):
            engine.evaluate(pred, source, [job, job2], keys, {}, save=save, stop_event=event)
        assert len(snapshots[-1]['rows']) == 1
        resumed = engine.evaluate(pred, source, [job, job2], keys, {}, saved=snapshots[-1])
        assert resumed['windows'] == 2 and calls == ['a0', 'a1']
        bad = copy.deepcopy(snapshots[-1]); bad['rows'][0]['key'][1] = 'bad'
        with pytest.raises(RuntimeError, match='prefix'):
            engine.evaluate(pred, source, [job], keys[:1], {}, saved=bad)
        bad = copy.deepcopy(snapshots[-1]); bad['rows'][0]['forecast_counts']['occ_gt/transport']['confusion'][0][0][0] += 1
        with pytest.raises(RuntimeError, match='integer'):
            engine.evaluate(pred, source, [job], keys[:1], {}, saved=bad)
        monkeypatch.setattr(engine, 'row_report', lambda *a: (_ for _ in ()).throw(ValueError('broken window')))
        saved = []
        with pytest.raises(ValueError, match='broken'):
            engine.evaluate(pred, source, [job], keys[:1], {}, save=saved.append)
        assert saved[-1]['rows'] == []
    finally: pred.close()


def test_real_compact_receipt_and_original_zip_bytes(tmp_path):
    from real_motion.waymo_i2world import fingerprint
    sem = np.full((200, 200, 16), 17, np.uint8); digest = hashlib.sha256(sem.tobytes()).hexdigest()
    files = {f'scene-0000/{i}': digest for i in range(6018)}; files['scene-0001/token'] = digest
    payload = dict(status='complete', frames=6019, scenes=150, files=files,
        contract=dict(protocol='stc_official_lossless_semantics_uint8_v1',
            official_drive_id='1dXB9mtROLWChycBZlhYIf_JBLshXogBs', shape=[200, 200, 16], masks_applied=False,
            archive_sha256='not_supplied'))
    payload['fingerprint'] = fingerprint(payload)
    path = tmp_path/'extraction_state.json'; path.write_text(json.dumps(payload))
    audit = input_audit.ArtifactAudit(tmp_path)
    try:
        audit.check('scene-0001', 'token', sem)
        assert audit.verify()['selected_history_frames_checked'] == 1
        changed = sem.copy(); changed[0, 0, 0] = 4
        with pytest.raises(RuntimeError, match='bytes'): audit.check('scene-0000', '0', changed)
        buffer = io.BytesIO(); np.savez_compressed(buffer, semantics=sem)
        archive = tmp_path/'small_test.zip'
        with zipfile.ZipFile(archive, 'w') as z: z.writestr('stc-results/scene-0000/1/labels.npz', buffer.getvalue())
        audit.zip = zipfile.ZipFile(archive)
        audit.check('scene-0000', '1', sem)
        path.write_text('changed')
        with pytest.raises(RuntimeError, match='receipt changed'): audit.verify()
    finally: audit.close()


def test_actual_cli_frozen_sources_resume_without_training(tmp_path, monkeypatch):
    from dataclasses import asdict
    from tools.final_semantic_diagnostics import run
    from tools.shared_history_experiments.full_gt.model import PROTOCOL
    from tools.scene_ego_head.frozen_gt.training import PROTOCOL as HEAD_PROTOCOL
    from real_motion.waymo_i2world import file_sha256
    pred, source, job, _ = scene()
    try:
        value = engine.row_report(pred, source, job); value['seconds'] = 1.
        config, info, cache = (tmp_path/n for n in ('config', 'info', 'cache'))
        for p in (config, info, cache): p.write_text('fixture')
        ckdir = tmp_path/'GT'; ckdir.mkdir(); ck = ckdir/'epoch2.pt'
        model = pred.body
        train_keys = [['TRAIN', str(i)] for i in range(20430)]
        eval_keys = [['VAL', str(i)] for i in range(512)]
        contract = dict(protocol=PROTOCOL, modes=['gt'], epochs=2, train_keys=train_keys, eval_keys=eval_keys,
            window_batch=4, source_budget=128, total_updates=4, model=model.configs(),
            data=dict(dev_info=file_sha256(info), dev_cache=file_sha256(cache)))
        torch.save(dict(protocol=PROTOCOL, contract=contract, epoch=2, updates=4, batch=0,
                        models=dict(gt=model.state_dict())), ck)
        head_dir = tmp_path/'head_source'; head_dir.mkdir(); hp = head_dir/'head10.pt'
        hcontract = dict(stage=dict(source_sha256=file_sha256(ck)), body_state_sha256=tensor_state_fingerprint(model.state_dict()),
            schedule=dict(epochs=10, batch_size=64), source_train_keys=train_keys, head=asdict(pred.head.config))
        torch.save(dict(protocol=HEAD_PROTOCOL, contract=hcontract, epochs=10, updates=3200, final_epoch_only=True,
            head=pred.head.state_dict(), state_sha256=tensor_state_fingerprint(pred.head.state_dict())), hp)
        before = (file_sha256(ck), file_sha256(hp))
        pcfg = pred.provider.pcfg; pcfg.free_label = 17
        monkeypatch.setattr(run, 'make_prepare_config', lambda _: pcfg)
        monkeypatch.setattr(run, 'load_runtime_config', lambda *a: {})
        monkeypatch.setattr(run, 'validate_grid', lambda _: None)
        monkeypatch.setattr(run, 'NuscInputs', lambda *a, **kw: source)
        monkeypatch.setattr(run, 'load_cache', lambda _: ({}, [{}]))
        selected = [SimpleNamespace(**{**vars(job), 't0_token': str(i)}) for i in range(64)]
        monkeypatch.setattr(run, 'nusc_jobs', lambda *a: (selected, dict(windows=64)))
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
        argv = ['--config', str(config), '--gt-checkpoint', str(ck), '--head10-checkpoint', str(hp),
            '--dataroot', str(tmp_path), '--dev-info', str(info), '--dev-cache', str(cache),
            '--stc-root', str(tmp_path/'compact'), '--out-dir', str(out), '--device', 'cpu']
        assert run.main(argv=argv) == 130
        assert run.main(argv=argv+['--resume']) == 0
        result = json.loads((out/'evaluation.json').read_text())
        assert result['windows'] == 64 and result['scientific_updates'] == 0
        assert (file_sha256(ck), file_sha256(hp)) == before
        assert (out/'summary.txt').exists()
        with pytest.raises(FileExistsError): run.main(argv=argv)
    finally: pred.close()
