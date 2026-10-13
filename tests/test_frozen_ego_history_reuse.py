import copy
import json
from pathlib import Path
import torch
import pytest
from test_full_gt_training import fixture
from test_frozen_gt_scene_ego import small_bank
from tools.scene_ego_head.frozen_gt import reuse, run, bank, training, engine
from tools.joint_occ_experiments import run as legacy


def contract():
    paths = ('tools/scene_ego_head/frozen_gt/bank.py', 'tools/scene_ego_head/model.py',
        'real_motion/ego_navigation.py', 'tools/ego_experiments/ego_kinematic.py',
        'tools/ego_experiments/train_surface_ego_full.py')
    return dict(stage=dict(args=dict(dataroot='data', out_dir='old', head_epochs=1),
        source_sha256='source', implementation={p: 'unchanged' for p in paths}),
        body_state_sha256='body', source_train_keys=[['TRAIN', str(i)] for i in range(6)],
        source_runtime={'torch': 'fixture'}, encoder_precision='FP32', features='pure F',
        head={'pool_side': 2, 'feature_width': 16}, bank_chunk=64,
        schedule=dict(epochs=1, batch_size=64, seed=21, lr=.0003))


def publish(directory, c, values):
    directory.mkdir(); tokens = directory/'history_tokens'; tokens.mkdir()
    fingerprint = legacy.stable_json_fingerprint(c)
    # Old weights are deliberately unusable: importer must ONLY inspect contract.
    torch.save({'contract': c, 'head': 'DO NOT LOAD', 'adam': 'DO NOT LOAD'}, directory/'head_last.pt')
    torch.save(dict(contract=fingerprint, keys=c['source_train_keys'], values=values), tokens/'0000.pt')
    legacy.write_json(tokens/'manifest.json', dict(contract=fingerprint, keys=c['source_train_keys'], complete=True,
        chunks=[dict(index=0, windows=6, sha256=legacy.sha256(tokens/'0000.pt'))]))


def test_reuse_changes_schedule_not_features_and_writes_nothing(tmp_path):
    torch.set_num_threads(1); body, _, p, row = fixture()
    values = small_bank(body, row, p.pcfg.grid)
    old = contract(); stage = tmp_path/'old'; publish(stage, old, values)
    files = {p: p.read_bytes() for p in stage.rglob('*') if p.is_file()}
    new = copy.deepcopy(old); new['schedule']['epochs'] = 10; new['schedule']['lr'] = .001
    new['stage']['args'].update(out_dir='new', head_epochs=10, final_population='all')
    loaded, receipt = reuse.load_readonly(stage, new)
    for k in values: assert torch.equal(loaded[k], values[k])
    assert receipt['fresh_windows'] == 0 and receipt['read_only']
    for path, data in files.items(): assert path.read_bytes() == data
    for field in ('body_state_sha256', 'encoder_precision'):
        invalid = copy.deepcopy(new); invalid[field] = 'different'
        with pytest.raises(RuntimeError, match='differs'): reuse.load_readonly(stage, invalid)
    invalid = copy.deepcopy(new); invalid['head']['pool_side'] = 8
    with pytest.raises(RuntimeError, match='differs'): reuse.load_readonly(stage, invalid)
    chunk = stage/'history_tokens/0000.pt'; chunk.write_bytes(chunk.read_bytes()+b'corrupt')
    with pytest.raises(RuntimeError, match='receipt'): reuse.load_readonly(stage, new)


def test_random_ten_epochs_not_import_old_head(tmp_path):
    torch.set_num_threads(1); body, _, p, row = fixture()
    values = small_bank(body, row, p.pcfg.grid)
    from tools.scene_ego_head.model import SceneEgoHead, SceneEgoConfig
    torch.manual_seed(21); head = SceneEgoHead(SceneEgoConfig(16, 32, 4, 2))
    c = contract(); c['schedule']['epochs'] = 10
    result = training.train(head, values, tmp_path, c)
    assert result['complete_epochs'] == 10 and result['updates'] == 10
    assert (tmp_path/'head_epoch10.pt').is_file()
    saved = torch.load(tmp_path/'head_last.pt', weights_only=False)
    assert saved['epoch'] == 10 and len(saved['history']) == 10


def test_all_val_cli_reuses_tokens_fresh_head_resume_and_old_files_unchanged(tmp_path, monkeypatch):
    from test_full_gt_training import test_real_cli_full_population_epoch_snapshots_resume_and_eval_readonly
    test_real_cli_full_population_epoch_snapshots_resume_and_eval_readonly(tmp_path, monkeypatch)
    source = tmp_path/'pilot/epoch_0002.pt'; out = tmp_path/'new10'; old = tmp_path/'old'
    old.mkdir(); (old/'untouched').write_bytes(b'OLD BANK AND HEAD')
    body, _, p, row = fixture()
    values = small_bank(body, row, p.pcfg.grid)
    source_bytes, old_bytes = source.read_bytes(), (old/'untouched').read_bytes()
    monkeypatch.setattr(engine, 'entrance_audit', lambda *a: dict(pass_gate=True))
    monkeypatch.setattr(bank, 'build', lambda *a, **k: pytest.fail('must not extract new history'))
    calls = []
    def imported(directory, c, **kw):
        assert Path(directory) == old and c['schedule']['epochs'] == 10
        calls.append('read_only'); return values, dict(tensor_mib=1., windows=6, fresh_windows=0, read_only=True)
    monkeypatch.setattr(reuse, 'load_readonly', imported)
    def evaluated(body, head, provider, source, rows, **kw):
        keys = [[r[0]['scene_name'], r[0]['t0_token']] for group in rows for r in group]
        assert keys[513] == ['DEV', 'dev513'] and keys[-1] == ['DEV', 'dev4368'] and len(keys) == 4369
        return dict(windows=len(keys), population_keys=keys, variants={'internal': {}}, trajectory={})
    monkeypatch.setattr(engine, 'evaluate', evaluated)
    args = [v for n, path in (('config', tmp_path/'config.yaml'), ('base-checkpoint', tmp_path/'base.pt'),
        ('train-cache', tmp_path/'train-cache'), ('dev-cache', tmp_path/'dev-cache'),
        ('population-manifest', tmp_path/'population-manifest'), ('dataroot', tmp_path),
        ('train-info', tmp_path/'train-info'), ('dev-info', tmp_path/'dev-info'),
        ('ccr-history-cache', tmp_path), ('ccr-val-history-cache', tmp_path), ('out-dir', out),
        ('resume', source), ('reuse-stage-dir', old)) for v in ('--'+n, str(path))]
    args += ['--head-epochs', '10', '--head-pool-side', '2', '--final-population', 'all']
    before = legacy.prefetched_batches, legacy.load_cache
    assert run.main(argv=args) == 0
    saved = torch.load(out/'head_last.pt', weights_only=False)
    assert saved['epoch'] == 10 and saved['updates'] == 10
    result = json.loads((out/'results.json').read_text())
    assert result['actual_eval_population'] == 'ALL_VAL4369' and result['evaluation']['windows'] == 4369
    assert len(json.loads((out/'population.json').read_text())['keys']) == 4369
    exported = (out/'head_epoch10.pt').read_bytes()
    assert run.main(argv=args+['--resume-stage']) == 0
    assert (out/'head_epoch10.pt').read_bytes() == exported
    assert source.read_bytes() == source_bytes and (old/'untouched').read_bytes() == old_bytes
    assert (legacy.prefetched_batches, legacy.load_cache) == before and calls == ['read_only']*2
