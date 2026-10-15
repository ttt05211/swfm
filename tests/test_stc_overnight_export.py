import json
from pathlib import Path
from types import SimpleNamespace
import zipfile
import numpy as np
import torch
from tools.stc_overnight import export
from test_shared_source_track import fixture
from test_dense_checkpoint_average import make_bundle


def test_selected_roles_are_scene_disjoint():
    dev = [dict(scene_name=f'v{i%150}', t0_token=str(i)) for i in range(4369)]
    train = [dict(scene_name=f't{i%100}', t0_token=str(i)) for i in range(20430)]
    keys = [[r['scene_name'], r['t0_token']] for r in dev[:64]]
    rows = export.choose_records(dev, train, keys)
    assert len(rows) == 320
    scenes = {role: {r['scene_name'] for r, what, _ in rows if what == role}
              for role in ('reference64', 'stc_fit', 'stc_selection', 'clean_train')}
    for role in scenes:
        for other in scenes:
            if role != other: assert not scenes[role] & scenes[other]
    assert len({(r['scene_name'], r['t0_token']) for r, _, _ in rows}) == 320
    full = export.choose_records(dev, train, keys, val_count=len(dev))
    assert len(full) == 4369+64
    assert {r['scene_name'] for r, role, _ in full if role in ('reference64', 'reference_scene')} == scenes['reference64']
    assert not {r['scene_name'] for r, role, _ in full if role.startswith('stc_')} & scenes['reference64']
    assert len({(r['scene_name'], r['t0_token']) for r, _, split in full if split == 'val'}) == 4369


def test_real_export_weights_frames_zip_no_optimizer_or_source_mutation(tmp_path, monkeypatch):
    p, _, _ = fixture()
    try:
        parent, _, _, paths, _ = make_bundle(tmp_path, p.body)
        cache = tmp_path/'train_cache'; cache.write_text('fixture')
        dense = parent/'epoch_0002.pt'; saved = torch.load(dense, weights_only=False)
        saved['contract']['data']['train_cache'] = export.base.sha256(cache); torch.save(saved, dense)
        original = dense.read_bytes()
        def record(scene, token):
            return dict(scene_name=scene, t0_token=token, history_tokens=['h0', 'h1', 'h2', token],
                        future_tokens=[f'f{i}' for i in range(6)])
        dev = [record('VAL' if i < 64 else 'EXTRA', f't{i}') for i in range(4369)]
        train = [record('TRAIN', f'train{i}') for i in range(20430)]
        selected = export.choose_records(dev, train, saved['contract']['eval_keys'], 65, 1)
        for r, _, split in selected:
            for token in r['history_tokens']+r['future_tokens']:
                path = tmp_path/'gts'/r['scene_name']/token/'labels.npz'; path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, semantics=np.zeros((2, 2, 2), np.uint8), mask_lidar=np.ones((2, 2, 2), bool))
            if split == 'val':
                for token in r['history_tokens']:
                    path = tmp_path/'stc'/r['scene_name']/token/'labels.npz'; path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, semantics=np.zeros((2, 2, 2), np.uint8))
        monkeypatch.setattr(export.base, 'load_cache', lambda path: ({}, train if Path(path) == cache else dev))
        from real_motion.prepared import PrepareConfig
        monkeypatch.setattr(export.base, 'load_runtime_config', lambda *a: None)
        monkeypatch.setattr(export.base, 'make_prepare_config', lambda _: PrepareConfig())
        monkeypatch.setattr(export, 'NuScenesWindowSource', lambda *a, **kw: SimpleNamespace(
            pose=lambda _: np.eye(4), nusc=SimpleNamespace(get=lambda *a: dict(timestamp=0))))
        out = tmp_path/'bundle'; args = ['--dense-dir', str(parent), '--out-dir', str(out), '--dataroot', str(tmp_path),
            '--stc-root', str(tmp_path/'stc'), '--train-cache', str(cache), '--val-windows', '65', '--train-windows', '1']
        for name, path in paths.items(): args += ['--'+name.replace('_', '-'), path]
        assert export.main(args) == 0
        receipt = json.loads((out/'bundle_receipt.json').read_text())
        assert receipt['windows']['reference64'] == 64 and receipt['models'] == ['dense_epoch2']
        with zipfile.ZipFile(receipt['archive']) as z:
            assert z.testzip() is None
            manifest = json.loads(z.read('manifest.json'))
            assert len(manifest['records']) == 66 and len(manifest['inventory']) > 0
            assert not any('sample_data.json' in name or name.endswith('.jpg') for name in z.namelist())
        packed = torch.load(out/'models/dense_epoch2.pt', weights_only=False)
        assert 'optimizers' not in packed and packed['optimizer_included'] is False
        assert dense.read_bytes() == original
    finally: p.close()
