"""Public-code parity, no future-input leakage, early-start safety and downloads."""
import copy
import hashlib
import io
import os
import pickle
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from real_motion.geometry import OccupancyGrid
from real_motion.prepared import PrepareConfig
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.joint_causal_columns import JointCausalColumns
from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
from real_motion.causal_column_completion import ColumnConfig
from tools.real_motion import geniedrive_eval_alignment as genie
from tools.real_motion import download_geniedrive_eval_info as downloader
from tools.real_motion import joint_long_rollout_common as rollout


def metadata_fixture():
    rows = []; sample = {}; scenes = {}
    for si, scene in enumerate(('s0', 's1')):
        scenes[scene] = dict(name=scene)
        for i in range(40):
            token = f'{scene}_{i}'
            row = dict(token=token, prev=f'{scene}_{i-1}' if i else '', scene_token=scene,
                timestamp=si*100000000+i*500000, occ_path=f'./data/nuscenes/gts/{scene}/{token}')
            rows.append(row)
            sample[token] = {**row, 'next': f'{scene}_{i+1}' if i < 39 else ''}
    def get(table, token):
        assert table in ('sample', 'scene')  # NO future annotation queries.
        return (sample if table == 'sample' else scenes)[token]
    return rows, SimpleNamespace(nusc=SimpleNamespace(get=get), allowed_scenes=set(scenes),
                                 info_by_token={r['token']: r for r in rows})


def official_padding_reference(rows):
    """Literal get_data_info repeat-on-failure + evaluate unique-index check."""
    rows = sorted(rows, key=lambda r: r['timestamp']); starts = []
    for i in range(len(rows)):
        indices = [i]; previous = i
        for j in (i-1, i-2, i-3):
            if j >= 0 and rows[j]['token'] == rows[previous]['prev']: previous = j
            indices.insert(0, previous)
        future = i
        for j in range(i+1, i+21):
            if j < len(rows) and rows[j]['prev'] == rows[future]['token']: future = j
            indices.append(future)
        if len(indices) == len(set(indices)): starts.append(i)
    return starts


@pytest.mark.parametrize('gap', (False, True))
def test_complete_population_matches_literal_official_padding_and_dropping(gap):
    rows, _ = metadata_fixture()
    if gap: rows[12]['prev'] = 'missing'
    rows.reverse()  # Sorting not arrival order.
    ordered, starts = genie.complete_start_indices(rows)
    assert starts == official_padding_reference(rows)
    if not gap: assert starts == list(range(3, 20))+list(range(43, 60))
    assert all(len(ordered[i-3:i+21]) == 24 for i in starts)


def test_duplicate_metadata_rejected():
    rows, _ = metadata_fixture()
    with pytest.raises(RuntimeError, match='duplicate'): genie.complete_start_indices(rows+[rows[0]])


def test_official_population_includes_early_starts_not_in_six_history_cache(tmp_path, monkeypatch):
    rows, source = metadata_fixture(); path = tmp_path/'info.pkl'
    path.write_bytes(pickle.dumps(dict(infos=rows, metadata={'version': 'v1.0-trainval'})))
    monkeypatch.setattr(genie, 'verify_info', lambda path: 'pinned')
    monkeypatch.setattr(genie, 'validate_start_manifest', lambda keys: None)  # small synthetic population
    cached = [dict(scene_name=f's{si}', t0_token=f's{si}_{i}',
        history_tokens=tuple(f's{si}_{j}' for j in range(i-5, i+1)),
        future_tokens=tuple(f's{si}_{j}' for j in range(i+1, i+7))) for si in range(2) for i in range(5, 34)]
    selected, audit = genie.select_population(path, source, cached)
    assert len(selected) == 34 and audit['cached_windows'] == 30
    assert len(audit['missing_six_history_cache_keys']) == 4
    assert selected[0][0].history_tokens == ('s0_0', 's0_1', 's0_2', 's0_3')
    assert len(selected[0][0].future_tokens) == 12
    assert len(audit['selected_sequence_tokens_4_plus20'][0]) == 24
    assert not audit['paper_table_population_verified']
    source.allowed_scenes.add('train')
    with pytest.raises(RuntimeError, match='scene populations differ'): genie.select_population(path, source, cached)
    source.allowed_scenes.remove('train'); cached[0]['future_tokens'] = cached[0]['future_tokens'][::-1]
    with pytest.raises(RuntimeError, match='order mismatch'): genie.select_population(path, source, cached)


def test_metric_compatibility_explicitly_drops_zero_but_standard_keeps_it():
    raw = rollout.legacy._new_raw()
    raw['sem_union'][:, :2] = 4; raw['sem_inter'][:, 1] = 2
    raw['occ_union'][:] = 3; raw['occ_inter'][:] = 2
    before = copy.deepcopy(raw)
    compatible = genie.compatibility_metrics(raw)
    assert compatible['per_horizon']['4.0']['mIoU'] == 50
    assert compatible['per_horizon']['4.0']['IoU'] == 66.67
    assert compatible['per_horizon']['4.0']['excluded_exact_zero_class_ids'] == [0]
    assert compatible['average_4s_5s_6s']['mIoU'] == 50
    assert rollout.finalize_metrics(raw)['per_horizon']['4.0']['mIoU'] == 25
    assert all(np.array_equal(raw[k], before[k]) for k in raw)
    raw['sem_inter'][:] = 0
    assert genie.compatibility_metrics(raw)['per_horizon']['4.0']['mIoU'] is None
    assert rollout.finalize_metrics(raw)['per_horizon']['4.0']['mIoU'] == 0


def test_grid_fail_closed():
    genie.validate_grid(PrepareConfig())
    with pytest.raises(RuntimeError, match='full unmasked'):
        genie.validate_grid(PrepareConfig(grid=OccupancyGrid(shape_hwd=(100, 200, 16))))


def test_pinned_start_manifest_rejects_different_population():
    with pytest.raises(RuntimeError, match='ordered start manifest'):
        genie.validate_start_manifest([['s', 't']])


def test_early_start_real_network_prepare_uses_only_four_history_and_future_poses():
    grid = OccupancyGrid(-6.4, -6.4, -1, (.4, .4, .4), (32, 32, 4))
    history = np.full((4, *grid.shape_hwd), 17, np.uint8); history[:, :, :, 0] = 11
    for f in range(4): history[f, 10+f:13+f, 10:13, 1:3] = 4
    joint = JointCausalColumns(LocalSTWMV17Config(history_frames=4, d_model=16, semantic_dim=4,
        blocks=1, decoder_blocks=1), ColumnConfig(width=16, semantic_dim=4, layers=1, z_bins=4))
    joint.eval().requires_grad_(False)
    provider = genie.AlignedColumnProvider.__new__(genie.AlignedColumnProvider)
    provider.joint = joint; provider.model = joint.transport; provider.latents_checked = True
    provider.pcfg = PrepareConfig(grid=grid); provider.strong = StrongW2DetConfig()
    provider.workers = 1; provider.device = torch.device('cpu')
    tokens = tuple(f'h{i}' for i in range(4)); loaded = []
    class Source:
        def load_occ3d(self, scene, token, **kwargs):
            assert token in tokens
            loaded.append(token); return history[tokens.index(token)], np.ones(grid.shape_hwd, bool)
        def pose(self, token): return np.eye(4)
        def official_trajectory(self, *args, **kwargs): pytest.fail('must not request six-history trajectory ABI')
        def load_semantics(self, *args): pytest.fail('future GT must not be loaded during forecast')
    seed = dict(scene_name='s', t0_token='h3', history_tokens=tokens,
                future_tokens=tuple(f'f{i}' for i in range(6)))
    before = copy.deepcopy(seed); weights = {k: v.clone() for k, v in joint.state_dict().items()}
    prior_threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        raw = provider.load_raw_columns(Source(), seed, include_gt=False)
        prepared = provider.prepare_columns(None, seed, include_gt=False, raw_window=raw)
        assert 'features' in prepared.state['rec'] and len(prepared.state['current']) == 1
        predictions, _ = rollout.predict_joint_block(prepared, joint.columns, grid, provider.device, 256)
        assert len(predictions) == 6 and provider.columns_checked
        # The ordinary deployment reference must also accept the newly rebuilt
        # four-history record, not just the first bespoke preparation.
        _, deployed = rollout.columns.forecast_columns(provider, Source(), prepared.state['rec'], joint.columns,
                                                       rollout.THRESHOLDS, 256)
        rollout.assert_dense_equal(predictions, deployed)
        assert seed == before and loaded == list(tokens)*2
        assert all(torch.equal(joint.state_dict()[k], v) for k, v in weights.items())
    finally: torch.set_num_threads(prior_threads)


def pinned_test_download(monkeypatch, content):
    monkeypatch.setattr(genie, 'INFO_BYTES', len(content)); monkeypatch.setattr(downloader, 'INFO_BYTES', len(content))
    monkeypatch.setattr(genie, 'INFO_SHA256', hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(downloader.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(content))


def test_download_verified_atomic_and_existing_file_not_overwritten(tmp_path, monkeypatch):
    pinned_test_download(monkeypatch, b'valid metadata')
    target = tmp_path/'official.pkl'; downloader.download(target)
    assert target.read_bytes() == b'valid metadata'
    downloader.download(target)
    target.write_bytes(b'wrong artifact')
    with pytest.raises(RuntimeError, match='mismatch'): downloader.download(target)
    assert target.read_bytes() == b'wrong artifact' and not list(tmp_path.glob('*.part'))


def test_failed_download_does_not_publish_or_delete_existing_temp(tmp_path, monkeypatch):
    pinned_test_download(monkeypatch, b'valid')
    monkeypatch.setattr(downloader.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(b'bad'))
    target = tmp_path/'official.pkl'
    with pytest.raises(RuntimeError, match='mismatch'): downloader.download(target)
    assert not target.exists() and not list(tmp_path.glob('*.part'))
    temporary = tmp_path/f'official.pkl.{os.getpid()}.part'; temporary.write_bytes(b'belongs to someone else')
    with pytest.raises(FileExistsError): downloader.download(target)
    assert temporary.read_bytes() == b'belongs to someone else'


def test_official_hash_is_checked_before_unpickle(tmp_path, monkeypatch):
    path = tmp_path/'unsafe.pkl'; path.write_bytes(b'wrong file')
    monkeypatch.setattr(genie.pickle, 'load', lambda *a: pytest.fail('untrusted pickle deserialized'))
    with pytest.raises(RuntimeError, match='size mismatch'): genie.select_population(path, None, [])
