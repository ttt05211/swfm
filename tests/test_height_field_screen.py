import copy
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from real_motion.height_causal_field import HeightCausalField
from tools.real_motion.height_field_screen_common import epoch_groups, learning_rate, train_step, probabilities, rows_for
from tools.real_motion.height_field_screen_recovery import payload, restore, validate_cursor


def training_fixture(device='cpu'):
    from test_source_repair_pilot import real_training_fixture
    teacher, provider, rows, _ = real_training_fixture()
    provider.device = torch.device(device); teacher = teacher.to(device).eval().requires_grad_(False)
    provider.workers = 1
    head = HeightCausalField('shared_field', z_bins=teacher.columns.config.z_bins,
                             source_dim=teacher.columns.source_dim).to(device)
    return teacher, provider, rows, head


def contract():
    return dict(epoch_batches=[2, 2], epoch_batch_sizes=[[1, 1], [1, 1]], schedule_steps=4, teacher='fixed', epochs=2)


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


def test_epoch_population_exact_and_source_budget_no_truncation():
    records = [dict(features=np.zeros((n, 1)), id=i) for i, n in enumerate((100, 129, 1, 2, 3, 10, 90))]
    for epoch in range(3):
        batches = epoch_groups(records, 19, epoch)
        assert sorted(r['id'] for b in batches for r in b) == list(range(len(records)))
        assert [[r['id'] for r in b] for b in batches] == [[r['id'] for r in b] for b in epoch_groups(records, 19, epoch)]
        assert all(len(b) <= 4 and (len(b) == 1 or sum(len(r['features']) for r in b) <= 128) for b in batches)
    assert learning_rate(0, 100, .002) == .002
    assert learning_rate(100, 100, .002) == pytest.approx(.0002)
    assert learning_rate(50, 100, .002) == pytest.approx(.0011)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_actual_train_step_fresh_encoder_backward_and_frozen_motion(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('actual CUDA required')
    teacher, provider, rows, head = training_fixture(device)
    old = copy.deepcopy(teacher.state_dict())
    before = head.spatial[0][0].weight.detach().clone()
    opt = torch.optim.AdamW(head.parameters(), lr=.001); rng = np.random.default_rng(11)
    calls = []
    handle = head.column.register_forward_hook(lambda *args: calls.append(True))
    with ThreadPoolExecutor(max_workers=2) as pool:
        for _ in range(3):
            stat = train_step(provider, rows*2, teacher, head, opt, rng, candidate_pool=pool)
            assert stat['windows'] == 2 and stat['sampled_columns'] > 0 and np.isfinite(stat['loss'])
            assert stat['generation_bce'] >= 0 and stat['refine_action_ce'] >= 0
            assert stat['transport_frozen'] and stat['KD'] is False
    handle.remove()
    assert len(calls) == 6
    assert not torch.equal(before, head.spatial[0][0].weight)
    assert all(p.grad is None for p in head.parameters())  # released after completed update
    assert all(p.grad is None for p in teacher.parameters())
    assert_nested_equal(old, teacher.state_dict())


def test_resume_reproduces_next_real_update_optimizer_sampling_and_torch_rng():
    teacher, provider, rows, head = training_fixture()
    opt = torch.optim.AdamW(head.parameters(), lr=.001); rng = np.random.default_rng(22)
    train_step(provider, rows, teacher, head, opt, rng)
    saved = copy.deepcopy(payload(head, opt, rng, contract(), epoch=0, batch=1, updates=1, executed=1,
                                  reports={'train_prior': {'population': 'TRAIN'}, 'epochs': []}))
    expected_torch = torch.rand(3)
    train_step(provider, rows, teacher, head, opt, rng)
    other = copy.deepcopy(head); other_opt = torch.optim.AdamW(other.parameters(), lr=.1)
    other_rng = np.random.default_rng(100)
    cursor, reports = restore(saved, other, other_opt, other_rng, contract())
    assert cursor == (0, 1, 1, 1) and reports['train_prior']['population'] == 'TRAIN'
    assert torch.equal(torch.rand(3), expected_torch)
    train_step(provider, rows, teacher, other, other_opt, other_rng)
    assert_nested_equal(head.state_dict(), other.state_dict())
    assert_nested_equal(opt.state_dict(), other_opt.state_dict())
    np.testing.assert_array_equal(rng.random(10), other_rng.random(10))
    for bad in ({**contract(), 'teacher': 'different'}, {**contract(), 'epochs': 3}):
        with pytest.raises(RuntimeError, match='identical'):
            restore(saved, other, other_opt, other_rng, bad)
    with pytest.raises(RuntimeError, match='identical'):
        restore({**saved, 'protocol': 'old_local_joint'}, other, other_opt, other_rng, contract())
    with pytest.raises(RuntimeError, match='counters'):
        restore({**saved, 'executed': 4}, other, other_opt, other_rng, contract())
    with pytest.raises(RuntimeError, match='calibration'):
        restore({**saved, 'reports': {}}, other, other_opt, other_rng, contract())


def test_all_six_inference_is_causal_and_encodes_history_only_once():
    teacher, provider, rows, head = training_fixture()
    record, raw = rows[0]; output = teacher.motion(record, provider.device)
    prep = provider.prepare_columns(None, record, include_gt=True, raw_window=raw, outputs=output)
    rows6 = rows_for(prep, provider.pcfg.grid, teacher.columns.config)
    calls = []
    hook = head.column.register_forward_hook(lambda *args: calls.append(True))
    expected = probabilities(head, prep, output, rows6, provider.pcfg.grid, teacher.columns.config, provider.device, chunk=3)
    assert len(calls) == 1 and len(expected) == 6
    prep.raw = {**prep.raw, 'future_gt_occ': 'POISON', 'future_annotations': 'POISON'}
    actual = probabilities(head, prep, output, rows6, provider.pcfg.grid, teacher.columns.config, provider.device, chunk=3)
    hook.remove()
    for (_, plan, _, _), a, b in zip(rows6, expected, actual):
        np.testing.assert_array_equal(a, b)
        assert a.shape == (*plan.base.shape, 3) and np.isfinite(a).all()
        assert not a[~plan.legal].any()


def test_recovery_epoch_boundaries_and_completed_cycle():
    validate_cursor(contract(), 1, 0, 2, 2)
    validate_cursor(contract(), 2, 0, 4, 4)
    for cursor in ((True, 0, 0, 0), (3, 0, 4, 4), (1, 3, 5, 5), (2, 1, 5, 5)):
        with pytest.raises(RuntimeError):
            validate_cursor(contract(), *cursor)


def test_train_rejects_accidental_live_motion():
    teacher, provider, rows, head = training_fixture(); teacher.requires_grad_(True)
    with pytest.raises(RuntimeError, match='frozen'):
        train_step(provider, rows, teacher, head, torch.optim.AdamW(head.parameters()), np.random.default_rng(1))


def mock_cli(monkeypatch, tmp_path):
    """Small actual network/optimizer, mocked external IO and dev metrics only."""
    from tools.real_motion import train_p0_f9_height_shared_field as cli
    from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import sha256
    teacher, provider, rows, _ = training_fixture()
    record, raw = rows[0]
    paths = {}
    for name in ('checkpoint', 'config', 'train-cache', 'dev-cache', 'population-manifest',
                 'base-checkpoint', 'train-info', 'dev-info'):
        paths[name] = tmp_path/name; paths[name].write_bytes(b'fixture')
    digest = sha256(paths['base-checkpoint'])
    train = [{**record, 'scene_name': 'train', 't0_token': str(i)} for i in range(20430)]
    dev = [{**record, 'scene_name': 'dev', 't0_token': str(i)} for i in range(512)]
    parent = [('dev', str(i)) for i in range(512)]; selected = parent[:64]
    ck = dict(cursor_epoch=19, model_configs={'adaptive_context': None},
              cache_fingerprints={'train': digest, 'dev': digest}, info_fingerprints={'train': digest, 'dev': digest},
              dev_manifest_fingerprint='fixed', dev_keys=parent, train_keys=[('train', str(i)) for i in range(20430)])
    monkeypatch.setattr(cli, 'require_cuda', lambda _: torch.device('cpu'))
    monkeypatch.setattr(cli, 'CLEAN_SHA256', digest)
    monkeypatch.setattr(cli, 'load_runtime_config', lambda *args: {})
    monkeypatch.setattr(cli, 'load_joint', lambda *args, **kwargs: (copy.deepcopy(ck), copy.deepcopy(teacher)))
    monkeypatch.setattr(cli, 'load_manifest', lambda _: ({'parent_keys': parent, 'manifest_fingerprint': 'fixed'}, selected, None))
    monkeypatch.setattr(cli, 'load_cache', lambda path: ({}, train if str(path) == str(paths['train-cache']) else dev))
    monkeypatch.setattr(cli, 'select_population', lambda *args, **kwargs: ([('train', '0'), ('train', '1')], {}))
    monkeypatch.setattr(cli, 'make_prepare_config', lambda _: provider.pcfg)
    monkeypatch.setattr(cli, 'PilotProvider', lambda *args: provider)
    monkeypatch.setattr(cli, 'NuScenesWindowSource', lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(cli, 'CachedColumnSource', lambda source, _: source)
    monkeypatch.setattr(cli, 'prefetch_column_batches', lambda prov, source, records, *args, **kwargs: iter([[(r, raw) for r in records]]))
    def prior(*args, **kwargs):
        return {'population': 'TRAIN'}
    monkeypatch.setattr(cli, 'calibrate_train', prior)
    def evaluation(provider, source, records, teacher, head, **kwargs):
        def variant(value):
            m = dict(mIoU=value, IoU=50., MovingMicro=30., per_horizon={h: {'MovingMicro': 30.} for h in ('1.0', '2.0', '3.0')})
            return dict(metrics=m, delta_vs_v18_pp={'mIoU': .5}, quality={'addition_semantic_precision': .7})
        variants = {n: variant(40.) for n in ('generation', 'refine', 'joint')}
        if kwargs.get('include_old'):
            variants['old_joint'] = variant(40.)
        return dict(windows=len(records), variants=variants)
    monkeypatch.setattr(cli, 'evaluate', evaluation)
    monkeypatch.setattr(cli, 'six_frame_speed', lambda *args, **kwargs: dict(
        speedup=5., six_frame_mean_seconds={'old_joint': .5, 'shared_field': .1},
        six_frame_amortized_FPS={'old_joint': 12., 'shared_field': 60.}, boundary='fixture', excludes='fixture'))
    argv = [item for name, path in paths.items() for item in ('--'+name, str(path))]
    argv += ['--dataroot', str(tmp_path), '--epochs', '2', '--prior-windows', '2', '--eval-windows', '2', '--fps-windows', '1', '--speed-repeats', '1']
    return cli, argv


def test_cli_finite_budget_resume_is_exact_and_rejects_old_contract(monkeypatch, tmp_path):
    cli, argv = mock_cli(monkeypatch, tmp_path)
    full = tmp_path/'full'; stopped = tmp_path/'stopped'; resumed = tmp_path/'resumed'
    assert cli.main(argv=argv+['--out-dir', str(full)]) == 0
    assert cli.main(argv=argv+['--out-dir', str(stopped), '--max-updates', '1']) == 0
    stop_ck = torch.load(stopped/'last.pt', weights_only=False)
    assert (stop_ck['epoch'], stop_ck['batch'], stop_ck['updates'], stop_ck['executed']) == (1, 0, 1, 2)
    assert cli.main(argv=argv+['--out-dir', str(resumed), '--resume', str(stopped/'last.pt')]) == 0
    a = torch.load(full/'last.pt', weights_only=False); b = torch.load(resumed/'last.pt', weights_only=False)
    assert (b['epoch'], b['batch'], b['updates'], b['executed']) == (2, 0, 2, 4)
    assert_nested_equal(a['head'], b['head']); assert_nested_equal(a['optimizer'], b['optimizer'])
    assert a['numpy_rng'] == b['numpy_rng'] and torch.equal(a['torch_rng'], b['torch_rng'])
    assert [r['epoch'] for r in b['reports']['epochs']] == [1, 2]
    with pytest.raises(RuntimeError, match='identical'):
        cli.main(argv=argv+['--out-dir', str(tmp_path/'bad'), '--resume', str(stopped/'last.pt'), '--epochs', '3'])


def test_cli_error_does_not_publish_half_failed_update(monkeypatch, tmp_path):
    cli, argv = mock_cli(monkeypatch, tmp_path)
    original = cli.train_step; calls = []
    def fail_second(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            # Deliberately corrupt live weights: must NOT replace prior saved boundary.
            with torch.no_grad():
                args[3].generation.bias.fill_(99.)
            raise RuntimeError('injected failed update')
        return original(*args, **kwargs)
    monkeypatch.setattr(cli, 'train_step', fail_second)
    out = tmp_path/'failed'
    with pytest.raises(RuntimeError, match='injected'):
        cli.main(argv=argv+['--out-dir', str(out)])
    ck = torch.load(out/'last.pt', weights_only=False)
    assert ck['updates'] == 1 and ck['epoch'] == 1
    assert ck['head']['generation.bias'].item() != 99.
    assert (out/'last.previous.pt').is_file()
