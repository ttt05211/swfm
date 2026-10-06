import copy
from types import SimpleNamespace
import threading

import numpy as np
import pytest
import torch

from real_motion.canonical_causal_repair import CanonicalRepairHead
from real_motion.canonical_repair_context import FixedCanonicalCache
from tools.real_motion import ccr_screen_common as screen
from tools.real_motion.height_field_screen_recovery import payload, restore
from test_canonical_causal_repair import scene
from test_height_field_screen import assert_nested_equal, contract, mock_cli


class FrozenTeacher(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.context = torch.nn.Parameter(torch.randn(1, 8, device=device), requires_grad=False)
        self.future = torch.nn.Parameter(torch.randn(1, 6, 8, device=device), requires_grad=False)
        self.columns = SimpleNamespace(source_dim=8)

    def motion(self, record, device):
        return dict(history_source_context=self.context, future_transport_queries=self.future)


def fixture(device='cpu'):
    device = torch.device(device); grid, prep = scene()
    prep.raw['future_gt_occ'] = prep.baseline.copy()
    prep.raw['future_gt_occ'][:, 3, 2, 1] = 4
    provider = SimpleNamespace(device=device, pcfg=SimpleNamespace(grid=grid), workers=1,
        ccr_cache=FixedCanonicalCache(1, neighbors=False), ccr_samples_per_role=8)
    provider.prepare_columns = lambda *args, **kwargs: prep
    teacher = FrozenTeacher(device)
    return teacher, provider, [(dict(scene_name='train', t0_token='a'), prep.raw)], CanonicalRepairHead(8).to(device)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_actual_ccr_train_frozen_motion_updates_new_head_and_releases_graph(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('actual CUDA required')
    teacher, provider, rows, head = fixture(device)
    before = copy.deepcopy(head.state_dict()); teacher_before = copy.deepcopy(teacher.state_dict())
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001); rng = np.random.default_rng(41)
    for _ in range(3):
        stats = screen.train_step(provider, rows*2, teacher, head, optimizer, rng)
        assert stats['windows'] == 2 and stats['sampled_points'] > 0
        assert stats['transport_frozen'] and stats['GT_only'] and stats['KD'] is False
        assert np.isfinite(stats['loss']) and all(p.grad is None for p in head.parameters())
    assert not torch.equal(before['encoder.0.weight'], head.encoder[0].weight)
    assert_nested_equal(teacher_before, teacher.state_dict())
    assert provider.ccr_cache.stats()['hits'] >= 5
    assert provider.ccr_cache.stats()['future_supervision_cached'] is False
    provider.ccr_cache.close()


def test_ccr_resume_next_actual_update_is_bit_exact_and_rejects_height_checkpoint():
    teacher, provider, rows, head = fixture()
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001); rng = np.random.default_rng(3)
    screen.train_step(provider, rows, teacher, head, optimizer, rng)
    saved = copy.deepcopy(payload(head, optimizer, rng, contract(), epoch=0, batch=1, updates=1, executed=1,
        reports={'train_prior': {'TRAIN_only': True}}, protocol=screen.PROTOCOL))
    other = copy.deepcopy(head); other_opt = torch.optim.AdamW(other.parameters(), lr=99.)
    other_rng = np.random.default_rng(9)
    screen.train_step(provider, rows, teacher, head, optimizer, rng)
    assert restore(saved, other, other_opt, other_rng, contract(), protocol=screen.PROTOCOL)[0] == (0, 1, 1, 1)
    screen.train_step(provider, rows, teacher, other, other_opt, other_rng)
    assert_nested_equal(head.state_dict(), other.state_dict()); assert_nested_equal(optimizer.state_dict(), other_opt.state_dict())
    assert rng.bit_generator.state == other_rng.bit_generator.state
    with pytest.raises(RuntimeError, match='identical'):
        restore(saved, other, other_opt, other_rng, contract())
    with pytest.raises(RuntimeError, match='identical'):
        restore(saved, other, other_opt, other_rng, {**contract(), 'samples_per_role': 33}, protocol=screen.PROTOCOL)


def test_train_prior_is_unsampled_train_only_and_stop_fail_closed(monkeypatch):
    teacher, provider, rows, head = fixture()
    monkeypatch.setattr(screen, 'prefetch_raw_columns', lambda *args, **kwargs: iter(rows))
    prior = screen.calibrate_train(provider, None, [rows[0][0]], teacher, head)
    counts = np.asarray(prior['counts']); assert counts.shape == (2, 2, 2) and counts.sum() > 0
    assert prior['windows'] == 1 and 'TRAIN' in prior['population']
    np.testing.assert_array_equal(head.positive_weight.numpy(), np.asarray(prior['positive_weights'], np.float32))
    event = threading.Event(); event.set()
    with pytest.raises(InterruptedError, match='prior interrupted'):
        screen.calibrate_train(provider, None, [rows[0][0]], teacher, head, stop_event=event)


def test_real_metrics_evaluation_uses_full_domain_no_gt_candidate_selection(monkeypatch):
    teacher, provider, rows, head = fixture()
    prep = provider.prepare_columns(); prep.window = SimpleNamespace(t0_token='a', future_tokens=tuple(str(i) for i in range(6)))
    source = SimpleNamespace(nusc=None)
    monkeypatch.setattr(screen, 'prefetch_raw_columns', lambda *args, **kwargs: iter(rows))
    monkeypatch.setattr(screen, 'gt_moving_support_sequence', lambda *args, **kwargs: None)
    monkeypatch.setattr(screen, 'moving_support_masks', lambda *args: [np.zeros(prep.baseline[0].shape, bool)]*6)
    result = screen.evaluate(provider, source, [rows[0][0]], teacher, head)
    assert set(result['variants']) == {'static_repair', 'dynamic_repair', 'joint'}
    assert result['windows'] == 1 and result['variants']['joint']['metrics']['mIoU'] > 0
    assert set(result['variants']['joint']['metrics']) >= {'IoU', 'mIoU', 'MovingMacro', 'MovingMicro', 'per_horizon'}


def test_ccr_orchestration_exact_three_epochs_resume_and_old_height_unaffected(monkeypatch, tmp_path):
    cli, argv = mock_cli(monkeypatch, tmp_path)
    argv += ['--epochs', '3']
    # External IO/metrics are mocked; actual frozen motion, point head and
    # optimizer/sampler steps run through the shared finite-budget engine.
    def setup(provider, args):
        provider.ccr_cache = FixedCanonicalCache(1, neighbors=False)
        provider.ccr_samples_per_role = 8
    monkeypatch.setattr(screen, 'setup', setup)
    monkeypatch.setattr(screen, 'calibrate_train', cli.calibrate_train)
    def evaluation(*args, **kwargs):
        result = cli.evaluate(*args, **kwargs)
        for row in result['variants'].values():
            row['metrics']['MovingMacro'] = 30.
        return result
    monkeypatch.setattr(screen, 'evaluate', evaluation)
    monkeypatch.setattr(screen, 'six_frame_speed', cli.six_frame_speed)
    full, stop, resume = [tmp_path/n for n in ('full', 'stop', 'resume')]
    assert cli.main(argv=argv+['--out-dir', str(full)], backend=screen) == 0
    assert cli.main(argv=argv+['--out-dir', str(stop), '--max-updates', '1'], backend=screen) == 0
    assert cli.main(argv=argv+['--out-dir', str(resume), '--resume', str(stop/'last.pt')], backend=screen) == 0
    a = torch.load(full/'last.pt', weights_only=False); b = torch.load(resume/'last.pt', weights_only=False)
    assert b['protocol'] == screen.PROTOCOL and b['epoch'] == 3 and b['executed'] == 6 and b['updates'] == 3
    assert [r['epoch'] for r in b['reports']['epochs']] == [1, 2, 3]
    assert_nested_equal(a['head'], b['head']); assert_nested_equal(a['optimizer'], b['optimizer'])
    assert a['numpy_rng'] == b['numpy_rng'] and torch.equal(a['torch_rng'], b['torch_rng'])
    assert 'CCR' in (resume/'summary.txt').read_text(encoding='utf-8')
    with pytest.raises(RuntimeError, match='identical'):
        cli.main(argv=argv+['--epochs', '5', '--out-dir', str(tmp_path/'bad'), '--resume', str(stop/'last.pt')], backend=screen)


def test_failed_update_does_not_replace_completed_ccr_checkpoint(monkeypatch, tmp_path):
    cli, argv = mock_cli(monkeypatch, tmp_path)
    monkeypatch.setattr(screen, 'setup', lambda provider, args: (
        setattr(provider, 'ccr_cache', FixedCanonicalCache(1, neighbors=False)), setattr(provider, 'ccr_samples_per_role', 8)))
    monkeypatch.setattr(screen, 'calibrate_train', cli.calibrate_train)
    monkeypatch.setattr(screen, 'evaluate', cli.evaluate)
    original = screen.train_step; calls = []
    def failed(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            with torch.no_grad():
                args[3].readout[-1].bias.fill_(99.)
            raise RuntimeError('injected failed CCR update')
        return original(*args, **kwargs)
    monkeypatch.setattr(screen, 'train_step', failed)
    out = tmp_path/'failed'
    with pytest.raises(RuntimeError, match='injected'):
        cli.main(argv=argv+['--out-dir', str(out)], backend=screen)
    saved = torch.load(out/'last.pt', weights_only=False)
    assert saved['updates'] == 1 and saved['epoch'] == 1
    assert not (saved['head']['readout.3.bias'] == 99.).any()


def test_cache_capacity_and_per_horizon_quality_guard(tmp_path):
    args = SimpleNamespace(descriptor_ram_mib=8193, descriptor_disk_mib=0, descriptor_cache=None, samples_per_role=1)
    with pytest.raises(ValueError, match='8GiB'):
        screen.setup(SimpleNamespace(), args)
    new = dict(mIoU=40., MovingMicro=30., per_horizon={h: {'MovingMicro': 30.} for h in ('1.0', '2.0', '3.0')})
    old = copy.deepcopy(new); old['per_horizon']['3.0']['MovingMicro'] += .3
    gate = screen.gate(new, old, {'speedup': 4.})
    assert gate['mIoU_within_0_20pp'] and gate['MovingMicro_within_0_20pp']
    assert gate['all_horizons_Moving_within_0_20pp'] is False


def test_async_descriptor_write_roundtrip_exact_and_future_gt_not_cached(tmp_path):
    grid, prep = scene()
    first = FixedCanonicalCache(1, neighbors=False, disk_root=tmp_path/'fixed', max_disk_mib=32, async_writes=True)
    e, graph = first.get(prep, grid); assert graph is None
    first.disk.flush()
    assert first.stats()['disk']['writes'] == 1 and first.stats()['disk']['pending_writes'] == 0
    before = e.features.copy(); first.close()
    prep.raw['future_gt_occ'] = 'POISON'
    second = FixedCanonicalCache(0, neighbors=False, disk_root=tmp_path/'fixed', max_disk_mib=32, async_writes=True)
    actual, g = second.get(prep, grid)
    np.testing.assert_array_equal(before, actual.features)
    assert second.stats()['disk']['hits'] == 1 and g is None
    assert second.stats()['future_supervision_cached'] is False
    second.close()
