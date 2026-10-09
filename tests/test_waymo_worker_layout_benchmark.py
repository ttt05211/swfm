"""Same-v2 layout fairness, correctness and read-only scientific prefix."""
from copy import deepcopy
import json
from pathlib import Path
from threading import Event

import numpy as np
import pytest
import torch

from real_motion.waymo_i2world import file_sha256, fingerprint
from tools.real_motion import waymo_worker_layout_benchmark as bench
from tools.real_motion import benchmark_p0_f9_waymo10_worker_layout as cli
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from test_waymo_parallel_execution import spawned_fixture


def fake_pools(monkeypatch, *, wrong=None, durations=(2., 1., 1., 2.)):
    calls, pools = [], []
    def row(index, processes, hashes):
        signature = {k: 'same' for k in ('transport', 'dense', 'probability', 'motion')}
        if wrong == 'hash' and processes == 4:
            signature['dense'] = 'different'
        return dict(index=index, signatures=signature if hashes else None,
            counts={'transport': [index], 'joint': [index + (wrong == 'counts' and processes == 4)]},
            edits={'added': 1, 'removed': int(wrong == 'remove')},
            exactness_passed=wrong != 'unchecked', pid=processes,
            model_stages={'head': 1.})
    class FakePool:
        def __init__(self, spec, processes):
            self.spec, self.processes, self.closed = spec, processes, False
            pools.append(self)
        def reset(self, indices, chunk):
            calls.append(('reset', self.processes, tuple(indices), chunk))
        def batches(self, indices, chunk, *, hashes=False, stop_event=None):
            calls.append(('pass', self.processes, hashes))
            indices = indices[:-1] if wrong == 'short' else indices
            yield [row(i, self.processes, hashes) for i in indices]
        def close(self):
            self.closed = True
    monkeypatch.setattr(bench, 'SpawnPool', FakePool)
    ticks = [0., 1., 2., 3.]
    now = 3.
    for duration in durations:
        ticks.extend((now, now + duration)); now += duration
    clock = iter(ticks)
    monkeypatch.setattr(bench.time, 'perf_counter', lambda: next(clock))
    return pools, calls


def test_layout_balanced_order_same_backend_budget_resets_and_scope(monkeypatch):
    pools, calls = fake_pools(monkeypatch)
    progress = []
    spec = dict(workers=2, backend='v2')
    d = bench.paired_worker_layout_speed(spec, list(range(10, 74)), repeats=2, progress=progress.append)
    assert [p.spec['workers'] for p in pools] == [2, 1]
    assert all(p.spec['backend'] == 'v2' and p.spec['workers'] * p.processes == 4 for p in pools)
    assert spec == dict(workers=2, backend='v2')  # no caller mutation
    assert [c[1] for c in calls if c[0] == 'pass'] == [2, 4, 2, 4, 4, 2]
    assert len([c for c in calls if c[0] == 'reset']) == 6
    assert len(progress) == 4 and all(p.closed for p in pools)
    assert d['speedup_4x1_vs_2x2'] == 2 and d['recommended_layout'] == 'parallel_4x1'
    assert d['counts_exact'] and d['no_metric_cursor_updates']
    assert d['no_automatic_resume_or_backend_promotion'] and 'NOT formal FPS' in d['scope']


@pytest.mark.parametrize('durations', [(2., 1.95, 1.95, 2.), (2., 1., 3., 2.)])
def test_marginal_or_inconsistent_gain_keeps_two_processes(monkeypatch, durations):
    fake_pools(monkeypatch, durations=durations)
    d = bench.paired_worker_layout_speed(dict(workers=2), list(range(32)))
    assert d['recommended_layout'] == 'parallel_2x2'


@pytest.mark.parametrize('wrong', ['hash', 'counts', 'remove', 'unchecked', 'short'])
def test_failed_gate_closes_pools_without_timing_or_selection(monkeypatch, wrong):
    pools, calls = fake_pools(monkeypatch, wrong=wrong)
    with pytest.raises((RuntimeError, InterruptedError)):
        bench.paired_worker_layout_speed({}, list(range(32)))
    assert all(p.closed for p in pools)
    assert all(c[2] for c in calls if c[0] == 'pass')


@pytest.mark.parametrize('indices,repeats', [([], 2), ([1, 2, 3], 2), ([1, 2, 2, 3], 2),
    ([1, 3, 4, 5], 2), (list(range(129)), 2), (list(range(32)), 3)])
def test_invalid_or_unbalanced_workload_rejected(indices, repeats):
    with pytest.raises(ValueError):
        bench.paired_worker_layout_speed({}, indices, repeats=repeats)


def test_prestart_interrupt_never_creates_worker(monkeypatch):
    monkeypatch.setattr(bench, 'SpawnPool', lambda *a: pytest.fail('no new workers on stop'))
    stopped = Event(); stopped.set()
    with pytest.raises(InterruptedError):
        bench.paired_worker_layout_speed({}, list(range(32)), stop_event=stopped)


def test_actual_four_vs_two_spawned_cpu_models_same_bytes_and_counts(tmp_path):
    _, spec = spawned_fixture(tmp_path, 'cpu')
    d = bench.paired_worker_layout_speed(spec, list(range(8)), chunk=2, repeats=2)
    assert d['counts_exact'] and d['motion_bytes_exact'] and d['probability_and_six_dense_bytes_exact']
    assert len(d['correctness_worker_pids']['parallel_2x2']) == 2
    assert len(d['correctness_worker_pids']['parallel_4x1']) == 4
    assert file_sha256(spec['checkpoint']) == spec['checkpoint_sha256']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='actual CUDA required')
def test_actual_four_vs_two_cuda_models_graphs_same_bytes_and_counts(tmp_path):
    _, spec = spawned_fixture(tmp_path, 'cuda')
    d = bench.paired_worker_layout_speed(spec, list(range(8)), chunk=2, repeats=2)
    assert d['counts_exact'] and d['motion_bytes_exact'] and d['probability_and_six_dense_bytes_exact']
    assert len(d['correctness_worker_pids']['parallel_4x1']) == 4
    assert spec['graphs'] and file_sha256(spec['checkpoint']) == spec['checkpoint_sha256']


def cli_fixture(tmp_path, monkeypatch):
    source = tmp_path / 'old'; source.mkdir()
    checkpoint = tmp_path / 'weights' / 'mean.pt'; checkpoint.parent.mkdir(); checkpoint.write_bytes(b'MEAN')
    data = tmp_path / 'data'; data.mkdir()
    contract = dict(windows=128, data_root=str(data), checkpoint=str(checkpoint),
        checkpoint_sha256=file_sha256(checkpoint), fast_execution=dict(parallel_chunk=8))
    counts = np.zeros((3,18,18), np.int64); counts[:,17,17] = 16 * int(np.prod(cli.SHAPE))
    saved = dict(completed_windows=16, counts={k: counts.tolist() for k in ('transport', 'joint')},
        exactness_passed=True, stage_seconds={}, edits={}, contract_fingerprint=fingerprint(contract))
    saved['fingerprint'] = fingerprint(saved)
    for name, value in (('contract.json', contract), ('state.json', saved)):
        (source / name).write_text(json.dumps(value), encoding='utf-8')
    monkeypatch.setattr(cli.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(cli, 'load_spec', lambda *a: dict(checkpoint=str(checkpoint),
        checkpoint_sha256=file_sha256(checkpoint)))
    return source, checkpoint


def test_cli_only_new_speed_files_source_prefix_and_weights_unchanged(tmp_path, monkeypatch):
    source, ckpt = cli_fixture(tmp_path, monkeypatch)
    originals = {p.name: p.read_bytes() for p in source.iterdir()}
    out = tmp_path / 'probe'; calls = []
    def speed(spec, indices, **kw):
        calls.append(indices)
        return dict(seconds_per_window={'parallel_2x2': .25, 'parallel_4x1': .2},
            speedup_4x1_vs_2x2=1.25, recommended_layout='parallel_4x1')
    monkeypatch.setattr(cli, 'paired_worker_layout_speed', speed)
    assert cli.main(argv=['--continue-from-dir', str(source), '--out-dir', str(out)]) == 0
    assert calls == [list(range(16,80))]
    assert not (out / 'state.json').exists() and not (out / 'waymo_validation.json').exists()
    assert (out / 'speed.json').is_file() and (out / 'summary.txt').is_file()
    assert all((source / k).read_bytes() == v for k,v in originals.items())
    assert ckpt.read_bytes() == b'MEAN'


def test_cli_active_evaluator_lock_blocks_probe(tmp_path, monkeypatch):
    source, _ = cli_fixture(tmp_path, monkeypatch)
    out = tmp_path / 'probe'
    with evaluation_lock(source), pytest.raises(RuntimeError, match='another evaluator'):
        cli.main(argv=['--continue-from-dir', str(source), '--out-dir', str(out)])
    assert not out.exists()


def test_cli_gate_failure_records_no_recommendation_and_no_cursor(tmp_path, monkeypatch):
    source, _ = cli_fixture(tmp_path, monkeypatch)
    before = (source / 'state.json').read_bytes(); out = tmp_path / 'probe'
    def fail(*a, **kw):
        raise RuntimeError('byte gate failure')
    monkeypatch.setattr(cli, 'paired_worker_layout_speed', fail)
    with pytest.raises(RuntimeError, match='gate'):
        cli.main(argv=['--continue-from-dir', str(source), '--out-dir', str(out)])
    assert (out / 'failure.json').is_file() and not (out / 'speed.json').exists()
    assert not (out / 'state.json').exists() and (source / 'state.json').read_bytes() == before


def test_probe_rejects_changed_original_implementation_before_data_access(tmp_path, monkeypatch):
    path = tmp_path / 'config.yaml'; path.write_bytes(b'CONFIG')
    environment = {k:v for k,v in sorted(cli.os.environ.items()) if k.startswith('SWFM_')}
    contract = dict(protocol=cli.PROTOCOL, execution='native_parallel', thresholds=[.5,None],
        fast_execution=dict(protocol=cli.PARALLEL_PROTOCOL), torch_version=str(torch.__version__),
        runtime_environment=environment, config_sha256=file_sha256(path),
        implementation={'AGENTS.md': 'changed'})
    with pytest.raises(RuntimeError, match='implementation changed'):
        cli.load_spec(contract, path)
