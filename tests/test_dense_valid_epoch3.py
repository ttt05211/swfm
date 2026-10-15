"""Continuation LR/Adam/RNG and actual two-rank runner with tiny simulated data."""
import copy
import json
import os
from pathlib import Path
from threading import Event
import time
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
from tools.dense_valid_epoch3 import run, engine
from tools.real_motion.joint_training_distributed import DistributedTraining


def test_monotone_tail_starts_at_actual_source_lr_and_fixed_cumulative_eval():
    start = [5.0001e-6, 3.0001e-6]
    values = [engine.tail_lrs(start, i, 17) for i in range(1, 18)]
    assert values[0] == start
    assert values[-1] == pytest.approx([v*.1 for v in start])
    assert all(a[0] >= b[0] for a, b in zip(values, values[1:]))
    assert [u for u in range(10293, 15440) if run.due(u, 15439)] == [12000, 14000, 15439]
    with pytest.raises(ValueError): engine.tail_lrs(start, 18, 17)


def _runner_rank(rank, directory):
    from test_full_gt_training import fixture
    from tools.dense_valid_ddp import engine as old_engine
    from tools.perception_robustness.evaluation import ROUTES
    from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import Metrics
    os.environ.update(WORLD_SIZE='2', RANK=str(rank), LOCAL_RANK=str(rank))
    ctx = DistributedTraining(True, backend='gloo', init_method=Path(directory, 'rendezvous').resolve().as_uri())
    monkey = pytest.MonkeyPatch()
    try:
        torch.set_num_threads(1); torch.manual_seed(21)
        model, _, provider, row = fixture()
        opt = torch.optim.AdamW([dict(params=model.transport.parameters(), lr=1e-3, initial_lr=1e-3),
                                dict(params=model.columns.parameters(), lr=2e-3, initial_lr=2e-3)])
        # Populate real Adam moments/steps before source import.
        with monkey.context() as m:
            m.setenv('WORLD_SIZE', '1'); single = DistributedTraining(False)
        for update in range(1, 5):
            old_engine.train_step(model, opt, provider, [row], single, global_windows=1,
                                  update=update, total_updates=4)
        rng = np.random.default_rng(21+rank); states = ctx.gather(ctx.rng_state(rng))
        root = Path(directory); out = root/'epoch3'; source = root/'original/epoch_0002.pt'
        if ctx.primary:
            source.parent.mkdir(); out.mkdir()
            for n in ('config', 'train_cache', 'dev_cache', 'base_checkpoint', 'train_info', 'dev_info'):
                (root/n).write_text('fixture')
            prior = dict(protocol=run.original.PROTOCOL, epochs=2, training_known='dense_valid',
                source_policy='shared_pinned', global_batch=4, source_budget=128, total_updates=4,
                train_keys=[['TRAIN', f't{i}'] for i in range(20430)],
                eval_keys=[['DEV', f'd{i}'] for i in range(64)], model=model.configs(), seed=21,
                objective=dict(CE=1., Lovasz=1., motion=0., existence=0., ego=0.),
                data={n: run.base.sha256(root/n) for n in ('config', 'train_cache', 'dev_cache',
                    'base_checkpoint', 'train_info', 'dev_info')}, implementation={}, source_implementation={},
                cache_namespace=dict(train='train', val='val'), STC_inventory=[])
            torch.save(dict(protocol=run.original.PROTOCOL, contract=prior, epoch=2, batch=0, updates=4,
                rank_rng=states, models={'gt': model.state_dict()}, optimizers={'gt': opt.state_dict()}), source)
        ctx.barrier()
        source_bytes = source.read_bytes()
        records = [dict(row[0], scene_name='TRAIN', t0_token=f't{i}') for i in range(20430)]
        dev = [dict(row[0], scene_name='DEV', t0_token=f'd{i}') for i in range(64)]
        monkey.setattr(run.base, 'load_cache', lambda p: ({}, records if str(p) == str(root/'train_cache') else dev))
        monkey.setattr(run.base, 'load_runtime_config', lambda *a: None)
        monkey.setattr(run.base, 'make_prepare_config', lambda _: provider.pcfg)
        def groups(records, seed, epoch, batch, budget):
            assert (seed, epoch, batch, budget) == (21, 2, 4, 128)
            return [records[:4], records[4:5]]
        monkey.setattr(run.base, 'epoch_groups', groups)
        def factory(*args):
            provider.joint = args[-2]; return provider
        monkey.setattr(run, 'TrackProvider', factory)
        provider.attach_tracks = lambda *a, **k: None
        provider.close_tracks = lambda: None
        monkey.setattr(run.base, 'NuScenesWindowSource', lambda *a, **k: SimpleNamespace(nusc=None))
        monkey.setattr(run.base, 'CachedColumnSource', lambda s, *a, **k: s)
        def caches(p, *a):
            p.ccr_history_cache = p.ccr_val_history_cache = SimpleNamespace(close=lambda: None)
            return dict(train=dict(namespace='train'), val=dict(namespace='val'))
        monkey.setattr(run.base, 'attach_caches', caches)
        class Disposable:
            kernels = pool = None
            def __init__(self, *a): pass
            def close(self): pass
        monkey.setattr('real_motion.canonical_repair_execution.CanonicalCpuExecution', Disposable)
        monkey.setattr('real_motion.strong_majority_execution.ParallelNativeMajority', Disposable)
        def prefetch(p, s, groups, reader, **kw):
            assert isinstance(reader, run.DenseTrainHistory)
            for group in groups: yield [(r, row[1], row[2], reader.get(r, row[1])) for r in group]
        monkey.setattr(run.data, 'prefetched_batches', prefetch)
        def evaluator(*a, **kw):
            def evaluate(models, p, source, batches, *, save, **kw):
                keys = [[r[0]['scene_name'], r[0]['t0_token']] for rows in batches for r in rows]
                m = Metrics()
                for h in range(3): m.update(h, np.array([4]), np.array([4]), np.array([True]))
                save(dict(protocol=run.EVAL_PROTOCOL, windows=len(keys), keys=keys,
                    metrics={n: {k: v.tolist() for k, v in vars(m).items()} for n in ROUTES}))
                torch.rand(2)  # Evaluation must not consume training RNG.
            return evaluate
        monkey.setattr(run, 'evaluator', evaluator)
        a = SimpleNamespace(out_dir=str(out), resume=False, config=str(root/'config'),
            train_cache=str(root/'train_cache'), dev_cache=str(root/'dev_cache'),
            base_checkpoint=str(root/'base_checkpoint'), train_info=str(root/'train_info'),
            dev_info=str(root/'dev_info'), cpu_workers=1, frame_cache_mib=1, dataroot=str(root),
            track_cache=str(root/'tracks'), descriptor_ram_mib=0, prefetch_workers=1,
            stc_root=str(root/'stc'), checkpoint_every=1, max_updates=1)
        run.launch(a, Event(), ctx, source)
        first = torch.load(out/'last.pt', weights_only=False)
        assert (first['updates'], first['epoch'], first['batch'], first['additional_updates']) == (5, 2, 1, 1)
        assert first['contract']['start_lrs'] == [g['lr'] for g in opt.param_groups]
        assert set(first['evaluations']) == {'4'}
        # Resume imports staged Adam rather than resetting; includes empty second rank.
        a.resume = True; a.max_updates = 0
        run.launch(a, Event(), ctx, source)
        final = torch.load(out/'last.pt', weights_only=False)
        assert (final['updates'], final['epoch'], final['batch']) == (6, 3, 0)
        assert set(final['evaluations']) == {'4', '6'}
        assert (out/'epoch_0003.pt').exists()
        for key, state in final['optimizers']['gt']['state'].items():
            assert state['step'] == first['optimizers']['gt']['state'][key]['step']+1
        assert [g['lr'] for g in final['optimizers']['gt']['param_groups']] == pytest.approx(
            [g['lr']*.1 for g in opt.param_groups])
        assert final['contract']['OCC_eval_known'] == 'dense_valid'
        assert final['contract']['STC_eval_known'] == 'dense_valid'
        assert source.read_bytes() == source_bytes
        assert json.loads((out/'results.json').read_text())['status'] == 'complete'
    finally: monkey.undo(); ctx.close()


def test_real_two_rank_continuation_resume_dense_eval_and_unchanged_source(tmp_path):
    workers = mp.spawn(_runner_rank, args=(str(tmp_path),), nprocs=2, join=False)
    deadline = time.monotonic()+90
    try:
        while not workers.join(timeout=1):
            if time.monotonic() > deadline: raise AssertionError('epoch3 collective timeout')
    finally:
        for child in workers.processes:
            if child.is_alive(): child.terminate(); child.join(timeout=5)
