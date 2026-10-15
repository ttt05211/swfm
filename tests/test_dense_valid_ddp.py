"""Two actual CPU/Gloo ranks: global means, ragged shards and Adam/RNG resume."""
import copy
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
from tools.dense_valid_ddp import engine, run
from tools.real_motion.joint_training_distributed import DistributedTraining
from tools.real_motion.joint_training_recovery import save_resume_checkpoint


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.common = torch.nn.Linear(2, 18)
        self.rare = torch.nn.Linear(2, 18)
        self.unused = torch.nn.Linear(2, 1)


def forward(model, provider, row):
    x = row[1]['x']
    value = model.common(x)
    if row[0] % 2: value = value+model.rare(x)
    return dict(logits=value.T[None].expand(6, -1, -1))


def _rank(rank, directory):
    torch.set_num_threads(1)
    os.environ.update(WORLD_SIZE='2', RANK=str(rank), LOCAL_RANK=str(rank))
    ctx = DistributedTraining(True, backend='gloo', init_method=Path(directory, 'rendezvous').resolve().as_uri())
    try:
        with patch.dict(os.environ, WORLD_SIZE='1'):
            single = DistributedTraining(False)
        torch.manual_seed(21)
        model = Tiny(); reference = copy.deepcopy(model)
        adam = torch.optim.AdamW([dict(params=model.parameters(), lr=.001, initial_lr=.001)])
        expected_adam = torch.optim.AdamW([dict(params=reference.parameters(), lr=.001, initial_lr=.001)])
        rows = [(i, dict(x=torch.tensor([[i*.1, .2], [.1, -.3], [.4, i*.2]]),
            future_gt_occ=np.array([[0, 4, 17]]*6)), None, None) for i in range(4)]
        provider = SimpleNamespace(device=torch.device('cpu'))
        unused = copy.deepcopy(model.unused.state_dict())
        for update, ids in enumerate(([0, 1, 2, 3], [0, 1, 2], [0]), 1):
            group = [rows[i] for i in ids]
            expected = engine.train_step(reference, expected_adam, provider, group, single,
                global_windows=len(group), update=update, total_updates=4, forward=forward)
            value = engine.train_step(model, adam, provider, group[rank::2], ctx,
                global_windows=len(group), update=update, total_updates=4, forward=forward)
            assert value['CE'] == pytest.approx(expected['CE'], abs=2e-6)
            for key in reference.state_dict():
                torch.testing.assert_close(model.state_dict()[key], reference.state_dict()[key], atol=1e-7, rtol=1e-6)
            for key, item in expected_adam.state_dict()['state'].items():
                for name, tensor in item.items():
                    torch.testing.assert_close(adam.state_dict()['state'][key][name], tensor, atol=2e-8, rtol=1e-5)
        for key, value in unused.items(): torch.testing.assert_close(model.unused.state_dict()[key], value, atol=0, rtol=0)
        rng = np.random.default_rng(21+rank); torch.manual_seed(21+rank)
        states = ctx.gather(ctx.rng_state(rng))
        expected_np, expected_torch = rng.random(3), torch.rand(3)
        ctx.restore_rng(states, rng)
        np.testing.assert_array_equal(rng.random(3), expected_np)
        torch.testing.assert_close(torch.rand(3), expected_torch, atol=0, rtol=0)
        if ctx.primary:
            save_resume_checkpoint(Path(directory)/'last.pt', dict(model=model.state_dict(), adam=adam.state_dict(), rank_rng=states))
        ctx.barrier()
        saved = torch.load(Path(directory)/'last.pt', weights_only=False)
        other = Tiny(); other.load_state_dict(saved['model'])
        other_adam = torch.optim.AdamW([dict(params=other.parameters(), lr=.001, initial_lr=.001)])
        other_adam.load_state_dict(saved['adam']); ctx.restore_rng(saved['rank_rng'], rng)
        for target, opt in ((model, adam), (other, other_adam)):
            engine.train_step(target, opt, provider, rows[rank::2], ctx, global_windows=4,
                              update=4, total_updates=4, forward=forward)
        for key in model.state_dict(): torch.testing.assert_close(model.state_dict()[key], other.state_dict()[key], atol=0, rtol=0)
        assert ctx.stop(rank == 1)
        Path(directory, f'rank{rank}.json').write_text(json.dumps(dict(ok=True)))
    finally: ctx.close()


def test_actual_two_rank_global_updates_and_resume(tmp_path):
    workers = mp.spawn(_rank, args=(str(tmp_path),), nprocs=2, join=False)
    deadline = time.monotonic()+90
    try:
        while not workers.join(timeout=1):
            if time.monotonic() > deadline: raise AssertionError('Gloo collective timeout')
    finally:
        for child in workers.processes:
            if child.is_alive(): child.terminate(); child.join(timeout=5)
    assert all(json.loads((tmp_path/f'rank{r}.json').read_text())['ok'] for r in (0, 1))


def test_real_small_grid_matches_original_h6_step():
    from test_full_gt_training import fixture
    torch.set_num_threads(1)
    original, adam, provider, row = fixture()
    model = copy.deepcopy(original)
    opt = torch.optim.AdamW([dict(params=model.transport.parameters(), lr=1e-3, initial_lr=1e-3),
        dict(params=model.columns.parameters(), lr=1e-3, initial_lr=1e-3),
        dict(params=model.ego.parameters(), lr=1e-3, initial_lr=1e-3)])
    torch.manual_seed(100); before = torch.get_rng_state()
    expected = engine.gt.train_arm(original, adam, provider, [row], 'gt', update=1, total_updates=4)
    torch.set_rng_state(before)
    actual = engine.train_step(model, opt, provider, [row], DistributedTraining(False),
                               global_windows=1, update=1, total_updates=4)
    assert actual['CE'] == pytest.approx(expected['CE'], abs=1e-7)
    for k in model.state_dict(): torch.testing.assert_close(model.state_dict()[k], original.state_dict()[k], atol=0, rtol=0)


def test_dense_train_only_and_invalid_labels_preserved():
    raw = dict(history_occ=np.array([[[[17, 255, 4]]]]*4, np.uint8),
               history_observed=np.zeros((4, 1, 1, 3), bool), future_gt_occ=object())
    row = ({}, raw, None, None)
    changed = engine.dense_rows([row])[0][1]
    assert changed['history_observed'].tolist() == [[[[True, False, True]]]]*4
    assert not raw['history_observed'].any() and changed['future_gt_occ'] is raw['future_gt_occ']


def test_merge_counts_not_average_rank_iou_and_cursor(tmp_path):
    from tools.perception_robustness.evaluation import ROUTES, PROTOCOL
    from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import Metrics
    keys = [['s', 'a'], ['s', 'b']]; states = []
    ref = Metrics()
    for i, (n, wrong) in enumerate(((2, False), (20, True))):
        m = Metrics(); target = np.full(n, 4); pred = np.full(n, 17 if wrong else 4)
        for h in range(3): m.update(h, pred, target, np.ones(n, bool)); ref.update(h, pred, target, np.ones(n, bool))
        states.append(dict(protocol=PROTOCOL, windows=1, keys=[keys[i]],
            metrics={name: {k: v.tolist() for k, v in vars(m).items()} for name in ROUTES}))
    result = engine.merge_evaluation(states, keys)
    assert result['variants']['gt']['occ_gt_transport']['mIoU'] == ref.compute()['mIoU']
    with pytest.raises(RuntimeError, match='dropped/duplicated'): engine.merge_evaluation([states[0]]*2, keys)
    contract = dict(total_updates=4)
    value = dict(protocol=run.PROTOCOL, contract=contract, epoch=0, batch=1, updates=1, rank_rng=[{}, {}])
    assert run.validate_cursor(value, contract, [[0, 1], [2, 3]]) == (0, 1, 1)
    value['updates'] = 2
    with pytest.raises(RuntimeError, match='cursor'): run.validate_cursor(value, contract, [[0, 1], [2, 3]])
    path = tmp_path/'reference.json'; path.write_text(json.dumps(dict(contract=dict(eval_keys=[['s', str(i)] for i in range(64)]))))
    assert len(run.eval_keys(path)) == 64
    assert [u for u in range(10293) if run.evaluation_due(u, 10292)] == [2000, 4000, 6000, 8000, 10000, 10292]


def _cli_rank(rank, directory):
    """Actual runner/backward/checkpoint across ranks; simulated dataset/metrics."""
    from dataclasses import asdict
    from threading import Event
    from tools.perception_robustness.evaluation import ROUTES, PROTOCOL
    from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import Metrics
    from test_full_gt_training import fixture
    os.environ.update(WORLD_SIZE='2', RANK=str(rank), LOCAL_RANK=str(rank))
    ctx = DistributedTraining(True, backend='gloo', init_method=Path(directory, 'cli_rendezvous').resolve().as_uri())
    monkey = pytest.MonkeyPatch()
    try:
        torch.set_num_threads(1)
        model, _, provider, row = fixture(); provider.pcfg.free_label = 17
        root = Path(directory); out = root/'run'
        if rank == 0:
            for name in ('config', 'train_cache', 'dev_cache', 'train_info', 'dev_info', 'manifest'):
                (root/name).write_text('fixture')
            torch.save(dict(model_config=asdict(model.transport.config)), root/'architecture.pt')
            (root/'keys.json').write_text(json.dumps(dict(contract=dict(eval_keys=[['DEV', f'dev{i}'] for i in range(64)]))))
            for token in row[0]['history_tokens'][-4:]:
                path = root/'stc/DEV'/str(token); path.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path/'labels.npz', semantics=row[1]['history_occ'][-1])
            out.mkdir()
        ctx.barrier()
        train = [{**row[0], 'scene_name': 'TRAIN', 't0_token': f'train{i}'} for i in range(20430)]
        dev = [{**row[0], 'scene_name': 'DEV', 't0_token': f'dev{i}'} for i in range(4369)]
        monkey.setattr(run.base, 'load_cache', lambda path: ({}, train if str(path) == str(root/'train_cache') else dev))
        monkey.setattr(run.base, 'load_runtime_config', lambda *a: None)
        monkey.setattr(run.base, 'make_prepare_config', lambda _: provider.pcfg)
        monkey.setattr(run.base, 'epoch_groups', lambda records, *a: [records[:4], records[4:5]])
        def factory(*args): provider.joint = args[-2]; return provider
        monkey.setattr(run.base, 'PilotProvider', factory)
        monkey.setattr(run.base, 'NuScenesWindowSource', lambda *a, **k: SimpleNamespace(nusc=None))
        monkey.setattr(run.base, 'CachedColumnSource', lambda source, *a, **k: source)
        def caches(p, *args):
            p.ccr_history_cache = p.ccr_val_history_cache = SimpleNamespace(close=lambda: None)
            return dict(train=dict(namespace='train'), val=dict(namespace='val'))
        monkey.setattr(run.base, 'attach_caches', caches)
        class Disposable:
            kernels = pool = None
            def __init__(self, *a): pass
            def close(self): pass
        monkey.setattr('real_motion.canonical_repair_execution.CanonicalCpuExecution', Disposable)
        monkey.setattr('real_motion.strong_majority_execution.ParallelNativeMajority', Disposable)
        original = run.GTBody
        monkey.setattr(run, 'GTBody', lambda cfg: original(cfg, feature_width=16, z_bins=4, memory_width=4, width=8, chunk_size=32))
        def prefetch(p, source, groups, reader, **kw):
            for group in groups: yield [(rec, row[1], row[2], reader.get(rec, row[1])) for rec in group]
        monkey.setattr(run.data, 'prefetched_batches', prefetch)
        def evaluator(*a, **kw):
            def evaluate(models, p, source, batches, *, save, **kw):
                keys = [[r[0]['scene_name'], r[0]['t0_token']] for rows in batches for r in rows]
                m = Metrics()
                for h in range(3): m.update(h, np.array([4]), np.array([4]), np.array([True]))
                save(dict(protocol=PROTOCOL, windows=len(keys), keys=keys,
                          metrics={n: {k: v.tolist() for k, v in vars(m).items()} for n in ROUTES}))
            return evaluate
        monkey.setattr(run, 'evaluator', evaluator)
        args = ['--config', str(root/'config'), '--base-checkpoint', str(root/'architecture.pt'),
            '--dataroot', str(root), '--population-manifest', str(root/'manifest'),
            '--ccr-history-cache', str(root/'oldtrain'), '--ccr-val-history-cache', str(root/'oldval'),
            '--out-dir', str(out), '--stc-root', str(root/'stc'), '--eval-reference', str(root/'keys.json'),
            '--source-policy', 'baseline', '--track-cache', str(root/'newtracks')]
        for name in ('train_cache', 'dev_cache', 'train_info', 'dev_info'):
            args += ['--'+name.replace('_', '-'), str(root/name)]
        a = run.parser().parse_args(args+['--max-updates', '1'])
        assert run.launch(a, Event(), ctx) == 0
        ck = torch.load(out/'last.pt', weights_only=False)
        assert (ck['epoch'], ck['batch'], ck['updates']) == (0, 1, 1)
        a = run.parser().parse_args(args+['--resume', str(out/'last.pt')])
        assert run.launch(a, Event(), ctx) == 0
        ck = torch.load(out/'last.pt', weights_only=False)
        assert (ck['epoch'], ck['batch'], ck['updates']) == (2, 0, 4)
        assert set(ck['evaluations']) == {'4'}
        if rank == 0:
            result = json.loads((out/'results.json').read_text())
            assert result['status'] == 'complete' and result['evaluations']['4']['windows'] == 64
            assert (out/'epoch_0001.pt').exists() and (out/'epoch_0002.pt').exists()
        ctx.barrier()
    finally: monkey.undo(); ctx.close()


def test_runner_two_rank_training_resume_and_final64(tmp_path):
    workers = mp.spawn(_cli_rank, args=(str(tmp_path),), nprocs=2, join=False)
    deadline = time.monotonic()+90
    try:
        while not workers.join(timeout=1):
            if time.monotonic() > deadline: raise AssertionError('runner collective timeout')
    finally:
        for child in workers.processes:
            if child.is_alive(): child.terminate(); child.join(timeout=5)
