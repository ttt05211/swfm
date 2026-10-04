"""Global-normalized real losses, Gloo/NCCL synchronization and rank RNG."""
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
from torch import nn
from unittest.mock import patch

from real_motion.causal_column_model import column_loss
from tools.real_motion.joint_column_common import motion_loss
from tools.real_motion.joint_training_distributed import (
    DistributedTraining, prefetch_distributed_batches, merge_rank_stats,
)
from tools.real_motion.joint_training_extension import set_extension_lr


def _worker(rank, directory, empty_rank, cuda):
    torch.set_num_threads(1)
    os.environ.update(WORLD_SIZE='2', RANK=str(rank), LOCAL_RANK=str(rank))
    method = Path(directory, 'rendezvous').resolve().as_uri()
    context = DistributedTraining(True, backend=None if cuda else 'gloo', init_method=method)
    device = context.device
    try:
        torch.manual_seed(17)
        model = nn.Module(); model.transport = nn.Linear(2, 4); model.columns = nn.Linear(2, 12)
        model.columns.register_buffer('generation_pos_weight', torch.tensor(2.))
        model.columns.register_buffer('refine_class_weights', torch.tensor([.2, 3., 4.]))
        model.to(device); original = copy.deepcopy(model)
        x = torch.arange(8, device=device).float().reshape(4, 2)/5
        valid = torch.ones(4, 6, dtype=torch.bool, device=device); valid[1, 2:] = False
        masks = torch.ones(4, 4, 4, 4, device=device); masks[2] = 0
        record = {'supervised_source': torch.tensor([1, 0, 1, 1], device=device).bool(),
            'se2_target_valid': valid, 'target_source_residual_xy_m': torch.full((4, 6, 2), .2, device=device),
            'existence': torch.ones(4, 6, device=device), 'target_yaw_rad': torch.full((4, 6), .1, device=device),
            'yaw_enabled': torch.tensor([1, 1, 0, 1], device=device).bool(), 'yaw_label_valid': valid,
            'kta_displacement_xy_m': torch.zeros(4, 6, 2, device=device),
            'target_source_displacement_xy_m': torch.full((4, 6, 2), .2, device=device),
            'target_source_mask_tube': masks}
        xc = torch.tensor([[.1, .3], [.5, .2], [.9, .1]], device=device)
        kind = torch.tensor([0, 1, 1], device=device)
        target = torch.tensor([[1, 0, 1], [2, 1, 0], [0, 2, 1]], device=device)
        legal = torch.ones(3, 3, 3, dtype=torch.bool, device=device)
        weights = torch.tensor([2., 1., 5.], device=device)
        def losses(m, source_ids, column_ids, normalizer=None):
            values = m.transport(x[source_ids])
            output = {'residual_xy_m': values[:, None, :2].expand(-1, 6, -1),
                      'yaw_delta_rad': values[:, None, 2].expand(-1, 6),
                      'existence_logits': values[:, None, 3].expand(-1, 6)}
            rec = {k: v[source_ids] for k, v in record.items()}
            lm, _ = motion_loss(output, rec, device, distributed=normalizer)
            out = m.columns(xc[column_ids]).reshape(-1, 3, 4)
            lc, _ = column_loss(m.columns, out[..., 0], out[..., 1:], kind[column_ids], legal[column_ids],
                                target[column_ids], weights[column_ids], distributed=normalizer)
            return lm+lc
        expected = losses(original, list(range(4)), list(range(3))); expected.backward()
        if empty_rank:
            si, ci = (list(range(4)), list(range(3))) if rank == 0 else ([], [])
        else:
            si, ci = ([0], [0]) if rank == 0 else ([1, 2, 3], [1, 2])
        local = losses(model, si, ci, context)
        if local.requires_grad: local.backward()
        assert context.synchronize_gradients(model)
        assert abs(float(context.sum(local))-float(expected)) < 2e-5
        for a, b in zip(model.parameters(), original.parameters()):
            assert torch.allclose(a.grad, b.grad, atol=2e-5, rtol=2e-5)
        opt = torch.optim.AdamW(model.parameters(), lr=.0001)
        reference_opt = torch.optim.AdamW(original.parameters(), lr=.0001)
        for m in (model, original):
            nn.utils.clip_grad_norm_(m.transport.parameters(), 5.)
            nn.utils.clip_grad_norm_(m.columns.parameters(), 1.)
        opt.step(); reference_opt.step()
        assert all(torch.allclose(a, b, atol=2e-6, rtol=2e-5) for a, b in zip(model.parameters(), original.parameters()))
        rng = np.random.default_rng(73+rank); torch.manual_seed(73+rank)
        states = context.gather(context.rng_state(rng))
        expected_np, expected_torch = rng.integers(0, 1000, 10), torch.randn(10)
        expected_cuda = torch.randn(10, device=device) if cuda else None
        context.restore_rng(states, rng)
        assert np.array_equal(expected_np, rng.integers(0, 1000, 10))
        assert torch.equal(expected_torch, torch.randn(10))
        if cuda: assert torch.equal(expected_cuda, torch.randn(10, device=device))
        assert context.stop(rank == 1)
        Path(directory, f'rank{rank}.json').write_text(json.dumps({'ok': True}), encoding='utf-8')
    finally: context.close()


def _spawn(directory, empty, cuda=False):
    directory.mkdir()
    process = mp.spawn(_worker, args=(str(directory), empty, cuda), nprocs=2, join=False)
    deadline = time.monotonic()+90
    try:
        while not process.join(timeout=1):
            if time.monotonic() > deadline: raise AssertionError('distributed test timeout (possible collective deadlock)')
    finally:
        for child in process.processes:
            if child.is_alive(): child.terminate(); child.join(timeout=5)
    assert all(json.loads((directory/f'rank{r}.json').read_text())['ok'] for r in (0, 1))


@pytest.mark.parametrize('empty', [False, True])
def test_real_global_losses_and_gradients_match_single_batch_gloo(tmp_path, empty):
    _spawn(tmp_path/'gloo', empty)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='two actual CUDA GPUs required')
def test_actual_dual_cuda_nccl_losses_gradients_and_rng(tmp_path):
    _spawn(tmp_path/'nccl', True, cuda=True)


def test_shards_keep_every_global_step_no_padding_drop_or_source_duplication():
    from types import SimpleNamespace
    records = [{'id': i} for i in range(7)]; groups = [(0, 1, 2, 3), (4,), (5, 6)]
    provider = SimpleNamespace(load_raw_columns=lambda source, row, include_gt: row['id'])
    ranks = []
    for rank in (0, 1):
        context = SimpleNamespace(rank=rank, world_size=2)
        ranks.append(list(prefetch_distributed_batches(provider, None, records, groups, context)))
    assert len(ranks[0]) == len(ranks[1]) == 3 and ranks[1][1] == []
    for step, group in enumerate(groups):
        assert sorted(row['id'] for rows in ranks for row, _ in rows[step]) == list(group)


def test_resume_extension_lr_is_monotone_and_does_not_reset_initial_lr():
    param = nn.Parameter(torch.ones(1))
    opt = torch.optim.AdamW([{'params': [param], 'lr': .00005, 'initial_lr': .0005}])
    c = {'original_updates': 100, 'extension_updates': 5, 'start_learning_rates': [.00005]}
    rates = []
    for update in range(100, 105): set_extension_lr(opt, update, c); rates.append(opt.param_groups[0]['lr'])
    assert rates[0] == .00005 and np.isclose(rates[-1], .000005)
    assert all(a > b for a, b in zip(rates, rates[1:])) and opt.param_groups[0]['initial_lr'] == .0005
    with pytest.raises(RuntimeError, match='outside'): set_extension_lr(opt, 99, c)


def test_log_aggregation_uses_global_sums_but_max_waits():
    rows = [{'loss': 1., 'windows': 2, 'sources': 7, 'sampled_columns': 23, 'input_wait_seconds': .1},
            {'loss': 2., 'windows': 1, 'sources': 1, 'sampled_columns': 4, 'input_wait_seconds': .2}]
    rows[0]['host_stage_seconds'] = {'backward': .1}; rows[1]['host_stage_seconds'] = {'backward': .2, 'motion_forward': .3}
    rows[1]['gpu_feature_fallback_horizons'] = 1
    value = merge_rank_stats(rows)
    assert value['loss'] == 3 and value['windows'] == 3 and value['sources'] == 8
    assert value['sampled_columns'] == 27 and value['input_wait_seconds'] == .2
    assert value['host_stage_seconds'] == {'backward': .2, 'motion_forward': .3}
    assert value['gpu_feature_fallback_horizons'] == 1


def test_globally_unused_parameters_keep_none_and_no_weight_decay():
    model = nn.Linear(2, 2); context = DistributedTraining(False)
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.1, weight_decay=1.)
    assert not context.synchronize_gradients(model)
    optimizer.step()
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())


def _cli_worker(rank, directory, parent):
    from test_joint_causal_columns_full import full_cli_fixture
    from tools.real_motion import train_p0_f9_joint_causal_columns_full as trainer
    directory = Path(directory)
    local = directory/f'inputs{rank}'; local.mkdir()
    run, _, _ = full_cli_fixture.__wrapped__(local)
    os.environ.update(WORLD_SIZE='2', RANK=str(rank), LOCAL_RANK=str(rank))
    for phase in ('baseline', 'paused', 'resumed'):
        context = lambda enabled: DistributedTraining(enabled, backend='gloo',
            init_method=(directory/f'rendezvous_{phase}').resolve().as_uri())
        source = Path(parent) if phase != 'resumed' else directory/'paused'/'last.pt'
        with patch.object(trainer, 'DistributedTraining', side_effect=context):
            value = run(directory/phase, 2, source, history_frames=4, extend=phase != 'resumed',
                        distributed=True, stop_update=14 if phase == 'paused' and rank == 0 else None, profile_every=1)
        assert value == (0 if phase == 'paused' else None if rank == 0 else 0)
    (directory/f'rank{rank}.json').write_text(json.dumps({'ok': True}))


def test_full_dual_extension_interrupt_resume_is_exact_and_rank0_publishes_once(tmp_path):
    from test_joint_causal_columns_full import full_cli_fixture
    inputs = tmp_path/'inputs'; inputs.mkdir()
    run, _, _ = full_cli_fixture.__wrapped__(inputs)
    parent = tmp_path/'original'; run(parent, 1, history_frames=4)
    original = (parent/'last.pt').read_bytes()
    directory = tmp_path/'dual'; directory.mkdir()
    process = mp.spawn(_cli_worker, args=(str(directory), str(parent/'last.pt')), nprocs=2, join=False)
    deadline = time.monotonic()+120
    try:
        while not process.join(timeout=1):
            if time.monotonic() > deadline: raise AssertionError('full dual recovery timeout')
    finally:
        for child in process.processes:
            if child.is_alive(): child.terminate(); child.join(timeout=5)
    assert (parent/'last.pt').read_bytes() == original
    a, b = (torch.load(directory/phase/'last.pt', weights_only=False) for phase in ('baseline', 'resumed'))
    assert a['cursor_epoch'] == b['cursor_epoch'] == 2 and a['attempted_updates'] == b['attempted_updates'] == 20
    assert a['executed_windows'] == b['executed_windows'] == 80
    assert a['distributed_training']['world_size'] == 2 and len(a['distributed_rng_states']) == 2
    assert all(torch.equal(v, b['state_dict'][k]) for k, v in a['state_dict'].items())
    assert a['optimizer']['param_groups'] == b['optimizer']['param_groups']
    for k, state in a['optimizer']['state'].items():
        assert all(torch.equal(v, b['optimizer']['state'][k][n]) for n, v in state.items())
    for first, second in zip(a['distributed_rng_states'], b['distributed_rng_states']):
        assert first['sampling'] == second['sampling'] and torch.equal(first['torch'], second['torch'])
    lines = [json.loads(row) for row in (directory/'resumed'/'progress.jsonl').read_text().splitlines()]
    updates = [row for row in lines if row['event'] == 'train_full']
    assert [row['update'] for row in updates] == list(range(15, 21))
    assert all(row['windows'] == 4 and row['rank_windows'] == [2, 2] for row in updates)
