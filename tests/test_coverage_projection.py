"""Causal support, gradient signals, fresh-init match and exact recovery guards."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from tools.coverage_projection import model, adapter, geometry, memory
from tools.coverage_projection.projection import VolumeProjection
from tools.coverage_projection.run import validate_cursor, PROTOCOL, prefetch, train_step
from tools.causal_forward_train.model import make_model as fresh_control
from tools.shared_history_experiments.refine.geometry import TransportPrior
from tools.source_scene_ego_conditioned.geometry import aligned_memory as old_memory


def test_common_fresh_tensors_RNG_and_only432_added_parameters():
    from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
    cfg = LocalSTWMV17Config(history_frames=4, decoder_blocks=2)
    torch.manual_seed(20261011); control = fresh_control(cfg); rng = torch.get_rng_state().clone()
    torch.manual_seed(20261011); candidate = model.make_model(cfg)
    assert torch.equal(torch.get_rng_state(), rng)
    assert sum(p.numel() for p in candidate.parameters()) == 2816554
    for name, value in control.state_dict().items():
        new = candidate.state_dict()[name]
        if name == 'columns.fine.0.weight':
            torch.testing.assert_close(new[:, :19], value, rtol=0, atol=0)
            assert new[:, 19:].count_nonzero() == 0
        else: torch.testing.assert_close(new, value, rtol=0, atol=0)
    restored = model.from_spec(candidate.configs(), candidate=True)
    restored.load_state_dict(candidate.state_dict(), strict=True)
    assert restored.configs() == candidate.configs()


def test_unknown_prior_has_no_free_bias_and_can_predict_occupied():
    net = model.CoverageRefiner(feature_width=16, memory_width=4, z_bins=4).eval()
    shape = (8, 8, 4); p = torch.zeros(6, 18, np.prod(shape)); p[:, 17] = 1
    m = torch.randn(6, 4, 4, 4, 4); mc = torch.zeros(6, 1, 4, 4, 4)
    pc = torch.zeros(6, 1, np.prod(shape), requires_grad=True)
    none = net(m, p, range(6), shape, prior_coverage=pc, memory_coverage=mc)
    assert none.count_nonzero() == 0  # unknown is neutral, not strong free.
    valid = net(m, p, range(6), shape, prior_coverage=torch.ones_like(pc), memory_coverage=mc)
    torch.testing.assert_close(valid, net.prior_logits(p), rtol=0, atol=0)
    with torch.no_grad(): net.head.bias[11] = 2.
    output = net(m, p, range(6), shape, prior_coverage=pc, memory_coverage=mc)
    assert bool((output.argmax(1) == 11).all())  # no output coverage mask
    output.sum().backward(); assert pc.grad is None


def tiny_scene(device='cpu'):
    from test_source_scene_stage2 import fixture
    old, runner, row = fixture(device)
    from tools.causal_forward_recovery.adapter import RecoveryPredictor
    body = model.make_model(replace(old.transport.config, decoder_blocks=2),
        feature_width=16, z_bins=4, memory_width=4).to(device)
    new = RecoveryPredictor(body, runner.provider.pcfg, device, workers=1)
    raw = {k: row[1][k] for k in ('history_occ', 'history_observed', 'history_poses')}
    prepared = new.prepare(row[0], raw, times=np.arange(4)*.5)
    poses, labels = row[1]['future_poses'], row[1]['future_gt_occ']
    runner.close()
    return body, new, prepared, poses, labels


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_real_forward_prior_parity_memory_parity_and_combined_backward(device):
    body, runner, row, poses, labels = tiny_scene(device)
    try:
        body.train()
        with torch.no_grad(): body.columns.head.weight.normal_(std=.02)
        fw = adapter.forward_window(body, runner.provider, row, poses=poses)
        raw, state = row[1], row[3][0]['state']
        from tools.shared_source_track.adapter import backgrounds
        state = dict(state, column_backgrounds=backgrounds(raw, state, fw['targets'].new_tensor(poses),
            runner.provider, raw['_shared_source_track']))
        prior = TransportPrior(raw, state, runner.provider.pcfg.grid, runner.provider.device).render(
            fw['targets'].new_tensor(poses), fw['targets'], fw['yaws'])
        torch.testing.assert_close(prior, fw['prior'], rtol=0, atol=0)
        assert fw['prior_coverage'].max() <= 1 and fw['prior_coverage'].min() >= 0
        assert not fw['prior_coverage'].requires_grad and not fw['memory_coverage'].requires_grad
        sem = torch.as_tensor(raw['history_occ'], device=device)
        known = torch.as_tensor(raw['history_observed'], device=device)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=device == 'cuda'):
            fields = body.transport.history_encoder(sem, known); lifted = body.columns.lift_history(fields[0])
        args = (body.columns, lifted, fw['output'], row[3], raw, fw['targets'], fw['yaws'],
            fw['targets'].new_tensor(poses), runner.provider.pcfg.grid)
        value, coverage = memory.aligned_memory(*args)
        old_value = old_memory(*args)
        repeated = old_memory(*args)
        exact = torch.equal(value, old_value)
        print('MEMORY_NUMERICS', device, dict(strict_byte=exact,
            old_repeat_strict_byte=torch.equal(old_value, repeated),
            candidate_max_abs=float((value-old_value).abs().max().detach()),
            old_repeat_max_abs=float((old_value-repeated).abs().max().detach())))
        # CUDA scatter_add itself is not bit deterministic. This check asserts
        # floating-point agreement, while reporting exact failures separately.
        torch.testing.assert_close(value, old_value, rtol=1e-6 if device == 'cuda' else 0,
            atol=1e-7 if device == 'cuda' else 0)
        assert coverage.dtype == torch.bool
        from tools.shared_history_experiments.full_gt.engine import semantic_loss
        from tools.joint_occ_experiments.soft_geometry import grid_arrays
        projector = VolumeProjection(*grid_arrays(runner.provider.pcfg.grid), side=8)
        base, _, _ = semantic_loss(fw['logits'], labels)
        projection, _, _ = projector.loss(fw['logits'][0], labels[0])
        (base.mean()+projection).backward()
        assert body.columns.fine[0].weight.grad[:, 19:].abs().sum() > 0
        assert body.transport.history_encoder.stem[0].weight.grad.abs().sum() > 0
        assert body.columns.importance.weight.grad.abs().sum() > 0
        assert body.transport.residual_head.weight.grad.abs().sum() > 0
        assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in body.parameters())
    finally: runner.close()


def test_source_outside_static_footprint_keeps_coverage():
    # Use a real tiny scene, then project the source far beyond the t0 field.
    body, runner, row, poses, labels = tiny_scene()
    try:
        fw = adapter.forward_window(body, runner.provider, row, poses=poses)
        p = torch.eye(4, dtype=torch.float64).repeat(6, 1, 1); p[:, 0, 3] = 100
        state = dict(row[3][0]['state'], column_backgrounds=[np.full(fw['shape'], 17)]*6)
        targets = fw['targets'].detach().clone(); targets[:, :, 0] += 100
        scene = geometry.CoveredPrior(row[1], state, runner.provider.pcfg.grid, torch.device('cpu'))
        prior, support = scene.render(p, targets, fw['yaws'].detach())
        dynamic = prior[:, 4:5] > .01
        assert dynamic.any() and bool((support[dynamic] > 0).all())
        assert support.min() == 0  # support is not a full-grid mask
    finally: runner.close()


def test_projection_perfect_target_and_gradients_for_missing_and_false_surface():
    shape = (12, 12, 4); renderer = VolumeProjection((-2.4, -2.4, -.8), (.4, .4, .4), shape, side=16)
    gt = torch.full(shape, 17, dtype=torch.long); gt[2:10, 2:10, 0] = 11
    target = torch.nn.functional.one_hot(gt, 18).permute(3, 0, 1, 2).float()
    for view in ('top', 'oblique'):
        s, depth, hit = renderer.render(target, view)
        torch.testing.assert_close(s.sum(1), torch.ones_like(hit), atol=1e-6, rtol=1e-6)
        assert torch.isfinite(depth).all() and (hit > 0).any()
    # Missing road: GT hit supplies a depth term even when prediction is free.
    logits = torch.full((18, np.prod(shape)), -3.); logits[17] = 3.; logits.requires_grad_()
    loss, semantic, depth = renderer.loss(logits, gt)
    loss.backward(); assert depth > 0 and semantic > 0
    gradient = logits.grad.reshape(18, *shape)
    assert gradient[11, 2:10, 2:10, 0].sum() < 0
    # Empty GT still penalizes phantom occupied projections via semantic BG.
    empty = torch.full(shape, 17); phantom = torch.zeros_like(logits).requires_grad_()
    total, semantic, depth = renderer.loss(phantom, empty)
    total.backward(); assert semantic > 0 and depth == 0 and phantom.grad[17].sum() < 0
    # Class0 remains occupied for supervision, even though COME hides it.
    zero = torch.zeros_like(target); zero[0] = 1
    assert bool((renderer.render(zero, 'top')[2] > .99).all())
    with pytest.raises(ValueError): renderer.loss(logits, torch.full(shape, 255))


def test_resume_guards_and_bounded_prefetch():
    c = dict(any='fixed'); saved = dict(protocol=PROTOCOL, contract=c, updates=64, rank_rng=[{}])
    assert validate_cursor(saved, c) == 64
    for change in (dict(updates=1032), dict(protocol='old'), dict(contract={}), dict(rank_rng=[])):
        with pytest.raises(ValueError): validate_cursor(dict(saved, **change), c)
    import tools.coverage_projection.run as run
    from unittest.mock import patch
    with patch.object(run, 'prepare_row', side_effect=lambda i, p, r, **kw: r):
        with prefetch([[1, 2], [3]], None, None) as it:
            assert list(it) == [[1, 2], [3]]


def test_actual_training_step_optimizer_and_rng_resume(tmp_path):
    import copy
    from tools.joint_occ_experiments.soft_geometry import grid_arrays
    from tools.real_motion.joint_training_distributed import DistributedTraining
    body, runner, row, poses, labels = tiny_scene()
    try:
        projector = VolumeProjection(*grid_arrays(runner.provider.pcfg.grid), side=8)
        def adam(net):
            return torch.optim.AdamW([
                dict(params=net.transport.parameters(), lr=5e-4, initial_lr=5e-4, weight_decay=1e-4),
                dict(params=net.columns.parameters(), lr=3e-4, initial_lr=3e-4, weight_decay=.01)])
        optimizer = adam(body); rows = [(row, poses, labels)]
        first = train_step(body, optimizer, runner, rows, projector, update=1, total=1031)
        assert first['projection_semantic'] > 0 and first['grad_norm'] > 0
        ctx = DistributedTraining(False); rng = np.random.default_rng(20261011)
        payload = copy.deepcopy(dict(models={'gt': body.state_dict()}, optimizer=optimizer.state_dict(),
            rank_rng=[ctx.rng_state(rng)]))
        second = train_step(body, optimizer, runner, rows, projector, update=2, total=1031)
        expected = copy.deepcopy(body.state_dict())
        body.load_state_dict(payload['models']['gt']); optimizer.load_state_dict(payload['optimizer'])
        ctx.restore_rng(payload['rank_rng'], rng)
        resumed = train_step(body, optimizer, runner, rows, projector, update=2, total=1031)
        assert resumed == second
        for key, value in body.state_dict().items(): torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    finally: runner.close()


def test_visual_export_palette_arrays_and_size(tmp_path):
    from tools.coverage_projection.visuals import render, bev
    shape = (8, 8, 4); vol = np.full(shape, 17, dtype=np.uint8); vol[2:6, 2:6, 0] = 11
    assert bev(vol).size == (320, 320)
    root = tmp_path/'visuals'; root.mkdir()
    (root/'manifest.json').write_text(json.dumps(dict(cases=[dict(index=1, scene='test', t0='test')])))
    for kind in ('occ', 'stc'):
        for label in ('control', 'candidate'):
            np.savez_compressed(root/f'{kind}_01_{label}.npz', prediction=np.stack([vol]*3),
                transport=np.stack([vol]*3), gt=np.stack([vol]*3), t0=vol)
    render(root)
    assert (root/'occ_01_bev.png').is_file() and (root/'index.html').is_file()
    assert (tmp_path/'visuals_and_metrics.zip').stat().st_size < 2**20


def test_workflow_shell_continuation_and_no_geometry_cache():
    text = (Path(__file__).resolve().parents[1]/'tools/coverage_projection/workflow.sh').read_text()
    assert '--dataroot "$DATA" \\\n' in text
    assert '--ccr-history-cache' not in text and 'tools.coverage_projection.visuals' in text


def test_same64_evaluation_label_order_integer_counts_and_resume(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import threading
    from tools.coverage_projection import evaluate as ev
    from tools.coverage_projection.run import sha
    from tools.joint_occ_experiments import run as base
    shape = (4, 4, 2); volume = int(np.prod(shape)); stop = threading.Event()
    out = tmp_path/'run'; out.mkdir(); control_dir = tmp_path/'control'; control_dir.mkdir()
    config = tmp_path/'config'; config.write_text('fixed config')
    dev = tmp_path/'dev'; dev.write_text('fixed dev'); info = tmp_path/'info'; info.write_text('fixed info')
    report_path = control_dir/'results.json'; report_path.write_text('{}')
    dummy = torch.nn.Linear(1, 1)
    torch.save(dict(contract=dict(model={}), models={'gt': dummy.state_dict()}), control_dir/'epoch_0001.pt')
    records = [dict(scene_name=f's{i}', t0_token=f't{i}', history_tokens=[f'h{i}_{j}' for j in range(4)],
        future_tokens=[f'f{i}_{j}' for j in range(6)]) for i in range(4369)]
    keys = [[r['scene_name'], r['t0_token']] for r in records[:64]]
    cc = dict(eval_keys=keys, overrides=[], data=dict(dev_info=sha(info), dev_cache=sha(dev)))
    c = dict(implementations={}, control_dir=str(control_dir), control_sha256=sha(control_dir/'epoch_0001.pt'),
        control_results_sha256=sha(report_path), eval_keys=keys, train_data=dict(config=sha(config)))
    torch.save(dict(protocol=PROTOCOL, updates=1031, contract=dict(c, model={}),
        models={'gt': dummy.state_dict()}), out/'epoch_0001.pt')
    monkeypatch.setattr(ev, 'reference', lambda _: dict(contract=cc))
    monkeypatch.setattr(base, 'load_cache', lambda _: ({}, records))
    monkeypatch.setattr(ev, 'load_runtime_config', lambda *a: {})
    grid = SimpleNamespace(x_min=0, y_min=0, z_min=0, voxel_size=[1, 1, 1], shape_hwd=shape)
    monkeypatch.setattr(ev, 'make_prepare_config', lambda _: SimpleNamespace(grid=grid))
    monkeypatch.setattr(ev, 'from_spec', lambda *a, **k: torch.nn.Linear(1, 1))
    counters = dict(forward=0, labels=0)
    gt = np.full(shape, 17, dtype=np.uint8); gt[0, 0, 0] = 11
    class Inputs:
        def __init__(self, *a, **k): pass
        def preflight(self, *a): return []
        def targets(self, w, hs):
            assert counters['forward'] > counters['labels']  # six readouts already exist
            counters['labels'] += 1
            if counters['labels'] == 3: stop.set()
            return [gt for _ in hs]
    class Predictor:
        def __init__(self, *a, **k): self.provider = None
        def close(self): pass
    monkeypatch.setattr(ev, 'DenseInputs', Inputs); monkeypatch.setattr(ev, 'RecoveryPredictor', Predictor)
    row = ({}, dict(history_occ=np.full((4, *shape), 17, np.uint8)), None, None)
    monkeypatch.setattr(ev, 'prepare_row', lambda *a, **k: (row, np.stack([np.eye(4)]*6), None))
    def forward(*a, candidate=False, **k):
        counters['forward'] += 1
        logits = torch.zeros(6, 18, volume); logits[:, 17] = 2
        prior = logits.clone()
        if candidate: logits[:, 11, 0] = 3
        return dict(shape=shape, prior=prior, logits=logits, prior_coverage=torch.ones(6, 1, volume))
    monkeypatch.setattr(ev, 'original_forward', forward)
    monkeypatch.setattr(ev, 'forward_window', lambda *a, **k: forward(*a, candidate=True, **k))
    args = SimpleNamespace(device='cpu', out_dir=out, dev_cache=dev, dev_info=info, config=config,
        dataroot=tmp_path, stc_root=tmp_path, cpu_workers=1)
    with pytest.raises(InterruptedError): ev.run(args, stop)
    state = json.loads((out/'evaluation_state.json').read_text())
    assert state['progress'] == {'control/occ': 3}
    stop.clear(); ev.run(args, stop)
    state = json.loads((out/'evaluation_state.json').read_text())
    assert set(state['progress'].values()) == {64} and counters == dict(forward=256, labels=256)
    for kind in state['counts'].values():
        for route in kind.values(): assert np.asarray(route['confusion']).sum() == 3*64*volume
    result = json.loads((out/'evaluation.json').read_text())
    assert result['delta']['occ']['semantic_refine']['mIoU'] == 100
    assert result['reports']['candidate/stc']['semantic_refine']['average']['IoU'] == 100
    assert len(list((out/'visuals').glob('*.npz'))) == 32


def _dual_rank(rank, directory):
    """Actual Gloo reduction + projection gradients + ragged ranks + Adam/RNG."""
    import copy
    import os
    from types import SimpleNamespace
    from unittest.mock import patch
    from tools.coverage_projection import run
    from tools.real_motion.joint_training_distributed import DistributedTraining
    from tools.real_motion.joint_training_recovery import save_resume_checkpoint
    torch.set_num_threads(1)
    os.environ.update(WORLD_SIZE='2', RANK=str(rank), LOCAL_RANK=str(rank))
    ctx = DistributedTraining(True, backend='gloo', init_method=Path(directory, 'rendezvous').resolve().as_uri())
    try:
        with patch.dict(os.environ, WORLD_SIZE='1'): single = DistributedTraining(False)
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.common = torch.nn.Linear(2, 18)
                self.rare = torch.nn.Linear(2, 18); self.unused = torch.nn.Linear(2, 1)
        def forward(net, provider, row, **kwargs):
            value = net.common(row['x'])
            if row['id'] % 2: value = value+net.rare(row['x'])
            bias = value.new_zeros(18, 8); bias[11] = .2
            return dict(logits=torch.stack([value.T+h*bias for h in range(6)]))
        torch.manual_seed(21); net = Tiny(); reference = copy.deepcopy(net)
        untouched = copy.deepcopy(net.unused.state_dict())
        def adam(net): return torch.optim.AdamW([dict(params=net.parameters(), lr=.001, initial_lr=.001)])
        opt, ref_opt = adam(net), adam(reference)
        records = [dict(scene_name='test', t0_token=str(i), id=i, features=[None]*n)
            for i, n in enumerate((50, 5, 20, 1))]
        all_groups = [records, records[:3], records[:1], records]
        local, slots = run.local_plan(all_groups, 2, rank)
        provider = SimpleNamespace(provider=SimpleNamespace(device=torch.device('cpu')))
        renderer = VolumeProjection((0, 0, 0), (.4, .4, .4), (2, 2, 2), side=4)
        def rows(group):
            result = []
            for r in group:
                x = torch.tensor([[r['id']*.1+.1, j*.03] for j in range(8)])
                gt = np.array([[0, 4, 17, 11, 17, 17, 17, 17]]*6).reshape(6, 2, 2, 2)
                result.append((dict(r, x=x), None, gt))
            return result
        calls = []
        class Spy:
            def loss(self, logits, labels):
                calls.append(logits.storage_offset()//(18*8))
                return renderer.loss(logits, labels)
        with patch.object(run, 'forward_window', forward):
            for update in range(1, 4):
                i = update-1; expected = run.train_step(reference, ref_opt, provider, rows(all_groups[i]),
                    renderer, update=update, total=4, context=single)
                calls.clear()
                actual = run.train_step(net, opt, provider, rows(local[i]), Spy(), update=update, total=4,
                    context=ctx, global_windows=len(all_groups[i]), window_slots=slots[i])
                assert calls == [(4*i+s)%6 for s in slots[i]]
                all_slots = ctx.gather(slots[i]); assert sorted(s for shard in all_slots for s in shard) == list(range(len(all_groups[i])))
                for name in ('CE', 'Lovasz', 'projection_semantic', 'projection_depth', 'loss'):
                    assert actual[name] == pytest.approx(expected[name], abs=2e-6)
                for name, value in net.state_dict().items():
                    torch.testing.assert_close(value, reference.state_dict()[name], rtol=1e-5, atol=2e-7)
                for key, state in ref_opt.state_dict()['state'].items():
                    for name, value in state.items():
                        torch.testing.assert_close(opt.state_dict()['state'][key][name], value, rtol=1e-5, atol=2e-7)
            for name, value in net.unused.state_dict().items():
                torch.testing.assert_close(value, untouched[name], rtol=0, atol=0)
            rng = np.random.default_rng(21+rank); torch.manual_seed(21+rank)
            states = ctx.gather(ctx.rng_state(rng)); expect_np, expect_torch = rng.random(3), torch.rand(3)
            ctx.restore_rng(states, rng)
            np.testing.assert_array_equal(rng.random(3), expect_np)
            torch.testing.assert_close(torch.rand(3), expect_torch, rtol=0, atol=0)
            if ctx.primary:
                save_resume_checkpoint(Path(directory)/'last.pt', dict(protocol=PROTOCOL, contract=dict(world_size=2),
                    updates=3, models={'gt': net.state_dict()}, optimizer=opt.state_dict(), rank_rng=states))
            ctx.barrier()
            saved = torch.load(Path(directory)/'last.pt', map_location='cpu', weights_only=False)
            assert run.validate_cursor(saved, dict(world_size=2)) == 3
            restored = Tiny(); restored.load_state_dict(saved['models']['gt']); restored_opt = adam(restored)
            restored_opt.load_state_dict(saved['optimizer']); ctx.restore_rng(saved['rank_rng'], rng)
            for model_, optimizer_ in ((net, opt), (restored, restored_opt)):
                run.train_step(model_, optimizer_, provider, rows(local[3]), renderer, update=4, total=4,
                    context=ctx, global_windows=4, window_slots=slots[3])
            for name, value in net.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
        assert ctx.stop(rank == 1)
        Path(directory, f'rank{rank}.json').write_text(json.dumps(dict(ok=True, windows=list(map(len, local)))))
    finally: ctx.close()


def test_actual_dual_rank_projection_mean_ragged_and_resume(tmp_path):
    import time
    import torch.multiprocessing as mp
    workers = mp.spawn(_dual_rank, args=(str(tmp_path),), nprocs=2, join=False)
    deadline = time.monotonic()+90
    try:
        while not workers.join(timeout=1):
            if time.monotonic() > deadline: raise AssertionError('dual projection collective timeout')
    finally:
        for child in workers.processes:
            if child.is_alive(): child.terminate(); child.join(timeout=5)
    values = [json.loads((tmp_path/f'rank{rank}.json').read_text()) for rank in (0, 1)]
    assert all(v['ok'] for v in values)
    assert sorted(v['windows'][2] for v in values) == [0, 1]


def test_dual_workflow_and_world_size_resume_guard():
    root = Path(__file__).resolve().parents[1]
    workflow = (root/'tools/coverage_projection/workflow.sh').read_text()
    assert '--nproc_per_node=2' in workflow and 'PILOT_GPUS:-0,1' in workflow
    assert workflow.index('export CUDA_VISIBLE_DEVICES="$EVAL_GPU"') < workflow.index('tools.coverage_projection.evaluate')
    c = dict(world_size=2)
    saved = dict(protocol=PROTOCOL, contract=c, updates=64, rank_rng=[{}, {}])
    assert validate_cursor(saved, c) == 64
    with pytest.raises(ValueError): validate_cursor(dict(saved, rank_rng=[{}]), c)


def completed_control_fixture():
    from tools.coverage_projection.control import CONTROL_PROTOCOL
    c = dict(protocol=CONTROL_PROTOCOL, epochs=1, total_updates=1031,
        source_policy='causal_forward_CV_history', registration='none',
        known='TRAIN_OCC_STC_dense_valid', parameters=2816122, world_size=1,
        seed=20261011, learning_rates=[5e-4, 3e-4], schedule='whole_1_epoch_cosine_floor0.01',
        objective=dict(CE=1., Lovasz=1., ego=0., motion=0., existence=0.),
        train_keys=[['train_scene', str(i)] for i in range(4086)],
        eval_keys=[['val_scene', str(i)] for i in range(64)],
        model=dict(source_policy='causal_forward_CV_history'))
    saved = dict(protocol=CONTROL_PROTOCOL, contract=c, epoch=1, batch=0, updates=1031,
        models={'gt': {'weight': torch.ones(2)}})
    # This is the historical failure mode: train finished, eval crashed, a
    # separate recovery report completed later; original report is read only.
    result = dict(status='running', epoch=1, updates=1031, contract=c, evaluations={})
    return saved, result


def test_complete_control_checkpoint_accepts_stale_eval_status_read_only(tmp_path):
    from tools.coverage_projection.control import reference, sha
    saved, result = completed_control_fixture()
    path = tmp_path/'epoch_0001.pt'; torch.save(saved, path)
    report = tmp_path/'results.json'; report.write_text(json.dumps(result))
    before = (sha(path), sha(report))
    value = reference(tmp_path)
    assert value['status'] == 'running' and value['evaluations'] == {}
    proof = value['source_checkpoint_completion']
    assert proof['training_complete'] and proof['updates'] == 1031
    assert proof['original_report_status'] == 'running' and not proof['old_evaluation_reused']
    assert proof['control_will_be_evaluated_fresh']
    assert (sha(path), sha(report)) == before


@pytest.mark.parametrize('name,value', [('updates', 1024), ('epoch', 0), ('batch', 4), ('protocol', 'old')])
def test_partial_control_checkpoint_never_accepted_despite_complete_report(name, value):
    from tools.coverage_projection.control import validate
    saved, report = completed_control_fixture(); report = dict(report, status='complete')
    with pytest.raises(ValueError, match='checkpoint.'+name): validate(dict(saved, **{name:value}), report)


@pytest.mark.parametrize('name,value', [('source_policy','shared_pinned'), ('total_updates',10292),
    ('world_size',2), ('parameters',1194146), ('learning_rates',[1e-5,1e-4]), ('epochs',2)])
def test_control_wrong_architecture_budget_or_settings_are_still_rejected(name, value):
    import copy
    from tools.coverage_projection.control import validate
    saved, report = copy.deepcopy(completed_control_fixture())
    saved['contract'][name] = value  # saved/report share this same contract
    with pytest.raises(ValueError, match='contract.'+name): validate(saved, report)


def test_control_report_mismatch_and_missing_source_have_actionable_errors(tmp_path):
    import copy
    from tools.coverage_projection.control import validate, reference
    saved, report = completed_control_fixture(); report = copy.deepcopy(report)
    report['contract']['seed'] = 123
    with pytest.raises(ValueError, match='results.json contract differs'): validate(saved, report)
    with pytest.raises(ValueError, match='epoch_0001.pt missing'): reference(tmp_path)
    workflow = (Path(__file__).resolve().parents[1]/'tools/coverage_projection/workflow.sh').read_text()
    assert workflow.index('tools.coverage_projection.control') < workflow.index('torch.distributed.run')
