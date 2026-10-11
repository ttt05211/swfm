import copy
from pathlib import Path
import numpy as np
import pytest
import torch

from tools.joint_occ_experiments.soft_geometry import occupancy_loss
from tools.joint_occ_experiments.performance.loss import (
    prepare_target, occupancy_loss_prepared, prepared_loss_execution,
)
from tools.joint_occ_experiments import engine
from test_joint_occ_ego import fixture


@pytest.mark.parametrize('case', ['random', 'tied', 'onehot', 'free', 'unknown', 'mixed_unknown'])
@pytest.mark.parametrize('weight', [0., .3, 1.])
@pytest.mark.parametrize('sparse', [False, True])
def test_combined_loss_preserves_every_voxel_value_and_gradient(case, weight, sparse):
    torch.set_num_threads(1)
    rng = torch.Generator().manual_seed(857)
    labels = torch.arange(198) % 18
    probability = torch.rand(18, len(labels), generator=rng)
    probability /= probability.sum(0)
    if case == 'tied': probability[:] = 1/18
    if case == 'onehot': probability = torch.eye(18)[:, labels]
    if case == 'free': labels[:] = 17
    if case == 'unknown': labels[:] = 255
    if case == 'mixed_unknown': labels[::3] = 255; labels[1::9] = -2
    probability.requires_grad_()
    prepared = prepare_target(labels, probability.device)
    old = occupancy_loss(probability, labels, lovasz_weight=weight)
    new = occupancy_loss_prepared(probability, prepared, lovasz_weight=weight, sparse=sparse)
    for a, b in zip(old, new): torch.testing.assert_close(a, b, rtol=2e-6 if sparse else 0, atol=2e-7 if sparse else 0)
    for i in range(3):
        a = torch.autograd.grad(old[i], probability, retain_graph=True)[0]
        b = torch.autograd.grad(new[i], probability, retain_graph=True)[0]
        torch.testing.assert_close(a, b, rtol=3e-6, atol=2e-8)


def test_prepared_supervision_and_loss_only_patch_restored_on_exception():
    _, _, provider, row = fixture()
    previous = engine.occupancy_loss
    target = row[1]['future_gt_occ'][0]
    prepared = prepare_target(target, provider.device)
    assert prepared.positions is None and prepared.volume == target.size
    with pytest.raises(RuntimeError, match='interrupted'):
        with prepared_loss_execution([row], provider.device):
            assert engine.occupancy_loss is not previous
            raise RuntimeError('interrupted')
    assert engine.occupancy_loss is previous
    with pytest.raises(RuntimeError, match='not all six'):
        with prepared_loss_execution([row], provider.device): pass
    assert engine.occupancy_loss is previous


def test_probability_gradients_follow_reference_normalization_order_exactly():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    labels = torch.arange(4007) % 18
    probability = torch.rand(18, len(labels))
    probability = (probability/probability.sum(0)).requires_grad_()
    prepared = prepare_target(labels, 'cpu')
    # Nonunit upstream catches reassociation of scale, CE and Lovasz division.
    for weight in (.3, 1.):
        old = occupancy_loss(probability, labels, lovasz_weight=weight)[0]*.0137
        new = occupancy_loss_prepared(probability, prepared, lovasz_weight=weight)[0]*.0137
        a = torch.autograd.grad(old, probability)[0]
        b = torch.autograd.grad(new, probability)[0]
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize('mode', ['gt', 'pred'])
@pytest.mark.parametrize('sparse', [False, True])
def test_actual_six_frame_adam_step_and_all_gradient_links_match_current_fused(mode, sparse):
    from real_motion.strong_w2det import StrongW2DetConfig
    model, optimizer, provider, row = fixture()
    provider.strong = StrongW2DetConfig(free_label=17)
    weights, moments, rng = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()), torch.get_rng_state()
    outcomes, states, reports = [], [], []
    for fast in (False, True):
        model.load_state_dict(weights); optimizer.load_state_dict(copy.deepcopy(moments)); torch.set_rng_state(rng)
        if fast:
            with prepared_loss_execution([row, row], provider.device, sparse=sparse):
                report = engine.train_arm(model, optimizer, provider, [row, row], mode,
                                          update=1, total_updates=3, probe=True)
        else:
            report = engine.train_arm(model, optimizer, provider, [row, row], mode,
                                      update=1, total_updates=3, probe=True)
        outcomes.append(copy.deepcopy(model.state_dict())); states.append(copy.deepcopy(optimizer.state_dict()))
        reports.append(report)
    for key in ('CE', 'Lovasz', 'source_geometry', 'existence', 'ego_geometry', 'grad_norm'):
        assert reports[0][key] == pytest.approx(reports[1][key], rel=3e-5, abs=1e-6)
    for key in reports[0]['gradient_links']:
        assert reports[0]['gradient_links'][key] == pytest.approx(reports[1]['gradient_links'][key], rel=3e-5, abs=1e-6)
    for key in outcomes[0]:
        torch.testing.assert_close(outcomes[0][key], outcomes[1][key], rtol=3e-5, atol=2e-6, msg=key)
    for key, state in states[0]['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(value, states[1]['state'][key][name], rtol=3e-5, atol=2e-6)


@pytest.mark.parametrize('seed', range(6))
def test_sparse_zero_tail_keeps_missed_foreground_gradients_ties_and_ignored_voxels(seed):
    rng = torch.Generator().manual_seed(seed)
    labels = torch.randint(18, (971,), generator=rng)
    labels[::11] = 255
    p = torch.zeros(18, len(labels))
    p.scatter_(0, torch.randint(18, (1, len(labels)), generator=rng), 1.)
    # Mix exact zeros, soft probabilities and tied variable errors.
    p[:, ::5] = .25*p[:, ::5]+.75*p[:, ::5].roll(1, 0)
    p[:, ::17] = p[:, ::17]*1e-8
    p.requires_grad_()
    old = occupancy_loss(p, labels)
    new = occupancy_loss_prepared(p, prepare_target(labels, 'cpu'), sparse=True)
    for a, b in zip(old, new): torch.testing.assert_close(a, b, rtol=2e-6, atol=3e-7)
    a = torch.autograd.grad(old[0], p)[0]
    b = torch.autograd.grad(new[0], p, retain_graph=True)[0]
    torch.testing.assert_close(a, b, rtol=3e-5, atol=2e-7)
    lovasz_gradient = torch.autograd.grad(new[2], p)[0]
    at = torch.arange(len(labels))
    safe_labels = labels.clamp_max(17)
    missing = (labels < 17) & (p.detach()[safe_labels, at] == 0)
    assert missing.any() and (lovasz_gradient[safe_labels[missing], at[missing]] < 0).all()


def test_legacy_training_fingerprint_and_source_files_untouched():
    root = Path(__file__).resolve().parents[1]
    import subprocess
    paths = list((root/'tools/joint_occ_experiments').glob('*.py'))
    for path in paths:
        relative = path.relative_to(root).as_posix()
        # Compare normalized working-tree bytes with pinned c57 source; Git
        # autocrlf must not make this a false claim about the resume hash.
        old = subprocess.check_output(['git', 'show', 'c57bd48:'+relative], cwd=root)
        assert path.read_bytes().replace(b'\r\n', b'\n') == old, relative
