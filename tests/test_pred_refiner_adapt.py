import copy
import json
import threading
from types import SimpleNamespace
import pytest
import torch
from test_full_gt_training import fixture
from test_scene_ego_head import causal_background
from tools.scene_ego_head import integration
from tools.scene_ego_head.pred_adapt.run import AdaptModel
from tools.scene_ego_head.pred_adapt import engine
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_GT_body_random_head_only_refiner_ego_update(monkeypatch, device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.set_num_threads(1); body, _, p, row = fixture()
    source = dict(contract=dict(model=body.configs()), models={'gt': copy.deepcopy(body.state_dict())})
    model = AdaptModel(body.transport.config, source, 'sourceSHA', pool_side=2).to(device)
    p.device = torch.device(device)
    for k, v in body.state_dict().items(): assert torch.equal(v, model.state_dict()[k].cpu())
    assert not model.ego.readout[-1].weight.count_nonzero()  # initial output is kinematic prior
    before = tensor_state_fingerprint(model.transport.state_dict())
    ref_before = copy.deepcopy(model.columns.state_dict()); head_before = copy.deepcopy(model.ego.state_dict())
    model.requires_grad_(False).requires_grad_(True).train()
    assert all(not p.requires_grad for p in model.transport.parameters()) and not model.transport.training
    assert all(p.requires_grad for p in model.columns.parameters()) and all(p.requires_grad for p in model.ego.parameters())
    monkeypatch.setattr(integration, 'rebuild_pred_backgrounds', causal_background)
    optimizer = torch.optim.AdamW([
        dict(params=model.transport.parameters(), lr=1e-5, initial_lr=1e-5),
        dict(params=model.columns.parameters(), lr=1e-5, initial_lr=1e-5),
        dict(params=model.ego.parameters(), lr=1e-4, initial_lr=1e-4)])
    a = engine.train_arm(model, optimizer, p, [row], 'pred', update=1, total_updates=500, probe=True)
    b = engine.train_arm(model, optimizer, p, [row], 'pred', update=2, total_updates=500)
    assert a['gradient_links']['occupancy_to_scene_ego'] > 0 and b['gradient_links']['occupancy_to_scene_ego'] > 0
    assert not a['source_auxiliary_computed']
    assert before == tensor_state_fingerprint(model.transport.state_dict())
    assert any(not torch.equal(v, model.columns.state_dict()[k]) for k, v in ref_before.items())
    assert any(not torch.equal(v, model.ego.state_dict()[k]) for k, v in head_before.items())
    assert all(p not in optimizer.state for p in model.transport.parameters())


def test_actual_partial_cli_source_readonly_and_resume(tmp_path, monkeypatch):
    from test_full_gt_training import test_real_cli_full_population_epoch_snapshots_resume_and_eval_readonly
    from tools.scene_ego_head.pred_adapt import run
    from tools.joint_occ_experiments import run as legacy
    test_real_cli_full_population_epoch_snapshots_resume_and_eval_readonly(tmp_path, monkeypatch)
    source = tmp_path/'pilot/epoch_0002.pt'; source_bytes = source.read_bytes()
    original = run.AdaptModel
    monkeypatch.setattr(run, 'AdaptModel', lambda cfg, saved, digest: original(cfg, saved, digest, pool_side=2))
    monkeypatch.setattr(integration, 'rebuild_pred_backgrounds', causal_background)
    def groups(records, *args):
        assert len(records) == 4086
        return [[records[i]] for i in range(600)]
    monkeypatch.setattr(legacy, 'epoch_groups', groups)
    out = tmp_path/'adapt'
    args = [v for n, p in (('config', tmp_path/'config.yaml'), ('base-checkpoint', tmp_path/'base.pt'),
        ('train-cache', tmp_path/'train-cache'), ('dev-cache', tmp_path/'dev-cache'),
        ('population-manifest', tmp_path/'population-manifest'), ('dataroot', tmp_path),
        ('train-info', tmp_path/'train-info'), ('dev-info', tmp_path/'dev-info'),
        ('ccr-history-cache', tmp_path), ('ccr-val-history-cache', tmp_path),
        ('out-dir', out), ('gt-checkpoint', source)) for v in ('--'+n, str(p))]
    before = legacy.train_arm, legacy.write_json, legacy.restore_checkpoint
    assert run.main(threading.Event(), args+['--max-updates', '2']) == 0
    saved = torch.load(out/'train/last.pt', weights_only=False)
    assert saved['updates'] == 2 and saved['contract']['total_updates'] == 500
    assert saved['contract']['initialization'] == 'GT_epoch2_body_RANDOM_ego_seed21_fresh_Adam'
    assert saved['contract']['lr'][1:] == [1e-5, 1e-4]
    assert saved['protocol'] == run.PROTOCOL
    assert run.main(threading.Event(), args+['--resume', str(out/'train/last.pt'), '--max-updates', '3']) == 0
    assert torch.load(out/'train/last.pt', weights_only=False)['updates'] == 3
    assert json.loads((out/'freeze_audit.json').read_text())['Encoder_STWM_unchanged']
    assert source.read_bytes() == source_bytes and before == (legacy.train_arm, legacy.write_json, legacy.restore_checkpoint)
