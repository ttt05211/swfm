import copy
import json
import pytest
import torch
import numpy as np
from test_full_gt_training import fixture
from tools.scene_ego_head import integration
from tools.scene_ego_head.frozen_gt_diagnostics import audit, run
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_real_one_window_controls_no_labels_weights_or_hooks_changed(monkeypatch, device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.set_num_threads(1); body, _, p, row = fixture()
    body.to(device); p.device = torch.device(device); body.eval().requires_grad_(False)
    with torch.no_grad(): body.columns.readout[-1].weight.normal_(std=.01)
    cached = row[1]['_column_causal_preparation']['prepared_state']['column_backgrounds']
    monkeypatch.setattr(integration, 'rebuild_pred_backgrounds', lambda *a: [np.array(v, copy=True) for v in cached])
    raw = {**row[1], 'future_gt_occ': object(), 'future_masks': object()}
    before = tensor_state_fingerprint(body.state_dict()); autocast = torch.autocast
    precision = torch.get_float32_matmul_precision()
    result = audit.audit(body, p, (row[0], raw, row[2], row[3]))
    assert result['windows'] == 1 and result['no_training'] and result['strict_FAIL_retained']
    assert result['background_changed_voxels_by_horizon'] == [0]*6
    assert result['cross_entrance']['fields']['field']['max_abs'] == 0
    assert result['identical_input_BF16_repeat']['logits']['pass_gate']
    assert result['FP32_cross_input_reference']['logits']['pass_gate']
    assert not result['automatic_precision_switch']
    trace = result['cross_input_BF16_refiner']['trace']['fields']
    assert set(trace) == {'input', 'time', 'blocks.0', 'blocks.1', 'blocks.2', 'readout.0', 'readout.2', 'readout.4'}
    assert trace['input']['dtype_a'] == ('torch.bfloat16' if device == 'cuda' else 'torch.float32')
    assert torch.autocast is autocast and tensor_state_fingerprint(body.state_dict()) == before
    assert torch.get_float32_matmul_precision() == precision
    assert all(not m._forward_hooks for m in body.modules())
    assert all(p.grad is None for p in body.parameters())


def test_fp32_scope_and_hook_cleanup_on_exception(monkeypatch):
    body, _, p, row = fixture(); autocast = torch.autocast
    fw = audit.gt.forward_window(body, p, row)
    def fail(*a): raise RuntimeError('diagnostic failure')
    monkeypatch.setattr(body.columns.readout, 'forward', fail)
    with pytest.raises(RuntimeError, match='diagnostic failure'):
        audit.readout(body.columns, fw['memory'], fw['prior'], fw['shape'], fp32=True, trace=True)
    assert torch.autocast is autocast and all(not m._forward_hooks for m in body.modules())
    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision('medium')
        with pytest.raises(RuntimeError):
            with audit.fp32_reference(): raise RuntimeError('precision failure')
        assert torch.get_float32_matmul_precision() == 'medium'
    finally: torch.set_float32_matmul_precision(precision)


def test_original_repeat_failure_is_reported_not_waived(monkeypatch):
    body, _, p, row = fixture()
    cached = row[1]['_column_causal_preparation']['prepared_state']['column_backgrounds']
    monkeypatch.setattr(integration, 'rebuild_pred_backgrounds', lambda *a: cached)
    original = body.columns.forward; calls = []
    def unstable(*args):
        v = original(*args).clone(); calls.append(1)
        if len(calls) >= 3: v[0, 0, 0] += .02
        return v
    monkeypatch.setattr(body.columns, 'forward', unstable)
    result = audit.audit(body, p, row)
    assert not result['GT_original_repeat']['fields']['logits']['pass_gate']
    assert result['strict_FAIL_retained'] and not result['automatic_precision_switch']


def test_cli_only_one_window_source_and_gate_files_unchanged(tmp_path, monkeypatch):
    from test_full_gt_training import test_real_cli_full_population_epoch_snapshots_resume_and_eval_readonly
    test_real_cli_full_population_epoch_snapshots_resume_and_eval_readonly(tmp_path, monkeypatch)
    _, _, p, row = fixture(); source = tmp_path/'pilot/epoch_0002.pt'
    receipts = {p: p.read_bytes() for p in (source, tmp_path/'pilot/last.pt', tmp_path/'pilot/summary.txt')}
    failed_gate = tmp_path/'failed_ego_gate.json'; failed_gate.write_text('{"pass_gate":false}')
    cached = row[1]['_column_causal_preparation']['prepared_state']['column_backgrounds']
    monkeypatch.setattr(integration, 'rebuild_pred_backgrounds', lambda *a: cached)
    out = tmp_path/'audit'
    args = [v for name, path in (
        ('config', tmp_path/'config.yaml'), ('base-checkpoint', tmp_path/'base.pt'), ('train-cache', tmp_path/'train-cache'),
        ('dev-cache', tmp_path/'dev-cache'), ('population-manifest', tmp_path/'population-manifest'), ('dataroot', tmp_path),
        ('train-info', tmp_path/'train-info'), ('dev-info', tmp_path/'dev-info'), ('ccr-history-cache', tmp_path),
        ('ccr-val-history-cache', tmp_path), ('out-dir', out), ('resume', source)) for v in ('--'+name, str(path))]
    assert run.main(argv=args) == 0
    result = json.loads((out/'audit.json').read_text())
    assert result['windows'] == 1 and result['body_unchanged']
    assert not (out/'head_last.pt').exists() and not (out/'last.pt').exists()
    for p, contents in receipts.items(): assert p.read_bytes() == contents
    assert failed_gate.read_text() == '{"pass_gate":false}'
