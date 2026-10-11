"""Explicit drift acceptance is NOT relabeling a failed parity report."""
import copy
import json
from pathlib import Path
import pytest
import torch
from test_joint_occ_speed_probe import saved_fixture
from tools.joint_occ_experiments.horizons.continuation import run as approved


def receipt(source='source', fingerprint='kernel'):
    return dict(status='complete', source_snapshot_sha256=source,
        candidates=dict(sequential_h2=dict(execution_fingerprint=fingerprint, pass_gate=False,
            numerical_parity=False, baseline_repeat_parity=False, faster_each_pass=True,
            memory_headroom_ge_10percent=True, speedup=1.4692371)),
        warmup={name: dict(numerical_parity=True, loss_gradient_audit=dict(pass_gate=True))
                for name in ('current_fused', 'sequential_h2')},
        final_state_parity=dict(current_fused=dict(numerical_parity=False, max_state_abs_difference=.000402516),
            sequential_h2=dict(numerical_parity=False, max_state_abs_difference=.000817965)))


def test_explicit_acceptance_keeps_original_fail_report_unchanged():
    report = receipt(); old = copy.deepcopy(report)
    record = approved.validate_accepted_receipt(report, 'sequential_h2', 'source', 'kernel')
    assert report == old
    assert record['original_pass_gate'] is False
    assert record['original_numerical_parity'] is False
    assert record['original_baseline_repeat_parity'] is False
    assert record['lossless_claim'] is False and record['scientific_quality_verified'] is False


@pytest.mark.parametrize('error', ['single_step', 'gradient', 'memory', 'speed', 'large_drift', 'nan', 'source', 'code'])
def test_nonapproved_failures_remain_blocked(error):
    report = receipt()
    if error == 'single_step': report['warmup']['sequential_h2']['numerical_parity'] = False
    elif error == 'gradient': report['warmup']['sequential_h2']['loss_gradient_audit']['pass_gate'] = False
    elif error == 'memory': report['candidates']['sequential_h2']['memory_headroom_ge_10percent'] = False
    elif error == 'speed': report['candidates']['sequential_h2']['speedup'] = .95
    elif error == 'large_drift': report['final_state_parity']['sequential_h2']['max_state_abs_difference'] = .02
    elif error == 'nan': report['final_state_parity']['current_fused']['max_state_abs_difference'] = float('nan')
    elif error == 'source': report['source_snapshot_sha256'] = 'different'
    else: report['candidates']['sequential_h2']['execution_fingerprint'] = 'different'
    with pytest.raises(RuntimeError): approved.validate_accepted_receipt(report, 'sequential_h2', 'source', 'kernel')


def test_original_149_import_and_resume_use_same_kernel_adam_rng_source(monkeypatch, tmp_path):
    strict, legacy = approved.strict, approved.strict.legacy
    model, optimizer, provider, row, saved = saved_fixture()
    root = Path(strict.__file__).resolve().parents[3]
    fingerprint = strict.execution_fingerprint(root)
    saved.update(updates=149, batch=149)
    saved['contract'].update(implementation=strict.legacy_fingerprint(root), window_batch=4, source_budget=128)
    original = tmp_path/'original'; original.mkdir()
    source = original/'last.pt'; torch.save(saved, source)
    before = source.read_bytes()
    proof = tmp_path/'speed.json'
    proof.write_text(json.dumps(receipt(legacy.sha256(source), fingerprint)), encoding='utf-8')
    proof_before = proof.read_bytes()
    previous = legacy.parser, legacy.write_json, legacy.summarize, strict.validate_receipt
    def fake_main(event, argv):
        a = legacy.parser().parse_args(argv)
        state = torch.load(a.resume, weights_only=False)
        current = {**state['contract'], 'implementation': legacy.stable_json_fingerprint(
            {'tools/joint_occ_experiments/run.py': 'unchanged_original_code'})}
        assert current['implementation'] == fingerprint
        cursor = legacy.restore_checkpoint(state, {'gt': model}, {'gt': optimizer}, current)
        assert cursor[:3] == (state['epoch'], state['batch'], state['updates'])
        from tools.joint_occ_experiments.performance.probe import assert_close_state
        assert_close_state(model.state_dict(), state['models']['gt'])
        assert_close_state(optimizer.state_dict(), state['optimizers']['gt'])
        assert torch.equal(legacy.rng_state()['torch'], state['rng']['torch'])
        legacy.train_arm(model, optimizer, provider, [row, row], 'gt', update=state['updates']+1, total_updates=500)
        destination = Path(a.out_dir); destination.mkdir(exist_ok=True)
        torch.save(legacy.checkpoint({'gt': model}, {'gt': optimizer}, current, 0,
            state['batch']+1, state['updates']+1, {}), destination/'last.pt')
        result = dict(status='stopped_at_complete_update', contract=current, updates=state['updates']+1)
        legacy.write_json(destination/'training.json', result)
        assert 'NOT lossless' in legacy.summarize(result)
        return 0
    monkeypatch.setattr(legacy, 'main', fake_main)
    out = tmp_path/'approved'
    args = []
    for name in ('config', 'train-cache', 'dev-cache', 'population-manifest', 'base-checkpoint',
                 'dataroot', 'train-info', 'dev-info', 'ccr-history-cache', 'ccr-val-history-cache'):
        args += ['--'+name, str(tmp_path/name)]
    args += ['--mode', 'gt', '--out-dir', str(out), '--resume', str(source), '--horizon-proof', str(proof)]
    with pytest.raises(ValueError, match='explicit'): approved.main(argv=args)
    assert not out.exists()
    # The unchanged strict entry MUST still reject this failed receipt.
    with pytest.raises(RuntimeError): strict.main(argv=args)
    args += ['--accept-numerical-drift']
    assert approved.main(argv=args) == 0
    acceptance = json.loads((out/'float_drift_acceptance.json').read_text())
    assert acceptance['original_pass_gate'] is False
    assert acceptance['measured_implementation'] == fingerprint
    assert json.loads((out/'horizon_import.json').read_text())['source_updates'] == 149
    assert json.loads((out/'training.json').read_text())['execution_acceptance'] == acceptance
    args[args.index('--resume')+1] = str(out/'last.pt')
    del args[args.index('--horizon-proof'):args.index('--horizon-proof')+2]
    args[args.index('--out-dir')+1] = str(tmp_path/'wrong_resume_directory')
    with pytest.raises(RuntimeError, match='SAME output directory'): approved.main(argv=args)
    assert not (tmp_path/'wrong_resume_directory').exists()
    args[args.index('--out-dir')+1] = str(out)
    assert approved.main(argv=args) == 0
    assert torch.load(out/'last.pt', weights_only=False)['updates'] == 151
    assert source.read_bytes() == before and proof.read_bytes() == proof_before
    assert (legacy.parser, legacy.write_json, legacy.summarize, strict.validate_receipt) == previous
