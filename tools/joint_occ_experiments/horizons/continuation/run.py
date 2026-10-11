"""Resume measured two-horizon GT training with EXPLICIT float-drift acceptance.

The original parity report remains FAIL. This is not a lossless equivalence
claim; original training kernels, weights, losses and schedule are untouched.
"""
import json
import math
from pathlib import Path
import signal
import threading
from .. import run as strict

OBSERVED_DRIFT_LIMIT = 1e-3
POLICY = 'explicit_user_accepts_small_multistep_float_drift_v1'


def validate_accepted_receipt(receipt, backend, source_sha, fingerprint):
    if backend != 'sequential_h2': raise RuntimeError('only the measured sequential two-horizon backend is allowed')
    if receipt.get('status') != 'complete' or receipt.get('source_snapshot_sha256') != source_sha:
        raise RuntimeError('complete report for THIS exact source checkpoint required')
    candidate = receipt.get('candidates', {}).get(backend, {})
    if candidate.get('execution_fingerprint') != fingerprint:
        raise RuntimeError('measured training implementation differs; no silent code switch')
    for key in ('faster_each_pass', 'memory_headroom_ge_10percent'):
        if candidate.get(key) is not True: raise RuntimeError('failed '+key)
    speedup = float(candidate.get('speedup', 0))
    if not math.isfinite(speedup) or speedup < 1.1: raise RuntimeError('insufficient measured speedup')
    for name in ('current_fused', backend):
        warm = receipt.get('warmup', {}).get(name, {})
        if warm.get('numerical_parity') is not True or warm.get('loss_gradient_audit', {}).get('pass_gate') is not True:
            raise RuntimeError('single-step/loss/gradient checks must still pass: '+name)
        state = receipt.get('final_state_parity', {}).get(name, {})
        difference = float(state.get('max_state_abs_difference', math.inf))
        if not math.isfinite(difference) or not 0 <= difference <= OBSERVED_DRIFT_LIMIT:
            raise RuntimeError('unmeasured or larger state drift is NOT approved: '+name)
    return dict(policy=POLICY, lossless_claim=False, scientific_quality_verified=False,
        original_pass_gate=candidate.get('pass_gate'),
        original_numerical_parity=candidate.get('numerical_parity'),
        original_baseline_repeat_parity=candidate.get('baseline_repeat_parity'),
        original_final_state_parity=receipt['final_state_parity'],
        observed_drift_acceptance_cap=OBSERVED_DRIFT_LIMIT, speedup=speedup,
        note='User accepted small measured drift for training; original equivalence gates/tolerances NOT changed.')


def main(stop_event=None, argv=None):
    legacy = strict.legacy
    original_parser, original_validator = legacy.parser, strict.validate_receipt
    original_write, original_summary = legacy.write_json, legacy.summarize
    def parser():
        p = original_parser()
        p.description = __doc__
        p.add_argument('--accept-numerical-drift', action='store_true',
            help='explicitly accept measured <=0.001 state drift; NOT a lossless parity pass')
        return p
    check = parser()
    check.add_argument('--horizon-proof')
    a = check.parse_args(argv)
    if not a.accept_numerical_drift:
        raise ValueError('explicit --accept-numerical-drift required; strict gate remains unchanged')
    if not a.resume or a.mode != 'gt' or a.training_backend != 'fused' or a.evaluate_only:
        raise ValueError('existing fused GT training checkpoint required')
    out = Path(a.out_dir).resolve()
    marker = out/'float_drift_acceptance.json'
    previous_marker = Path(a.resume).resolve().parent/'float_drift_acceptance.json'
    if previous_marker.is_file() and previous_marker.resolve() != marker:
        raise RuntimeError('approved continuation must resume in the SAME output directory')
    if out.exists() and not marker.is_file():
        raise RuntimeError('existing continuation directory must retain its acceptance record')
    acceptance = json.loads(marker.read_text(encoding='utf-8')) if marker.exists() else None
    if acceptance is not None and acceptance.get('policy') != POLICY:
        raise RuntimeError('different acceptance policy in output directory')
    def validator(receipt, backend, source_sha, fingerprint):
        nonlocal acceptance
        acceptance = validate_accepted_receipt(receipt, backend, source_sha, fingerprint)
        acceptance.update(source_checkpoint=str(Path(a.resume).resolve()), source_sha256=source_sha,
            measured_implementation=fingerprint, proof=str(Path(a.horizon_proof).resolve()),
            proof_sha256=legacy.sha256(a.horizon_proof))
    def write(path, value):
        path = Path(path)
        if path.resolve() == out/'horizon_import.json':
            if acceptance is None: raise RuntimeError('missing original measured receipt')
            original_write(marker, acceptance)
            value = {**value, 'execution_acceptance': acceptance}
        elif path.resolve() == out/'training.json':
            value = {**value, 'execution_acceptance': acceptance}
        return original_write(path, value)
    def summarize(result):
        return ('EXECUTION: sequential_h2, user-accepted small floating-point drift; NOT lossless.\n'
                'Original multi-step numerical/repeat FAIL remains recorded; final quality not guaranteed.\n'
                +original_summary(result))
    try:
        legacy.parser, strict.validate_receipt = parser, validator
        legacy.write_json, legacy.summarize = write, summarize
        print('EXPLICIT FLOAT-DRIFT ACCEPTANCE: keep old FAIL report; reuse measured kernel, '
              'weights/Adam/RNG/cursor/LR intact; NO new probe.', flush=True)
        return strict.main(stop_event, argv)
    finally:
        legacy.parser, strict.validate_receipt = original_parser, original_validator
        legacy.write_json, legacy.summarize = original_write, original_summary


if __name__ == '__main__':
    event = threading.Event()
    def stop(*_):
        event.set()
        print('Stop requested: save after completed Adam update, please wait.', flush=True)
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    main(event)
