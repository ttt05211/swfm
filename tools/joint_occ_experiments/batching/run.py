"""Explicit, gated import of old GT Adam/RNG into a NEW execution directory."""
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
from pathlib import Path
import signal
import threading
import torch
from .. import run as legacy
from ..performance.probe import legacy_fingerprint
from .probe import execution_fingerprint
from .engine import train_arm


def import_contract(saved, current, old_implementation):
    """Only execution implementation may differ; every scientific field stays."""
    if saved.get('protocol') != legacy.PROTOCOL: raise RuntimeError('checkpoint protocol differs')
    previous = saved['contract']
    if previous == current: return saved, False
    if previous.get('implementation') != old_implementation:
        raise RuntimeError('not the exact original GT implementation; no silent backend switch')
    expected = {**previous, 'implementation': current['implementation']}
    if expected != current: raise RuntimeError('data/model/batch/schedule/loss/precision differs')
    return {**saved, 'contract': expected}, True


def validate_receipt(receipt, backend, source_sha, fingerprint):
    if receipt.get('status') != 'complete' or receipt.get('source_snapshot_sha256') != source_sha:
        raise RuntimeError('speed probe must be complete and use THIS exact source checkpoint')
    candidate = receipt.get('candidates', {}).get(backend, {})
    if candidate.get('execution_fingerprint') != fingerprint or not candidate.get('pass_gate'):
        raise RuntimeError('selected batching execution has NOT passed numeric/speed/memory gates')
    for key in ('numerical_parity', 'baseline_repeat_parity', 'faster_each_pass', 'memory_headroom_ge_10percent'):
        if candidate.get(key) is not True: raise RuntimeError('failed '+key)
    if candidate.get('speedup', 0) < 1.1: raise RuntimeError('insufficient measured speedup')


def main(stop_event=None, argv=None):
    original_parser = legacy.parser
    parser = original_parser()
    parser.add_argument('--horizon-group', type=int, choices=(1, 2), default=2)
    parser.add_argument('--batching-proof', help='successful read-only batching speed.json, required for original import')
    a, remaining = parser.parse_known_args(argv)
    if remaining: raise ValueError('unknown arguments: '+str(remaining))
    if a.mode != 'gt' or a.training_backend != 'fused' or not a.resume or a.evaluate_only:
        raise ValueError('GT-only fused RESUME required; original evaluator remains unchanged')
    if a.window_batch_size != 4 or a.source_budget != 128:
        raise ValueError('keep global batch4/source128; do not alter optimizer step count')
    root = Path(__file__).resolve().parents[3]
    source, out = Path(a.resume).resolve(), Path(a.out_dir).resolve()
    saved = torch.load(source, map_location='cpu', weights_only=False)
    previous = saved['contract']['implementation']
    old_fingerprint = legacy_fingerprint(root)
    fingerprint = execution_fingerprint(root, a.horizon_group)
    imported = previous == old_fingerprint
    if imported:
        if out.exists() or out == source.parent or source.parent in out.parents:
            raise ValueError('original import requires NEW directory OUTSIDE source run')
        if not a.batching_proof: raise ValueError('run the read-only batching speed probe first')
        validate_receipt(json.loads(Path(a.batching_proof).read_text(encoding='utf-8')),
                         'packed_h'+str(a.horizon_group), legacy.sha256(source), fingerprint)
    elif previous != fingerprint:
        raise RuntimeError('execution code/group differs from checkpoint; explicit re-audit needed')
    del saved
    old_hash, old_restore, old_step = legacy.stable_json_fingerprint, legacy.restore_checkpoint, legacy.train_arm
    # Isolated process adapter: original trainer files, contract and hard eval
    # remain intact. New checkpoints are fingerprinted as a different execution.
    def hash_contract(value):
        return fingerprint if isinstance(value, dict) and 'tools/joint_occ_experiments/run.py' in value else old_hash(value)
    def restore(saved, models, optimizers, current):
        converted, did_import = import_contract(saved, current, old_fingerprint)
        result = old_restore(converted, models, optimizers, current)
        if did_import:
            out.mkdir(parents=True, exist_ok=True)
            legacy.write_json(out/'batching_import.json', dict(source=str(source), source_sha256=legacy.sha256(source),
                source_updates=saved['updates'], original_implementation=old_fingerprint,
                execution_implementation=fingerprint, horizon_group=a.horizon_group, proof=a.batching_proof,
                unchanged='weights/Adam/RNG/cursor/data/global_batch/source_budget/LR/cosine/loss',
                equivalence='numerical, NOT byte identity; inference unchanged'))
        return result
    def patched_parser():
        # Already validated additional arguments; legacy gets the same values.
        result = original_parser()
        result.add_argument('--horizon-group', type=int, choices=(1, 2), default=2)
        result.add_argument('--batching-proof')
        return result
    try:
        with ThreadPoolExecutor(max_workers=min(4, a.cpu_workers)) as pool:
            legacy.parser = patched_parser
            legacy.stable_json_fingerprint, legacy.restore_checkpoint = hash_contract, restore
            legacy.train_arm = partial(train_arm, horizon_group=a.horizon_group, projection_pool=pool)
            print(f'PACKED GT RESUME: global_batch=4 source128 horizon_group={a.horizon_group}; '
                  'same Adam/RNG/cosine, original hard inference; no silent OOM retry', flush=True)
            return legacy.main(stop_event, argv)
    finally:
        legacy.parser = original_parser
        legacy.stable_json_fingerprint, legacy.restore_checkpoint, legacy.train_arm = old_hash, old_restore, old_step


if __name__ == '__main__':
    event = threading.Event()
    def stop(*_):
        event.set()
        print('Stop requested: save after completed Adam update, please wait.', flush=True)
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    main(event)
