"""Gated resume: ONLY two-horizon rendering; original window-network order."""
import json
from pathlib import Path
import signal
import threading
import torch
from .. import run as legacy
from ..performance.probe import legacy_fingerprint
from ..batching.run import import_contract, validate_receipt
from .probe import execution_fingerprint
from .engine import train_arm


def main(stop_event=None, argv=None):
    original_parser = legacy.parser
    def patched_parser():
        result = original_parser()
        result.description = __doc__
        result.add_argument('--horizon-proof', help='successful two-horizon speed.json for original import')
        return result
    a = patched_parser().parse_args(argv)
    if a.mode != 'gt' or a.training_backend != 'fused' or not a.resume or a.evaluate_only:
        raise ValueError('GT-only fused RESUME required; hard evaluator remains unchanged')
    if a.window_batch_size != 4 or a.source_budget != 128:
        raise ValueError('keep global batch4/source128 and optimizer step count')
    root = Path(__file__).resolve().parents[3]
    source, out = Path(a.resume).resolve(), Path(a.out_dir).resolve()
    saved = torch.load(source, map_location='cpu', weights_only=False)
    previous = saved['contract']['implementation']
    old_fingerprint, fingerprint = legacy_fingerprint(root), execution_fingerprint(root)
    if previous == old_fingerprint:
        if out.exists() or out == source.parent or source.parent in out.parents:
            raise ValueError('original import requires NEW directory OUTSIDE source run')
        if not a.horizon_proof: raise ValueError('run the read-only two-horizon probe first')
        validate_receipt(json.loads(Path(a.horizon_proof).read_text(encoding='utf-8')),
                         'sequential_h2', legacy.sha256(source), fingerprint)
    elif previous != fingerprint:
        raise RuntimeError('execution differs from checkpoint; packed/other backend cannot silently resume here')
    del saved
    old_hash, old_restore, old_step = legacy.stable_json_fingerprint, legacy.restore_checkpoint, legacy.train_arm
    def hash_contract(value):
        return fingerprint if isinstance(value, dict) and 'tools/joint_occ_experiments/run.py' in value else old_hash(value)
    def restore(saved, models, optimizers, current):
        converted, imported = import_contract(saved, current, old_fingerprint)
        result = old_restore(converted, models, optimizers, current)
        if imported:
            out.mkdir(parents=True, exist_ok=True)
            legacy.write_json(out/'horizon_import.json', dict(source=str(source), source_sha256=legacy.sha256(source),
                source_updates=saved['updates'], original_implementation=old_fingerprint,
                execution_implementation=fingerprint, proof=a.horizon_proof,
                unchanged='weights/Adam/RNG/cursor/data/global_batch/source_budget/LR/cosine/loss/network_execution',
                changed='two-horizon rendering/loss/backward only; numerical, NOT byte identity'))
        return result
    try:
        legacy.parser = patched_parser
        legacy.stable_json_fingerprint, legacy.restore_checkpoint, legacy.train_arm = hash_contract, restore, train_arm
        print('TWO-HORIZON GT RESUME: original sequential WM/CCR+per-window backward; '
              'global batch4/source128, same Adam/RNG/cosine; hard inference unchanged.', flush=True)
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
