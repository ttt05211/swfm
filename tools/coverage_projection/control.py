"""Verify completed control WEIGHTS, independently of its old eval status.

The original C run finished training, then its evaluation failed. Recovery
deliberately wrote a separate report instead of rewriting C_train/results.json.
This pilot evaluates the frozen control afresh, so stale eval status is neither
training-completion evidence nor grounds for rejecting completed weights.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch

CONTROL_PROTOCOL = 'AdaLN_forward_RANDOM_TRAIN4086x1_singleGPU_raw_FREEreadout_v2'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''): digest.update(block)
    return digest.hexdigest()


def canonical(value): return json.loads(json.dumps(value))


def validate(saved, result):
    """Do not turn a partial training checkpoint into a completed control."""
    c = canonical(saved.get('contract', {})); errors = []
    expected = dict(protocol=CONTROL_PROTOCOL, epochs=1, total_updates=1031,
        source_policy='causal_forward_CV_history', registration='none',
        known='TRAIN_OCC_STC_dense_valid', parameters=2816122, world_size=1,
        seed=20261011, learning_rates=[5e-4, 3e-4], schedule='whole_1_epoch_cosine_floor0.01',
        objective=dict(CE=1., Lovasz=1., ego=0., motion=0., existence=0.))
    for name, value in expected.items():
        if c.get(name) != value:
            errors.append(f'contract.{name}: actual={c.get(name)!r}, expected={value!r}')
    for name, value in dict(protocol=CONTROL_PROTOCOL, epoch=1, batch=0, updates=1031).items():
        if saved.get(name) != value:
            errors.append(f'checkpoint.{name}: actual={saved.get(name)!r}, expected={value!r}')
    for name, count in (('train_keys', 4086), ('eval_keys', 64)):
        keys = c.get(name, [])
        if (not isinstance(keys, list) or len(keys) != count
                or any(not isinstance(k, list) or len(k) != 2
                    or any(not isinstance(v, str) or not v for v in k) for k in keys)
                or len(set(map(tuple, keys))) != count):
            errors.append(f'contract.{name}: require {count} unique (scene,t0) keys')
    if c.get('model', {}).get('source_policy') != 'causal_forward_CV_history':
        errors.append('contract.model.source_policy: latest forward model required')
    if set(saved.get('models', {})) != {'gt'} or not saved.get('models', {}).get('gt'):
        errors.append('checkpoint.models: nonempty GT model state required')
    if canonical(result.get('contract', {})) != c:
        errors.append('results.json contract differs from the frozen checkpoint contract')
    if errors:
        raise ValueError('CONTROL checkpoint rejected (no training started):\n  '+'\n  '.join(errors))
    # Keep the original status/evaluations AS RECORDED. Completion here refers
    # solely to training; this does not pretend its old evaluation completed.
    return dict(result, contract=c, source_checkpoint_completion=dict(
        protocol=saved['protocol'], epoch=1, batch=0, updates=1031,
        training_complete=True, original_report_status=result.get('status'),
        original_report_updates=result.get('updates'), old_evaluation_reused=False,
        control_will_be_evaluated_fresh=True))


def reference(folder):
    folder = Path(folder).resolve(); checkpoint = folder/'epoch_0001.pt'
    if not checkpoint.is_file():
        raise ValueError('CONTROL completed epoch_0001.pt missing: '+str(checkpoint))
    result = json.loads((folder/'results.json').read_text(encoding='utf-8'))
    digest = sha(checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    verified = validate(saved, result)
    if sha(checkpoint) != digest: raise ValueError('CONTROL checkpoint changed while reading')
    verified['source_checkpoint_completion'].update(checkpoint=str(checkpoint), sha256=digest)
    return verified


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--control-dir', type=Path, required=True)
    value = reference(p.parse_args().control_dir)
    print('CONTROL_PREFLIGHT '+json.dumps(value['source_checkpoint_completion']), flush=True)


if __name__ == '__main__': main()
