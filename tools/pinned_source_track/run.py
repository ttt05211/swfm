"""Train ONLY pinned-pair track arm; reuse the completed matched baseline read-only."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import signal
import sys
import threading
import torch
from tools.joint_occ_experiments import run as legacy
from tools.unified_source_training import train as old, evaluation as old_eval, workflow as old_workflow
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from . import geometry

PROTOCOL = 'pinned_original_pair_track_train20pct_1epoch_v1'


def find_reference(search_root):
    candidates = []
    for root in Path(search_root).glob('track_matched_20pct_1ep_*'):
        path = root/'baseline/results.json'
        if path.is_file() and (root/'workflow.json').is_file():
            value = json.loads(path.read_text(encoding='utf-8'))
            if value.get('status') == 'complete' and value.get('contract', {}).get('protocol') == old.PROTOCOL:
                candidates.append(root)
    if not candidates: raise RuntimeError('set --reference-dir to original completed matched baseline root')
    # Chronological selection only, never a metric/epoch search. Exact provenance
    # is validated before training and bound into the new workflow for resume.
    return max(candidates, key=lambda p: (p.stat().st_mtime_ns, str(p)))


def reference(directory):
    root = Path(directory).resolve(); manifest = json.loads((root/'workflow.json').read_text(encoding='utf-8'))
    if manifest.get('protocol') != 'matched_track_KTA_one_epoch_workflow_v1': raise RuntimeError('original matched track workflow required')
    baseline = old_workflow.completed(root/'baseline')
    if baseline is None: raise RuntimeError('completed baseline required; will NOT retrain it')
    c = baseline['contract']
    if len(c['train_keys']) != 4086 or len(c['eval_keys']) != 64 or c['epochs'] != 1 or c['initialization'] != 'identical_RANDOM_no_teacher':
        raise RuntimeError('original TRAIN4086 x1 / DEV64 RANDOM budget required')
    for key, digest in manifest['data'].items():
        if legacy.sha256(manifest['args'][key]) != digest: raise RuntimeError('reference data changed: '+key)
    code = Path(__file__).resolve().parents[2]
    for file, digest in manifest['implementation'].items():
        if legacy.sha256(code/str(file).replace('\\', '/')) != digest: raise RuntimeError('original implementation changed: '+file)
    return manifest, baseline


def dependencies():
    return sorted(set(old.dependency_files()+list(Path(__file__).parent.glob('*.py'))))


@contextmanager
def training_scope(ref):
    """Dedicated child process only; always restore imported bindings."""
    names = ('parser', 'PROTOCOL', 'history_inputs', 'prefetch_adapter', 'dependency_files', 'GTBody')
    saved = {n: getattr(old, n) for n in names}
    eval_names = ('Predictor', 'live_variant', 'external_forward', 'PROTOCOL')
    original_eval = {n: getattr(old_eval, n) for n in eval_names}
    old_main = legacy.main; prior_groups = legacy.epoch_groups
    original_dependencies = old.dependency_files
    def parser():
        p = legacy.parser(); p.set_defaults(mode='gt', epochs=1, train_fraction=.2, eval_population='dev64')
        p.add_argument('--track-arm', choices=('unified_pair',), default='unified_pair')
        p.add_argument('--stc-root', required=True)
        return p
    def model(config):
        body = saved['GTBody'](config)
        if tensor_state_fingerprint(body.state_dict()) != ref['random_initializer']:
            raise RuntimeError('initializer differs from completed baseline; no optimizer step')
        return body
    def groups(records, seed, epoch, batch, sources):
        if [[str(r['scene_name']), str(r['t0_token'])] for r in records] != ref['contract']['train_keys']:
            raise RuntimeError('TRAIN order/population differs from baseline')
        result = prior_groups(records, seed, epoch, batch, sources)
        if len(result) != ref['contract']['total_updates']: raise RuntimeError('update budget differs from baseline')
        return result
    def invoke(event, argv):
        summary = legacy.summarize
        legacy.summarize = lambda r: summary(r).replace('Baseline vs unified track + actual-time multi-frame KTA. Original Strong/CLEAR fixed.',
            'Original t-1/t0 pair PINNED; original KTA/anchors unchanged; extend older frames only.')
        try: return old_main(event, argv)
        finally: legacy.summarize = summary
    try:
        old.parser, old.PROTOCOL = parser, PROTOCOL
        old.history_inputs = lambda arm: geometry.PinnedHistory
        old.prefetch_adapter = geometry.prefetch_adapter
        old.dependency_files = lambda: sorted(set(original_dependencies()+list(Path(__file__).parent.glob('*.py'))))
        old.GTBody = model
        old_eval.Predictor = geometry.PinnedPredictor
        old_eval.live_variant = lambda row, *a: row
        old_eval.external_forward = geometry.forward_pair
        old_eval.PROTOCOL = PROTOCOL+'_DEV64'
        legacy.main, legacy.epoch_groups = invoke, groups
        yield
    finally:
        legacy.main, legacy.epoch_groups = old_main, prior_groups
        for n, v in saved.items(): setattr(old, n, v)
        for n, v in original_eval.items(): setattr(old_eval, n, v)


def publish(out, ref):
    path = out/'unified_pair/results.json'
    if not path.is_file(): return
    result = json.loads(path.read_text(encoding='utf-8'))
    if result.get('status') != 'complete': return
    c = result['contract']; candidate = result['evaluation']
    if c['protocol'] != PROTOCOL or result['epoch'] != 1 or result['updates'] != c['total_updates'] or candidate['windows'] != 64:
        raise RuntimeError('candidate is not the completed one-epoch arm')
    a = torch.load(out/'unified_pair/last.pt', map_location='cpu', weights_only=False)
    if (a['contract'] != c or a['random_initializer'] != ref['random_initializer'] or a['epoch'] != 1 or a['batch'] != 0):
        raise RuntimeError('candidate initializer/checkpoint mismatch')
    if set(c) != set(ref['contract']): raise RuntimeError('scientific contract schema changed')
    for key in c:
        if key not in ('protocol', 'implementation') and c[key] != ref['contract'][key]:
            raise RuntimeError('candidate scientific budget differs: '+key)
    m = candidate['variants']['gt']; lines = ['===== PINNED ORIGINAL PAIR / BASELINE REUSED =====',
        'Only unified_pair newly RANDOM trained, TRAIN4086 x1; baseline is existing completed same-budget run.',
        'Original last-pair identity/KTA fixed; older-only track extension. No multiframe fit or baseline retraining.',
        'Same init/order/CE+Lovasz/cosine/DEV64. No mask/alignment, no automatic promotion.',
        'arm route mIoU IoU MovingMicro']
    for arm, values in (('baseline', ref['metrics']), ('unified_pair', m)):
        for name in old_eval.ROUTES:
            v = values[name]; lines.append(f'{arm} {name} {v["mIoU"]:.6f} {v["IoU"]:.6f} {v["MovingMicro"]:.6f}')
    delta = {n: {k: m[n][k]-ref['metrics'][n][k] for k in ('mIoU', 'IoU', 'MovingMicro')} for n in old_eval.ROUTES}
    lines.append('unified_pair_minus_baseline='+json.dumps(delta))
    legacy.write_json(out/'comparison.json', dict(baseline=ref, candidate=result,
        unified_pair_minus_baseline=delta, baseline_retrained=False, automatic_promotion=False))
    (out/'summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8'); print('\n'.join(lines), flush=True)


def main(event=None, argv=None):
    event = event or threading.Event(); p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-dir'); p.add_argument('--out-dir', required=True)
    p.add_argument('--search-root', default='/root/nas/occ/swfm/outputs/p0_f9_shared_history')
    p.add_argument('--resume', action='store_true'); p.add_argument('--max-updates', type=int, default=0)
    a = p.parse_args(argv)
    if a.max_updates < 0: raise ValueError('nonnegative optional safe-stop budget required')
    if not a.reference_dir and a.resume:
        a.reference_dir = json.loads((Path(a.out_dir)/'workflow.json').read_text(encoding='utf-8'))['reference']
    a.reference_dir = a.reference_dir or str(find_reference(a.search_root))
    manifest, ref = reference(a.reference_dir)
    out, source = Path(a.out_dir).resolve(), Path(a.reference_dir).resolve()
    if out == source or source in out.parents or out in source.parents: raise ValueError('NEW independent output outside reference')
    code = Path(__file__).resolve().parents[2]
    identity = dict(protocol=PROTOCOL, reference=str(source), baseline_sha256=ref['checkpoint_sha256'],
        workflow_sha256=legacy.sha256(source/'workflow.json'), result_sha256=legacy.sha256(source/'baseline/results.json'),
        implementation={str(f.relative_to(code)): legacy.sha256(f) for f in dependencies()})
    lock = out.parent/'.pinned_pair_locks'/legacy.stable_json_fingerprint(str(out)); lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(lock):
        if out.exists():
            if not a.resume or json.loads((out/'workflow.json').read_text(encoding='utf-8')) != identity:
                raise RuntimeError('NEW output, or SAME unchanged workflow with --resume')
        else:
            if a.resume: raise RuntimeError('missing new workflow')
            out.mkdir(); legacy.write_json(out/'workflow.json', identity)
        print(f'PINNED_PAIR reference={source}; baseline READ ONLY; NEW training only={out}/unified_pair', flush=True)
        fields = dict(manifest['args']); fields.pop('out_dir', None)
        child = out/'unified_pair'; args = []
        for k, v in fields.items():
            if k in ('resume', 'evaluate_only', 'override') or v is None: continue
            if isinstance(v, bool):
                if v: args += ['--'+k.replace('_', '-')]
            else: args += ['--'+k.replace('_', '-'), str(v)]
        for v in fields.get('override') or []: args += ['--override', v]
        args += ['--out-dir', str(child), '--track-arm', 'unified_pair']
        if a.max_updates: args += ['--max-updates', str(a.max_updates)]
        if (child/'last.pt').is_file(): args += ['--resume', str(child/'last.pt')]
        if not (child/'results.json').is_file() or json.loads((child/'results.json').read_text())['status'] != 'complete':
            with training_scope(ref): old.main(event, args)
        if legacy.sha256(source/'baseline/last.pt') != identity['baseline_sha256']: raise RuntimeError('reference baseline changed')
        publish(out, ref)
    return 0


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
