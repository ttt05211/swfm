"""Independent RANDOM clean/robust TRAIN20% x2, identical GT-body and loss."""
from collections import Counter
import hashlib
from pathlib import Path
import signal
import sys
import threading
import time
import torch
from tools.joint_occ_experiments import run as legacy, data
from tools.shared_history_experiments.full_gt.model import GTBody
from tools.shared_history_experiments.full_gt import engine as gt
from tools.shared_history_experiments.refine.history import ResidualHistoryInputs
from tools.shared_history_experiments.run import NoLegacyHeadGraph
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from .augmentation import RECIPE
from .data import finish_backgrounds, prefetch_adapter
from .evaluation import evaluator

PROTOCOL = 'shared_gt_body_perception_robustness_train20pct_2epochs_v1'


def parser():
    p = legacy.parser(); p.description = __doc__
    p.set_defaults(mode='gt', epochs=2, train_fraction=.2, eval_population='dev64')
    p.add_argument('--augmentation', choices=('clean', 'robust'), required=True)
    p.add_argument('--stc-root', required=True)
    return p


def stc_inventory(a):
    """Verify the existing fixed population; never substitute another window."""
    _, dev64, _ = legacy.load_manifest(a.population_manifest)
    _, records = legacy.load_cache(a.dev_cache)
    selected = legacy.align_records(records, dev64)
    if len(selected) != 64: raise RuntimeError('exact existing DEV64 required')
    inventory = []
    for record in selected:
        for token in record['history_tokens'][-4:]:
            path = Path(a.stc_root)/record['scene_name']/str(token)/'labels.npz'
            st = path.stat()
            if not path.is_file() or not st.st_size: raise RuntimeError('empty/non-file STC history: '+str(path))
            inventory.append([str(path.resolve()), st.st_size, st.st_mtime_ns])
    return inventory


def main(event=None, argv=None):
    # Only this isolated process changes imported bindings; no repository core
    # file is edited, and every binding is restored after success/failure.
    p = parser(); a = p.parse_args(argv)
    if (a.mode != 'gt' or a.epochs != 2 or a.train_fraction != .2 or a.eval_population != 'dev64'
            or a.window_batch_size != 4 or a.source_budget != 128 or a.seed != 20261011
            or a.training_backend != 'fused' or (a.object_weight, a.ego_weight, a.lovasz_weight) != (1., 1., 1.)):
        raise ValueError('fixed GT-body / TRAIN20% x2 / B4-source128 / original seed / CE+Lovasz / DEV64')
    out = Path(a.out_dir).resolve()
    if a.evaluate_only or a.resume and Path(a.resume).resolve().parent != out:
        raise ValueError('same-directory resume only; use workflow for both trained arms')
    if not Path(a.stc_root).is_dir(): raise RuntimeError('existing compact STC directory required; no download/extraction')
    # Preflight EXACT fixed DEV64 before spending any training budget.
    inventory = stc_inventory(a)
    names = ('parser', 'train_arm', 'evaluate', 'JointOccEgo', 'HistoryInputs', 'BothActionsExecution',
             'PROTOCOL', 'summarize', 'stable_json_fingerprint', 'save_resume_checkpoint', 'prefetched_batches')
    original = {n: getattr(legacy, n) for n in names}; old_ego = data.ego_inputs
    initial_epoch = torch.load(a.resume, map_location='cpu', weights_only=False)['epoch'] if a.resume else 0
    root = Path(__file__).resolve().parents[2]
    files = list(Path(__file__).parent.glob('*.py')) + [root/x for x in (
        'tools/shared_history_experiments/full_gt/model.py', 'tools/shared_history_experiments/full_gt/engine.py',
        'tools/shared_history_experiments/refine/history.py', 'tools/shared_history_experiments/refine/model.py',
        'tools/shared_history_experiments/refine/geometry.py', 'tools/shared_history_experiments/model.py',
        'tools/shared_history_experiments/history.py', 'tools/final_semantic_evaluation/predictor.py',
        'tools/scene_ego_head/integration.py', 'tools/scene_ego_head/diagnostics/engine.py')]
    def fingerprint(value):
        if isinstance(value, dict) and 'tools/joint_occ_experiments/run.py' in value:
            value = dict(base=value, augmentation=a.augmentation, recipe=RECIPE.specification(),
                STC_inventory=inventory, prototype={str(f.relative_to(root)): legacy.sha256(f) for f in files})
        return original['stable_json_fingerprint'](value)
    def train(model, optimizer, provider, rows, mode, **kwargs):
        tick = time.perf_counter(); finished = [finish_backgrounds(provider, row) for row in rows]
        background_seconds = time.perf_counter()-tick
        stats = gt.train_arm(model, optimizer, provider, finished, mode, **kwargs)
        stats['seconds'] += background_seconds
        stats['stages_seconds']['fresh_corrupted_Strong'] = background_seconds
        audits = [row[1].get('_perception_augmentation', {}) for row in rows]
        stats['augmentation'] = dict(arm=a.augmentation,
            families=dict(Counter(v.get('family', 'clean') for v in audits)),
            selected_windows=sum(bool(v.get('selected')) for v in audits),
            changed_windows=sum(not v.get('clean_geometry_reusable', True) for v in audits),
            changed_voxels=sum(sum(v.get('changed_voxels', [])) for v in audits),
            CPU_geometry_seconds_sum=sum(v.get('geometry_prepare_seconds', 0.) for v in audits),
            actual_sources=sum(len(row[0]['source_class_id']) for row in finished),
            nominal_sources=sum(row[1].get('_perception_augmentation', {}).get('original_sources',
                                len(row[0]['source_class_id'])) for row in rows))
        return stats
    def summary(result):
        text = ['===== PERCEPTION ROBUSTNESS / TWO-EPOCH GT BODY =====',
            f'status={result["status"]}; arm={a.augmentation}; completed_epochs={result.get("epoch", 0)}; updates={result.get("updates", 0)}',
            'RANDOM; same TRAIN20% / order / B4-source128 nominal grouping / 2-epoch cosine.',
            'SharedEncoder/STWM/SemanticRefiner unchanged; ONLY full6 CE+Lovasz; no ego head.',
            'GT ego conditioned; corruption only HISTORY, future targets clean; known unchanged.',
            'Changed inputs rebuild components/association/KTA/Strong/memory; old caches READ ONLY.',
            'Final fixed DEV64 OCC+GT / STC+GT; no counterfactuals, masks, threshold or best-epoch selection.']
        if 'performance' in result: text.append('PERFORMANCE '+legacy.json.dumps(result['performance']))
        for name, m in result.get('evaluation', {}).get('variants', {}).get('gt', {}).items():
            text.append(f'{name} mIoU={m["mIoU"]:.6f} IoU={m["IoU"]:.6f} MovingMicro={m["MovingMicro"]:.6f}')
            for h, v in m.get('per_horizon', {}).items():
                text.append(f'  {h}s mIoU={v["mIoU"]:.6f} IoU={v["IoU"]:.6f} MovingMicro={v["MovingMicro"]:.6f}')
        return '\n'.join(text)+'\n'
    def save(path, payload):
        original['save_resume_checkpoint'](path, payload)
        if Path(path).name == 'last.pt' and payload['epoch'] > 0 and payload['batch'] == 0:
            target = Path(path).with_name(f'epoch_{payload["epoch"]:04d}.pt')
            if not target.exists(): original['save_resume_checkpoint'](target, payload)
    try:
        legacy.parser = lambda: p
        legacy.train_arm, legacy.evaluate = train, evaluator(a.stc_root, inventory=inventory)
        legacy.JointOccEgo, legacy.HistoryInputs = GTBody, ResidualHistoryInputs
        legacy.BothActionsExecution, legacy.PROTOCOL = NoLegacyHeadGraph, PROTOCOL
        legacy.summarize, legacy.stable_json_fingerprint, legacy.save_resume_checkpoint = summary, fingerprint, save
        legacy.prefetched_batches = prefetch_adapter(original['prefetched_batches'], a.augmentation, a.seed, initial_epoch)
        data.ego_inputs = gt.history_inputs
        lock = out.parent/'.robustness_locks'/hashlib.sha256(str(out).encode()).hexdigest()
        lock.mkdir(parents=True, exist_ok=True)
        with evaluation_lock(lock): return legacy.main(event, argv)
    finally:
        data.ego_inputs = old_ego
        for name, value in original.items(): setattr(legacy, name, value)


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
