"""Matched RANDOM GT-body TRAIN20% x2: original known vs dense-valid."""
import hashlib
from pathlib import Path
import signal
import sys
import threading
import numpy as np
from tools.joint_occ_experiments import run as legacy, data
from tools.shared_history_experiments.full_gt.model import GTBody
from tools.shared_history_experiments.full_gt import engine as gt
from tools.shared_history_experiments.run import NoLegacyHeadGraph
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from tools.perception_robustness.run import stc_inventory
from .data import POLICIES, history_inputs, prefetch_adapter
from .evaluation import evaluator

PROTOCOL = 'shared_gt_history_known_train20pct_2epochs_v1'


def dependency_files():
    root = Path(__file__).resolve().parents[2]
    files = list(Path(__file__).parent.glob('*.py')) + list((root/'tools/joint_occ_experiments').glob('*.py'))
    files += [root/x for x in (
        'tools/shared_history_experiments/full_gt/model.py', 'tools/shared_history_experiments/full_gt/engine.py',
        'tools/shared_history_experiments/refine/history.py', 'tools/shared_history_experiments/refine/model.py',
        'tools/shared_history_experiments/refine/geometry.py', 'tools/shared_history_experiments/model.py',
        'tools/shared_history_experiments/history.py', 'tools/final_semantic_evaluation/predictor.py',
        'tools/scene_ego_head/integration.py', 'tools/scene_ego_head/diagnostics/engine.py',
        'tools/perception_robustness/evaluation.py', 'tools/perception_robustness/data.py',
        'tools/perception_robustness/run.py', 'tools/perception_robustness/augmentation.py',
        'tools/joint_occ_experiments/batching/geometry.py',
        'tools/shared_history_experiments/actions/validation/workflow.py')]
    return sorted(set(files))


def parser():
    p = legacy.parser(); p.description = __doc__
    p.set_defaults(mode='gt', epochs=2, train_fraction=.2, eval_population='dev64')
    p.add_argument('--known-policy', choices=POLICIES, required=True)
    p.add_argument('--stc-root', required=True)
    return p


def validate(a):
    if (a.mode != 'gt' or a.epochs != 2 or a.train_fraction != .2 or a.eval_population != 'dev64'
            or a.window_batch_size != 4 or a.source_budget != 128 or a.seed != 20261011
            or a.training_backend != 'fused' or (a.object_weight, a.ego_weight, a.lovasz_weight) != (1., 1., 1.)):
        raise ValueError('fixed RANDOM GT-body TRAIN20% x2 / B4-source128 / full6 CE+Lovasz / DEV64')


def main(event=None, argv=None):
    p = parser(); a = p.parse_args(argv); validate(a)
    out = Path(a.out_dir).resolve()
    if a.evaluate_only or a.resume and Path(a.resume).resolve().parent != out:
        raise ValueError('same-directory training resume only')
    inventory = stc_inventory(a)  # Check fixed STC coverage before training.
    names = ('parser', 'train_arm', 'evaluate', 'JointOccEgo', 'HistoryInputs', 'BothActionsExecution',
             'PROTOCOL', 'summarize', 'stable_json_fingerprint', 'save_resume_checkpoint', 'prefetched_batches')
    original = {n: getattr(legacy, n) for n in names}; old_ego = data.ego_inputs
    root = Path(__file__).resolve().parents[2]
    files = dependency_files()
    def fingerprint(value):
        if isinstance(value, dict) and any(str(k).replace('\\', '/') == 'tools/joint_occ_experiments/run.py' for k in value):
            value = dict(base=value, known_policy=a.known_policy, STC_inventory=inventory,
                prototype={str(f.relative_to(root)): legacy.sha256(f) for f in files})
        return original['stable_json_fingerprint'](value)
    def train(model, optimizer, provider, rows, mode, **kwargs):
        result = gt.train_arm(model, optimizer, provider, rows, mode, **kwargs)
        result['history_known'] = dict(policy=a.known_policy,
            fractions=[float(np.asarray(row[1]['history_observed']).mean()) for row in rows],
            corruption=False, learned_prior_gate=False)
        return result
    def summary(result):
        text = ['===== HISTORY KNOWN / MATCHED GT BODY =====',
            f'status={result["status"]}; policy={a.known_policy}; epochs={result.get("epoch", 0)}; updates={result.get("updates", 0)}',
            'RANDOM; same TRAIN4086 x2 / order / B4-source128 / H6 / two-epoch cosine.',
            'GT future ego conditioned; ONLY full6 CE+Lovasz; no ego/motion/existence auxiliary loss.',
            'Only learned history known/coverage changes. Geometry/KTA/Strong/Transport math unchanged.',
            'STC full semantics/dense validity; scoring full grid, NO visibility mask or alignment.',
            'No noise augmentation, learned prior gate, new feature bank or old checkpoint/cache writes.']
        if 'performance' in result: text.append('PERFORMANCE '+legacy.json.dumps(result['performance']))
        for name, m in result.get('evaluation', {}).get('variants', {}).get('gt', {}).items():
            text.append(f'{name} mIoU={m["mIoU"]:.6f} IoU={m["IoU"]:.6f} MovingMicro={m["MovingMicro"]:.6f}')
            for h, v in m.get('per_horizon', {}).items():
                text.append(f'  {h}s mIoU={v["mIoU"]:.6f} IoU={v["IoU"]:.6f} MovingMicro={v["MovingMicro"]:.6f}')
        return '\n'.join(text)+'\n'
    initializer = {}
    def model(config):
        value = GTBody(config)
        initializer['sha256'] = tensor_state_fingerprint(value.state_dict())
        return value
    def save(path, payload):
        payload = dict(payload, history_known_policy=a.known_policy, random_initializer=initializer['sha256'])
        original['save_resume_checkpoint'](path, payload)
        if Path(path).name == 'last.pt' and payload['epoch'] > 0 and payload['batch'] == 0:
            target = Path(path).with_name(f'epoch_{payload["epoch"]:04d}.pt')
            if not target.exists(): original['save_resume_checkpoint'](target, payload)
    try:
        legacy.parser = lambda: p
        legacy.train_arm, legacy.evaluate = train, evaluator(a.stc_root, a.known_policy, inventory=inventory)
        legacy.JointOccEgo, legacy.HistoryInputs = model, history_inputs(a.known_policy)
        legacy.BothActionsExecution, legacy.PROTOCOL = NoLegacyHeadGraph, PROTOCOL
        legacy.summarize, legacy.stable_json_fingerprint, legacy.save_resume_checkpoint = summary, fingerprint, save
        legacy.prefetched_batches = prefetch_adapter(original['prefetched_batches'], a.known_policy)
        data.ego_inputs = gt.history_inputs
        lock = out.parent/'.known_locks'/hashlib.sha256(str(out).encode()).hexdigest()
        lock.mkdir(parents=True, exist_ok=True)
        with evaluation_lock(lock): return legacy.main(event, argv)
    finally:
        data.ego_inputs = old_ego
        for name, value in original.items(): setattr(legacy, name, value)


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
