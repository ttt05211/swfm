"""RANDOM joint500; evaluate only INTERNAL ego on the complete fixed DEV512."""
import hashlib
from pathlib import Path
import signal
import sys
import threading
import torch
from tools.joint_occ_experiments import run as legacy
from tools.shared_history_experiments.run import NoLegacyHeadGraph
from tools.shared_history_experiments.refine.history import ResidualHistoryInputs
from tools.shared_history_experiments.refine import engine as quality
from tools.shared_history_experiments.loss_controls.run import capped_groups
from tools.scene_ego_head.integration import SceneEgoRefiner, forward_pred_window
from tools.scene_ego_head.experiment.engine import train_arm
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock

PROTOCOL = 'shared_scene_ego_random_joint500_internal_only_v1'
EVAL_PROTOCOL = 'shared_scene_ego_internal_only_full_grid_eval_v1'


@torch.no_grad()
def evaluate(models, provider, source, batches, **kwargs):
    if set(models) != {'pred'}: raise ValueError('internal Pred model required')
    original_forward, original_protocol = quality.forward_window, quality.EVAL_PROTOCOL
    def forward(model, provider, row):
        value = forward_pred_window(model, provider, row)
        return {**value, 'ego': value['ego']['se2'][0], 'ego_target': row[2][2]}
    try:
        quality.forward_window, quality.EVAL_PROTOCOL = forward, EVAL_PROTOCOL
        # Reuse only integer metrics/recovery; the prediction callable is INTERNAL.
        result = quality.evaluate({'gt': models['pred']}, provider, source, batches, **kwargs)
    finally:
        quality.forward_window, quality.EVAL_PROTOCOL = original_forward, original_protocol
    result['variants'] = {'internal': result['variants']['gt']}
    result['trajectory'] = {'internal': result['trajectory']['gt']} if result['trajectory'] else {}
    result.update(scoring='full grid, INTERNAL predicted ego; NO GT pose alignment',
                  evaluation_routes=['internal'], planner_used=False)
    return result


def summarize(result):
    lines = ['===== SCENE EGO / RANDOM JOINT500 / INTERNAL ONLY =====', 'status='+result['status'],
        f"updates={result.get('updates', 0)}/500; complete_dataset_epochs=0",
        'Entire Shared Encoder/STWM/Refiner/scene ego RANDOM jointly optimized; original H6.',
        'ONLY full6 CE+Lovasz+ego R10; motion/existence auxiliary losses OFF.',
        'Train AND eval: INTERNAL predicted ego everywhere; NO external planner or GT-ego reference inference.',
        'GT-derived destination navigation-conditioned; NOT navigation-free; no GT pose alignment.',
        f"scene_ego_parameters={result['contract']['model']['ego']['parameters']}",
        'Original GT models/trainer/checkpoints/history caches unchanged; no new learned bank.']
    if 'performance' in result: lines.append('PERFORMANCE '+legacy.json.dumps(result['performance']))
    report = result.get('evaluation')
    if report:
        lines += [f"===== FIXED {result['contract']['eval_population']} / {report['windows']} WINDOWS =====",
                  'INTERNAL ONLY; population NOT trimmed by planner coverage.']
        for name, m in report['variants']['internal'].items():
            lines.append(f"internal_{name} mIoU={m['mIoU']:.6f} IoU={m['IoU']:.6f} MovingMicro={m['MovingMicro']:.6f}")
            for h, row in m['per_horizon'].items():
                lines.append(f"  {h}s mIoU={row['mIoU']:.6f} IoU={row['IoU']:.6f} MovingMicro={row['MovingMicro']:.6f}")
        lines.append('ego_errors='+legacy.json.dumps(report['trajectory']))
    if 'error' in result: lines.append('error='+result['error'])
    lines.append('500-step partial-pass pilot; final step only; no automatic extension or inherited FPS.')
    return '\n'.join(lines)+'\n'


def main(stop_event=None, argv=None):
    names = ('parser', 'train_arm', 'evaluate', 'JointOccEgo', 'HistoryInputs', 'BothActionsExecution',
             'PROTOCOL', 'summarize', 'stable_json_fingerprint', 'epoch_groups')
    original = {name: getattr(legacy, name) for name in names}
    def parser():
        p = original['parser'](); p.description = __doc__
        p.set_defaults(mode='pred', epochs=1, train_fraction=.2)
        return p
    a = parser().parse_args(argv)
    if (a.mode != 'pred' or a.epochs != 1 or a.train_fraction != .2 or a.window_batch_size != 4
            or a.source_budget != 128 or a.seed != 20261011 or a.training_backend != 'fused'
            or a.eval_population != 'dev512' or a.max_updates > 500
            or (a.object_weight, a.ego_weight, a.lovasz_weight) != (1., 1., 1.)):
        raise ValueError('fixed RANDOM joint500 internal Pred/full6 loss/DEV512, original B4/source128/seed')
    if a.resume and not a.evaluate_only and Path(a.resume).resolve().parent != Path(a.out_dir).resolve():
        raise ValueError('resume SAME internal-only output; no silent import from old experiments')
    if a.evaluate_only and (not a.resume or Path(a.resume).resolve().parent == Path(a.out_dir).resolve()):
        raise ValueError('read-only evaluation requires own checkpoint and separate output')
    class Joint(SceneEgoRefiner):
        def configs(self):
            return {**super().configs(), 'experiment': dict(protocol=PROTOCOL, joint_RANDOM=True,
                motion_aux=False, existence_aux=False, fixed_updates=500, evaluation_routes=['internal'], planner_used=False)}
    root = Path(__file__).resolve().parents[4]
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += list((root/'tools/scene_ego_head').glob('*.py'))
    paths += list((root/'tools/shared_history_experiments/refine').glob('*.py'))
    paths += [root/p for p in ('tools/scene_ego_head/experiment/engine.py',
        'tools/shared_history_experiments/model.py', 'tools/shared_history_experiments/history.py',
        'tools/shared_history_experiments/loss_controls/run.py', 'tools/joint_occ_experiments/batching/geometry.py',
        'tools/joint_occ_experiments/fast_autograd.py', 'tools/joint_occ_experiments/soft_geometry.py',
        'real_motion/ego_trajectory_head.py', 'tools/real_motion/benchmark_p0_f9_v18_runtime.py')]
    def fingerprint(value):
        base = original['stable_json_fingerprint'](value)
        if isinstance(value, dict) and 'tools/joint_occ_experiments/run.py' in value:
            return original['stable_json_fingerprint'](dict(base=base, internal_only={str(p.relative_to(root)): legacy.sha256(p) for p in paths}))
        return base
    try:
        legacy.parser, legacy.PROTOCOL = parser, PROTOCOL
        legacy.JointOccEgo, legacy.HistoryInputs, legacy.BothActionsExecution = Joint, ResidualHistoryInputs, NoLegacyHeadGraph
        legacy.train_arm, legacy.evaluate, legacy.summarize = train_arm, evaluate, summarize
        legacy.epoch_groups = lambda *args: capped_groups(original['epoch_groups'], *args)
        legacy.stable_json_fingerprint = fingerprint
        output = Path(a.out_dir).resolve()
        lock = output.parent/'.scene_ego_locks'/hashlib.sha256(str(output).encode()).hexdigest()
        lock.mkdir(parents=True, exist_ok=True)
        print('SCENE_EGO_INTERNAL_ONLY: RANDOM joint500; full fixed DEV512; no planner files or GT-reference forecasting.', flush=True)
        with evaluation_lock(lock): return legacy.main(stop_event, argv)
    finally:
        for name, value in original.items(): setattr(legacy, name, value)


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
