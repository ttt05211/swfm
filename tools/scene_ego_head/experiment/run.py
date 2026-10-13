"""One RANDOM complete-model 500-step Pred pilot, then same-model planner comparison."""
import hashlib
from pathlib import Path
import signal
import sys
import threading
from tools.joint_occ_experiments import run as legacy
from tools.shared_history_experiments.run import NoLegacyHeadGraph
from tools.shared_history_experiments.refine.history import ResidualHistoryInputs
from tools.shared_history_experiments.loss_controls.run import capped_groups
from tools.scene_ego_head.integration import SceneEgoRefiner
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from .planner import PlannerCache
from . import engine

PROTOCOL = 'shared_scene_ego_random_joint500_ce_lovasz_ego_v1'


def summarize(result):
    c = result['contract']
    lines = ['===== SCENE EGO / RANDOM JOINT500 / SAME-MODEL PLANNER =====',
        'status='+result['status'], f"updates={result.get('updates', 0)}/500; complete_dataset_epochs=0",
        'Entire Shared Encoder/STWM/Refiner/scene ego RANDOM and jointly optimized; NO old weights.',
        'FOUR histories -> SIX futures; original H6; ONLY CE+Lovasz+ego R10; motion/existence aux OFF.',
        'Train: INTERNAL ego everywhere; hard Strong/CLEAR background rebuilt with detached predicted ego.',
        'Continuous prior/memory carry occupancy gradients into ego and shared F; no alignment.',
        'GT-derived destination navigation-conditioned; NOT navigation-free. No camera/STC evaluation.',
        f"scene_ego_parameters={c['model']['ego']['parameters']}; one 500-step partial TRAIN4086 pass.",
        'Evaluation changes ONLY ego condition of this one trained model; no separate external-planner model trained.',
        'GT reference is conditioned intervention, NOT a separately trained GT-model result.',
        'History caches read only; no learned bank, threshold tuning, best-head selection or automatic extension.']
    if 'performance' in result: lines.append('PERFORMANCE '+legacy.json.dumps(result['performance']))
    report = result.get('evaluation')
    if report:
        p = report['population']
        lines += [f"requested={p['requested']}; same_population_windows={report['windows']}; missing_planner={len(p['missing_keys'])}",
                  'route/readout                 mIoU       IoU MovingMicro']
        for route, variants in report['variants'].items():
            for name, m in variants.items():
                lines.append(f"{route+'_'+name:28s} {m['mIoU']:.6f} {m['IoU']:.6f} {m['MovingMicro']:.6f}")
                for h, row in m['per_horizon'].items():
                    lines.append(f"  {h}s mIoU={row['mIoU']:.6f} IoU={row['IoU']:.6f} MovingMicro={row['MovingMicro']:.6f}")
        lines += ['internal_minus_external='+legacy.json.dumps(report['same_model_internal_minus_external']),
                  'trajectory_errors='+legacy.json.dumps(report['trajectory'])]
    if 'error' in result: lines.append('error='+result['error'])
    lines.append('Pilot quality evaluation, NOT converged accuracy or inherited FPS. Original runs/checkpoints unchanged.')
    return '\n'.join(lines)+'\n'


def main(stop_event=None, argv=None):
    names = ('parser', 'train_arm', 'evaluate', 'JointOccEgo', 'HistoryInputs', 'BothActionsExecution',
             'PROTOCOL', 'summarize', 'stable_json_fingerprint', 'epoch_groups', 'load_manifest')
    original = {name: getattr(legacy, name) for name in names}
    def parser():
        p = original['parser'](); p.description = __doc__
        p.set_defaults(mode='pred', epochs=1, train_fraction=.2)
        p.add_argument('--plan-cache', required=True, help='exact existing v4 external BEV-Planner+yaw cache')
        return p
    a = parser().parse_args(argv)
    if (a.mode != 'pred' or a.epochs != 1 or a.train_fraction != .2 or a.window_batch_size != 4
            or a.source_budget != 128 or a.seed != 20261011 or a.training_backend != 'fused'
            or a.eval_population not in ('dev64', 'dev512') or a.max_updates > 500
            or (a.object_weight, a.ego_weight, a.lovasz_weight) != (1., 1., 1.)):
        raise ValueError('fixed RANDOM joint500 Pred/full6 CE+Lovasz+ego, original B4/source128/seed')
    if a.resume and not a.evaluate_only and Path(a.resume).resolve().parent != Path(a.out_dir).resolve():
        raise ValueError('resume SAME scene-ego output only; no old model/Adam import')
    if a.evaluate_only and (not a.resume or Path(a.resume).resolve().parent == Path(a.out_dir).resolve()):
        raise ValueError('read-only evaluation needs own checkpoint and separate output')
    planner = PlannerCache(a.plan_cache)
    def manifest(path):
        result = original['load_manifest'](path)
        keys = result[1] if a.eval_population == 'dev64' else result[0]['parent_keys']
        planner.prepare(keys)
        print(f"PLANNER_COVERAGE requested={len(keys)} common={planner.metadata['covered']} missing={len(planner.metadata['missing_keys'])}; ALL routes same intersection", flush=True)
        return result
    class Joint(SceneEgoRefiner):
        def configs(self):
            return {**super().configs(), 'experiment': dict(protocol=PROTOCOL, joint_RANDOM=True,
                motion_aux=False, existence_aux=False, fixed_updates=500, planner=planner.metadata)}
    root = Path(__file__).resolve().parents[3]
    paths = list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.parent.glob('*.py'))
    paths += list((root/'tools/shared_history_experiments/refine').glob('*.py'))
    paths += [root/p for p in ('tools/shared_history_experiments/model.py', 'tools/shared_history_experiments/history.py',
        'tools/shared_history_experiments/loss_controls/run.py', 'tools/joint_occ_experiments/batching/geometry.py',
        'tools/joint_occ_experiments/fast_autograd.py', 'tools/joint_occ_experiments/soft_geometry.py',
        'real_motion/stc_camera_protocol.py', 'real_motion/ego_trajectory_head.py',
        'tools/real_motion/benchmark_p0_f9_v18_runtime.py')]
    def fingerprint(value):
        base = original['stable_json_fingerprint'](value)
        if isinstance(value, dict) and 'tools/joint_occ_experiments/run.py' in value:
            return original['stable_json_fingerprint'](dict(base=base, scene_ego={str(p.relative_to(root)): legacy.sha256(p) for p in paths}))
        return base
    try:
        legacy.parser, legacy.PROTOCOL = parser, PROTOCOL
        legacy.JointOccEgo, legacy.HistoryInputs, legacy.BothActionsExecution = Joint, ResidualHistoryInputs, NoLegacyHeadGraph
        legacy.train_arm = engine.train_arm
        legacy.evaluate = lambda *args, **kwargs: engine.evaluate(*args, planner=planner, **kwargs)
        legacy.epoch_groups = lambda *args: capped_groups(original['epoch_groups'], *args)
        legacy.load_manifest, legacy.summarize, legacy.stable_json_fingerprint = manifest, summarize, fingerprint
        output = Path(a.out_dir).resolve()
        lock = output.parent/'.scene_ego_locks'/hashlib.sha256(str(output).encode()).hexdigest()
        lock.mkdir(parents=True, exist_ok=True)
        print('SCENE_EGO: whole model RANDOM joint500; internal Pred full6 CE+Lovasz+ego; old models untouched.', flush=True)
        with evaluation_lock(lock): return legacy.main(stop_event, argv)
    finally:
        for name, value in original.items(): setattr(legacy, name, value)


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
