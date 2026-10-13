"""Full TRAIN20430 x2 RANDOM GT body; CE+Lovasz only; legacy paths unchanged."""
import hashlib
from pathlib import Path
import signal
import sys
import threading
from tools.joint_occ_experiments import run as legacy
from tools.shared_history_experiments.run import NoLegacyHeadGraph
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from .model import ResidualJoint, PROTOCOL
from tools.shared_history_experiments.refine.history import ResidualHistoryInputs
from . import engine
from tools.joint_occ_experiments import data


def main(stop_event=None, argv=None):
    names = ('parser', 'train_arm', 'evaluate', 'JointOccEgo', 'HistoryInputs', 'BothActionsExecution',
             'PROTOCOL', 'summarize', 'stable_json_fingerprint', 'save_resume_checkpoint')
    original = {name: getattr(legacy, name) for name in names}
    original_inputs = data.ego_inputs
    def parser():
        p = original['parser'](); p.description = __doc__
        p.set_defaults(mode='gt', epochs=2, train_fraction=1.)
        return p
    a = parser().parse_args(argv)
    if (a.mode != 'gt' or a.training_backend != 'fused' or a.train_fraction != 1. or a.epochs != 2
            or a.window_batch_size != 4 or a.source_budget != 128 or a.seed != 20261011):
        raise ValueError('GT / full TRAIN / fixed 2 epochs / original batch4-source128 / original seed')
    if a.resume and not a.evaluate_only and Path(a.resume).resolve().parent != Path(a.out_dir).resolve():
        raise ValueError('resume SAME new refiner directory; no old weight/Adam import')
    if a.evaluate_only and (not a.resume or Path(a.resume).resolve().parent == Path(a.out_dir).resolve()):
        raise ValueError('read-only evaluation requires own checkpoint and a separate output directory')
    if a.eval_population != 'dev512' or (a.object_weight, a.ego_weight, a.lovasz_weight) != (1., 1., 1.):
        raise ValueError('fixed DEV512, CE/Lovasz weight1; legacy aux weights are unused compatibility placeholders')
    root = Path(__file__).resolve().parents[3]
    dependencies = list(Path(__file__).parent.glob('*.py')) + [root/p for p in (
        'tools/shared_history_experiments/model.py', 'tools/shared_history_experiments/history.py',
        'tools/shared_history_experiments/refine/model.py', 'tools/shared_history_experiments/refine/geometry.py',
        'tools/shared_history_experiments/refine/history.py',
        'tools/joint_occ_experiments/soft_geometry.py', 'tools/joint_occ_experiments/fast_autograd.py',
        'tools/joint_occ_experiments/batching/geometry.py', 'real_motion/local_st_world_model.py')]
    def fingerprint(value):
        result = original['stable_json_fingerprint'](value)
        if isinstance(value, dict) and 'tools/joint_occ_experiments/run.py' in value:
            return original['stable_json_fingerprint'](dict(base=result, variant='semantic_residual_field',
                prototype={str(p.relative_to(root)): legacy.sha256(p) for p in dependencies}))
        return result
    def summary(result):
        text = original['summarize'](result)
        text = text.replace('===== JOINT OCCUPANCY / INTERNAL EGO =====', '===== FULL TRAIN GT / CE + LOVASZ =====')
        text = text.replace('GT uses GT future poses; Pred uses INTERNAL predicted poses everywhere (no alignment).',
            'GT future ego conditioned; no ego prediction head or ego loss.')
        text = text.replace('Navigation-conditioned, GT-derived destination commands; NOT navigation-free.',
            'No navigation command consumed by model; external planner is a later inference setting.')
        text = text.replace('Final full-grid CE+Lovasz + direct source geometry/existence + direct R10 ego geometry.',
            'ONLY complete-six full-grid CE+Lovasz; no direct motion/existence/ego loss.')
        text = text.replace('TRAIN uses a continuous surrogate; INFERENCE uses hard Strong/A1/CCR; soft != hard.',
            'TRAIN and EVAL use SAME continuous Transport prior + 18-class logit residual; final argmax.')
        text = text.replace('ADD=0.5; REMOVE=0.95; ADD-only and ADD+REMOVE both reported; no threshold search.',
            'No action heads/thresholds, canonical whitelist/halo, Surface descriptors or owner/fallback repair.')
        text = text.replace('This is a 3-epoch PILOT, not a converged paper result or verification of legacy FPS.',
            f'Fixed full TRAIN20430 x2 RANDOM; final epoch2, no DEV-best selection or inherited FPS.')
        return text + ('Full six-horizon CE/Lovasz only; source motion remains learned through occupancy gradients.\n'
            'Stable log-softmax CE; finite smoothed Transport log-prior, epsilon=1e-4.\n'
            'Transport source overlay order remains; feature memory uses normalized learned evidence pooling.\n'
            'History geometry caches READ ONLY; no Surface fit/old feature materialization/new feature bank.\n'
            'Zero last layer preserves INITIAL Transport argmax only, NOT later quality or raw probability bytes.\n')
    def save(path, payload):
        original['save_resume_checkpoint'](path, payload)
        if Path(path).name == 'last.pt' and payload['epoch'] > 0 and payload['batch'] == 0:
            epoch_path = Path(path).with_name(f'epoch_{payload["epoch"]:04d}.pt')
            if not epoch_path.exists(): original['save_resume_checkpoint'](epoch_path, payload)
    try:
        legacy.parser, legacy.train_arm, legacy.evaluate = parser, engine.train_arm, engine.evaluate
        legacy.JointOccEgo, legacy.HistoryInputs = ResidualJoint, ResidualHistoryInputs
        legacy.BothActionsExecution, legacy.PROTOCOL = NoLegacyHeadGraph, PROTOCOL
        legacy.summarize, legacy.stable_json_fingerprint, legacy.save_resume_checkpoint = summary, fingerprint, save
        data.ego_inputs = engine.history_inputs
        output = Path(a.out_dir).resolve()
        lock = output.parent/'.shared_pilot_locks'/hashlib.sha256(str(output).encode()).hexdigest()
        lock.mkdir(parents=True, exist_ok=True)
        print('FULL_GT: new RANDOM TRAIN20430 x2 / original H6 / CE+Lovasz ONLY / no old weights or Adam.', flush=True)
        with evaluation_lock(lock): return legacy.main(stop_event, argv)
    finally:
        data.ego_inputs = original_inputs
        for name, value in original.items(): setattr(legacy, name, value)


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
