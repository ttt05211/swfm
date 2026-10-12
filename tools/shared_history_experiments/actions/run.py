"""Independent RANDOM GT / TRAIN20% / two epochs / continuous ADD+REMOVE."""
import hashlib
from pathlib import Path
import signal
import sys
import threading
from tools.joint_occ_experiments import run as legacy
from tools.shared_history_experiments.history import SharedHistoryInputs
from tools.shared_history_experiments.run import NoLegacyHeadGraph
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from .model import SharedJointBoth, PROTOCOL
from . import engine


def main(stop_event=None, argv=None):
    names = ('parser', 'train_arm', 'evaluate', 'JointOccEgo', 'HistoryInputs', 'BothActionsExecution',
             'PROTOCOL', 'summarize', 'stable_json_fingerprint', 'save_resume_checkpoint')
    original = {name: getattr(legacy, name) for name in names}

    def parser():
        p = original['parser'](); p.description = __doc__
        p.set_defaults(mode='gt', epochs=2, train_fraction=.2)
        for action in p._actions:
            if action.dest == 'mode': action.choices = ('gt',)
            if action.dest == 'training_backend': action.choices = ('fused',)
            if action.dest == 'remove_threshold':
                action.help = 'compatibility metadata only; NOT used by continuous ADD/REMOVE'
        return p

    a = parser().parse_args(argv)
    if (a.mode != 'gt' or a.training_backend != 'fused' or a.train_fraction != .2 or a.epochs not in (1, 2)
            or a.window_batch_size != 4 or a.source_budget != 128 or a.seed != 20261011):
        raise ValueError('dual-action pilot: GT / TRAIN20% / 1 or 2 epochs / batch4-source128 / original seed')
    if a.resume and not a.evaluate_only and Path(a.resume).resolve().parent != Path(a.out_dir).resolve():
        raise ValueError('resume SAME dual-action directory; no ADD-only checkpoint import')
    if a.evaluate_only and (not a.resume or Path(a.resume).resolve().parent == Path(a.out_dir).resolve()):
        raise ValueError('evaluate-only requires a dual-action checkpoint and a separate output directory')

    root = Path(__file__).resolve().parents[3]
    dependencies = list(Path(__file__).parent.glob('*.py')) + list((root/'tools/shared_history_experiments').glob('*.py')) + [
        root/'tools/shared_history_experiments/speed/engine.py',
        root/'tools/joint_occ_experiments/batching/geometry.py',
        root/'tools/joint_occ_experiments/soft_inference/fast.py',
        root/'tools/joint_occ_experiments/soft_inference/engine.py',
        root/'real_motion/canonical_causal_repair.py', root/'real_motion/local_st_world_model.py']

    def fingerprint(value):
        result = original['stable_json_fingerprint'](value)
        if isinstance(value, dict) and 'tools/joint_occ_experiments/run.py' in value:
            return original['stable_json_fingerprint'](dict(base=result, variant='shared_add_remove',
                prototype={str(p.relative_to(root)): legacy.sha256(p) for p in dependencies}))
        return result

    def summary(result):
        text = original['summarize'](result)
        text = text.replace('===== JOINT OCCUPANCY / INTERNAL EGO =====', '===== SHARED HISTORY GT / SOFT ADD+REMOVE PILOT =====')
        text = text.replace('GT uses GT future poses; Pred uses INTERNAL predicted poses everywhere (no alignment).',
            'GT future ego conditioned; ego head auxiliary only, no scene-head upgrade.')
        text = text.replace('Navigation-conditioned, GT-derived destination commands; NOT navigation-free.',
            'GT-derived navigation only enters the auxiliary ego head, not STWM/CCR scene inputs.')
        text = text.replace('TRAIN uses a continuous surrogate; INFERENCE uses hard Strong/A1/CCR; soft != hard.',
            'TRAIN and EVAL use continuous ADD+REMOVE; final argmax only at inference.')
        text = text.replace('ADD=0.5; REMOVE=0.95; ADD-only and ADD+REMOVE both reported; no threshold search.',
            'No hard action thresholds; fixed main=soft ADD+REMOVE, ADD-only is an ablation.')
        text = text.replace('This is a 3-epoch PILOT, not a converged paper result or verification of legacy FPS.',
            f'This is a fixed {a.epochs}-epoch TRAIN20% pilot, not a converged result or formal FPS.')
        return text + ('Ownership-constrained REMOVE transfers class mass to fallback; not arbitrary relabeling or a second-pass ADD undo.\n'
            'New RANDOM model/protocol; no old Adam/weights imported. Same selection/seed/budget, NOT byte-identical ADD-only initializer.\n'
            'History geometry reused READ ONLY; no learned bank. Sequential H6 by explicit user float-drift acceptance; original numerical/replay FAIL retained.\n'
            'New REMOVE H6 quality and L40S speed remain unverified; no Packed execution or loss subsampling.\n')

    def save(path, payload):
        payload = dict(payload, execution_policy=engine.execution_policy())
        original['save_resume_checkpoint'](path, payload)
        if Path(path).name == 'last.pt' and payload['epoch'] > 0 and payload['batch'] == 0:
            epoch_path = Path(path).with_name(f"epoch_{payload['epoch']:04d}.pt")
            if not epoch_path.exists(): original['save_resume_checkpoint'](epoch_path, payload)

    try:
        legacy.parser, legacy.train_arm, legacy.evaluate = parser, engine.train_arm, engine.evaluate
        legacy.JointOccEgo, legacy.HistoryInputs = SharedJointBoth, SharedHistoryInputs
        legacy.BothActionsExecution, legacy.PROTOCOL = NoLegacyHeadGraph, PROTOCOL
        legacy.summarize, legacy.stable_json_fingerprint, legacy.save_resume_checkpoint = summary, fingerprint, save
        print(f'SHARED_ACTIONS: RANDOM GT / TRAIN20% x{a.epochs}; continuous ADD+REMOVE; sequential H6; no old resume.', flush=True)
        print('H6_EXECUTION: explicit user float-drift acceptance, original probe/replay FAIL retained.', flush=True)
        output = Path(a.out_dir).resolve()
        lock_dir = output.parent/'.shared_pilot_locks'/hashlib.sha256(str(output).encode()).hexdigest()
        lock_dir.mkdir(parents=True, exist_ok=True)
        with evaluation_lock(lock_dir):
            return legacy.main(stop_event, argv)
    finally:
        for name, value in original.items(): setattr(legacy, name, value)


if __name__ == '__main__':
    event = threading.Event()
    def stop(*_):
        event.set(); print('Stop requested; save after complete update/window.', flush=True)
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, stop)
    sys.exit(main(event) or 0)
