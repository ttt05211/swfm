"""Independent RANDOM GT / TRAIN20% / one or two epochs / soft ADD-only."""
from pathlib import Path
import hashlib
import signal
import sys
import threading
from tools.joint_occ_experiments import run as legacy
from . import engine
from .history import SharedHistoryInputs
from .model import SharedJoint, PROTOCOL
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock


class NoLegacyHeadGraph:
    """The shared head owns a live F/E readout, not the legacy encode signature."""
    def __init__(self, *args, **kwargs): pass
    def stats(self): return dict(graphs=False, route='shared_live_history_eager_fast_soft_composition')
    def close(self): pass


def main(stop_event=None, argv=None):
    names = ('parser', 'train_arm', 'evaluate', 'JointOccEgo', 'HistoryInputs', 'BothActionsExecution',
             'PROTOCOL', 'summarize', 'stable_json_fingerprint', 'save_resume_checkpoint')
    original = {name: getattr(legacy, name) for name in names}
    def parser():
        p = original['parser'](); p.description = __doc__
        p.set_defaults(mode='gt', epochs=2, train_fraction=.2)
        p.add_argument('--variant', choices=('shared', 'legacy_control'), default='shared',
            help='control is a SEPARATE same-budget old architecture, also soft ADD-only in train/eval')
        return p
    a = parser().parse_args(argv)
    if (a.mode != 'gt' or a.training_backend != 'fused' or a.train_fraction != .2 or a.epochs not in (1, 2)
            or a.window_batch_size != 4 or a.source_budget != 128 or a.seed != 20261011):
        raise ValueError('first prototype: GT / TRAIN20% / 1 or 2 epochs / batch4-source128 / original seed')
    if a.resume and not a.evaluate_only and Path(a.resume).resolve().parent != Path(a.out_dir).resolve():
        raise ValueError('resume SAME shared/control output directory; no old checkpoint import')
    if a.evaluate_only and not a.resume:
        raise ValueError('evaluate-only requires this prototype epoch/last checkpoint')
    if a.evaluate_only and Path(a.resume).resolve().parent == Path(a.out_dir).resolve():
        raise ValueError('use a separate eval subdirectory to preserve the training summary')
    root = Path(__file__).resolve().parents[2]
    dependencies = list(Path(__file__).parent.glob('*.py'))+[root/'tools/joint_occ_experiments/batching/geometry.py',
        root/'tools/joint_occ_experiments/soft_inference/fast.py',
        root/'tools/joint_occ_experiments/soft_inference/engine.py',
        root/'real_motion/canonical_causal_repair.py', root/'real_motion/local_st_world_model.py']
    def fingerprint(value):
        result = original['stable_json_fingerprint'](value)
        if isinstance(value, dict) and 'tools/joint_occ_experiments/run.py' in value:
            return original['stable_json_fingerprint'](dict(base=result, variant=a.variant,
                prototype={str(p.relative_to(root)): legacy.sha256(p) for p in dependencies}))
        return result
    def summary(result):
        text = original['summarize'](result)
        text = text.replace('===== JOINT OCCUPANCY / INTERNAL EGO =====', '===== SHARED HISTORY GT / SOFT ADD-ONLY PILOT =====')
        text = text.replace('GT uses GT future poses; Pred uses INTERNAL predicted poses everywhere (no alignment).',
            'GT future ego conditioned; existing ego head is auxiliary only, no scene-head upgrade.')
        text = text.replace('Navigation-conditioned, GT-derived destination commands; NOT navigation-free.',
            'GT-derived navigation only enters the auxiliary ego head, not STWM/CCR scene inputs.')
        text = text.replace('TRAIN uses a continuous surrogate; INFERENCE uses hard Strong/A1/CCR; soft != hard.',
            'TRAIN and EVAL use continuous ADD-only; final argmax only at inference. REMOVE=0 in BOTH.')
        text = text.replace('ADD=0.5; REMOVE=0.95; ADD-only and ADD+REMOVE both reported; no threshold search.',
            'No hard action thresholds, fitted role priors or sampled weighted ADD loss.')
        text = text.replace('This is a 3-epoch PILOT, not a converged paper result or verification of legacy FPS.',
            f'This is a fixed {a.epochs}-epoch TRAIN20% pilot, not a converged result or verification of old FPS.')
        text += f'variant={a.variant}; same TRAIN4086 selection/epoch ordering as existing pilot; RANDOM, no old weights.\n'
        text += ('Shared variant: height-aware F replaces semantic tube stem; point E reads SAME F; Surface12 retained.\n'
                 'CC/source associations/SE(2)/support/Strong/owner remain.\n'
                 'Legacy-control is optional and separately trained, NOT automatically run. Do not compare against old 3/20-epoch models as an equal-budget ablation.\n'
                 'Only compatible history geometry reused READ ONLY; no learned feature bank, auto-extension or promotion.\n')
        text += ('Shared: old density/neighbour/age descriptors NOT calculated or consumed.\n' if a.variant == 'shared'
                 else 'Legacy-control: original feature construction retained; only action policy aligned to soft ADD-only.\n')
        return text
    def save(path, payload):
        original['save_resume_checkpoint'](path, payload)
        if Path(path).name == 'last.pt' and payload['epoch'] > 0 and payload['batch'] == 0:
            epoch_path = Path(path).with_name(f"epoch_{payload['epoch']:04d}.pt")
            if not epoch_path.exists(): original['save_resume_checkpoint'](epoch_path, payload)
    try:
        legacy.parser, legacy.train_arm, legacy.evaluate = parser, engine.train_arm, engine.evaluate
        legacy.PROTOCOL = PROTOCOL+'_'+a.variant
        if a.variant == 'shared': legacy.JointOccEgo, legacy.HistoryInputs = SharedJoint, SharedHistoryInputs
        legacy.BothActionsExecution = NoLegacyHeadGraph
        legacy.summarize, legacy.stable_json_fingerprint, legacy.save_resume_checkpoint = summary, fingerprint, save
        print(f'SHARED_HISTORY variant={a.variant}: RANDOM GT, TRAIN20% x{a.epochs}, strict4->6, '
              'whole-cycle cosine, soft ADD-only in train/eval; original caches READ ONLY.', flush=True)
        # Lock lives outside the new output: fresh-run validation must still
        # reject existing experiment directories; no stale-PID lock deletion.
        output = Path(a.out_dir).resolve()
        identity = hashlib.sha256(str(output).encode()).hexdigest()
        lock_dir = output.parent/'.shared_pilot_locks'/identity
        lock_dir.mkdir(parents=True, exist_ok=True)
        with evaluation_lock(lock_dir):
            return legacy.main(stop_event, argv)
    finally:
        for name, value in original.items(): setattr(legacy, name, value)


if __name__ == '__main__':
    event = threading.Event()
    def stop(*_):
        event.set(); print('Stop requested; saving at completed update/window boundary.', flush=True)
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, stop)
    sys.exit(main(event) or 0)
