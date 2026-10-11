"""TRAIN20% x THREE epochs, independent RANDOM Pred model, fixed DEV512."""
from pathlib import Path
import signal
import threading
from .. import run as legacy
from . import engine


def implementation(root, base):
    files = [*Path(__file__).parent.glob('*.py'),
        root/'tools/joint_occ_experiments/batching/geometry.py',
        root/'tools/joint_occ_experiments/soft_inference/engine.py',
        root/'tools/joint_occ_experiments/soft_inference/fast.py']
    return legacy.stable_json_fingerprint(dict(base=base,
        pred={str(p.relative_to(root)): legacy.sha256(p) for p in files}))


def main(stop_event=None, argv=None):
    parser, step, evaluate, summarize, fingerprint = (
        legacy.parser, legacy.train_arm, legacy.evaluate, legacy.summarize, legacy.stable_json_fingerprint)
    def pred_parser():
        p = parser(); p.description = __doc__; p.set_defaults(mode='pred')
        return p
    a = pred_parser().parse_args(argv)
    if (a.mode != 'pred' or a.training_backend != 'fused' or a.epochs != 3 or a.train_fraction != .2
            or a.window_batch_size != 4 or a.source_budget != 128 or a.eval_population != 'dev512'
            or a.seed != 20261011):
        raise ValueError('keep INTERNAL Pred / TRAIN20% / 3 epochs / batch4 / source128 / DEV512 / original seed')
    if a.resume and Path(a.resume).resolve().parent != Path(a.out_dir).resolve():
        raise ValueError('resume THIS Pred experiment in the SAME directory; no GT checkpoint import')
    root = Path(__file__).resolve().parents[3]
    def hash_contract(value):
        base = fingerprint(value)
        if isinstance(value, dict) and 'tools/joint_occ_experiments/run.py' in value:
            return implementation(root, base)
        return base
    def summary(result):
        text = summarize(result).replace(
            'TRAIN uses a continuous surrogate; INFERENCE uses hard Strong/A1/CCR; soft != hard.',
            'TRAIN: two-horizon continuous composition. EVAL: full DEV512 soft + hard six readouts; main=soft ADD-only.')
        text += ('Independent RANDOM Pred, SAME TRAIN20%/3-epoch seed/order as GT. No GT weights or optimizer imported.\n'
                 'Only history geometry reused; Pred Strong/background/owner/phase rebuilt with INTERNAL ego.\n'
                 'GT future pose is supervision ONLY; GT-derived navigation commands remain declared inputs.\n'
                 'Floating-point two-horizon training, NOT a byte-identical training claim. No new speed probe.\n')
        if result.get('evaluation'):
            text += 'fast Pred six-frame parity: '+str(result['evaluation']['fast_six_frame_byte_parity'])+'\n'
        return text
    try:
        legacy.parser, legacy.train_arm, legacy.evaluate = pred_parser, engine.train_arm, engine.evaluate
        legacy.summarize, legacy.stable_json_fingerprint = summary, hash_contract
        print('INTERNAL PRED: independent RANDOM, TRAIN20% x3, original batch4/source128; '
              'all future geometry predicted; final DEV512 soft/hard, no new bank.', flush=True)
        return legacy.main(stop_event, argv)
    finally:
        legacy.parser, legacy.train_arm, legacy.evaluate = parser, step, evaluate
        legacy.summarize, legacy.stable_json_fingerprint = summarize, fingerprint


if __name__ == '__main__':
    event = threading.Event()
    def stop(*_):
        event.set()
        print('Stop requested: save after completed optimizer update/evaluation window.', flush=True)
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    main(event)
