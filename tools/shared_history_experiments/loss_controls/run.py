"""Matched RANDOM 500-update control, not a complete TRAIN epoch."""
import argparse
import json
from pathlib import Path
import signal
import sys
import threading
import torch
from tools.joint_occ_experiments import run as legacy
from tools.shared_history_experiments.refine import run as refiner

STEPS = 500
ARMS = {'full': (1., 1.), 'no_motion': (0., 1.), 'no_existence': (1., 0.)}


def protocol(arm):
    return 'shared_semantic_refiner_loss500_v1_' + arm


def capped_groups(original, records, seed, epoch, batch, sources):
    if epoch != 0:
        raise ValueError('one fixed 500-update partial pass only')
    groups = original(records, seed, epoch, batch, sources)
    if len(groups) < STEPS:
        raise RuntimeError('insufficient original TRAIN20% groups for 500 updates')
    return groups[:STEPS]


def train_control(original, auxiliary, arm, *args, **kwargs):
    """Drop only the selected direct loss; occupancy gradients stay connected."""
    weights = ARMS[arm]
    raw = []
    def selected(output, record, state=None):
        geometry, existence = auxiliary(output, record, state)
        raw.append((geometry.detach(), existence.detach()))
        # Do not connect a disabled existence head even with a zero multiplier:
        # AdamW must not decay unused head weights through zero-valued gradients.
        zero = output['residual_xy_m'].sum() * 0
        return (geometry if weights[0] else zero,
                existence if weights[1] else zero)
    saved = refiner.engine.source_auxiliary
    try:
        refiner.engine.source_auxiliary = selected
        result = original(*args, **kwargs)
    finally:
        refiner.engine.source_auxiliary = saved
    result.update(loss_arm=arm, motion_weight=weights[0], existence_weight=weights[1],
        raw_source_geometry=float(torch.stack([g for g, _ in raw]).mean()),
        raw_existence=float(torch.stack([e for _, e in raw]).mean()))
    return result


def main(stop_event=None, argv=None):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--loss-arm', choices=tuple(ARMS), required=True)
    control, rest = p.parse_known_args(argv)
    a = legacy.parser().parse_args(rest)
    if (a.epochs != 1 or a.mode != 'gt' or a.eval_population != 'dev512'
            or a.object_weight != 1. or a.ego_weight != 1. or a.lovasz_weight != 1.
            or a.evaluate_only or a.max_updates):
        raise ValueError('fixed 500-update cycle / GT / DEV512 / original unit weights; no early budget override')
    saved = (refiner.PROTOCOL, refiner.ResidualJoint, refiner.engine.train_arm,
             legacy.epoch_groups, legacy.summarize, legacy.stable_json_fingerprint)
    original_auxiliary = refiner.engine.source_auxiliary
    class ControlJoint(saved[1]):
        def configs(self):
            return {**super().configs(), 'loss_control': dict(arm=control.loss_arm,
                motion_weight=ARMS[control.loss_arm][0], existence_weight=ARMS[control.loss_arm][1],
                fixed_updates=STEPS, complete_dataset_epoch=False)}
    def summary(result):
        text = saved[4](result)
        text = text.replace('completed_epochs=', 'completed_500_update_cycles=')
        text = text.replace('Final full-grid CE+Lovasz + direct source geometry/existence + direct R10 ego geometry.',
            f'CE+Lovasz+ego retained; arm={control.loss_arm}, motion/existence weights={ARMS[control.loss_arm]}.')
        return text + '\n500 updates from identical RANDOM initialization; NOT a complete TRAIN epoch.\n'
    root = Path(__file__).resolve().parents[3]
    files = sorted(Path(__file__).parent.glob('*.py'))
    def fingerprint(value):
        base = saved[5](value)
        if isinstance(value, dict) and 'tools/joint_occ_experiments/run.py' in value:
            return saved[5](dict(base=base, loss_control={str(f.relative_to(root)): legacy.sha256(f) for f in files}))
        return base
    try:
        refiner.PROTOCOL, refiner.ResidualJoint = protocol(control.loss_arm), ControlJoint
        refiner.engine.train_arm = lambda *args, **kw: train_control(
            saved[2], original_auxiliary, control.loss_arm, *args, **kw)
        legacy.epoch_groups = lambda *args: capped_groups(saved[3], *args)
        legacy.summarize, legacy.stable_json_fingerprint = summary, fingerprint
        print(f'LOSS500 arm={control.loss_arm}; same RANDOM/500-step cosine/order; '
              'CE+Lovasz+ego retained; final DEV512; partial TRAIN20% pass, NOT one full epoch.', flush=True)
        return refiner.main(stop_event, rest)
    finally:
        (refiner.PROTOCOL, refiner.ResidualJoint, refiner.engine.train_arm,
         legacy.epoch_groups, legacy.summarize, legacy.stable_json_fingerprint) = saved
        path = Path(a.out_dir)/'results.json'
        if path.is_file():
            result = json.loads(path.read_text(encoding='utf-8'))
            if result.get('contract', {}).get('protocol') == protocol(control.loss_arm):
                result['loss_control'] = dict(arm=control.loss_arm, fixed_updates=STEPS,
                    motion_weight=ARMS[control.loss_arm][0], existence_weight=ARMS[control.loss_arm][1],
                    complete_dataset_epochs=0, cursor_epoch_means_partial_update_cycle=True)
                legacy.write_json(path, result)
                text_path = path.with_name('summary.txt')
                if text_path.is_file():
                    text = text_path.read_text(encoding='utf-8')
                    text = text.replace('Fixed TRAIN20% x1 RANDOM pilot;', 'Fixed 500-update RANDOM partial-pass pilot;')
                    text = text.replace('Full six-horizon CE/Lovasz + original source geometry/existence + ego R10.',
                        f'Full six-horizon CE/Lovasz + ego R10; motion/existence={ARMS[control.loss_arm]}.')
                    text_path.write_text(text, encoding='utf-8')


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
