"""Frozen epoch2 only: DEV512 edit audit and 32 x 3 fresh six-frame timings."""
import json
from pathlib import Path
import signal
import sys
import threading
import numpy as np
import torch
from tools.joint_occ_experiments import run as legacy
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from ..refine import run as training, engine as original_engine
from ..refine.model import PROTOCOL
from . import engine


def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def report(out, result, speed_windows, speed_repeats, reference):
    evaluation = result.get('evaluation')
    lines = ['===== FROZEN SEMANTIC REFINER / EDITS + FRESH SIX-FRAME TIMING =====',
        f'status={result["status"]}; scientific_updates=0; fixed epoch2; no method/threshold selection.']
    if evaluation:
        variants = evaluation['variants']['gt']
        metrics_match = reference is not None and all(
            variants[n][k] == reference['evaluation']['variants']['gt'][n][k]
            for n in variants for k in ('mIoU', 'IoU', 'MovingMicro'))
        lines += [f'windows={evaluation["windows"]}; source_epoch2_quality_reproduced={metrics_match}',
            'route                 mIoU       IoU MovingMicro']
        for name, values in variants.items():
            lines.append(f'{name:20s} {values["mIoU"]:9.6f} {values["IoU"]:9.6f} {values["MovingMicro"]:9.6f}')
        totals = {}
        for h in ('1.0', '2.0', '3.0'): engine.add_counts(totals, evaluation['edits'][h])
        lines.append('EDITS cumulative 1/2/3s, disjoint added/removed/relabelled; ignore invalid GT:')
        for group, values in totals.items():
            lines.append(group+' '+json.dumps(values))
        speed = engine.speed_summary(evaluation['speed'], speed_windows, speed_repeats,
            result['contract']['execution']['device_type'] == 'cuda')
        speed['source_epoch2_quality_reproduced'] = metrics_match
        if not metrics_match:
            speed['valid_same_quality_route_timing'] = False
            speed['window_fps'] = speed['output_frame_fps'] = None
        lines.append('SPEED '+json.dumps(speed))
        legacy.write_json(out/'speed.json', speed)
        legacy.write_json(out/'edit_audit.json', dict(windows=evaluation['windows'],
            all_six_horizons=evaluation['edits'], cumulative_1_2_3s=totals,
            variants=variants, source_epoch2_quality_reproduced=metrics_match))
    else:
        lines.append('Incomplete evaluation: completed whole windows saved; resume SAME diagnostic directory.')
    lines += ['Same old prepared-history timing boundary; NEW learned Encoder/memory/Refiner included.',
        'Initial history staging outside, projection-internal materialization INSIDE.',
        'Fresh Strong/KTA and ALL SIX CPU output frames INSIDE; quality cache never reused for timing.',
        '32x3 is WARM diagnostic latency, NOT old-model 54.945 FPS or raw-camera end-to-end FPS.',
        'No optimizer steps, training/cache/source checkpoint writes, alignment, mask or threshold search.']
    text = '\n'.join(lines)+'\n'; (out/'summary.txt').write_text(text, encoding='utf-8'); print(text, flush=True)


def main(event=None, argv=None):
    event = event or threading.Event()
    p = legacy.parser(); p.set_defaults(mode='gt', epochs=2, eval_population='dev512')
    p.add_argument('--diagnostic-resume', action='store_true')
    p.add_argument('--speed-windows', type=int, default=32)
    p.add_argument('--speed-repeats', type=int, default=3)
    a = p.parse_args(argv)
    if (not a.resume or a.mode != 'gt' or a.epochs != 2 or a.eval_population != 'dev512' or a.max_updates
            or not 1 <= a.speed_windows <= 64 or not 2 <= a.speed_repeats <= 4):
        raise ValueError('explicit frozen epoch2 source, fixed DEV512, 1..64 speed windows / 2..4 repeats')
    out, source = Path(a.out_dir).resolve(), Path(a.resume).resolve()
    if out == source.parent or source.parent in out.parents:
        raise ValueError('NEW diagnostics directory OUTSIDE original training directory')
    lock = out.parent/'.shared_pilot_locks'/('refiner_audit_'+legacy.stable_json_fingerprint(str(out)))
    lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(lock):
        if a.diagnostic_resume:
            if not out.is_dir(): raise RuntimeError('existing diagnostic output required')
        else:
            out.mkdir(parents=True, exist_ok=False); (out/'frozen').mkdir()
            snapshot_checkpoint(source, out/'frozen/epoch2.pt')
        snapshot = out/'frozen/epoch2.pt'
        saved = torch.load(snapshot, map_location='cpu', weights_only=False)
        if (saved.get('protocol') != PROTOCOL or saved['epoch'] != 2 or saved['batch'] != 0
                or saved['updates'] != saved['contract']['total_updates'] or saved['contract']['modes'] != ['gt']):
            raise RuntimeError('completed TWO-epoch GT Refiner ONLY; no legacy or selected-best checkpoint')
        keys = [tuple(k) for k in saved['contract']['eval_keys']]
        if len(keys) != 512 or len(set(keys)) != 512: raise RuntimeError('exact frozen DEV512 required')
        speed_keys = set(legacy.select_scene_balanced_round_robin(keys, a.speed_windows))
        fields = vars(a).copy()
        for key in ('diagnostic_resume', 'evaluate_only'): fields.pop(key)
        repo = Path(__file__).resolve().parents[3]
        dependencies = ('tools/joint_occ_experiments/soft_inference/engine.py',
            'real_motion/final_dataflow.py', 'tools/real_motion/benchmark_p0_f9_v18_runtime.py',
            'real_motion/runtime_fastpath.py', 'real_motion/strong_majority_execution.py')
        identity = dict(protocol=engine.PROTOCOL, args=fields, checkpoint_sha256=legacy.sha256(snapshot),
            code={f.name: legacy.sha256(f) for f in Path(__file__).parent.glob('*.py')},
            dependencies={name: legacy.sha256(repo/name) for name in dependencies},
            runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda, numpy=np.__version__),
            speed_keys=[list(k) for k in sorted(speed_keys)])
        identity = json.loads(json.dumps(identity))
        path = out/'diagnostic_contract.json'
        if a.diagnostic_resume:
            if load(path) != identity: raise RuntimeError('diagnostic source/population/implementation/settings differ')
        else: legacy.write_json(path, identity)
        reference = load(source.parent/'results.json') if (source.parent/'results.json').is_file() else None
        if reference and (reference.get('updates') != saved['updates'] or reference.get('epoch') != 2
                or reference.get('contract') != saved['contract'] or reference.get('evaluation', {}).get('windows') != 512):
            raise RuntimeError('source epoch2 quality report differs')
        inner = vars(a).copy()
        for name in ('diagnostic_resume', 'speed_windows', 'speed_repeats'): inner.pop(name)
        inner.update(resume=str(snapshot), evaluate_only=True)
        args = []
        for name, value in inner.items():
            if name == 'override' or value is None: continue
            if isinstance(value, bool):
                if value: args += ['--'+name.replace('_', '-')]
            else: args += ['--'+name.replace('_', '-'), str(value)]
        for value in a.override or []: args += ['--override', value]
        original = original_engine.evaluate
        original_engine.evaluate = lambda *pos, **kw: engine.evaluate(*pos, **kw,
            speed_keys=speed_keys, speed_repeats=a.speed_repeats)
        try:
            status = training.main(event, args)
        finally: original_engine.evaluate = original
        if legacy.sha256(snapshot) != identity['checkpoint_sha256']: raise RuntimeError('frozen snapshot changed')
        report(out, load(out/'results.json'), a.speed_windows, a.speed_repeats, reference)
        return status


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
