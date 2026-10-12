"""Reuse completed DEV512 edits/quality READ ONLY; remeasure ALL fixed 32 windows."""
import copy
import json
from pathlib import Path
import signal
import sys
import threading
import numpy as np
import torch
from tools.joint_occ_experiments import run as legacy
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from ..refine import run as training, engine as quality
from ..refine.model import PROTOCOL as MODEL_PROTOCOL
from . import engine

PROTOCOL = 'semantic_refiner_reuse_quality_complete_retime_v2'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def completed_audit(directory):
    root = Path(directory).resolve()
    result, contract = read(root/'results.json'), read(root/'diagnostic_contract.json')
    saved = torch.load(root/'frozen/epoch2.pt', map_location='cpu', weights_only=False)
    evaluation = result.get('evaluation', {})
    if (contract.get('protocol') not in ('semantic_refiner_frozen_edits_fresh_fps_v1', engine.PROTOCOL)
            or result.get('status') != 'complete' or evaluation.get('windows') != 512
            or saved.get('protocol') != MODEL_PROTOCOL or saved['epoch'] != 2 or saved['batch'] != 0
            or result['contract'] != saved['contract'] or result['updates'] != saved['updates']
            or saved['updates'] != saved['contract']['total_updates']
            or legacy.sha256(root/'frozen/epoch2.pt') != contract['checkpoint_sha256']):
        raise RuntimeError('completed compatible epoch2 DEV512 audit required')
    if set(evaluation['variants']['gt']) != {'transport', 'semantic_refine'} or not evaluation.get('edits'):
        raise RuntimeError('missing original quality/edit counts')
    state_file = root/f'eval_state_update{saved["updates"]}.json'
    state = read(state_file)
    if (state['update'] != saved['updates'] or state['population'] != legacy.stable_json_fingerprint(saved['contract']['eval_keys'])
            or state['state']['protocol'] != contract['protocol']
            or state['state']['windows'] != 512 or state['state']['edits'] != evaluation['edits']
            or set(state['state']['metrics']) != {'transport', 'semantic_refine'}):
        raise RuntimeError('integer audit provenance differs')
    for name, fields in state['state']['metrics'].items():
        metric = engine.Metrics()
        if set(fields) != set(vars(metric)): raise RuntimeError('metric schema differs')
        for key, values in fields.items():
            values = np.asarray(values)
            if values.shape != getattr(metric, key).shape or np.any(values < 0) or not np.array_equal(values, values.astype('int64')):
                raise RuntimeError('invalid original integer metrics')
            setattr(metric, key, values.astype('int64'))
        computed = metric.compute()
        for key in ('mIoU', 'IoU', 'MovingMicro'):
            a, b = computed[key], evaluation['variants']['gt'][name][key]
            if a != b and not (np.isnan(a) and np.isnan(b)): raise RuntimeError('original quality/counts differ')
    return result, contract, saved


@torch.no_grad()
def evaluate(models, provider, source, batches, *, imported, speed_keys, repeats, saved=None,
             save=None, stop_event=None, progress=None, **_):
    speed = copy.deepcopy(saved['speed']) if saved else {}
    if saved and (saved.get('protocol') != PROTOCOL or not set(speed) <= {'|'.join(k) for k in speed_keys}):
        raise RuntimeError('retime cursor/population differs')
    if any(len(r['seconds']) != repeats or r.get('timing_policy') != engine.TIMING_POLICY for r in speed.values()):
        raise RuntimeError('incomplete retime window or old biased timing')
    def snapshot(): return dict(protocol=PROTOCOL, speed=copy.deepcopy(speed))
    for rows in batches:
        for row in rows:
            key = (row[0]['scene_name'], row[0]['t0_token']); encoded = '|'.join(key)
            if key not in speed_keys: raise RuntimeError('non-selected timing window supplied')
            if encoded in speed: continue
            if stop_event is not None and stop_event.is_set():
                if save: save(snapshot())
                raise InterruptedError('complete retime windows saved')
            fw = quality.forward_window(models['gt'], provider, row)
            expected = {n: v.argmax(1).reshape(6, *fw['shape']).byte().cpu().numpy()
                for n, v in (('transport', fw['prior']), ('semantic_refine', fw['logits']))}
            reference = fw['logits'].float().cpu().numpy(); del fw
            timing = engine.measure(models['gt'], provider, row, expected, repeats, reference_logits=reference)
            speed[encoded] = timing
            if save: save(snapshot())
            print(f'REFINER_RETIME {len(speed)}/{len(speed_keys)} byte_parity={timing["six_frame_byte_parity"]}', flush=True)
            if progress: progress(dict(event='refiner_retime', windows=len(speed), strict_byte_parity=timing['six_frame_byte_parity']))
    if len(speed) != len(speed_keys): raise RuntimeError('missing selected retime windows')
    result = copy.deepcopy(imported)
    result.update(speed=speed, quality_reused=True, no_DEV512_rescoring=True,
        speed_summary=engine.speed_summary(speed, len(speed_keys), repeats, provider.device.type == 'cuda'))
    return result


def main(event=None, argv=None):
    event = event or threading.Event()
    p = legacy.parser(); p.set_defaults(mode='gt', epochs=2, eval_population='dev512')
    p.add_argument('--reuse-audit', required=True)
    p.add_argument('--retime-resume', action='store_true')
    p.add_argument('--speed-repeats', type=int, default=3)
    a = p.parse_args(argv)
    if not a.resume or a.max_updates or not 2 <= a.speed_repeats <= 4: raise ValueError('frozen retime only')
    root, parent = Path(a.out_dir).resolve(), Path(a.reuse_audit).resolve()
    if root == parent or parent in root.parents: raise RuntimeError('NEW retime directory OUTSIDE original audit')
    parent_lock = parent.parent/'.shared_pilot_locks'/('refiner_audit_'+legacy.stable_json_fingerprint(str(parent)))
    parent_lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(parent_lock):
        old, old_contract, saved = completed_audit(parent)
        if old_contract['runtime'] != dict(torch=str(torch.__version__), cuda=torch.version.cuda, numpy=np.__version__):
            raise RuntimeError('use original audit Torch/CUDA/NumPy environment')
        if Path(a.resume).resolve() != parent/'frozen/epoch2.pt': raise RuntimeError('use audited frozen snapshot')
        keys = {tuple(k) for k in old_contract['speed_keys']}
        if len(keys) != 32: raise RuntimeError('reuse EXACT old fixed32 population; no score-dependent sampling')
        receipts = {name: legacy.sha256(parent/name) for name in
            ('results.json', 'diagnostic_contract.json', f'eval_state_update{saved["updates"]}.json', 'frozen/epoch2.pt')}
        fields = vars(a).copy(); fields.pop('retime_resume')
        identity = dict(protocol=PROTOCOL, args=fields, source_receipts=receipts,
            code={f.name: legacy.sha256(f) for f in Path(__file__).parent.glob('*.py')},
            fresh_dependencies=old_contract['dependencies'], runtime=old_contract['runtime'])
        repo = Path(__file__).resolve().parents[3]
        for path, digest in old_contract['dependencies'].items():
            if legacy.sha256(repo/path) != digest: raise RuntimeError('original fresh inference dependency differs: '+path)
        if a.retime_resume:
            if read(root/'retime_contract.json') != json.loads(json.dumps(identity)): raise RuntimeError('retime contract changed')
        else:
            root.mkdir(parents=True, exist_ok=False); (root/'frozen').mkdir()
            snapshot_checkpoint(parent/'frozen/epoch2.pt', root/'frozen/epoch2.pt')
            legacy.write_json(root/'retime_contract.json', identity)
        if legacy.sha256(root/'frozen/epoch2.pt') != receipts['frozen/epoch2.pt']: raise RuntimeError('snapshot mismatch')
        inner = vars(a).copy()
        for name in ('reuse_audit', 'retime_resume', 'speed_repeats'): inner.pop(name)
        inner.update(resume=str(root/'frozen/epoch2.pt'), evaluate_only=True)
        args = []
        for name, value in inner.items():
            if value is None or name == 'override': continue
            if isinstance(value, bool):
                if value: args.append('--'+name.replace('_', '-'))
            else: args += ['--'+name.replace('_', '-'), str(value)]
        for value in a.override or []: args += ['--override', value]
        original_eval, original_batches = quality.evaluate, legacy.prefetched_batches
        def selected_batches(provider, source, groups, descriptors, **kw):
            selected = [g for g in groups if all((r['scene_name'], r['t0_token']) in keys for r in g)]
            return original_batches(provider, source, selected, descriptors, **kw)
        quality.evaluate = lambda *pos, **kw: evaluate(*pos, **kw, imported=old['evaluation'], speed_keys=keys, repeats=a.speed_repeats)
        legacy.prefetched_batches = selected_batches
        try: status = training.main(event, args)
        finally: quality.evaluate, legacy.prefetched_batches = original_eval, original_batches
        if any(legacy.sha256(parent/name) != digest for name, digest in receipts.items()): raise RuntimeError('source audit changed')
        result = read(root/'results.json'); evaluation = result.get('evaluation')
        lines = ['===== FROZEN REFINER / COMPLETE32 RETIME, QUALITY REUSED =====',
            f'status={result["status"]}; no training, no DEV512 re-scoring, original edit counts unchanged.',
            f'ALL32 windows x{a.speed_repeats} timed; differences NEVER filter population; strict byte FAIL retained independently.']
        if evaluation:
            lines.append('SPEED '+json.dumps(evaluation['speed_summary']))
            legacy.write_json(root/'speed.json', evaluation['speed_summary'])
        else: lines.append('Saved complete timing windows; resume SAME retime directory.')
        text = '\n'.join(lines)+'\n'; (root/'summary.txt').write_text(text, encoding='utf-8'); print(text, flush=True)
        return status


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
