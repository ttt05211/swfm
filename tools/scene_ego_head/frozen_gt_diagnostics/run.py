"""One frozen DEV window only; no ego optimization, metric scores or gate waiver."""
import json
from pathlib import Path
import signal
import sys
import threading
import torch
from tools.joint_occ_experiments import run as legacy
from tools.shared_history_experiments.full_gt import run as gt_run, engine as gt
from tools.scene_ego_head.frozen_gt.run import validate_completed_source
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from .audit import audit


def summary(result):
    lines = ['===== FROZEN GT / ONE-WINDOW LOGITS DRIFT AUDIT =====',
        'No training / no DEV quality scores / no gate or tolerance changes.']
    if result:
        lines.append('background_changed='+str(result['background_changed_voxels_by_horizon']))
        for name in ('cross_entrance', 'GT_original_repeat', 'external_original_repeat'):
            r = result[name]
            lines.append(f'{name}: first_FAIL={r["earliest_numerical_FAIL"]}; changed_voxels={r["semantic"]["changed_voxels"]}')
        for name in ('identical_input_BF16_repeat', 'cross_input_BF16_refiner', 'FP32_cross_input_reference', 'original_vs_FP32'):
            r = result[name]
            lines.append(f'{name}: logits_pass={r["logits"]["pass_gate"]}; max_abs={r["logits"]["max_abs"]}; '
                f'changed_voxels={r["semantic"]["changed_voxels"]}')
            if 'trace' in r: lines.append('  trace_first_difference='+str(r['trace']['first_exact_difference']))
        lines.append('BF16_quantization='+json.dumps(result['BF16_input_quantization']))
    lines.append('Original checkpoints/body/Adam/cache unchanged; FP32 is DIAGNOSTIC ONLY, not deployed.')
    return '\n'.join(lines)+'\n'


def main(event=None, argv=None):
    event = event or threading.Event(); p = legacy.parser(); p.description = __doc__
    p.set_defaults(mode='gt', epochs=2, train_fraction=1., eval_population='dev512')
    a = p.parse_args(argv)
    if not a.resume or a.max_updates: raise ValueError('frozen completed GT checkpoint only')
    source, out = Path(a.resume).resolve(), Path(a.out_dir).resolve()
    if out == source.parent or source.parent in out.parents: raise ValueError('NEW diagnostic directory outside source')
    out.mkdir(parents=True, exist_ok=False); (out/'frozen').mkdir()
    digest = snapshot_checkpoint(source, out/'frozen/body.pt')
    saved = torch.load(out/'frozen/body.pt', map_location='cpu', weights_only=False)
    validate_completed_source(saved)
    original_eval, original_train = gt.evaluate, gt.train_arm
    old_summary, old_save = legacy.summarize, legacy.save_resume_checkpoint
    holder = {}
    def evaluate(models, provider, dev_source, rows, **kwargs):
        first = next(iter(rows), None)
        if first is None or event.is_set(): raise InterruptedError('no diagnostic window evaluated')
        body = models['gt']; before = tensor_state_fingerprint(body.state_dict())
        report = audit(body, provider, first[0])
        if tensor_state_fingerprint(body.state_dict()) != before: raise RuntimeError('frozen body changed')
        report.update(source_checkpoint=str(source), source_sha256=digest, source_update=saved['updates'],
            body_state_sha256=before, body_unchanged=True)
        holder['report'] = report; legacy.write_json(out/'audit.json', report)
        return dict(windows=0, variants={}, trajectory={}, diagnostic_only=True)
    def forbidden(*a, **kw): raise RuntimeError('read-only audit cannot train/save training checkpoints')
    inner = vars(a).copy(); inner.update(resume=str(out/'frozen/body.pt'), evaluate_only=True)
    args = []
    for k, v in inner.items():
        if k == 'override' or v is None: continue
        if isinstance(v, bool):
            if v: args.append('--'+k.replace('_', '-'))
        else: args += ['--'+k.replace('_', '-'), str(v)]
    for v in a.override or []: args += ['--override', v]
    try:
        gt.evaluate, gt.train_arm = evaluate, forbidden
        legacy.summarize, legacy.save_resume_checkpoint = lambda _: summary(holder.get('report')), forbidden
        status = gt_run.main(event, args)
    finally:
        gt.evaluate, gt.train_arm = original_eval, original_train
        legacy.summarize, legacy.save_resume_checkpoint = old_summary, old_save
    if legacy.sha256(out/'frozen/body.pt') != digest: raise RuntimeError('snapshot changed')
    text = summary(holder.get('report')); (out/'summary.txt').write_text(text, encoding='utf-8')
    print(text, flush=True); return status


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
