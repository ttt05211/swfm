"""Full-population evaluation of the already fixed FREE=-1, no fitting."""
import argparse
import json
from pathlib import Path
import signal
import sys
import threading
import numpy as np
import torch
from tools.joint_occ_experiments import run as base
from tools.fpn_unet_all.run import AllPredictor, select_jobs
from tools.fpn_unet_eval.common import DenseInputs
from tools.fpn_unet_stc_age.run import select_source, load_frozen
from tools.fpn_stc_bias448.run import SOURCE_SHA256, FREE_BIAS, NAMES, HORIZONS, predict, restore
from tools.final_semantic_evaluation import metrics
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint

PROTOCOL = 'frozen_FULL_FPN_STC_age0.5_fixed_free_minus1_ALL4369_v1'


def summary(result):
    lines = ['===== FROZEN STC AGE / FIXED FREE -1 / ALL4369 =====', 'status='+result['status'],
        f'windows={len(result["keys"])}/{len(result["contract"]["keys"])}; scientific_updates=0',
        'SAME complete VAL4369 / 150 scenes as the original full evaluation; no planner/missing-STC trimming.',
        'FULL TRAIN20430x2 epoch2; STC memory age0.5; GT future ego conditioned; dense-valid/full-grid.',
        'ONE network forward per window; same SIX forecasts for baseline and fixed FREE=-1 readouts.',
        'Only FREE logit minus1; no norm/embedding/readout adaptation, motion changes or parameter search.',
        'Labeled target-domain output calibration: bias chosen on original FIRST32 scenes before this run.',
        'ALL4369 includes those calibration scenes; NOT an independent test or zero-calibration result.',
        'route mIoU IoU MovingMicro']
    fmt = lambda v: 'NA' if v is None else f'{v:.6f}'
    for n, report in result['reports'].items():
        lines.append(n+' '+' '.join(fmt(report['average'][k]) for k in ('mIoU', 'IoU', 'MovingMicro')))
        for h, row in report['per_horizon'].items():
            lines.append(f'  {h}s '+' '.join(f'{k}={fmt(row[k])}' for k in ('mIoU', 'IoU', 'MovingMicro')))
    if result['reports']:
        old, new = [result['reports'][n]['average'] for n in NAMES]
        lines.append('free_minus1_minus_baseline='+json.dumps({k: None if old[k] is None or new[k] is None
            else new[k]-old[k] for k in ('mIoU', 'IoU', 'MovingMicro')}))
    if result.get('error'): lines.append('error='+result['error'])
    lines.append('Integer counts accumulated across ALL windows, not averages of subset scores. Quality time NOT FPS.')
    lines.append('Original model/checkpoint/caches unchanged; no training, further tuning or automatic promotion.')
    return '\n'.join(lines)+'\n'


def main(argv=None, event=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir'); p.add_argument('--out-dir', required=True); p.add_argument('--resume', action='store_true')
    for n in ('config', 'dev-cache', 'dev-info', 'dataroot', 'stc-root'): p.add_argument('--'+n, required=True)
    p.add_argument('--device', default='cuda'); p.add_argument('--cpu-workers', type=int, default=4)
    a = p.parse_args(argv); event = event or threading.Event(); repo = Path(__file__).resolve().parents[2]
    if not 1 <= a.cpu_workers <= 4: raise ValueError('workers1..4 required')
    if a.device.startswith('cuda') and (torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('expose ONE BF16 GPU')
    torch.set_num_threads(1); torch.manual_seed(21); out = Path(a.out_dir).resolve()
    if a.resume and not a.source_dir:
        a.source_dir = str(Path(json.loads((out/'contract.json').read_text(encoding='utf-8'))['source']).parent)
    source = select_source(a.source_dir, repo/'outputs/p0_f9_shared_history'); sha = base.sha256(source)
    if sha != SOURCE_SHA256: raise RuntimeError('fixed FREE=-1 belongs to the supplied cedcf7c FULL epoch2 only')
    if out == source.parent or source.parent in out.parents: raise ValueError('NEW independent output outside source run required')
    lock = out.parent/'.fpn_stc_bias_all_locks'/base.stable_json_fingerprint(str(out)); lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(lock):
        if not a.resume:
            out.mkdir(parents=True, exist_ok=False); snapshot_checkpoint(source, out/'frozen_epoch2.pt')
        frozen = out/'frozen_epoch2.pt'
        if base.sha256(frozen) != sha: raise RuntimeError('snapshot differs')
        c, body = load_frozen(frozen, a.device)
        for n in ('config', 'dev_cache', 'dev_info'):
            if base.sha256(getattr(a, n)) != c['data'][n]: raise RuntimeError('original metadata differs: '+n)
        for n, digest in c['implementation'].items():
            if base.sha256(repo/n) != digest: raise RuntimeError('original implementation differs: '+n)
        _, records = base.load_cache(a.dev_cache)
        jobs, keys = select_jobs(records, c['eval_keys'], c['train_keys'])
        inputs = DenseInputs(a.dataroot, a.dev_info, stc_root=a.stc_root, cache_mib=256)
        inventory = inputs.preflight(jobs, 'stc')  # stat only, never silently drop missing windows
        current = {r[0]: tuple(r[1:]) for r in inventory}
        for path, size, stamp in c['STC_inventory']:
            if current.get(path) != (size, stamp): raise RuntimeError('original STC data differs: '+path)
        files = [*Path(__file__).parent.glob('*.py'), Path(__file__).parent/'run.sh']+[
            repo/n for n in ('tools/fpn_stc_bias448/run.py', 'tools/fpn_unet_all/run.py',
                'tools/fpn_unet_eval/common.py', 'tools/fpn_unet_stc_age/run.py',
                'tools/fpn_unet_stc_age/processing.py', 'tools/final_semantic_evaluation/metrics.py',
                'tools/final_semantic_evaluation/sources.py')]
        contract = dict(protocol=PROTOCOL, source=str(source), source_sha256=sha, source_contract=c,
            keys=keys, windows=len(keys), scenes=len({k[0] for k in keys}), known='STC_dense_valid', metric_mask='none',
            selection='SAME_original64_prefix_plus_ALL_sorted_remaining4305_no_filter',
            calibration=dict(type='labeled_target_domain_output_calibration', free_logit_bias=FREE_BIAS,
                residual_gain=1., fit_keys=c['eval_keys'][:32], fixed_before_full_evaluation=True, calibration_scenes_included=True),
            half_life_s=.5, inventory=inventory,
            implementation={f.relative_to(repo).as_posix(): base.sha256(f) for f in files},
            runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda, device=a.device,
                cpu_workers=a.cpu_workers, dataroot=str(Path(a.dataroot).resolve()), stc_root=str(Path(a.stc_root).resolve())))
        if a.resume and json.loads((out/'contract.json').read_text(encoding='utf-8')) != contract:
            raise RuntimeError('resume SAME full population/model/data/fixed bias only')
        base.write_json(out/'contract.json', contract)
        pcfg = base.make_prepare_config(base.load_runtime_config(a.config, c.get('overrides')))
        state_path = out/'state.json'
        saved = json.loads(state_path.read_text(encoding='utf-8')) if a.resume and state_path.exists() else None
        counts, completed = restore(saved, contract, keys, int(np.prod(pcfg.grid.shape_hwd)))
        predictor = AllPredictor(body, pcfg, a.device, workers=a.cpu_workers); predictor.kind = 'stc'
        before = tensor_state_fingerprint(body.state_dict())
        result = dict(status='running', contract=contract, keys=completed, reports={}, scientific_updates=0)
        def publish():
            base.write_json(state_path, dict(contract=contract, keys=completed,
                counts={n: {k: v.tolist() for k, v in fields.items()} for n, fields in counts.items()}))
            result['reports'] = {n: metrics.report(v, HORIZONS) for n, v in counts.items()}
            base.write_json(out/'evaluation.json', result)
            base.write_json(out/'progress.json', dict(status=result['status'], windows=len(completed), total=len(keys)))
            (out/'summary.txt').write_text(summary(result), encoding='utf-8')
        try:
            publish()
            with torch.no_grad():
                for i in range(len(completed), len(jobs)):
                    if event.is_set(): raise InterruptedError('complete window saved')
                    job = jobs[i]; rec, raw, times = inputs.history(job, 'stc')
                    row = predictor.prepare(rec, raw, times=times)
                    fw = predictor.forward(row, poses=np.asarray(inputs.poses(job)))
                    predictions = predict(fw['logits'], fw['shape']); del fw
                    # Both full six-frame readouts exist before future GT occupancy/Moving reads.
                    target = inputs.targets(job, (1, 3, 5))
                    moving = inputs.moving(job, (1, 3, 5), HORIZONS, pcfg.grid, predictor.provider.workers)
                    delta = {n: metrics.empty(3) for n in NAMES}
                    for n in NAMES:
                        for h, at in enumerate((1, 3, 5)): metrics.add(delta[n], h, predictions[n][at], target[h], moving[h])
                    for n in NAMES:
                        for field in counts[n]: counts[n][field] += delta[n][field]
                    completed.append(keys[i])
                    if len(completed) % 8 == 0:
                        publish(); print(f'STC_BIAS_ALL {len(completed)}/{len(keys)}; FREE=-1; no fitting', flush=True)
            result['status'] = 'complete'; return 0
        except InterruptedError:
            result['status'] = 'stopped_at_complete_window'; return 130
        except Exception as error:
            result.update(status='failed_original_unchanged', error=repr(error)); raise
        finally:
            predictor.close()
            unchanged = base.sha256(source) == sha and tensor_state_fingerprint(body.state_dict()) == before
            if not unchanged: result.update(status='failed_integrity', error='original checkpoint/model changed')
            publish(); print(summary(result), flush=True)
            if not unchanged: raise RuntimeError(result['error'])


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event=event) or 0)
