"""ONE fixed FREE-logit bias on 448 windows outside the original64 scenes."""
import argparse
import json
from pathlib import Path
import signal
import sys
import threading
import numpy as np
import torch
from tools.joint_occ_experiments import run as base
from tools.fpn_unet_all.run import AllPredictor
from tools.fpn_unet_eval.common import DenseInputs
from tools.fpn_unet_stc_age.run import select_source, load_frozen
from tools.fpn_unet_stc_age.processing import HALF_LIFE_SECONDS
from tools.final_semantic_evaluation.sources import window
from tools.final_semantic_evaluation import metrics
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint

PROTOCOL = 'frozen_FPN_STC_age_fixed_free_minus1_scene_disjoint448_v1'
SOURCE_SHA256 = 'cedcf7c35123489b0b7e3e0fdca1367241c3baf28c18fb1b8db6e0ff6bef138f'
FREE_BIAS = -1.
NAMES = ('baseline/stc_age_refine', 'free_minus1/stc_age_refine')
HORIZONS = (1., 2., 3.)


def select_jobs(records, original64, train_keys):
    keys = base.record_keys(records)
    original = [tuple(k) for k in original64]
    if len(keys) != 4369 or len(set(keys)) != 4369 or len({k[0] for k in keys}) != 150:
        raise RuntimeError('complete unique VAL4369 / 150 scenes required')
    excluded = {k[0] for k in original}
    if len(original) != 64 or len(set(original)) != 64 or len(excluded) != 64 or not set(original) <= set(keys):
        raise RuntimeError('exact original64 from 64 distinct scenes required')
    if {k[0] for k in keys} & {k[0] for k in train_keys}:
        raise RuntimeError('TRAIN/VAL scene overlap')
    selected = base.select_scene_balanced_round_robin(sorted(k for k in keys if k[0] not in excluded), 448)
    if len(selected) != 448 or len(set(selected)) != 448:
        raise RuntimeError('448 unique windows outside ALL original64 scenes required')
    return [window(r) for r in base.align_records(records, selected)], [list(k) for k in selected]


@torch.no_grad()
def predict(logits, shape):
    """One six-frame forward, two readouts; no in-place logit/weight changes."""
    if logits.shape != (6, 18, int(np.prod(shape))) or logits.dtype != torch.float32:
        raise ValueError('original FP32 full-grid SIX-frame logits required')
    # FREE is class17: occupied classes win ties, just as in full argmax.
    semantic, classes = logits[:, :17].max(1)
    calibrated = torch.where(semantic >= logits[:, 17]+FREE_BIAS, classes, 17)
    return {name: value.reshape(6, *shape).byte().cpu().numpy() for name, value in
        zip(NAMES, (logits.argmax(1), calibrated))}


def restore(saved, contract, keys, voxels):
    counts = {n: metrics.empty(3) for n in NAMES}
    completed = []
    if saved is not None:
        completed = saved['keys']
        if saved['contract'] != contract or len(completed) > len(keys) or completed != keys[:len(completed)]:
            raise RuntimeError('resume identity/population differs')
        if set(saved['counts']) != set(NAMES): raise RuntimeError('saved routes differ')
        for name in NAMES:
            if set(saved['counts'][name]) != {'confusion', 'moving'}: raise RuntimeError('count fields differ')
            for field in counts[name]:
                value = np.asarray(saved['counts'][name][field])
                if value.shape != (3, 18, 18) or value.dtype.kind not in 'iu' or np.any(value < 0):
                    raise RuntimeError('invalid integer counts')
                counts[name][field][:] = value
            if np.any(counts[name]['confusion'].sum((1, 2)) != len(completed)*voxels):
                raise RuntimeError('incomplete full-grid counts')
            if np.any(counts[name]['moving'] > counts[name]['confusion']):
                raise RuntimeError('invalid Moving support')
    return counts, list(completed)


def summary(result):
    lines = ['===== FROZEN STC AGE / FIXED FREE -1 / NEW-SCENE 448 =====', 'status='+result['status'],
        f'windows={len(result["keys"])}/{len(result["contract"]["keys"])}; scientific_updates=0',
        'Labeled target-domain OUTPUT CALIBRATION: FREE=-1 chosen using FIRST32 original scenes GT labels.',
        'ALL original64 scenes excluded; deterministic scene-balanced448. NOT zero-adaptation or ALL4369.',
        'Same frozen FULL TRAIN20430x2 epoch2; STC memory age0.5; GT future ego conditioned.',
        'FOUR histories -> SIX predictions BEFORE future GT/Moving reads; dense-valid, full-grid 1/2/3s.',
        'Only FREE logit offset; residual gain=1; no class/position/horizon rules or parameter search.',
        'route mIoU IoU MovingMicro']
    fmt = lambda v: 'NA' if v is None else f'{v:.6f}'
    for name, report in result['reports'].items():
        lines.append(name+' '+' '.join(fmt(report['average'][k]) for k in ('mIoU', 'IoU', 'MovingMicro')))
        for h, row in report['per_horizon'].items():
            lines.append(f'  {h}s '+' '.join(f'{k}={fmt(row[k])}' for k in ('mIoU', 'IoU', 'MovingMicro')))
    if result['reports']:
        old, new = [result['reports'][n]['average'] for n in NAMES]
        lines.append('free_minus1_minus_baseline='+json.dumps({k: None if old[k] is None or new[k] is None
            else new[k]-old[k] for k in ('mIoU', 'IoU', 'MovingMicro')}))
    if result.get('error'): lines.append('error='+result['error'])
    lines.append('No fitting on these448, threshold search, automatic promotion or full evaluation. Quality time NOT FPS.')
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
    if sha != SOURCE_SHA256: raise RuntimeError('fixed calibration belongs to the supplied cedcf7c FULL epoch2, not another model')
    if out == source.parent or source.parent in out.parents: raise ValueError('independent output outside source run required')
    lock = out.parent/'.fpn_stc_bias448_locks'/base.stable_json_fingerprint(str(out)); lock.mkdir(parents=True, exist_ok=True)
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
        _, records = base.load_cache(a.dev_cache); jobs, keys = select_jobs(records, c['eval_keys'], c['train_keys'])
        inputs = DenseInputs(a.dataroot, a.dev_info, stc_root=a.stc_root, cache_mib=256)
        inventory = inputs.preflight(jobs, 'stc')  # stat only; no missing-file population trimming
        files = [*Path(__file__).parent.glob('*.py'), Path(__file__).parent/'run.sh']+[
            repo/n for n in ('tools/fpn_unet_all/run.py', 'tools/fpn_unet_eval/common.py',
                'tools/fpn_unet_stc_age/run.py', 'tools/fpn_unet_stc_age/processing.py',
                'tools/final_semantic_evaluation/metrics.py', 'tools/final_semantic_evaluation/sources.py')]
        contract = dict(protocol=PROTOCOL, source=str(source), source_sha256=sha, source_contract=c,
            keys=keys, windows=len(keys), scenes=len({k[0] for k in keys}),
            excluded_scenes=sorted({k[0] for k in c['eval_keys']}),
            selection='sorted_scene_balanced448_excluding_ALL_original64_scenes_no_score_filter',
            calibration=dict(type='labeled_target_domain_output_calibration', free_logit_bias=FREE_BIAS,
                residual_gain=1., fit_keys=c['eval_keys'][:32], fixed_before_448=True),
            half_life_s=HALF_LIFE_SECONDS, known='STC_dense_valid', metric_mask='none', inventory=inventory,
            implementation={f.relative_to(repo).as_posix(): base.sha256(f) for f in files},
            runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda, device=a.device,
                cpu_workers=a.cpu_workers, dataroot=str(Path(a.dataroot).resolve()), stc_root=str(Path(a.stc_root).resolve())))
        if a.resume and json.loads((out/'contract.json').read_text(encoding='utf-8')) != contract:
            raise RuntimeError('resume SAME frozen data/model/fixed calibration only')
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
                    # Both complete six-frame predictions exist before ANY future occupancy/Moving labels.
                    target = inputs.targets(job, (1, 3, 5))
                    moving = inputs.moving(job, (1, 3, 5), HORIZONS, pcfg.grid, predictor.provider.workers)
                    delta = {n: metrics.empty(3) for n in NAMES}
                    for n in NAMES:
                        for h, at in enumerate((1, 3, 5)): metrics.add(delta[n], h, predictions[n][at], target[h], moving[h])
                    for n in NAMES:
                        for field in counts[n]: counts[n][field] += delta[n][field]
                    completed.append(keys[i])
                    if len(completed) % 8 == 0:
                        publish(); print(f'STC_BIAS448 {len(completed)}/{len(keys)}; fixed FREE=-1; no fitting', flush=True)
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
