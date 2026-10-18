"""Frozen full stage2 epoch2, same64 GT/STC future-motion cause audit."""
import argparse
import json
from pathlib import Path
import signal
import threading
import numpy as np
import torch
from tools.joint_occ_experiments import run as base
from tools.source_scene_full.model import checkpoint_model
from tools.source_scene_full.run import PROTOCOL as SOURCE_PROTOCOL
from tools.source_scene_stage2.adapter import DirectPredictor
from tools.final_semantic_evaluation.sources import NuscInputs, window
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from .engine import predictions, MODES
from .statistics import window_counts, finish, edits, box_motion_rows, source_summary, distribution

PROTOCOL = 'frozen_stage2_same64_future_motion_intervention_v1'


def summary(result):
    lines = ['===== FROZEN STAGE2 / FUTURE MOTION AUDIT =====', 'status='+result['status'],
        f'windows={result["windows"]}/64; scientific_updates=0; checkpoint/Adam/RNG/caches READ ONLY.',
        'Same dense-valid histories/GT future ego/full six forecasts. NO alignment/threshold/epoch tuning.',
        'KTA_future zeros FUTURE XY/yaw corrections ONLY. Learned Q/history yaw and backgrounds stay fixed.',
        'history_yaw statistics exclude the t0 zero gauge.',
        'motion_head_weight_norms='+json.dumps(result.get('head_weight_norms', {})),
        'mode               route             mIoU       IoU MovingMicro']
    for modality, routes in result.get('metrics', {}).items():
        lines.append('===== '+modality+' =====')
        if not routes: continue
        for route, m in routes.items():
            lines.append(f'{route:34s} {m["mIoU"]:9.6f} {m["IoU"]:9.6f} {m["MovingMicro"]:11.6f}')
        for readout in ('transport', 'semantic_refine'):
            a, b = routes['learned/'+readout], routes['KTA_future/'+readout]
            lines.append(readout+' learned_minus_KTA='+json.dumps({k: a[k]-b[k] for k in ('mIoU', 'IoU', 'MovingMicro')}))
        s = result['source_motion'][modality]
        for group, v in [('all', s['all']), *s['groups'].items()]:
            lines.append(f'SOURCE {group}: n={v["samples"]}; correction_m={v["XY_correction_m"]}; future_yaw_deg={v["future_yaw_abs_deg"]}; XY_error learned/KTA={v["learned_error_m"]}/{v["KTA_error_m"]}; helped={v["XY_helped"]} harmed={v["XY_harmed"]}')
        lines.append('history_yaw_abs_deg='+json.dumps(result['history_yaw_abs_deg'][modality]))
        lines.append('original_repeat_changed_voxels='+json.dumps(result['reference_repeat'][modality]))
        for cls, v in s['classes'].items():
            lines.append(f'SOURCE_CLASS {cls}: XY_error learned/KTA={v["learned_error_m"]}/{v["KTA_error_m"]}; helped={v["XY_helped"]} harmed={v["XY_harmed"]}')
        for readout in ('transport', 'semantic_refine'):
            # Full-grid class deltas use common-valid reporting horizons only.
            aa, bb = routes['learned/'+readout], routes['KTA_future/'+readout]
            delta = {}
            for cls in range(17):
                values = []
                for h in ('1.0', '2.0', '3.0'):
                    av = aa['per_horizon'][h]['semantic_per_class'][str(cls)]
                    bv = bb['per_horizon'][h]['semantic_per_class'][str(cls)]
                    if av is not None and bv is not None and np.isfinite(av) and np.isfinite(bv): values.append(av-bv)
                if values: delta[str(cls)] = float(np.mean(values))
            lines.append(readout+' fullgrid_class_delta_learned_minus_KTA='+json.dumps(delta))
        lines.append('semantic_edit_counts learned_vs_KTA='+json.dumps(result['edit_counts'][modality]))
    lines += ['Source motion errors: conditional GT-box-attributed component displacement, NOT perfect point-motion truth; unresolved/fragmented support disclosed.',
        'GT boxes and future labels read ONLY AFTER all intervention forecasts. Static/moving source groups use fixed GT interval speed0.5m/s.',
        'Outside-moving-support metric region includes background and is NOT an exact stationary-instance mask.',
        'Per-class/per-horizon/full+moving+outside counts, source errors and history yaw in results.json/windows.jsonl.',
        'Original-repeat byte drift is reported, NOT waived or filtered. Counterfactual changes near that scale are inconclusive.',
        'Repeated DEV mechanism diagnostic, NOT independent test, model selection, training or FPS.']
    return '\n'.join(lines)+'\n'


def restore(saved, contract):
    if saved['contract'] != contract or saved.get('protocol') != PROTOCOL: raise RuntimeError('audit resume contract differs')
    if saved['keys'] != contract['keys'][:len(saved['keys'])] or len(saved['keys']) != len(saved['rows']):
        raise RuntimeError('invalid completed-window prefix')
    for modality, routes in saved['counts'].items():
        for route, fields in routes.items():
            for field in ('full', 'moving'):
                a = np.asarray(fields[field])
                if a.shape != (3, 18, 18) or a.dtype.kind not in 'iu' or np.any(a < 0): raise RuntimeError('invalid integer confusion')
                fields[field] = a.copy()
            if np.any(fields['moving'] > fields['full']): raise RuntimeError('moving confusion exceeds full')
    return saved


@torch.no_grad()
def main(argv=None, event=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'out', 'config', 'dev-cache', 'dev-info', 'dataroot', 'stc-root'): ap.add_argument('--'+name, required=True)
    ap.add_argument('--resume', action='store_true'); ap.add_argument('--cpu-workers', type=int, default=4)
    a = ap.parse_args(argv); event = event or threading.Event()
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported(): raise RuntimeError('ONE BF16 GPU required')
    if not 1 <= a.cpu_workers <= 4: raise ValueError('bounded CPU workers')
    torch.set_num_threads(1); device = torch.device('cuda:0')
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False); c = saved['contract']
    if (saved.get('protocol') != SOURCE_PROTOCOL or saved['epoch'] != 2 or saved['batch'] != 0
            or saved['updates'] != c['total_updates'] or c['known'] != 'TRAIN_OCC_STC_dense_valid'):
        raise RuntimeError('completed frozen full-stage2 epoch2 required')
    root = Path(__file__).resolve().parents[2]
    for file, sha in c['implementation'].items():
        if base.sha256(root/file) != sha: raise RuntimeError('source implementation changed: '+file)
    for name in ('config', 'dev_cache', 'dev_info'):
        if base.sha256(getattr(a, name)) != c['data'][name]: raise RuntimeError('source data/config changed: '+name)
    source_sha = base.sha256(a.checkpoint); model = checkpoint_model(saved, device)
    weights = {k: float(v.float().norm()) for k, v in saved['models']['gt'].items()
               if any(x in k for x in ('residual_head', 'yaw_head', 'history_yaw'))}
    del saved
    pcfg = base.make_prepare_config(base.load_runtime_config(a.config, c['overrides']))
    _, records = base.load_cache(a.dev_cache); jobs = base.align_records(records, c['eval_keys'])
    if len(jobs) != 64: raise RuntimeError('same remaining64 required')
    inputs = NuscInputs(a.dataroot, a.dev_info, stc_root=a.stc_root, cache_mib=128)
    files = []
    for rec in jobs:
        w = window(rec)
        for kind in ('occ', 'stc'):
            for token in w.history_tokens:
                path = inputs.path(kind, w.scene_name, token); files.append((str(path), *inputs.stat(path)))
        for h in (1, 3, 5):
            path = inputs.path('occ', w.scene_name, w.future_tokens[h]); files.append((str(path), *inputs.stat(path)))
    contract = dict(protocol=PROTOCOL, checkpoint_sha256=source_sha, source_update=c['total_updates'], keys=c['eval_keys'],
        files=sorted(set(files)), modes=MODES, fixed_Q_and_history_yaw=True, cpu_workers=a.cpu_workers,
        runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)),
        metadata={n: base.sha256(Path(a.dataroot)/'v1.0-trainval'/n) for n in
            ('sample.json', 'sample_annotation.json', 'instance.json', 'category.json', 'sample_data.json', 'ego_pose.json')},
        implementation={p.name: base.sha256(p) for p in Path(__file__).parent.glob('*.py')})
    contract = json.loads(json.dumps(contract)); out = Path(a.out).resolve()
    lock = out.parent/'.motion_audit_locks'/base.stable_json_fingerprint(str(out)); lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(lock):
        if a.resume: state = restore(json.loads((out/'state.json').read_text()), contract)
        else:
            out.mkdir(parents=True, exist_ok=False)
            state = dict(protocol=PROTOCOL, contract=contract, counts={'occ_gt': {}, 'stc_gt': {}}, keys=[], rows=[])
        runner = DirectPredictor(model, pcfg, device, workers=a.cpu_workers)
        def publish(status):
            result = dict(status=status, contract=contract, windows=len(state['keys']), head_weight_norms=weights,
                metrics={kind: {r: finish(v) for r, v in routes.items()} for kind, routes in state['counts'].items()},
                history_yaw_abs_deg={kind: distribution([abs(y) for w in state['rows'] for y in w[kind]['valid_history_yaw_deg']])
                    for kind in ('occ_gt', 'stc_gt')},
                reference_repeat={kind: {r: dict(failed_windows=sum(w[kind]['repeat_changed_voxels'][r] > 0 for w in state['rows']),
                    max_changed_voxels=max((w[kind]['repeat_changed_voxels'][r] for w in state['rows']), default=0))
                    for r in ('transport', 'semantic_refine')} for kind in ('occ_gt', 'stc_gt')},
                source_motion={kind: source_summary([r for w in state['rows'] for r in w[kind]['sources']]) for kind in ('occ_gt', 'stc_gt')},
                edit_counts={kind: {readout: {key: sum(w[kind]['edits'][readout].get(key, 0) for w in state['rows'])
                    for key in {k for w in state['rows'] for k in w[kind]['edits'][readout]}}
                    for readout in ('transport', 'semantic_refine')} for kind in ('occ_gt', 'stc_gt')})
            serial_state = dict(state, counts={kind: {r: {k: v.tolist() for k, v in fields.items()}
                for r, fields in routes.items()} for kind, routes in state['counts'].items()})
            base.write_json(out/'state.json', serial_state); base.write_json(out/'results.json', result)
            (out/'summary.txt').write_text(summary(result), encoding='utf-8')
            return result
        try:
            for rec in jobs[len(state['keys']):]:
                if event.is_set(): break
                w = window(rec); forecasts = {}; diagnostics = {}; rows = {}
                for kind, modality in (('occ_gt', 'occ'), ('stc_gt', 'stc')):
                    record, historical, times = inputs.history(w, modality)
                    sem = np.asarray(historical['history_occ'])
                    historical = dict(historical, history_observed=(sem >= 0)&(sem < 18))
                    rows[kind] = runner.prepare(record, historical, times=times)
                    forecasts[kind], diagnostics[kind] = predictions(model, runner.provider, rows[kind], inputs.poses(w))
                # ALL six-frame interventions for BOTH modalities precede GT
                # target/annotation reads. Targets never enter model arguments.
                targets = inputs.targets(w, (1, 3, 5))
                masks = np.asarray(inputs.moving(w, (1, 3, 5), (1., 2., 3.), pcfg.grid, a.cpu_workers), bool)
                window_row = dict(key=[str(rec['scene_name']), str(rec['t0_token'])]); delta = {}
                for kind in forecasts:
                    delta[kind] = window_counts(forecasts[kind], targets, masks)
                    d = diagnostics[kind]
                    window_row[kind] = dict(sources=box_motion_rows(rows[kind], d, inputs.source.nusc, pcfg.grid),
                        edits=edits(forecasts[kind], targets, masks),
                        repeat_changed_voxels=d['reference_repeat_changed_voxels'],
                        valid_history_yaw_deg=(d['history_yaw_rad'][:, :-1][d['history_valid'][:, :-1]]*180/np.pi).tolist())
                for kind, routes in delta.items():
                    for route, fields in routes.items():
                        if route not in state['counts'][kind]: state['counts'][kind][route] = fields
                        else:
                            for field, v in fields.items(): state['counts'][kind][route][field] += v
                state['keys'].append(window_row['key']); state['rows'].append(window_row)
                with (out/'windows.jsonl').open('a', encoding='utf-8') as f: f.write(json.dumps(window_row)+'\n')
                if len(state['keys']) % 8 == 0:
                    publish('running'); print(f'MOTION_AUDIT {len(state["keys"])}/64', flush=True)
            if base.sha256(a.checkpoint) != source_sha: raise RuntimeError('source checkpoint changed during audit')
            result = publish('complete' if len(state['keys']) == 64 else 'stopped_at_complete_window')
            print(summary(result), flush=True)
        finally: runner.close()


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    main(event=event)
