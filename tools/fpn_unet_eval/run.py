"""Frozen final FPN+UNet: original64 + fixed448 quality, same-window warm FPS."""
import argparse
import json
import math
from pathlib import Path
import signal
import sys
import threading
import torch
from tools.joint_occ_experiments import run as base
from tools.final_semantic_evaluation.engine import evaluate
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from .common import select_source, load_frozen, select_jobs, DenseInputs, ModalInputs, FPNPredictor, merge_counts
from . import speed

PROTOCOL = 'frozen_FPN_UNet_final1_DEV64plus448_512_and_legacy_FPS_v1'


def summary(result):
    c = result['contract']
    lines = ['===== FROZEN FPN + 3D U-NET / EXTENDED DEV512 + WARM FPS =====', 'status='+result['status'],
        'Fixed TRAIN4086 x1 final checkpoint; scientific_updates=0; NO retraining/threshold change/model selection.',
        'Population=original64 + deterministic scene-balanced additional448 from VAL4369; NOT historical planner-trimmed DEV.',
        'GT future ego Query and all geometry; dense-valid OCC/STC; full-grid 1/2/3s; NO Pred/visibility mask/alignment.',
        f'source={c["source"]}; parameters={c["source_contract"]["parameters"]}',
        'population       route                         mIoU       IoU MovingMicro']
    for group, reports in result['reports'].items():
        for route, report in reports.items():
            m = report['average']
            show = lambda v: 'NA' if v is None else f'{v:.6f}'
            lines.append(f'{group:16s} {route:28s} {show(m["mIoU"]):>10s} {show(m["IoU"]):>10s} {show(m["MovingMicro"]):>11s}')
    original = result['reports'].get('original64', {})
    if original:
        for route, report in original.items():
            old = result.get('source64_metrics', {}).get(route)
            if old:
                delta = {k: None if report['average'][k] is None else report['average'][k]-old[k] for k in ('mIoU', 'IoU', 'MovingMicro')}
                lines.append('recomputed_original64_minus_training_report '+route+' '+json.dumps(delta))
    if result.get('speed_reports'):
        lines += ['===== SAME-WINDOW WARM INFERENCE / SIX FRAMES =====',
            f'windows={c["speed_windows"]}; repeats={c["speed_repeats"]}; ONE visible GPU; {result["hardware"]}']
        for kind, s in result['speed_reports'].items():
            lines.append(f'{kind}: six_frame_ms={s["six_frame_ms"]:.3f}; output_frames/s={s["output_frame_fps"]:.3f}; '
                f'p90_ms={s["p90_six_frame_ms"]:.3f}; peak_allocated_MiB={s["peak_allocated_mib"]:.1f}; '
                f'strict_byte_parity={s["strict_byte_parity"]}; max_changed={s["max_changed_voxels"]}; '
                f'reference_repeat_max_changed={s["reference_repeat_max_changed_voxels"]}')
        lines.append('Parity FAIL remains FAIL: elapsed samples retained, but failed timings are NOT a verified identical quality route.')
    lines += ['Timing IN: fresh KTA/Strong+CLEAR/FPN/WM/lift/prior/memory/U-Net/all6 CPU semantic outputs.',
        'Timing OUT: historical extraction/association/representation, initial upload, disk I/O, labels/metrics, warmup/startup.',
        'Same declared legacy prepared-history timing boundary, NOT raw-camera end-to-end FPS; no automatic promotion.',
        'Original checkpoint/model/caches read only; no reuse of cached future backgrounds or learned outputs.']
    if result.get('error'): lines.append('error='+result['error'])
    return '\n'.join(lines)+'\n'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir'); p.add_argument('--out-dir', required=True); p.add_argument('--resume', action='store_true')
    for n in ('config', 'dev-cache', 'dev-info', 'dataroot', 'stc-root'): p.add_argument('--'+n, required=True)
    p.add_argument('--device', default='cuda'); p.add_argument('--cpu-workers', type=int, default=4)
    p.add_argument('--speed-windows', type=int, default=32); p.add_argument('--speed-repeats', type=int, default=2)
    return p


def main(argv=None, event=None):
    a = parser().parse_args(argv); event = event or threading.Event(); repo = Path(__file__).resolve().parents[2]
    out = Path(a.out_dir).resolve()
    if not 1 <= a.cpu_workers <= 4 or not 1 <= a.speed_windows <= 32 or not 1 <= a.speed_repeats <= 2:
        raise ValueError('bounded workers1..4 / timing windows1..32 / repeats1..2 required')
    if a.device.startswith('cuda') and (torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('exactly ONE visible BF16 GPU required')
    torch.set_num_threads(1); torch.manual_seed(21)
    if a.resume and not a.source_dir:
        a.source_dir = str(Path(json.loads((out/'contract.json').read_text(encoding='utf-8'))['source']).parent)
    source = select_source(a.source_dir, repo/'outputs/p0_f9_shared_history')
    if out == source.parent or source.parent in out.parents: raise ValueError('independent output outside training directory required')
    original_sha = base.sha256(source)
    lock = out.parent/'.fpn_unet_eval_locks'/base.stable_json_fingerprint(str(out)); lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(lock):
        if not a.resume:
            out.mkdir(parents=True, exist_ok=False); snapshot_checkpoint(source, out/'epoch1_frozen.pt')
        frozen = out/'epoch1_frozen.pt'
        if base.sha256(frozen) != original_sha: raise RuntimeError('source differs from frozen snapshot')
        c, body = load_frozen(frozen, a.device)
        source_report_path = source.parent/'results.json'
        source_report_sha = base.sha256(source_report_path)
        source_report = json.loads(source_report_path.read_text(encoding='utf-8'))
        if source_report.get('contract') != c or source_report.get('updates') != c['total_updates']:
            raise RuntimeError('final training report and frozen checkpoint differ')
        for name in ('config', 'dev_cache', 'dev_info'):
            if base.sha256(getattr(a, name)) != c['data'][name]: raise RuntimeError('original metadata differs: '+name)
        for name, expected in c['implementation'].items():
            if base.sha256(repo/name) != expected: raise RuntimeError('training implementation changed: '+name)
        _, records = base.load_cache(a.dev_cache); jobs, keys = select_jobs(records, c['eval_keys'], c['train_keys'])
        inputs = DenseInputs(a.dataroot, a.dev_info, stc_root=a.stc_root, cache_mib=256)
        inventory = {kind: inputs.preflight(jobs, kind) for kind in ('occ', 'stc')}
        current_files = {row[0]: tuple(row[1:]) for row in inventory['stc']}
        for path, size, stamp in c['STC_inventory']:
            if current_files.get(path) != (size, stamp): raise RuntimeError('original64 STC input differs: '+path)
        files = [*Path(__file__).parent.glob('*.py'), Path(__file__).parent/'run.sh',
            *[repo/n for n in ('tools/final_semantic_evaluation/engine.py', 'tools/final_semantic_evaluation/metrics.py',
                'tools/final_semantic_evaluation/sources.py', 'tools/final_semantic_evaluation/predictor.py', 'tools/source_scene_speed/engine.py')]]
        contract = dict(protocol=PROTOCOL, source=str(source), source_sha256=original_sha, source_contract=c,
            source_results_sha256=source_report_sha,
            keys=keys, windows=512, strata=dict(original64=keys[:64], additional448=keys[64:]),
            selection='original64_prefix_plus_scene_balanced448_no_score_filter_no_planner_trim',
            known='OCC_STC_dense_valid', metric_mask='none', inventory=inventory,
            speed_windows=a.speed_windows, speed_repeats=a.speed_repeats, timing_boundary=speed.BOUNDARY,
            runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda, device=a.device, cpu_workers=a.cpu_workers,
                dataroot=str(Path(a.dataroot).resolve()), stc_root=str(Path(a.stc_root).resolve())),
            implementation={f.relative_to(repo).as_posix(): base.sha256(f) for f in files})
        cp = out/'contract.json'
        if a.resume and json.loads(cp.read_text(encoding='utf-8')) != contract: raise RuntimeError('resume SAME frozen evaluation only')
        base.write_json(cp, contract)
        pcfg = base.make_prepare_config(base.load_runtime_config(a.config, c.get('overrides')))
        predictor = FPNPredictor(body, pcfg, a.device, workers=a.cpu_workers)
        before = tensor_state_fingerprint(body.state_dict())
        # epoch_0001.pt is saved BEFORE final DEV64; its scientific weights are
        # final, but only results.json/last.pt necessarily contains that report.
        old64 = source_report.get('evaluations', {}).get('1', {}).get('variants', {})
        result = dict(status='running', contract=contract, scientific_updates=0, reports={}, speed_reports={}, speed_samples=[],
            source64_metrics=old64, hardware=torch.cuda.get_device_name() if a.device.startswith('cuda') else 'CPU diagnostic')
        if a.resume:
            result = json.loads((out/'evaluation.json').read_text(encoding='utf-8'))
            if result['contract'] != contract: raise RuntimeError('result contract changed')
            result['status'] = 'running'
        def publish():
            base.write_json(out/'evaluation.json', result)
            (out/'summary.txt').write_text(summary(result), encoding='utf-8')
        publish()
        print(f'FPN_UNET_FROZEN source={source}; original64+fixed448=512; no optimizer; output={out}', flush=True)
        try:
            with torch.no_grad():
                for kind, prefix in (('occ', 'occ_gt'), ('stc', 'stc_gt')):
                    parts = []
                    for group, lo, hi in (('original64', 0, 64), ('additional448', 64, 512)):
                        path = out/f'{kind}_{group}_state.json'
                        state = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None
                        report = evaluate(predictor, ModalInputs(inputs, kind), jobs[lo:hi], 'gt', keys[lo:hi],
                            dict(base=contract, modality=kind, stratum=group), saved=state,
                            save=lambda value, path=path: base.write_json(path, value), stop_event=event)
                        result['reports'].setdefault(group, {}).update({prefix+'_'+n.split('/', 1)[1]: m for n, m in report['reports'].items()})
                        parts.append(json.loads(path.read_text(encoding='utf-8'))); publish()
                    merged = merge_counts(parts, keys, math.prod(pcfg.grid.shape_hwd))
                    result['reports'].setdefault('all512', {}).update({prefix+'_'+n.split('/', 1)[1]: m for n, m in merged.items()}); publish()
                result['speed_reports'] = speed.measure(body, predictor, inputs, jobs, keys,
                    samples=result['speed_samples'], windows=a.speed_windows, repeats=a.speed_repeats, save=publish, stop_event=event)
            result['status'] = 'complete'; return 0
        except InterruptedError:
            result['status'] = 'stopped_at_complete_window_or_timing_call'; return 130
        except Exception as error:
            result.update(status='failed_original_unchanged', error=repr(error)); raise
        finally:
            predictor.close()
            if (tensor_state_fingerprint(body.state_dict()) != before or base.sha256(source) != original_sha or
                    base.sha256(source_report_path) != source_report_sha):
                raise RuntimeError('read-only source/model changed')
            publish(); print(summary(result), flush=True)


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event=event) or 0)
