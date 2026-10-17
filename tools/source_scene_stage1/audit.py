"""CPU-only GT/STC last-pair identity/velocity/registration diagnostic."""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import signal
import threading
import numpy as np
from tools.joint_occ_experiments import run as base
from real_motion.nuscenes_adapter import sample_ego_to_world
from real_motion.strong_w2det import StrongW2DetConfig
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from .association import inspect_pair, aggregate, boxes_for_sample, PURITY, SECONDARY

PROTOCOL = 'read_only_GT_box_attributed_original_pair_audit_v1'


def summary(result):
    lines = ['===== ORIGINAL LAST-PAIR / GT BOX IDENTITY AUDIT =====',
        'status='+result['status'], f'windows={result["windows"]}/{len(result["contract"]["keys"])}',
        'CPU only. Same-class WORLD-XYZ mutual-nearest, nominal dt=.5 / gate25mps unchanged.',
        'GT boxes/instance_token used ONLY after causal matching. No future occupancy/model/training.',
        'GT and STC same history tokens/poses. Box attribution is approximate; unresolved support excluded from conditional precision.',
        f'Fixed diagnostic box purity>={PURITY}; secondary support>={SECONDARY} unresolved; no threshold search.']
    for kind, values in result['metrics'].items():
        lines += ['===== '+kind+' =====', json.dumps(values, ensure_ascii=False)]
    lines += ['Velocity error only for correctly attributed pairs. Recall uses distinct identifiable instances in BOTH frames.',
              'No association/gate/KTA corrections, automatic promotion or quality/FPS claim.']
    return '\n'.join(lines)+'\n'


def main(argv=None, event=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataroot', 'dev-cache', 'config', 'stc-root', 'out'): p.add_argument('--'+name, required=True)
    p.add_argument('--windows', type=int, default=128)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args(argv); event = event or threading.Event()
    if not 1 <= a.windows <= 4369: raise ValueError('positive bounded window count required')
    pcfg = base.make_prepare_config(base.load_runtime_config(a.config, []))
    if pcfg.frame_dt_s != .5: raise ValueError('original nominal dt=.5 protocol required')
    strong = StrongW2DetConfig(free_label=17)
    _, records = base.load_cache(a.dev_cache); base.record_keys(records)
    if len(records) != 4369: raise ValueError('original VAL4369 metadata required')
    keys = base.select_scene_balanced_round_robin(base.record_keys(records), a.windows)
    chosen = base.align_records(records, keys)
    data = Path(a.dataroot).resolve(); stc = Path(a.stc_root).resolve(); out = Path(a.out).resolve()
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(data), verbose=False)
    inputs = []
    for row in chosen:
        for token in row['history_tokens'][-2:]:
            for kind, root in (('gt', data/'gts'), ('stc', stc)):
                path = root/row['scene_name']/str(token)/'labels.npz'
                stat = path.stat(); inputs.append((str(path), stat.st_size, stat.st_mtime_ns))
    contract = dict(protocol=PROTOCOL, keys=[list(k) for k in keys], matcher=asdict(strong),
        prepare=asdict(pcfg), nominal_dt=.5, attribution=dict(purity=PURITY, secondary=SECONDARY),
        files=sorted(set(inputs)), config_sha=base.sha256(a.config), dev_cache_sha=base.sha256(a.dev_cache),
        metadata_sha={name: base.sha256(data/'v1.0-trainval'/name) for name in
            ('sample_annotation.json', 'sample.json', 'instance.json', 'category.json', 'sample_data.json', 'ego_pose.json')},
        implementation={str(path.relative_to(Path(__file__).resolve().parents[2])): base.sha256(path) for path in (
            Path(__file__), Path(__file__).with_name('association.py'),
            Path(__file__).resolve().parents[2]/'real_motion/shared_source_track.py',
            Path(runtime.__file__), Path(__file__).resolve().parents[2]/'real_motion/source_evidence_audit.py')})
    contract = json.loads(json.dumps(contract)); windows = []
    lock = out.parent/'.pair_audit_locks'/base.stable_json_fingerprint(str(out)); lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(lock):
        if not a.resume: out.mkdir(parents=True, exist_ok=False)
        else:
            saved = json.loads((out/'results.json').read_text(encoding='utf-8'))
            if saved['contract'] != contract: raise RuntimeError('audit resume population/data/code differs')
            windows = saved['rows']
            if [w['key'] for w in windows] != contract['keys'][:len(windows)]: raise RuntimeError('invalid complete-window prefix')
        def publish(status):
            result = dict(status=status, contract=contract, windows=len(windows), rows=windows,
                          metrics={kind: aggregate([w[kind] for w in windows]) for kind in ('gt', 'stc')})
            base.write_json(out/'results.json', result)
            (out/'summary.txt').write_text(summary(result), encoding='utf-8')
            return result
        for row in chosen[len(windows):]:
            if event.is_set(): break
            tokens = [str(t) for t in row['history_tokens'][-2:]]
            poses = [sample_ego_to_world(nusc, t) for t in tokens]
            times = [nusc.get('sample', t)['timestamp']/1e6 for t in tokens]
            boxes = [boxes_for_sample(nusc, t) for t in tokens]
            item = dict(key=[str(row['scene_name']), str(row['t0_token'])])
            for kind, root in (('gt', data/'gts'), ('stc', stc)):
                frames = []
                for token, pose in zip(tokens, poses):
                    with np.load(root/row['scene_name']/token/'labels.npz', allow_pickle=False) as payload:
                        sem = payload['semantics']
                        if sem.shape != tuple(pcfg.grid.shape_hwd) or sem.dtype.kind not in 'iu' or np.any(sem > 17) or np.any(sem < 0):
                            raise ValueError('semantic grid contract differs')
                        frames.append(runtime.extract_instances_cropped_exact(sem, pose, grid=pcfg.grid, cfg=strong))
                item[kind] = inspect_pair(*frames, *boxes, *poses, pcfg.grid,
                    actual_dt=times[1]-times[0], dt=.5, speed_gate=strong.max_match_speed_mps)
            windows.append(item)
            if len(windows) % 8 == 0:
                publish('running'); print(f'PAIR_AUDIT {len(windows)}/{len(chosen)}', flush=True)
        result = publish('complete' if len(windows) == len(chosen) else 'stopped_at_complete_window')
        print(summary(result), flush=True)
    return 0


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    raise SystemExit(main(event=event))
