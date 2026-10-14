"""Profile released STC errors on frozen development scenes. No model/training."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import numpy as np
import scipy
from tools.prior_strength_report import resolve_result
from .data import Source, receipt, split, file_sha, fingerprint
from . import statistics as stats

PROTOCOL = 'STC_VAL_development_error_fit_four_history_v1'


def write(path, value):
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')), encoding='utf-8')
    os.replace(temp, path)


@contextmanager
def lock(out):
    with (out/'.lock').open('a+b') as f:
        if os.name == 'nt':
            import msvcrt
            if not f.tell(): f.write(b'0'); f.flush()
            f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def profile_scene(source, scene, event):
    tick = time.perf_counter(); history = deque(maxlen=4); total, ledger = {}, []
    frames = source.frames[scene]
    for i, frame in enumerate(frames):
        if event.is_set(): raise InterruptedError('save complete scenes only')
        row = source.read(frame)
        stats.accumulate(total, stats.frame_statistics(row['pred'], row['gt'], row['known']))
        objects, object_counts = stats.object_statistics(row['gt'], row['pred'], row['pose'], row['annotations'])
        stats.accumulate(total, dict(objects=object_counts))
        history.append(dict(gt=row['gt'], pred=row['pred'], pose=row['pose'], objects=objects))
        if len(history) == 4: stats.accumulate(total, dict(temporal=stats.temporal_statistics(list(history))))
        ledger.append(dict(token=frame.token, GT_semantics_sha256=row['GT_semantics_sha256'], GT_known_sha256=row['GT_known_sha256']))
        if (i+1) % 8 == 0: print(f'STC_NOISE scene={scene} frames={i+1}/{len(frames)} CPU_ONLY', flush=True)
    return dict(scene=scene, statistics=total, frame_ledger=ledger, seconds=time.perf_counter()-tick)


def checked_state(state, contract, source):
    if state.get('contract') != contract or not isinstance(state.get('scenes'), list): raise ValueError('profile resume contract changed')
    fit = sorted(source.frames)
    if len(state['scenes']) > len(fit): raise ValueError('invalid completed-scene cursor')
    for row, scene in zip(state['scenes'], fit):
        frames = source.frames[scene]; counts = row['statistics']
        if (row['scene'] != scene or [v['token'] for v in row['frame_ledger']] != [f.token for f in frames]
                or counts['frames'] != len(frames) or counts['temporal']['windows4'] != len(frames)-3):
            raise ValueError('saved scene/frame prefix differs')
        # Check conditional/components/temporal counters, not just a cursor.
        stats.validate(counts, int(np.prod(stats.SHAPE)))
        if not np.isfinite(row['seconds']) or row['seconds'] < 0: raise ValueError('saved elapsed time invalid')


def summary(result):
    s = result['statistics']; c = np.asarray(s['confusion'], np.int64)
    fp, fn = int(c[17, :17].sum()), int(c[:17, 17].sum())
    predicted, truth, correct_occ = int(c[:, :17].sum()), int(c[:17].sum()), int(c[:17, :17].sum())
    pct = lambda a, b: f'{100*a/b:.3f}%' if b else 'NA'
    lines = ['===== STC REAL ERROR PROFILE / DEVELOPMENT SCENES =====',
        f"status=complete; fit_scenes={len(result['split']['fit_scenes'])}; remaining_scenes={len(result['split']['remaining_scenes'])}; frames={s['frames']}",
        'CPU only; no model/GPU/train/forecast. Same-frame VAL GT labels are used for error fitting.',
        f'FP={fp} FN={fn}; occupied_precision={pct(correct_occ,predicted)} occupied_recall={pct(correct_occ,truth)}',
        f'P(occupied prediction | GT free)={pct(fp,c[17].sum())}; P(free prediction | GT occupied)={pct(fn,truth)}',
        'family voxels components component_size_vox_P50/P90 bbox_extent_m_P50_X/Y/Z']
    for family, v in s['families'].items():
        quantiles = stats.weighted_quantiles(v['component_sizes'])
        extent_hist = [{}, {}, {}]
        for signature, n in v['component_joint'].items():
            fields = [int(x) for x in signature.split(',')]
            for axis in range(3):
                key = str(fields[axis+2]); extent_hist[axis][key] = extent_hist[axis].get(key, 0)+n
        dims = [stats.weighted_quantiles(h, (.5,))[0] for h in extent_hist]
        dims = [round(x*.4, 3) if x is not None else None for x in dims]
        lines.append(f"{family} {v['voxels']} {v['components']} {quantiles} {dims}")
        lines.append('  boundary_distance_hist='+str(v['boundary_distance_voxel_hist']))
    lines.append('distance bins m=[0,10,20,30,40,60,inf]; GT distance uses horizontal voxel-center range')
    dc = np.asarray(s['distance_confusion'])
    for i, m in enumerate(dc):
        fp, fn = int(m[17, :17].sum()), int(m[:17, 17].sum())
        lines.append(f"  bin{i}: FP_rate={pct(fp,m[17].sum())} FN_rate={pct(fn,m[:17].sum())}")
    for name in ('historical_known_confusion', 'historical_unknown_confusion'):
        m = np.asarray(s[name]); lines.append(f"{name}: voxels={m.sum()} FP={m[17,:17].sum()} FN={m[:17,17].sum()}")
    t = s['temporal']; lines.append(f"four-history windows={t['windows4']}; stable_static_overlap_voxels={t['static_stable_overlap_voxels']}")
    lines.append('Static bit pattern: bit0 oldest, bit3 latest; excludes GT dynamic classes/changed GT labels/out-of-grid.')
    for family, h in t['static_patterns'].items():
        n = sum(h[1:]); lengths = [sum(h[b] for b in range(1, 16) if b.bit_count() == k) for k in range(1, 5)]
        lines.append(f'  {family}: any_error={n} errors_in_1/2/3/4_frames={lengths}; patterns16={h}')
    o = s['objects']; d = t['dynamic']
    lines += [f"dynamic annotated boxes={o['annotation_boxes']}; GT_supported={o['GT_supported_boxes']}; missing_STC={o['missing_STC_boxes']}",
        'object-local STC-minus-GT centroid offset hist XYZ='+str(o['offset_xyz_hist']),
        'adjacent residual delta XY hist='+str(d['delta_offset_xy_hist'])+f"; tracks={d['residual_pair_tracks']}",
        'four-history STC missing-pattern16='+str(d['missing_STC_patterns'])+f"; GT_supported_all4={d['tracks_GT_supported_all4']}",
        'Annotation-crop residual is not a predicted-instance matcher, true velocity, or pure sensor noise.',
        'Four-frame patterns are overlapping conditional measurements, not full object/error lifetimes.',
        'Remaining-scene labels were not read. Prior full VAL scores were inspected; remaining is NOT a pristine independent test.',
        'noise_model.json contains distributions/counts only: no VAL voxel shapes/tokens copied; NO simulator/training yet.',
        f"elapsed_seconds={result['elapsed_seconds']:.3f}; quality/statistics time, NOT FPS"]
    return '\n'.join(lines)+'\n'


def main(argv=None, event=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('dataroot', 'stc-root', 'out-dir'): p.add_argument('--'+key, required=True)
    p.add_argument('--prior-result'); p.add_argument('--workers', type=int, default=2); p.add_argument('--resume', action='store_true')
    a = p.parse_args(argv); event = event or threading.Event()
    if not 1 <= a.workers <= 4: raise ValueError('bounded 1--4 CPU workers required')
    prior_path = resolve_result(a.prior_result); prior_sha = file_sha(prior_path)
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    extraction, sha, all_scenes = receipt(a.stc_root); development = split(prior, all_scenes)
    if sha != prior['contract']['STC_receipt_sha256']: raise ValueError('original prior diagnostic STC release changed')
    source = Source(a.dataroot, a.stc_root, development, extraction)
    out = Path(a.out_dir).resolve()
    for root in (Path(a.dataroot).resolve(), Path(a.stc_root).resolve(), prior_path.parent):
        if out == root or root in out.parents: raise ValueError('new output must be outside source data/results')
    if not a.resume: out.mkdir(parents=True, exist_ok=False)
    elif not out.is_dir(): raise FileNotFoundError('resume output directory missing')
    files = sorted(Path(__file__).parent.glob('*.py'))
    contract = dict(protocol=PROTOCOL, prior_result=str(prior_path), prior_sha256=prior_sha, STC_receipt_sha256=sha,
        source_model_declared_sha256=prior['contract']['body_sha256'], split=development,
        metadata=source.metadata, input_inventory_sha256=fingerprint(source.inventory),
        grid=dict(shape=list(stats.SHAPE), origin=list(stats.ORIGIN), step=list(stats.STEP)),
        implementation={f.name: file_sha(f) for f in files}, workers=a.workers,
        numpy=np.__version__, scipy=scipy.__version__, python=sys.version.split()[0],
        scientific_updates=0, fitting_uses_VAL_labels=True)
    begun = time.perf_counter()
    with lock(out):
        cp, sp = out/'contract.json', out/'state.json'
        if a.resume and json.loads(cp.read_text(encoding='utf-8')) != contract: raise ValueError('resume inputs/split/code changed')
        write(cp, contract); write(out/'split.json', development)
        state = json.loads(sp.read_text(encoding='utf-8')) if a.resume and sp.exists() else dict(contract=contract, scenes=[])
        checked_state(state, contract, source); write(sp, state)
        pending = sorted(source.frames)[len(state['scenes']):]
        try:
            with ThreadPoolExecutor(max_workers=a.workers) as pool:
                for row in pool.map(lambda name: profile_scene(source, name, event), pending):
                    state['scenes'].append(row); checked_state(state, contract, source); write(sp, state)
                    print(f"STC_NOISE_SCENES {len(state['scenes'])}/{len(source.frames)}", flush=True)
        except InterruptedError:
            write(out/'status.json', dict(status='stopped_at_complete_scene')); return 130
        if len(state['scenes']) != len(source.frames): raise ValueError('complete fitting scene population required')
        source.check_inventory()
        if file_sha(prior_path) != prior_sha or receipt(a.stc_root)[1] != sha: raise RuntimeError('source provenance changed during fit')
        total = {}
        for row in state['scenes']: stats.accumulate(total, row['statistics'])
        stats.validate(total, int(np.prod(stats.SHAPE)))
        result = dict(status='complete', protocol=PROTOCOL, split=development, contract=contract, statistics=total,
            scientific_updates=0, elapsed_seconds=time.perf_counter()-begun,
            accumulated_scene_compute_seconds=sum(r['seconds'] for r in state['scenes']))
        model = stats.noise_model(total)
        model.update(fitting_split={k: v for k, v in development.items() if k != 'seed_DEV64_keys'},
            contract_sha256=fingerprint(contract), bins=dict(
            distance_m=[None if not np.isfinite(x) else x for x in stats.DISTANCE],
            boundary_m=[None if not np.isfinite(x) else x for x in stats.BOUNDARY],
            offset_m=[None if not np.isfinite(x) else x for x in stats.OFFSET],
            jitter_m=[None if not np.isfinite(x) else x for x in stats.JITTER],
            voxel_ratio=[None if not np.isfinite(x) else x for x in stats.RATIO]),
            component_joint_fields=['semantic_class', 'radial_distance_bin', 'extent_x_vox', 'extent_y_vox', 'extent_z_vox', 'size_vox'])
        write(out/'noise_model.json', model); write(out/'evaluation.json', result)
        (out/'summary.txt').write_text(summary(result), encoding='utf-8')
        print(summary(result), flush=True); return 0


if __name__ == '__main__':
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: stop.set())
    sys.exit(main(event=stop) or 0)
