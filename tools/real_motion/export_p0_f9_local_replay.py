#!/usr/bin/env python3
"""Read-only export of 16 TRAIN + 16 dev real windows for local CUDA replay."""
import sys
from pathlib import Path
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import gc
import io
import json
import os
import platform
import shutil
import subprocess
import time

import numpy as np
import torch
import yaml

from real_motion.local_replay_bundle import (BundleWriter, ReplayBundle, file_digest, fingerprint,
    pack_window, select_records, validate_window)
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.prepared import load_nuscenes_window_raw
from real_motion.nuscenes_adapter import NuScenesWindowSource, gt_moving_support_sequence
from real_motion.source_repair_evidence import PROTOCOL as REPAIR_PROTOCOL
from real_motion.sparse_evidence_repair import SparseRepairHead
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import align_records, load_manifest
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from tools.real_motion.shared_evidence_pilot_common import moving_support_masks
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256
from tools.real_motion.train_p0_f9_causal_columns import record_keys
from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint
from tools.real_motion.v18_source_interaction_common import select_population


def check_student(saved, teacher_sha, config_sha, source_dim):
    contract = saved.get('contract', {})
    if (saved.get('protocol') != REPAIR_PROTOCOL or saved.get('transport_frozen') is not True
            or saved.get('deployable') is not False or contract.get('teacher_sha256') != teacher_sha
            or contract.get('config_fingerprint') != config_sha or contract.get('source_dim') != source_dim):
        raise RuntimeError('student/teacher/config protocol mismatch')
    cursor, successful = saved.get('cursor'), saved.get('successful')
    if (type(cursor) is not int or type(successful) is not int or not 0 <= successful <= cursor <= contract['schedule_steps']):
        raise RuntimeError('invalid student completed-update cursor')
    head = SparseRepairHead('local_consensus', source_dim=source_dim)
    head.load_state_dict(saved['head'], strict=True)
    if any(not torch.isfinite(v).all() for v in head.state_dict().values()):
        raise RuntimeError('nonfinite student weights')
    return dict(cursor=cursor, successful=successful, contract_fingerprint=stable_json_fingerprint(contract))


def git_state(repo):
    def run(*args):
        try:
            return subprocess.check_output(['git', '-c', 'safe.directory='+str(repo), *args],
                cwd=repo, text=True, stderr=subprocess.DEVNULL, timeout=15).strip()
        except (OSError, subprocess.SubprocessError): return None
    return dict(commit=run('rev-parse', 'HEAD'), status=run('status', '--porcelain'))


def log_tail(path, limit=512*2**10):
    """Bounded complete-line prefix of a live log tail; NOT a checkpoint boundary."""
    with Path(path).open('rb') as f:
        f.seek(0, os.SEEK_END); size = f.tell(); f.seek(max(0, size-limit)); block = f.read(limit)
    if size > limit: block = block.partition(b'\n')[2]
    if b'\n' in block: block = block[:block.rfind(b'\n')+1]
    else: block = b''
    return block


def main():
    p = argparse.ArgumentParser(description=__doc__); add_config_args(p)
    for name in ('checkpoint', 'base-checkpoint', 'train-cache', 'dev-cache', 'population-manifest',
                 'dataroot', 'train-info', 'dev-info', 'out-dir'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--pilot-dir', help='optional sparse migration directory (read-only)')
    p.add_argument('--train-windows', type=int, default=16)
    p.add_argument('--dev-windows', type=int, default=16)
    p.add_argument('--stress-per-split', type=int, default=4)
    p.add_argument('--cpu-workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=20261005)
    p.add_argument('--max-package-mib', type=int, default=4096, help='uncompressed archive cap, not dataset size')
    a = p.parse_args()
    for name in ('config', 'checkpoint', 'base_checkpoint', 'train_cache', 'dev_cache', 'population_manifest', 'train_info', 'dev_info'):
        if not Path(getattr(a, name) or '').is_file(): p.error('missing '+name)
    if not Path(a.dataroot).is_dir(): p.error('missing dataroot')
    if (a.cpu_workers < 1 or a.max_package_mib < 128 or not 0 <= a.stress_per_split < min(a.train_windows, a.dev_windows)
            or a.train_windows+a.dev_windows > 64): p.error('invalid bounded replay budgets')
    out = Path(a.out_dir).resolve()
    if out.exists(): p.error('refusing to overwrite existing output')
    out.parent.mkdir(parents=True, exist_ok=True)
    cap = a.max_package_mib*2**20
    if shutil.disk_usage(out.parent).free < cap+512*2**20:
        p.error('not enough free disk for the package cap plus 512MiB reserve; reduce --max-package-mib explicitly')
    if a.pilot_dir:
        pilot = Path(a.pilot_dir)
        if not (pilot/'migration_last.pt').is_file() or not (pilot/'contract.json').is_file():
            p.error('pilot directory needs migration_last.pt and contract.json; no silent fallback to an untrained head')
    torch.set_num_threads(1)
    begun = time.perf_counter(); out.mkdir(); writer = None
    status = dict(status='running',old_experiments='READ ONLY',cuda_used=False)
    try:
        cfg = load_runtime_config(a.config, a.override); config_sha = stable_json_fingerprint(cfg)
        pcfg = make_prepare_config(cfg)
        teacher_file = out/'epoch_0019.pt'; teacher_sha = snapshot_checkpoint(a.checkpoint, teacher_file)
        ck, teacher = load_joint(teacher_file, torch.device('cpu'), reference_sha=CLEAN_SHA256,
                                config_sha=config_sha, allow_diagnostic=True)
        if (teacher.transport.config.history_frames != 4 or ck['cursor_epoch'] != 19
                or ck['model_configs'].get('adaptive_context') is not None):
            raise RuntimeError('selected epoch19 Local / strict FOUR histories required')
        source_dim = teacher.columns.source_dim
        del teacher; gc.collect()
        base_file = out/'clean_e14.pt'; base_sha = snapshot_checkpoint(a.base_checkpoint, base_file)
        if base_sha != CLEAN_SHA256: raise RuntimeError('CleanE14 checkpoint changed')
        student_file = None; student_meta = None; student_contract = None
        if a.pilot_dir:
            student_file = out/'migration_last.pt'; student_sha = snapshot_checkpoint(pilot/'migration_last.pt', student_file)
            saved = torch.load(student_file, map_location='cpu', weights_only=False)
            student_meta = {**check_student(saved, teacher_sha, config_sha, source_dim), 'sha256': student_sha}
            student_contract = saved['contract']
            live_contract = json.loads((pilot/'contract.json').read_text(encoding='utf-8'))
            if stable_json_fingerprint(live_contract) != student_meta['contract_fingerprint']:
                raise RuntimeError('pilot contract file differs from immutable migration snapshot')
            del saved
        print('EXPORT checkpoint snapshots verified; hashing frozen data/info (CPU only)...', flush=True)
        paths = dict(train=a.train_cache, dev=a.dev_cache, train_info=a.train_info, dev_info=a.dev_info)
        expected = {**ck['cache_fingerprints'], 'train_info': ck['info_fingerprints']['train'], 'dev_info': ck['info_fingerprints']['dev']}
        provenance = {}
        for name, path in paths.items():
            digest = file_digest(path)
            if digest != expected[name]: raise RuntimeError('teacher/data fingerprint mismatch: '+name)
            provenance[name] = dict(path=str(Path(path).resolve()), sha256=digest)
        manifest, dev_keys, _ = load_manifest(a.population_manifest)
        if (manifest['manifest_fingerprint'] != ck['dev_manifest_fingerprint']
                or tuple(map(tuple, manifest['parent_keys'])) != tuple(map(tuple, ck['dev_keys']))):
            raise RuntimeError('frozen dev512 identity/order changed')
        selected = []
        for split, path in (('train', a.train_cache), ('dev', a.dev_cache)):
            _, records = load_cache(path); keys = record_keys(records)
            if split == 'train':
                if tuple(keys) != tuple(map(tuple, ck['train_keys'])): raise RuntimeError('TRAIN identity/order mismatch')
                train_keys = (student_contract['train_keys'] if student_contract else select_population(keys,
                    {s for s, _ in manifest['parent_keys']}, fraction=.2, seed=a.seed)[0])
                key_set = set(keys)
                if any(tuple(k) not in key_set for k in train_keys): raise RuntimeError('student TRAIN keys missing')
                rows = align_records(records, train_keys); count = a.train_windows
            else:
                rows = align_records(records, dev_keys); count = a.dev_windows
                if student_contract and tuple(map(tuple, student_contract['dev_keys'])) != tuple(map(tuple, dev_keys)):
                    raise RuntimeError('student dev population differs from frozen dev64')
            selected.extend((split, r, stratum) for r, stratum in select_records(rows, count, a.stress_per_split))
            del records, rows; gc.collect()
        if ({str(r['scene_name']) for s, r, _ in selected if s=='train'}
                & {str(r['scene_name']) for s, r, _ in selected if s=='dev'}):
            raise RuntimeError('TRAIN/dev scene overlap')
        resolved = out/'runtime.yaml'; resolved.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
        writer = BundleWriter(out/'replay.zip.partial', cap)
        for name, path in (('checkpoints/epoch_0019.pt', teacher_file), ('checkpoints/clean_e14.pt', base_file), ('runtime.yaml', resolved)):
            writer.add_file(name, path)
        if student_file:
            writer.add_file('checkpoints/migration_last.pt', student_file)
            writer.add_stream('pilot/contract.json', io.BytesIO(json.dumps(student_contract, ensure_ascii=False).encode()))
            for name in ('progress.jsonl', 'bundle.json', 'summary.txt'):
                if (pilot/name).is_file():
                    data = log_tail(pilot/name) if name=='progress.jsonl' else (pilot/name).read_bytes()
                    if len(data) > 2**20: raise RuntimeError('oversized pilot metadata: '+name)
                    writer.add_stream('pilot/'+name, io.BytesIO(data))
        windows = []; sources = {}
        for wi, (split, record, stratum) in enumerate(selected):
            if split not in sources:
                sources[split] = NuScenesWindowSource(a.dataroot,
                    info_pkl=a.train_info if split=='train' else a.dev_info, verbose=False)
            source = sources[split]; window = window_from_record(record)
            raw = load_nuscenes_window_raw(source, window, pcfg, include_gt=True,
                                          io_workers=a.cpu_workers, active_history_frames=4)
            moving = gt_moving_support_sequence(source.nusc, window.t0_token, window.future_tokens,
                tuple(.5*(h+1) for h in range(6)), grid=pcfg.grid, workers=a.cpu_workers)
            moving = moving_support_masks(moving, pcfg.grid.shape_hwd)
            inputs, labels = pack_window(record, raw, moving); validate_window(inputs, labels, pcfg.grid.shape_hwd)
            stem = f'windows/{wi:03d}'
            writer.add_tensors(stem+'/inputs.pt', inputs); writer.add_tensors(stem+'/labels.pt', labels)
            windows.append(dict(key=[str(record['scene_name']), str(record['t0_token'])], split=split, stratum=stratum,
                inputs=stem+'/inputs.pt', labels=stem+'/labels.pt', sources=len(record['features']),
                observed_road_sidewalk_voxels=int((np.isin(raw['history_occ'], (11, 13)) & raw['history_observed']).sum())))
            print(f'EXPORT window={wi+1}/{len(selected)} {split}/{stratum} sources={len(record["features"])}', flush=True)
            del inputs, labels, moving, raw
        metadata = dict(active_history_frames=4, future_frames=6, grid_shape=list(pcfg.grid.shape_hwd),
            raw_history_tokens='last FOUR of record history; legacy six-token metadata kept for official trajectory',
            windows=windows, teacher_sha256=teacher_sha, base_sha256=base_sha, config_fingerprint=config_sha,
            student=student_meta, source_dim=source_dim, data_provenance=provenance,
            population_manifest_fingerprint=manifest['manifest_fingerprint'], git=git_state(Path(__file__).resolve().parents[2]),
            versions=dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__),
            cached_learned_features=False, cached_geometry=False, frozen_baseline_predictions=False,
            trajectory='official known ego/planning conditioning, NOT future occupancy',
            label_isolation='future GT/motion labels/Moving support separate; target_source_mask_tube is causal t0 footprint',
            selection='scene/temporal identity sampling + high source-count stress; no GT-based window selection',
            pilot_logs='captured live diagnostic tail, NOT guaranteed same update boundary as immutable student snapshot',
            use='local correctness/performance development only; NOT quality validation, convergence or L40S speed prediction')
        writer.finish(metadata); writer = None
        check = ReplayBundle(out/'replay.zip.partial', max_bytes=cap)
        try:
            for i in range(len(windows)): check.window(i)
        finally: check.close()
        os.replace(out/'replay.zip.partial', out/'replay.zip')
        status.update(status='complete', windows=len(windows), archive=str(out/'replay.zip'),
            archive_mib=(out/'replay.zip').stat().st_size/2**20, sha256=file_digest(out/'replay.zip'),
            student=student_meta, seconds=time.perf_counter()-begun)
        print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
        print('仅下载上面的 replay.zip；旧模型/数据/缓存没有修改，不启动训练。', flush=True)
    except BaseException as error:
        status.update(status='failed', error=type(error).__name__+': '+str(error))
        raise
    finally:
        if writer is not None: writer.close()
        (out/'export_status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
