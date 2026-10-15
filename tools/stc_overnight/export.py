"""CPU-only portable replay bundle: frozen models, selected labels/poses, no raw sensors."""
import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import torch
from tools.joint_occ_experiments import run as base
from tools.dense_valid_epoch3.run import select_source, validate_source
from real_motion.nuscenes_adapter import NuScenesWindowSource

PROTOCOL = 'stc_overnight_portable_real_replay_v1'


def choose_records(dev, train, reference_keys, val_count=64, train_count=32):
    if len(reference_keys) != 64 or len(set(map(tuple, reference_keys))) != 64: raise ValueError('exact reference64 required')
    reference = base.align_records(dev, reference_keys); held_scenes = {str(r['scene_name']) for r in reference}
    # Full-VAL export includes additional windows from reference scenes, but
    # those scenes remain HELD OUT from all local fitting/selection.
    if val_count == len(dev):
        ref = set(map(tuple, reference_keys))
        remaining = [r for r in dev if (str(r['scene_name']), str(r['t0_token'])) not in ref]
    else: remaining = [r for r in dev if str(r['scene_name']) not in held_scenes]
    needed = val_count-64
    if needed < 0 or train_count < 1 or len(remaining) < needed or len(train) < train_count: raise ValueError('insufficient distinct export windows')
    extras = base.align_records(remaining, base.select_scene_balanced_round_robin(base.record_keys(remaining), needed)) if needed else []
    clean = base.align_records(train, base.select_scene_balanced_round_robin(base.record_keys(train), train_count))
    if {str(r['scene_name']) for r in clean} & {str(r['scene_name']) for r in dev}: raise ValueError('TRAIN/VAL overlap')
    rows = [(r, 'reference64', 'val') for r in reference]
    for record in extras:
        if str(record['scene_name']) in held_scenes:
            rows.append((record, 'reference_scene', 'val')); continue
        holdout = int(hashlib.sha256(('stc_overnight_split_v1:'+str(record['scene_name'])).encode()).hexdigest(), 16) % 4 == 0
        rows.append((record, 'stc_selection' if holdout else 'stc_fit', 'val'))
    rows += [(r, 'clean_train', 'train') for r in clean]
    return rows


def export_weight(path, name, out, *, required=False):
    if not path or not Path(path).is_file():
        if required: raise RuntimeError('required checkpoint missing: '+str(path))
        print(f'OPTIONAL_MODEL_MISSING {name}: {path}', flush=True); return None
    path = Path(path).resolve(); sha = base.sha256(path)
    value = torch.load(path, map_location='cpu', weights_only=False)
    if name == 'dense_epoch2': validate_source(value)
    if set(value.get('models', {})) != {'gt'}: raise RuntimeError('GT body checkpoint required: '+str(path))
    configs = value['contract']['model']; state = value['models']['gt']
    if not state or any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in state.values()):
        raise RuntimeError('invalid/nonfinite model weights')
    target = out/'models'/f'{name}.pt'; target.parent.mkdir(exist_ok=True)
    torch.save(dict(protocol=PROTOCOL, model_configs=configs, state_dict=state, source_protocol=value['protocol'],
        source_sha256=sha, source_checkpoint=str(path), source_update=value.get('updates'),
        source_contract=value['contract'], evaluation_only=True, optimizer_included=False), target)
    if base.sha256(path) != sha: raise RuntimeError('source checkpoint changed during export')
    return dict(role=name, file=str(target.relative_to(out)).replace('\\', '/'), sha256=base.sha256(target), source_sha256=sha)


def compact_label(path):
    with np.load(path, allow_pickle=False) as z:
        sem = z['semantics']
        if sem.dtype.kind not in 'ui' or sem.ndim != 3 or np.any(sem < 0) or np.any(sem > 255):
            raise ValueError('lossless uint8 semantic export required: '+str(path))
        values = dict(semantics=sem.astype(np.uint8, copy=False))
        if 'mask_lidar' in z:
            mask = np.asarray(z['mask_lidar'], bool)
            if mask.shape != sem.shape: raise ValueError('mask/semantic shape differs')
            values.update(mask_lidar_packed=np.packbits(mask.reshape(-1)), mask_lidar_shape=np.asarray(mask.shape, np.int32))
    buffer = io.BytesIO(); np.savez_compressed(buffer, **values)
    return buffer.getvalue()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dense-dir'); p.add_argument('--clean-checkpoint'); p.add_argument('--old-checkpoint')
    for name in ('out-dir', 'config', 'dataroot', 'dev-info', 'dev-cache', 'train-cache', 'stc-root'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--val-windows', type=int, default=64); p.add_argument('--train-windows', type=int, default=32)
    a = p.parse_args(argv); root = Path(__file__).resolve().parents[2]; out = Path(a.out_dir).resolve()
    torch.set_num_threads(1)
    dense = select_source(a.dense_dir, root/'outputs/p0_f9_shared_history')
    if out == dense.parent or dense.parent in out.parents: raise ValueError('NEW bundle directory outside source run required')
    source = torch.load(dense, map_location='cpu', weights_only=False); contract = validate_source(source); del source
    for name in ('config', 'dev_cache', 'dev_info', 'train_cache'):
        if base.sha256(getattr(a, name)) != contract['data'][name]: raise RuntimeError('source metadata differs: '+name)
    _, dev = base.load_cache(a.dev_cache); _, train = base.load_cache(a.train_cache)
    if len(dev) != 4369 or len(train) != 20430: raise RuntimeError('original full VAL4369/TRAIN20430 caches required')
    selected = choose_records(dev, train, contract['eval_keys'], a.val_windows, a.train_windows)
    # One complete nuScenes metadata reader. Only per-token ego poses/timestamps are exported.
    reader = NuScenesWindowSource(a.dataroot, info_pkl=a.dev_info)
    records, frame_paths = [], {}
    for record, role, split in selected:
        scene = str(record['scene_name']); hist = list(map(str, record['history_tokens'][-4:])); future = list(map(str, record['future_tokens']))
        if len(hist) != 4 or len(future) != 6 or hist[-1] != str(record['t0_token']): raise RuntimeError('FOUR -> SIX identity mismatch')
        row = dict(scene_name=scene, t0_token=str(record['t0_token']), history_tokens=hist, future_tokens=future,
            role=role, split=split, history_times_s=[reader.nusc.get('sample', t)['timestamp']/1e6 for t in hist],
            history_poses=[reader.pose(t).tolist() for t in hist], future_poses=[reader.pose(t).tolist() for t in future])
        records.append(row)
        for token in hist+future:
            frame_paths[f'frames/gt/{scene}/{token}/labels.npz'] = Path(a.dataroot)/'gts'/scene/token/'labels.npz'
        if split == 'val':
            for token in hist: frame_paths[f'frames/stc/{scene}/{token}/labels.npz'] = Path(a.stc_root)/scene/token/'labels.npz'
    inventory = []
    for target, path in sorted(frame_paths.items()):
        st = path.stat()
        if not path.is_file() or not st.st_size: raise RuntimeError('missing/empty frame: '+str(path))
        inventory.append(dict(file=target, source_size=st.st_size, source_sha256=base.sha256(path), source_mtime_ns=st.st_mtime_ns))
    total = sum(r['source_size'] for r in inventory)
    print(f'BUNDLE_INPUT windows={len(records)} unique_frames={len(inventory)} original_frame_bytes={total} ({total/2**20:.1f}MiB); compact export; CPU ONLY', flush=True)
    out.mkdir(parents=True, exist_ok=False)
    weights = [export_weight(dense, 'dense_epoch2', out, required=True)]
    clean = a.clean_checkpoint
    if not clean:
        options = list((root/'outputs/p0_f9_shared_history').glob('stc_clean_continue1000_*/train/last.pt'))
        if options: clean = max(options, key=lambda x: x.stat().st_mtime_ns)
    old = a.old_checkpoint  # Small default: only dense epoch2 + clean1000; no third model.
    for path, name in ((clean, 'clean1000'), (old, 'original_gt_epoch2')):
        weight = export_weight(path, name, out)
        if weight: weights.append(weight)
    pcfg = base.make_prepare_config(base.load_runtime_config(a.config, contract.get('overrides')))
    counts = {role: sum(r['role'] == role for r in records)
              for role in ('reference64', 'reference_scene', 'stc_fit', 'stc_selection', 'clean_train')}
    manifest = dict(protocol=PROTOCOL, source_checkpoint=str(dense), source_sha256=base.sha256(dense),
        prepare_config=asdict(pcfg), records=records, inventory=inventory, weights=weights, counts=counts,
        full_val_keys=[list(k) for k in base.record_keys(dev)] if a.val_windows == len(dev) else None,
        reference64_keys=contract['eval_keys'],
        policy='VAL supervised adaptation possible; fit/selection/reference scenes disjoint; NOT zero-shot or pristine independent test',
        no_raw_sensors=True, no_optimizer=True, future_labels_are_supervision_only=True,
        export_implementation={str(f.relative_to(root)): base.sha256(f) for f in sorted(Path(__file__).parent.glob('*.py'))})
    manifest['label_encoding'] = 'lossless_uint8_semantics_plus_packbits_lidar_no_camera_mask'
    archive = out/'stc_overnight_bundle.zip'
    with zipfile.ZipFile(archive, 'w', allowZip64=True) as z:
        for weight in weights:
            z.write(out/weight['file'], weight['file'], compress_type=zipfile.ZIP_DEFLATED, compresslevel=1)
        for i, entry in enumerate(inventory, 1):
            path = frame_paths[entry['file']]; st = path.stat()
            if (st.st_size, st.st_mtime_ns) != (entry['source_size'], entry['source_mtime_ns']): raise RuntimeError('source frame changed')
            payload = compact_label(path)
            entry.update(size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
            z.writestr(entry['file'], payload, compress_type=zipfile.ZIP_STORED)
            if i % 128 == 0 or i == len(inventory): print(f'BUNDLE_ZIP {i}/{len(inventory)}', flush=True)
        z.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False), compress_type=zipfile.ZIP_DEFLATED, compresslevel=1)
    base.write_json(out/'manifest.json', manifest)
    digest = base.sha256(archive)
    report = dict(archive=str(archive), bytes=archive.stat().st_size, sha256=digest, windows=counts, models=[w['role'] for w in weights])
    base.write_json(out/'bundle_receipt.json', report)
    print('DOWNLOAD_THIS_FILE: '+str(archive), flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == '__main__': main()
