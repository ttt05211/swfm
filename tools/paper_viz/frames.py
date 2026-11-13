"""Paper-figure frames bundle: GT labels + ego poses only, FOUR history -> TWELVE future (6 s).

READ-ONLY on the server: it reads nuScenes metadata, Occ3D GT labels, the frozen
paper epoch2 checkpoint (contract/keys only) and writes ONE NEW directory.
Nothing is deleted or overwritten (the output directory must not exist).

Population (default --population long6): the official GenieDrive-aligned long
6 s VAL windows used by the paper's long-horizon evaluation, scene-balanced
round-robin down to --windows.  --population val: the VAL4369 cache windows that
have 12 contiguous futures in the same scene.  Reference64 windows are tagged
role=reference64 (they were inspected repeatedly; not an independent test).

Optional --moving adds packed GT Moving masks at 1..6 s (needs nuScenes boxes;
slow on 2 CPU cores).

    PYTHONPATH=$PWD:$PWD/upstream_occfm python -u -m tools.paper_viz.frames \
        --out-dir outputs/paper_viz/frames_$(date +%Y%m%d_%H%M%S) --windows 400

Read it back with tools.paper_viz.frames.FramesBundle (same API as the
stc_overnight Bundle: history(), targets(), frame(), records, pcfg).
"""
import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
import numpy as np
import torch

PROTOCOL = 'paper_viz_frames_v1'
ROOT = Path(__file__).resolve().parents[2]
NUSCENES_ROOT = Path('/root/nas/occ/OccFM-NeurIPS2025-main/data/nuscenes')
CHECKPOINT = ROOT/'outputs/p0_f9_shared_history/adaln_forward_full_2ep_20261026_215351/epoch_0002.pt'
FUTURE = 12
MOVING_AT = (1, 3, 5, 7, 9, 11)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def compact_label(path):
    from tools.stc_overnight.export import compact_label as compact  # identical lossless encoding
    return compact(path)


def iter_json_array(path, chunk=1 << 22):
    """Stream objects of a top-level JSON array without loading the file (low RAM)."""
    dec = json.JSONDecoder(); buf = ''; started = False
    with open(path, 'r', encoding='utf-8') as f:
        while True:
            data = f.read(chunk); buf += data; pos = 0
            while True:
                while pos < len(buf) and buf[pos] in ' \t\r\n,': pos += 1
                if not started and pos < len(buf):
                    if buf[pos] != '[': raise ValueError('top-level JSON array expected: '+str(path))
                    started = True; pos += 1; continue
                if pos < len(buf) and buf[pos] == ']': return
                try: obj, end = dec.raw_decode(buf, pos)
                except json.JSONDecodeError:
                    if not data: raise
                    break
                yield obj; pos = end
            buf = buf[pos:]
            if not data: return


class LiteNusc:
    """Minimal read-only nuScenes tables (scene, sample, LIDAR_TOP keyframe ego poses): ~100 MB RAM
    instead of the multi-GB full devkit.  Same records/fields as NuScenes.get for these tables."""
    def __init__(self, dataroot, version='v1.0-trainval'):
        meta = Path(dataroot)/version; t = time.time()
        self.scene = json.loads((meta/'scene.json').read_text())
        self._scene = {r['token']: r for r in self.scene}
        self._sample = {r['token']: r for r in json.loads((meta/'sample.json').read_text())}
        lidar = {}
        for r in iter_json_array(meta/'sample_data.json'):
            if r.get('is_key_frame') and '/LIDAR_TOP/' in r['filename'] and r['sample_token'] in self._sample:
                lidar[r['sample_token']] = dict(token=r['token'], ego_pose_token=r['ego_pose_token'])
        need = {v['ego_pose_token'] for v in lidar.values()}
        self._ego = {r['token']: r for r in iter_json_array(meta/'ego_pose.json') if r['token'] in need}
        self._sd = {v['token']: v for v in lidar.values()}
        for token, rec in self._sample.items():
            if token in lidar: rec['data'] = dict(LIDAR_TOP=lidar[token]['token'])
        if len(lidar) != len(self._sample) or len(self._ego) != len(need):
            raise RuntimeError(f'incomplete LIDAR_TOP keyframes: samples={len(self._sample)} lidar={len(lidar)} ego={len(self._ego)}')
        print(f'LITE_NUSCENES samples={len(self._sample)} scenes={len(self.scene)} {time.time()-t:.0f}s', flush=True)

    def get(self, table, token):
        return {'scene': self._scene, 'sample': self._sample, 'sample_data': self._sd, 'ego_pose': self._ego}[table][token]


class LiteSource:
    """NuScenesWindowSource subset used here: nusc, dataroot, pose(), allowed_scenes, info_by_token."""
    def __init__(self, dataroot, info_pkl):
        import pickle
        self.dataroot = str(Path(dataroot).resolve()); self.nusc = LiteNusc(dataroot)
        with open(info_pkl, 'rb') as f: infos = pickle.load(f)['infos']
        self.allowed_scenes = set(infos.keys())
        self.info_by_token = {rec['token']: dict(token=rec['token']) for frames in infos.values() for rec in frames}
        del infos

    def pose(self, token):
        from real_motion.nuscenes_adapter import sample_ego_to_world
        return sample_ego_to_world(self.nusc, token)


def long_windows(nusc, records):
    """VAL cache windows extended to 12 contiguous futures inside the same scene."""
    from real_motion.nuscenes_adapter import WindowTokens
    out = []
    for r in records:
        hist = tuple(map(str, r['history_tokens'][-4:])); future = list(map(str, r['future_tokens']))
        token = future[-1]
        while len(future) < FUTURE:
            token = nusc.get('sample', token)['next']
            if not token: break
            future.append(token)
        if len(future) == FUTURE:
            out.append(WindowTokens(str(r['scene_name']), hist, str(r['t0_token']), tuple(future)))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--windows', type=int, default=400, help='0 = whole population')
    p.add_argument('--population', choices=('long6', 'val'), default='long6')
    p.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    p.add_argument('--config', type=Path, default=ROOT/'configs/real_motion_occfm.yaml')
    p.add_argument('--dataroot', type=Path, default=NUSCENES_ROOT)
    p.add_argument('--dev-info', type=Path)
    p.add_argument('--dev-cache', type=Path, default=ROOT/'data/p0_f9_v18_se2_val_all_4369.pt')
    p.add_argument('--geniedrive-info', type=Path, default=ROOT/'data/geniedrive/world-nuscenes_infos_val.pkl')
    p.add_argument('--moving', action='store_true', help='also export GT Moving masks (slow)')
    p.add_argument('--moving-workers', type=int, default=2)
    p.add_argument('--dry-run', action='store_true', help='select + size estimate only, write nothing')
    a = p.parse_args(argv)
    torch.set_num_threads(1)
    from tools.causal_forward_all.common import validate_source
    from real_motion.runtime_config import load_runtime_config, make_prepare_config
    from real_motion.v21_source_induction import select_scene_balanced_round_robin
    from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
    from tools.final_semantic_evaluation.sources import NuscInputs, nusc_jobs

    out = a.out_dir.resolve(); ckpt = a.checkpoint.resolve()
    if out.exists(): raise SystemExit('refusing: --out-dir already exists (nothing is ever overwritten)')
    if ckpt.parent == out or ckpt.parent in out.parents: raise SystemExit('--out-dir must be outside the checkpoint run')
    info = (a.dev_info or a.dataroot/'nuscenes_infos_val_temporal_v3_scene.pkl').resolve()
    saved = torch.load(ckpt, map_location='cpu', weights_only=False); contract = validate_source(saved); del saved
    for name, path in (('config', a.config), ('dev_info', info), ('dev_cache', a.dev_cache)):
        if sha256(path) != contract['data'][name]: raise SystemExit(f'{name} differs from the paper model contract: {path}')
    pcfg = make_prepare_config(load_runtime_config(a.config, contract.get('overrides')))
    t = time.time()
    if a.moving:  # GT Moving needs boxes -> full devkit (several GB RAM; not for a 4 GB box)
        inputs = NuscInputs(a.dataroot, info); reader = inputs.source
    else:
        inputs = None; reader = LiteSource(a.dataroot, info)
    nusc = reader.nusc
    print(f'NUSCENES_LOADED {time.time()-t:.0f}s lite={not a.moving}', flush=True)
    if a.population == 'long6':
        from tools.real_motion import geniedrive_eval_alignment as genie, joint_long_rollout_common as rollout
        pairs, _ = genie.select_population(a.geniedrive_info, reader, [])  # no cache needed: tokens from official info
        jobs = [w for w, _ in pairs]
        for w in jobs: rollout.validate_timestamps(nusc, w)
    else:
        _, records = load_cache(a.dev_cache)
        if len(records) != 4369: raise SystemExit('VAL4369 cache required')
        jobs = long_windows(nusc, records); del records
    train_scenes = {str(k[0]) for k in contract.get('train_keys', [])}
    if {w.scene_name for w in jobs} & train_scenes: raise SystemExit('TRAIN/VAL scene overlap')
    by = {(w.scene_name, w.t0_token): w for w in jobs}
    keys = list(by) if not a.windows or a.windows >= len(by) else \
        [tuple(k) for k in select_scene_balanced_round_robin(list(by), a.windows)]
    reference = {tuple(map(str, k)) for k in contract['eval_keys']}
    rows, frames = [], {}
    for key in keys:
        w = by[key]
        if len(w.history_tokens) != 4 or len(w.future_tokens) != FUTURE or w.history_tokens[-1] != w.t0_token:
            raise RuntimeError('FOUR -> TWELVE identity mismatch: '+str(key))
        rows.append(dict(scene_name=w.scene_name, t0_token=w.t0_token, history_tokens=list(w.history_tokens),
            future_tokens=list(w.future_tokens), role='reference64' if key in reference else a.population,
            split='val', history_times_s=[nusc.get('sample', x)['timestamp']/1e6 for x in w.history_tokens],
            future_times_s=[nusc.get('sample', x)['timestamp']/1e6 for x in w.future_tokens],
            history_poses=[reader.pose(x).tolist() for x in w.history_tokens],
            future_poses=[reader.pose(x).tolist() for x in w.future_tokens]))
        for x in (*w.history_tokens, *w.future_tokens):
            frames[f'frames/gt/{w.scene_name}/{x}/labels.npz'] = a.dataroot/'gts'/w.scene_name/x/'labels.npz'
    inventory = []
    for name, path in sorted(frames.items()):
        if not path.is_file() or not path.stat().st_size: raise RuntimeError('missing/empty label: '+str(path))
        st = path.stat(); inventory.append(dict(file=name, source_size=st.st_size, source_mtime_ns=st.st_mtime_ns))
    raw = sum(r['source_size'] for r in inventory)
    print(f'SELECTED population={a.population} available={len(by)} windows={len(rows)} '
          f'scenes={len({r["scene_name"] for r in rows})} reference64={sum(r["role"] == "reference64" for r in rows)} '
          f'unique_frames={len(inventory)} raw_label_bytes={raw/2**20:.1f}MiB '
          f'(compact zip usually ~5-15% of that)', flush=True)
    if a.dry_run: return 0
    out.mkdir(parents=True, exist_ok=False)
    archive = out/'paper_viz_frames.zip'
    with zipfile.ZipFile(archive, 'x', allowZip64=True) as z:
        for i, entry in enumerate(inventory, 1):
            path = frames[entry['file']]; st = path.stat()
            if (st.st_size, st.st_mtime_ns) != (entry['source_size'], entry['source_mtime_ns']): raise RuntimeError('label changed')
            payload = compact_label(path)
            entry.update(size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
            z.writestr(entry['file'], payload, compress_type=zipfile.ZIP_STORED)
            if i % 256 == 0 or i == len(inventory): print(f'FRAMES {i}/{len(inventory)}', flush=True)
        if a.moving:
            for i, (row, key) in enumerate(zip(rows, keys), 1):
                masks = inputs.moving(by[key], MOVING_AT, tuple(float(h) for h in range(1, 7)), pcfg.grid, a.moving_workers)
                buf = io.BytesIO(); stacked = np.stack(masks)
                np.savez_compressed(buf, packed=np.packbits(stacked.reshape(-1)), shape=np.asarray(stacked.shape, np.int32))
                name = f'moving/{row["scene_name"]}/{row["t0_token"]}.npz'; payload = buf.getvalue()
                z.writestr(name, payload, compress_type=zipfile.ZIP_STORED)
                row['moving'] = dict(file=name, sha256=hashlib.sha256(payload).hexdigest(), future_indices=list(MOVING_AT))
                if i % 16 == 0 or i == len(rows): print(f'MOVING {i}/{len(rows)}', flush=True)
        manifest = dict(protocol=PROTOCOL, prepare_config=asdict(pcfg), records=rows, inventory=inventory, weights=[],
            population=a.population, requested_windows=a.windows, available_windows=len(by),
            source_checkpoint_sha256=sha256(ckpt), future_frames=FUTURE, frame_dt_s=float(pcfg.frame_dt_s),
            label_encoding='lossless_uint8_semantics_plus_packbits_lidar_no_camera_mask', no_raw_sensors=True,
            note='future labels are for scoring/figures only; reference64 was inspected repeatedly (not independent)',
            implementation={str(Path(__file__).relative_to(ROOT)): sha256(__file__)})
        z.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False), compress_type=zipfile.ZIP_DEFLATED)
    receipt = dict(archive=str(archive), bytes=archive.stat().st_size, sha256=sha256(archive), windows=len(rows),
                   scenes=len({r['scene_name'] for r in rows}), moving=bool(a.moving))
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print('DOWNLOAD_THIS_FILE: '+str(archive), flush=True); print(json.dumps(receipt, indent=2), flush=True)
    return 0


class FramesBundle:
    """Reader with the stc_overnight Bundle API (GT only, 12 futures, optional Moving)."""
    def __init__(self, path, cache_mib=256):
        from tools.stc_overnight import bundle as legacy
        from collections import OrderedDict
        self.path = Path(path).resolve(); self.zip = zipfile.ZipFile(self.path)
        self.manifest = json.loads(self.zip.read('manifest.json'))
        if self.manifest['protocol'] != PROTOCOL: raise ValueError('not a paper_viz frames bundle')
        self.pcfg = legacy.prepare_config(self.manifest['prepare_config'])
        self.inventory = {r['file']: r for r in self.manifest['inventory']}
        self.records = self.manifest['records']; self.cache = OrderedDict(); self.bytes = 0; self.limit = int(cache_mib*2**20)
        self.by_key = {(r['scene_name'], r['t0_token']): r for r in self.records}
        self._legacy = legacy.Bundle

    frame = lambda self, *a: self._legacy.frame(self, *a)
    targets = lambda self, row, indices=(1, 3, 5): self._legacy.targets(self, row, indices)
    close = lambda self: (self.zip.close(), self.cache.clear())

    def history(self, row, kind='gt', policy='dense_valid'):
        if kind != 'gt': raise ValueError('frames bundle holds GT only')
        return self._legacy.history(self, row, kind, policy)

    def moving(self, row):
        if 'moving' not in row: return None
        payload = self.zip.read(row['moving']['file'])
        if hashlib.sha256(payload).hexdigest() != row['moving']['sha256']: raise RuntimeError('moving digest differs')
        with np.load(io.BytesIO(payload)) as z:
            shape = tuple(map(int, z['shape'])); return np.unpackbits(z['packed'])[:int(np.prod(shape))].reshape(shape).astype(bool)


def open_bundle(path, cache_mib=64):
    with zipfile.ZipFile(path) as z: protocol = json.loads(z.read('manifest.json'))['protocol']
    if protocol == PROTOCOL: return FramesBundle(path, cache_mib)
    from tools.stc_overnight.bundle import Bundle
    return Bundle(path, cache_mib=cache_mib)


if __name__ == '__main__': raise SystemExit(main())
