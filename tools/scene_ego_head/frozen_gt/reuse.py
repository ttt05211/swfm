"""Read-only import of legacy frozen history tokens, independent of head schedule."""
import json
from pathlib import Path
import sys
import torch
from .bank import FIELDS
from tools.joint_occ_experiments import run as legacy


def feature_identity(contract):
    # Do NOT bind epochs, LR, seed, head weights/Adam or output directory.
    stage = contract['stage']
    args = stage['args']
    extraction = ('tools/scene_ego_head/frozen_gt/bank.py', 'tools/scene_ego_head/model.py',
        'real_motion/ego_navigation.py', 'tools/ego_experiments/ego_kinematic.py',
        'tools/ego_experiments/train_surface_ego_full.py')
    implementation = {k.replace('\\', '/'): v for k, v in stage['implementation'].items()}
    return dict(protocol='frozen_scene_history_feature_identity_v1',
        source_sha256=stage['source_sha256'], body_state_sha256=contract['body_state_sha256'],
        train_keys=contract['source_train_keys'], runtime=contract['source_runtime'],
        encoder_precision=contract['encoder_precision'], features=contract['features'],
        pool_side=contract['head']['pool_side'], feature_width=contract['head']['feature_width'],
        chunk_size=contract['bank_chunk'],
        data={k: args.get(k) for k in ('config', 'override', 'dataroot', 'train_info', 'train_cache', 'ccr_history_cache')},
        implementation={k: implementation[k] for k in extraction})


def load_readonly(stage_dir, contract, *, stop_event=None):
    stage_dir = Path(stage_dir).resolve(); directory = stage_dir/'history_tokens'
    head_path = stage_dir/'head_last.pt'
    old = torch.load(head_path, map_location='cpu', weights_only=False)['contract']
    identity = feature_identity(contract)
    if feature_identity(old) != identity:
        raise RuntimeError('reused history features: body/data/extraction differs')
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    old_fp = legacy.stable_json_fingerprint(old)
    keys = contract['source_train_keys']
    if not manifest['complete'] or manifest['keys'] != keys or manifest['contract'] != old_fp:
        raise RuntimeError('reused bank incomplete or legacy provenance differs')
    values = {k: [] for k in FIELDS}; offset = used = 0
    for index, entry in enumerate(manifest['chunks']):
        if stop_event is not None and stop_event.is_set(): raise InterruptedError('history import stopped')
        path = directory/f'{index:04d}.pt'
        count = entry['windows']
        if (entry['index'] != index or type(count) is not int or not 0 < count <= contract['bank_chunk']
                or index < len(manifest['chunks'])-1 and count != contract['bank_chunk']
                or legacy.sha256(path) != entry['sha256']):
            raise RuntimeError('reused bank receipt differs')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload['contract'] != old_fp or payload['keys'] != keys[offset:offset+count] or set(payload['values']) != set(FIELDS):
            raise RuntimeError('reused bank identity/order differs')
        tokens, width = 4*identity['pool_side']**2, identity['feature_width']
        shapes = dict(features=(count, tokens, width), geometry=(count, tokens, 5), valid=(count, tokens),
            history=(count, 4, 10), commands=(count, 6), target=(count, 6, 3))
        for k in FIELDS:
            v = payload['values'][k]
            if (not isinstance(v, torch.Tensor) or tuple(v.shape) != shapes[k]
                    or k == 'valid' and v.dtype != torch.bool
                    or k == 'commands' and (v.dtype != torch.int64 or bool(((v < 0) | (v > 2)).any()))
                    or k not in ('valid', 'commands') and (v.dtype != torch.float32 or not bool(torch.isfinite(v).all()))):
                raise RuntimeError('reused bank tensors invalid')
            used += v.numel()*v.element_size()
            if used > 3072*2**20: raise RuntimeError('reused compact history exceeds 3GiB')
            values[k].append(v)
        offset += count
    if offset != len(keys) or not offset: raise RuntimeError('reused bank population differs')
    receipt = dict(reused=True, read_only=True, fresh_windows=0, windows=offset, tensor_mib=used/2**20,
        stage_directory=str(stage_dir), legacy_contract_sha256=old_fp,
        legacy_head_receipt_sha256=legacy.sha256(head_path), feature_identity=identity)
    return {k: torch.cat(v) for k, v in values.items()}, receipt


def find_stage(root, checkpoint):
    digest = legacy.sha256(Path(checkpoint))
    candidates = sorted(Path(root).glob('frozen_gt_scene_ego*/history_tokens/manifest.json'),
                        key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for path in candidates:
        head = path.parent.parent/'head_last.pt'
        if not head.is_file(): continue
        manifest = json.loads(path.read_text(encoding='utf-8'))
        c = torch.load(head, map_location='cpu', weights_only=False)['contract']
        if (manifest['complete'] and c['stage']['source_sha256'] == digest
                and c['head']['pool_side'] == 8 and manifest['contract'] == legacy.stable_json_fingerprint(c)):
            return str(path.parent.parent.resolve())
    raise RuntimeError('No matching completed history bank; set EGO_REUSE_DIR to the successful one-epoch directory')


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != '--find-stage': raise SystemExit('usage: --find-stage ROOT GT_CHECKPOINT')
    print(find_stage(sys.argv[2], sys.argv[3]))
