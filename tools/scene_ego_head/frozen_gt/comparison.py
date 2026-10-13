"""Four prespecified heads, same frozen body and exact DEV512, no selection."""
import json
from pathlib import Path
import sys
import torch
from . import engine, reuse, training
from tools.joint_occ_experiments import run as legacy
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint


def checked_head(path, contract, epoch):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    old = payload['contract']
    if (payload['protocol'] != training.PROTOCOL or payload['epochs'] != epoch
            or reuse.feature_identity(old) != reuse.feature_identity(contract)
            or old['head'] != contract['head']
            or tensor_state_fingerprint(payload['head']) != payload['state_sha256']):
        raise RuntimeError('comparison head epoch/body/features/structure/hash differs')
    count = len(old['source_train_keys']); batch = old['schedule']['batch_size']
    if payload['updates'] != epoch*((count+batch-1)//batch):
        raise RuntimeError('comparison head incomplete epoch')
    if epoch == 10 and (old['schedule']['epochs'] != 10 or not payload['final_epoch_only']):
        raise RuntimeError('baseline must be the completed original ten-epoch cycle')
    return payload


def evaluate(body, head, provider, source, records, descriptors, contract, directory,
             baseline, *, batches, workers, stop_event=None, progress=None):
    directory = Path(directory)
    keys = [[r['scene_name'], r['t0_token']] for r in records]
    if len(keys) != 512 or len({tuple(k) for k in keys}) != 512:
        raise RuntimeError('exact fixed DEV512 required')
    legacy.write_json(directory/'comparison_population.json', dict(keys=keys, windows=512))
    reports = {}
    for epoch in (10, 50, 80, 100):
        if stop_event is not None and stop_event.is_set(): raise InterruptedError('four-head comparison stopped')
        path = Path(baseline) if epoch == 10 else directory/f'head_epoch{epoch}.pt'
        payload = checked_head(path, contract, epoch)
        head.load_state_dict(payload['head'], strict=True); head.eval()
        identity = dict(protocol='frozen_scene_ego_fixed_four_head_DEV512_v1', epoch=epoch,
            checkpoint_sha256=legacy.sha256(path), body_state_sha256=contract['body_state_sha256'], keys=keys)
        state_path = directory/f'dev512_head{epoch}_state.json'
        saved = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else None
        if saved and saved['identity'] != identity: raise RuntimeError('comparison eval provenance differs')
        def save(state): legacy.write_json(state_path, dict(identity=identity, state=state))
        stream = batches(provider, source, [[r] for r in records], descriptors, workers=workers)
        try:
            print(f'FIXED_DEV512_HEAD epoch={epoch}', flush=True)
            report = engine.evaluate(body, head, provider, source, stream,
                saved=saved['state'] if saved else None, save=save, stop_event=stop_event, progress=progress)
        finally: stream.close()
        if report['windows'] != 512 or report['population_keys'] != keys:
            raise RuntimeError('comparison head evaluated different DEV512 windows')
        report.update(head_epoch=epoch, checkpoint_sha256=identity['checkpoint_sha256'])
        reports[str(epoch)] = report
        legacy.write_json(directory/'head_comparison.json', dict(status='running', keys=keys, reports=reports,
            selection='all four reported; no best-head promotion'))
    legacy.write_json(directory/'head_comparison.json', dict(status='complete', keys=keys, reports=reports,
        selection='all four reported; no best-head promotion'))
    return {**reports['100'], 'head_comparison': reports}


def find_head10(root, checkpoint):
    digest = legacy.sha256(Path(checkpoint))
    paths = sorted(Path(root).glob('frozen_gt_scene_ego*/head_epoch10.pt'),
        key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for p in paths:
        v = torch.load(p, map_location='cpu', weights_only=False)
        if (v.get('epochs') == 10 and v.get('final_epoch_only')
                and v['contract']['schedule']['epochs'] == 10
                and v['contract']['stage']['source_sha256'] == digest):
            return str(p.resolve())
    raise RuntimeError('No completed ten-epoch baseline found; set HEAD10_CKPT explicitly')


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != '--find-head10': raise SystemExit('--find-head10 ROOT GT_CKPT')
    print(find_head10(sys.argv[2], sys.argv[3]))
