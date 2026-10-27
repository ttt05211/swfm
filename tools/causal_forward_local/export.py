"""CPU-only small ZIP: CURRENT epoch2 weights/config/reference scores, no datasets/Adam."""
import argparse
from dataclasses import asdict
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile
import torch
from tools.joint_occ_experiments import run as base
from tools.causal_forward_all.common import select_source,validate_source

PROTOCOL='current_forward_AdaLN_epoch2_weights_only_local_v1'


def export(source,out,prepare_config,source64,expected_contract=None):
    source=Path(source).resolve();out=Path(out).resolve()
    if out==source.parent or source.parent in out.parents:raise RuntimeError('NEW export outside training run required')
    original_sha=base.sha256(source);saved=torch.load(source,map_location='cpu',weights_only=False);c=validate_source(saved)
    if expected_contract is not None and expected_contract!=c:raise RuntimeError('checkpoint and completed report contracts differ')
    if (source64.get('windows')!=64 or source64.get('keys')!=c['eval_keys']
            or set(source64.get('variants',{}))!={'occ_gt_transport','occ_gt_semantic_refine','stc_gt_transport','stc_gt_semantic_refine'}):
        raise RuntimeError('complete matching original reference64 report required')
    if any(not math.isfinite(source64['variants'][r][m]) for r in source64['variants'] for m in ('mIoU','IoU','MovingMicro')):
        raise RuntimeError('nonfinite reference scores')
    state=saved['models']['gt']
    if any(not isinstance(v,torch.Tensor) or not torch.isfinite(v).all() for v in state.values()):raise RuntimeError('nonfinite source weights')
    # Preserve the exact original model/config/training protocol, without Adam,
    # rank RNG, past checkpoint evaluations or any copied historical caches.
    payload={k:saved[k] for k in ('protocol','contract','epoch','batch','updates','models')}
    buffer=io.BytesIO();torch.save(payload,buffer);data=buffer.getvalue()
    manifest=dict(protocol=PROTOCOL,source_checkpoint=str(source),source_checkpoint_sha256=original_sha,
        source_protocol=saved['protocol'],source_update=10292,parameters=c['parameters'],
        checkpoint_member='epoch_0002.pt',checkpoint_payload_sha256=hashlib.sha256(data).hexdigest(),
        checkpoint_payload_bytes=len(data),prepare_config=prepare_config,reference64_keys=c['eval_keys'],
        optimizer_included=False,RNG_included=False,raw_data_included=False,evaluation_only=True,
        runtime_server=dict(torch=str(torch.__version__),cuda=torch.version.cuda))
    out.mkdir(parents=True,exist_ok=False);archive=out/'forward_adaln_epoch2_local.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.writestr('epoch_0002.pt',data)
        z.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False))
        z.writestr('source_dev64.json',json.dumps(source64,ensure_ascii=False))
    if base.sha256(source)!=original_sha:raise RuntimeError('source checkpoint changed during export')
    receipt=dict(archive=str(archive),bytes=archive.stat().st_size,MiB=archive.stat().st_size/2**20,
        sha256=base.sha256(archive),source_checkpoint_sha256=original_sha,parameters=c['parameters'],
        no_optimizer=True,no_RNG=True,no_dataset_copy=True)
    base.write_json(out/'receipt.json',receipt)
    print('DOWNLOAD_THIS_FILE: '+str(archive),flush=True);print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)
    return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-dir');p.add_argument('--out-dir',required=True)
    p.add_argument('--config',default='configs/real_motion_occfm.yaml');a=p.parse_args(argv)
    root=Path(__file__).resolve().parents[2];torch.set_num_threads(1)
    source=select_source(a.source_dir,root/'outputs/p0_f9_shared_history')
    report=json.loads((source.parent/'results.json').read_text(encoding='utf-8'));c=report['contract']
    if report.get('status')!='complete' or report.get('epoch')!=2 or report.get('updates')!=10292:
        raise RuntimeError('completed current full-training source required')
    if base.sha256(a.config)!=c['data']['config']:raise RuntimeError('original config changed')
    pcfg=base.make_prepare_config(base.load_runtime_config(a.config,c['overrides']))
    return export(source,a.out_dir,asdict(pcfg),report['evaluations']['10292'],expected_contract=c)


if __name__=='__main__':main()
