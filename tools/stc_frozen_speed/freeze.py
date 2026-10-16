"""Merge the verified norm values ONCE; never modify a source checkpoint."""
from pathlib import Path
import os
import torch
from tools.stc_norm_final.common import ROOT as ASSETS,BODY_SHA,norm_state,assign,read,write
from tools.joint_occ_experiments import run as base
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint

ROOT=Path(__file__).resolve().parent


def assets():
    release=read(ROOT/'release.json');patch=read(ASSETS/'norm192.json');posterior=read(ASSETS/'posterior_fit.json')
    if (release['source_sha256']!=BODY_SHA or patch['source_sha256']!=BODY_SHA or posterior['source_sha256']!=BODY_SHA
            or base.stable_json_fingerprint(patch)!=release['norm_patch_semantic_sha256']
            or base.stable_json_fingerprint(posterior)!=release['posterior_semantic_sha256']):
        raise ValueError('fixed release assets changed; no automatic new version')
    return release,patch,posterior


def validate_server(path):
    path=Path(path);result=read(path/'results.json');j=result['contract'];release,patch,posterior=assets()
    if (result['status']!='complete' or result['windows']!=4369 or j['population']!='ALL4369/150'
            or j['source_sha256']!=BODY_SHA or j['source_policy']!=release['source_policy']
            or len(j['keys'])!=4369 or len(set(map(tuple,j['keys'])))!=4369 or len({k[0] for k in j['keys']})!=150
            or j['metric_mask']!='none' or not j['GT_future_ego_conditioned']
            or j['known']!='STC dense-valid; NO GT visibility' or j['horizon_group']!=6
            or read(path/'contract.json')!=j):
        raise ValueError('completed matching STC norm192 ALL4369 evaluation required')
    for name,value in [('norm192.json',patch),('posterior_fit.json',posterior)]:
        source=path/'frozen'/name
        if base.sha256(source)!=j['assets'][name] or read(source)!=value:raise ValueError('evaluated assets differ')
    if any(round(result['metrics']['norm192/semantic_refine']['average'][k],6)!=release['verified_average'][k] for k in ('mIoU','IoU')):
        raise ValueError('not the fixed verified release result')
    return dict(directory=str(path.resolve()),results_sha256=base.sha256(path/'results.json'),
        contract_sha256=base.sha256(path/'contract.json'),verified_average=release['verified_average'])


def freeze(body,saved,folder,evidence=None):
    release,patch,posterior=assets();folder=Path(folder)
    if saved['source_sha256']!=BODY_SHA:raise ValueError('wrong source body')
    folder.mkdir(parents=True,exist_ok=False)
    parameters,original,adapted=norm_state(body,patch,saved['source_sha256'])
    assign(parameters,adapted)  # one-time merge, no per-window parameter swapping
    fingerprint=tensor_state_fingerprint(body.state_dict())
    payload=dict(protocol='frozen_STC_norm192_deployment_v1',release=release,model_configs=saved['model_configs'],
        state_dict={k:v.detach().cpu().clone() for k,v in body.state_dict().items()},state_sha256=fingerprint,
        source_sha256=BODY_SHA,posterior_bias=posterior['bias'],optimizer_included=False,training_allowed=False,
        source_contract=saved['source_contract'])
    temporary=folder/'final_model.pt.tmp';torch.save(payload,temporary);os.replace(temporary,folder/'final_model.pt')
    write(folder/'receipt.json',dict(release=release,merged_state_sha256=fingerprint,
        frozen_checkpoint_sha256=base.sha256(folder/'final_model.pt'),server_evaluation=evidence,
        source_files_written=False,optimizer_steps=0))
    return parameters,original,adapted,payload


def load_frozen(folder,device='cpu'):
    from tools.dense_checkpoint_average.weights import model_from_spec
    folder=Path(folder);receipt=read(folder/'receipt.json');release,_,_=assets()
    if receipt['release']!=release or base.sha256(folder/'final_model.pt')!=receipt['frozen_checkpoint_sha256']:
        raise ValueError('frozen checkpoint receipt/hash differs')
    saved=torch.load(folder/'final_model.pt',map_location='cpu',weights_only=False)
    if (saved['release']!=release or saved['source_sha256']!=BODY_SHA
            or saved['optimizer_included'] is not False or saved['training_allowed'] is not False):
        raise ValueError('frozen deployment checkpoint required')
    body=model_from_spec(saved['model_configs']);body.load_state_dict(saved['state_dict'],strict=True)
    if tensor_state_fingerprint(body.state_dict())!=receipt['merged_state_sha256']:
        raise ValueError('merged frozen model fingerprint differs')
    return body.to(device).eval().requires_grad_(False)
