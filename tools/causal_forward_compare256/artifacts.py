"""Read existing epoch2 and its completed mean3; never reconstruct the mean."""
import copy
import json
from pathlib import Path
import torch
from tools.causal_forward_average.weights import PROTOCOL as MEAN_PROTOCOL,STEPS,validate_contract,cursor
from tools.causal_forward_full.model import checkpoint_model
from tools.causal_forward_full.run import PROTOCOL as TRAIN_PROTOCOL
from tools.causal_forward_train.control import sha


def select_mean(explicit,root):
    if explicit:return Path(explicit).resolve()
    choices=[]
    for p in Path(root).glob('adaln_forward_mean3_*/results.json'):
        v=json.loads(p.read_text(encoding='utf-8'))
        if v.get('status')=='complete' and v.get('contract',{}).get('protocol')==MEAN_PROTOCOL:choices.append(p.parent.resolve())
    if len(choices)!=1:raise RuntimeError(f'expected ONE completed current mean3, found {len(choices)}; set FORWARD_MEAN_DIR')
    return choices[0]


def load_pair(mean_dir,full_dir=None):
    mean_dir=Path(mean_dir).resolve();mean_path=mean_dir/'mean3.pt';mean_sha=sha(mean_path)
    receipt=json.loads((mean_dir/'mean_receipt.json').read_text(encoding='utf-8'))
    if receipt['sha256']!=mean_sha:raise RuntimeError('existing mean3 receipt differs')
    mean=torch.load(mean_path,map_location='cpu',weights_only=False);mc=mean['contract']
    if (mean.get('protocol')!=MEAN_PROTOCOL or mc.get('protocol')!=MEAN_PROTOCOL
            or mean.get('checkpoint_role')!='evaluation_only' or mean.get('resume_allowed') is not False
            or mean.get('source_updates')!=list(STEPS) or mean.get('coefficients')!=[1/3]*3
            or mc.get('coefficients')!=[1/3]*3 or set(mean.get('models',{}))!={'gt'}
            or {'optimizers','rank_rng'} & set(mean)):
        raise RuntimeError('only the existing current mean(4000,10000,10292) artifact accepted')
    completed=json.loads((mean_dir/'results.json').read_text(encoding='utf-8'))
    if completed.get('status')!='complete' or completed.get('contract')!=mc:
        raise RuntimeError('completed same64 mean3 evaluation required')
    source=Path(mc['source_dir']).resolve()
    if full_dir and Path(full_dir).resolve()!=source:raise RuntimeError('epoch2 directory is not this mean3 parent')
    c=mc['source_contract'];validate_contract(c)
    if json.loads((source/'contract.json').read_text(encoding='utf-8'))!=c or mc['keys']!=c['eval_keys']:
        raise RuntimeError('mean3/full source contracts differ')
    if sha(source/'results.json')!=mc['source_report_sha256']:raise RuntimeError('source full report changed')
    if [s['step'] for s in mc['sources']]!=list(STEPS):raise RuntimeError('mean3 source steps differ')
    for item in mc['sources']:
        expected=source/('epoch_0002.pt' if item['step']==10292 else f'step_{item["step"]:07d}.pt')
        if Path(item['path']).resolve()!=expected or sha(expected)!=item['sha256']:
            raise RuntimeError('mean3 source checkpoint changed: '+str(expected))
    epoch_path=source/'epoch_0002.pt';epoch_sha=sha(epoch_path)
    saved=torch.load(epoch_path,map_location='cpu',weights_only=False)
    if (saved.get('protocol')!=TRAIN_PROTOCOL or saved.get('contract')!=c or saved.get('updates')!=10292
            or (saved.get('epoch'),saved.get('batch'))!=cursor(10292,c['epochs_updates']) or set(saved.get('models',{}))!={'gt'}):
        raise RuntimeError('exact completed source epoch2 checkpoint required')
    root=Path(__file__).resolve().parents[2]
    for manifest in (c['implementation'],mc['implementation']):
        for name,digest in manifest.items():
            file=(root/name).resolve()
            if root not in file.parents or sha(file)!=digest:raise RuntimeError('source implementation changed: '+name)
    body=checkpoint_model(saved,'cpu');other=copy.deepcopy(body);template=body.state_dict();state=mean['models']['gt']
    if set(state)!=set(template):raise RuntimeError('mean3 state keys differ')
    names=set(dict(body.named_parameters()))
    for name,v in state.items():
        if (not isinstance(v,torch.Tensor) or v.shape!=template[name].shape or v.dtype!=template[name].dtype
                or not torch.isfinite(v).all()):raise RuntimeError('mean3 tensor incompatible: '+name)
        if name not in names and not torch.equal(v,template[name]):raise RuntimeError('mean3 fixed buffer differs: '+name)
    other.load_state_dict(state,strict=True)
    if sha(mean_path)!=mean_sha or sha(epoch_path)!=epoch_sha:raise RuntimeError('source changed during loading')
    return dict(epoch2=body,mean3=other),c,dict(full_dir=str(source),mean_dir=str(mean_dir),
        checkpoints=dict(epoch2=dict(path=str(epoch_path),sha256=epoch_sha),mean3=dict(path=str(mean_path),sha256=mean_sha)),
        source_checkpoints=[{k:v for k,v in item.items() if k in ('step','path','sha256')} for item in mc['sources']],
        full_report=dict(path=str(source/'results.json'),sha256=mc['source_report_sha256']),scientific_updates=0)
