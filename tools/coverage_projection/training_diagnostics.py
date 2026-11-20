"""CPU-only checkpoint/log inspection; no model inference or optimizer update."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch


def progress_summary(path):
    path=Path(path)
    if not path.is_file(): return dict(available=False,path=str(path))
    updates={};invalid=[];duplicates=0
    for line_no,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
        try:
            row=json.loads(line);index=int(row['update'])
            duplicates+=int(index in updates);updates[index]=row
        except (ValueError,KeyError,TypeError): invalid.append(line_no)
    ordered=[updates[i] for i in sorted(updates)]
    def summarize(rows):
        names=('CE','Lovasz','projection_semantic','projection_depth','loss','grad_norm')
        result={k:dict(mean=float(np.mean([r[k] for r in rows if k in r])),
            median=float(np.median([r[k] for r in rows if k in r]))) for k in names if any(k in r for r in rows)}
        if all(k in result for k in names[:4]):
            main=result['CE']['mean']+result['Lovasz']['mean']
            projection=result['projection_semantic']['mean']+result['projection_depth']['mean']
            result['projection_over_main_value']=projection/main if main else None
        return result
    errors=[abs(r['loss']-r['CE']-r['Lovasz']-r['projection_semantic']-r['projection_depth'])
        for r in ordered if all(k in r for k in ('loss','CE','Lovasz','projection_semantic','projection_depth'))]
    return dict(available=True,path=str(path),unique_updates=len(updates),duplicate_log_updates=duplicates,
        invalid_lines=invalid,first_update=min(updates) if updates else None,last_update=max(updates) if updates else None,
        first32=summarize(ordered[:32]),last128=summarize(ordered[-128:]),
        loss_sum_max_abs_error=max(errors) if errors else None,
        note='term VALUE ratios are not gradient-norm ratios; last record wins on resumed duplicate update')


def parameter_summary(state):
    result={}
    for name in ('columns.head.weight','columns.head.bias','columns.fine.0.weight',
            'columns.importance.weight','columns.empty'):
        if name not in state: continue
        x=state[name].detach().float().cpu()
        result[name]=dict(shape=list(x.shape),L2=float(x.norm()),max_abs=float(x.abs().max()),
            finite=bool(torch.isfinite(x).all()))
    bias=state.get('columns.head.bias')
    if bias is not None:
        b=bias.detach().float().cpu()
        result['head_bias_by_class']=b.tolist()
        result['head_free_minus_road_bias']=float(b[17]-b[11])
    fine=state.get('columns.fine.0.weight')
    if fine is not None and fine.shape[1]==21:
        result['new_coverage_channel_weight_L2']={str(i):float(fine[:,i].detach().float().norm()) for i in (19,20)}
    return result


def optimizer_steps(body, optimizer):
    if optimizer is None: return dict(available=False)
    groups=optimizer['param_groups'];names=[['transport.'+n for n,_ in body.transport.named_parameters()],
        ['columns.'+n for n,_ in body.columns.named_parameters()]]
    if len(groups)!=2 or any(len(g['params'])!=len(n) for g,n in zip(groups,names)):
        raise ValueError('optimizer parameter order/count differs from saved architecture')
    steps={}
    for group,group_names in zip(groups,names):
        for name,index in zip(group_names,group['params']):
            value=optimizer['state'].get(index,{}).get('step')
            if name.startswith('columns.'):
                steps[name]=int(value.item() if torch.is_tensor(value) else value) if value is not None else None
    selected=('columns.head.weight','columns.head.bias','columns.fine.0.weight',
        'columns.lift.weight','columns.importance.weight','columns.empty')
    used=[v for v in steps.values() if v is not None]
    return dict(available=True,columns_steps={k:steps.get(k) for k in selected},
        total_columns_parameters=len(steps),minimum_step=min(used) if used else None,
        maximum_step=max(used) if used else None,parameters_without_optimizer_state=[k for k,v in steps.items() if v is None])


def inspect(folder):
    from .model import from_spec
    from .run import PROTOCOL,ROOT,sha
    folder=Path(folder).resolve();path=folder/'epoch_0001.pt';before=sha(path)
    saved=torch.load(path,map_location='cpu',weights_only=False);c=saved['contract']
    if saved.get('protocol')!=PROTOCOL or saved.get('updates')!=1031:
        raise ValueError('completed coverage pilot checkpoint required')
    for name,digest in c['implementations'].items():
        source=(ROOT/name).resolve()
        if ROOT not in source.parents or sha(source)!=digest:raise ValueError('frozen implementation changed: '+name)
    body=from_spec(c['model'],candidate=True)
    body.load_state_dict(saved['models']['gt'],strict=True)
    result=dict(scope='CPU checkpoint/log inspection ONLY; no inference/training/optimizer updates',
        checkpoint=str(path),checkpoint_sha256=before,updates=saved['updates'],
        candidate_parameters=parameter_summary(saved['models']['gt']),
        candidate_optimizer=optimizer_steps(body,saved.get('optimizer')),
        candidate_progress=progress_summary(folder/'progress.jsonl'))
    del body,saved
    control=Path(c['control_dir']);cp=control/'epoch_0001.pt'
    if sha(cp)!=c['control_sha256']:raise ValueError('frozen control source changed')
    old=torch.load(cp,map_location='cpu',weights_only=False)
    result['control_parameters']=parameter_summary(old['models']['gt'])
    result['control_progress']=progress_summary(control/'progress.jsonl')
    if sha(path)!=before:raise ValueError('candidate checkpoint changed during read-only inspection')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True,type=Path)
    a=p.parse_args();torch.set_num_threads(1);print(json.dumps(inspect(a.run_dir),indent=2))
