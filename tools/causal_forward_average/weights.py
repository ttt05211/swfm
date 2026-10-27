"""Current full lineage, fixed three steps, existing same64 scores and parameter mean."""
import json
import math
from pathlib import Path
import torch
from tools.joint_occ_experiments import run as base
from tools.causal_forward_full.run import PROTOCOL as TRAIN_PROTOCOL
from tools.causal_forward_full.model import SOURCE_POLICY
from tools.fpn_adaln_zero.model import VARIANT
from tools.real_motion.joint_surface_checkpoint_selection import average_named_parameters

PROTOCOL='forward_AdaLN_full_STC_mIoU_selected_mean3_4000_10000_10292_same64_v1'
STEPS=(4000,10000,10292)
METRICS=('mIoU','IoU','MovingMicro')
ROUTES=tuple(k+'_'+r for k in ('occ_gt','stc_gt') for r in ('transport','semantic_refine'))


def validate_contract(c):
    s=c['model'];ex=s.get('experiment') or {}
    if (c.get('protocol')!=TRAIN_PROTOCOL or c.get('epochs')!=2 or c.get('total_updates')!=10292
            or c.get('epochs_updates')!=[5145,5147] or c.get('parameters')!=2816122
            or c.get('source_policy')!=SOURCE_POLICY or c.get('registration')!='none'
            or c.get('known')!='TRAIN_OCC_STC_dense_valid' or c.get('world_size') not in (1,2)
            or c.get('objective')!=dict(CE=1.,Lovasz=1.,motion=0.,existence=0.,ego=0.)
            or c.get('precision')!='BF16_network_FP64_geometry_full6'
            or s.get('variant')!=VARIANT or s.get('source_policy')!=SOURCE_POLICY
            or s['motion']['history_frames']!=4 or s['motion']['decoder_blocks']!=2
            or ex.get('FREE_bias')!=0. or ex.get('history_age') is not False):
        raise RuntimeError('current full forward AdaLN x2 / no FREE-age / same architecture required')
    for field,count in (('train_keys',20430),('eval_keys',64)):
        keys=c[field]
        if len(keys)!=count or any(len(k)!=2 for k in keys) or len(set(map(tuple,keys)))!=count:
            raise RuntimeError('unique ordered population differs: '+field)
    if {k[0] for k in c['train_keys']} & {k[0] for k in c['eval_keys']}:raise RuntimeError('TRAIN/DEV scene overlap')


def select_source(explicit,search):
    if explicit:return Path(explicit).resolve()
    choices=[]
    for p in Path(search).glob('adaln_forward_full_2ep_*/results.json'):
        v=json.loads(p.read_text(encoding='utf-8'))
        if v.get('status')=='complete' and v.get('contract',{}).get('protocol')==TRAIN_PROTOCOL:
            choices.append(p.parent.resolve())
    if len(choices)!=1:raise RuntimeError(f'expected ONE completed current forward full run; found {len(choices)}; set FORWARD_FULL_DIR explicitly')
    return choices[0]


def cursor(step,budgets):
    epoch=0;batch=step
    for length in budgets:
        if batch<length:break
        batch-=length;epoch+=1
    return epoch,batch


def load_sources(folder):
    folder=Path(folder);path=folder/'results.json';report_sha=base.sha256(path)
    report=json.loads(path.read_text(encoding='utf-8'));c=report['contract'];validate_contract(c)
    if report.get('status')!='complete' or report.get('epoch')!=2 or report.get('updates')!=10292:
        raise RuntimeError('completed current full two-epoch report required')
    if json.loads((folder/'contract.json').read_text(encoding='utf-8'))!=c:raise RuntimeError('run/report contracts differ')
    controls={};sources=[];header=None
    for step in STEPS:
        p=folder/('epoch_0002.pt' if step==10292 else f'step_{step:07d}.pt')
        digest=base.sha256(p);saved=torch.load(p,map_location='cpu',weights_only=False)
        if (saved.get('protocol')!=TRAIN_PROTOCOL or saved.get('contract')!=c or saved.get('updates')!=step
                or (saved.get('epoch'),saved.get('batch'))!=cursor(step,c['epochs_updates'])
                or set(saved.get('models',{}))!={'gt'}):raise RuntimeError('checkpoint lineage/update/cursor differs: '+str(p))
        e=report['evaluations'][str(step)]
        if (e.get('windows')!=64 or e.get('keys')!=c['eval_keys'] or set(e['variants'])!=set(ROUTES)
                or any(not math.isfinite(e['variants'][r][m]) for r in ROUTES for m in METRICS)):
            raise RuntimeError('complete matching raw same64 controls required: '+str(step))
        controls[str(step)]=e['variants']
        sources.append(dict(step=step,path=str(p.resolve()),sha256=digest,state=saved['models']['gt']))
        if step==10292:header={k:saved[k] for k in ('protocol','contract','models')}
        if base.sha256(p)!=digest:raise RuntimeError('source checkpoint changed while loading')
    if base.sha256(path)!=report_sha:raise RuntimeError('source report changed while loading')
    return c,sources,controls,header,report_sha


def mean_payload(sources,body,contract):
    if tuple(s['step'] for s in sources)!=STEPS:raise RuntimeError('fixed mean(4000,10000,10292) required')
    return dict(protocol=PROTOCOL,checkpoint_role='evaluation_only',resume_allowed=False,
        coefficients=[1/3]*3,source_updates=list(STEPS),buffers='identical_copy',
        selection='three highest STC semantic_refine mIoU in user-provided same64 report; development-selected',
        arithmetic='named_parameters_float64_equal_mean_then_original_dtype',contract=contract,
        models={'gt':average_named_parameters([s['state'] for s in sources],body)})
