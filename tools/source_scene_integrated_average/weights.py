"""Fixed recipe, strict lineage and existing complete same-population controls."""
import json
import math
from pathlib import Path
import torch
from tools.joint_occ_experiments import run as base
from tools.source_scene_integrated_full.run import PROTOCOL as TRAIN_PROTOCOL
from tools.real_motion.joint_surface_checkpoint_selection import average_named_parameters

PROTOCOL = 'integrated22_fixed_mean3_8000_10000_10292_evaluation_v1'
STEPS = (8000, 10000, 10292)
METRICS = ('mIoU', 'IoU', 'MovingMicro')
ROUTES = tuple(k+'_'+r for k in ('occ_gt', 'stc_gt') for r in ('transport', 'semantic_refine'))


def select_source(explicit, search):
    if explicit:
        return Path(explicit).resolve()
    candidates = []
    for p in Path(search).glob('scene_integrated22_full_2ep_*/results.json'):
        v = json.loads(p.read_text(encoding='utf-8'))
        if v.get('status') == 'complete' and v.get('contract', {}).get('protocol') == TRAIN_PROTOCOL:
            candidates.append(p.parent.resolve())
    if len(candidates) != 1:
        raise RuntimeError(f'expected ONE completed latest Integrated run, found {len(candidates)}; set INTEGRATED_FULL_DIR explicitly')
    return candidates[0]


def validate_contract(c):
    s = c['model']
    if (c.get('protocol') != TRAIN_PROTOCOL or c.get('arm') != 'integrated22_full' or
            c.get('source_policy') != 'shared_pinned' or c.get('registration') != 'none' or
            c.get('known') != 'TRAIN_OCC_STC_dense_valid' or c.get('epochs') != 2 or
            c.get('total_updates') != 10292 or c.get('epochs_updates') != [5145, 5147] or
            c.get('objective') != dict(CE=1., Lovasz=1., motion=0., existence=0., ego=0.) or
            c.get('precision') != 'original_BF16_geometry_FP64_H6' or
            s.get('variant') != 'source_scene_integrated_AdaGN' or
            s['encoder']['blocks'] != 2 or s['motion']['decoder_blocks'] != 2 or
            s['reader']['layers'] != 2 or s['reader']['temporal_layers'] != 2 or
            s.get('kta_conditioning') is not True or s.get('registration') != 'none' or
            s['repair'].get('conditioning') != 'residual_branch_AdaGN_zero_projection' or
            s['repair'].get('GN_affine') is not False):
        raise RuntimeError('only latest FULL TRAIN x2 Integrated2+2 / KTA query / no ICP / dense-valid allowed')
    for field, count in (('train_keys', 20430), ('eval_keys', 64)):
        keys = c[field]
        if len(keys) != count or any(len(k) != 2 for k in keys) or len(set(map(tuple, keys))) != count:
            raise RuntimeError('wrong unique ordered population: '+field)
    if {k[0] for k in c['train_keys']} & {k[0] for k in c['eval_keys']}:
        raise RuntimeError('TRAIN/DEV scenes overlap')


def load_sources(folder):
    folder = Path(folder)
    path = folder/'results.json'; report_sha = base.sha256(path)
    report = json.loads(path.read_text(encoding='utf-8')); c = report['contract']
    validate_contract(c)
    if report.get('status') != 'complete' or report.get('updates') != 10292 or report.get('epoch') != 2:
        raise RuntimeError('complete latest full-training results required')
    if json.loads((folder/'contract.json').read_text(encoding='utf-8')) != c:
        raise RuntimeError('run contract differs from complete report')
    sources, controls = [], {}
    for step in STEPS:
        p = folder/('epoch_0002.pt' if step == 10292 else f'step_{step:07d}.pt')
        sha = base.sha256(p); v = torch.load(p, map_location='cpu', weights_only=False)
        cursor = (2, 0) if step == 10292 else (1, step-5145)
        if (v.get('protocol') != TRAIN_PROTOCOL or v.get('contract') != c or v.get('updates') != step or
                (v.get('epoch'), v.get('batch')) != cursor or set(v.get('models', {})) != {'gt'}):
            raise RuntimeError('checkpoint lineage/update/cursor differs: '+str(p))
        e = report['evaluations'][str(step)]
        if e.get('windows') != 64 or e.get('keys') != c['eval_keys'] or set(e['variants']) != set(ROUTES):
            raise RuntimeError('complete matching dense-valid remaining64 controls required: '+str(step))
        if any(not math.isfinite(e['variants'][r][m]) for r in ROUTES for m in METRICS):
            raise RuntimeError('nonfinite original control score')
        controls[str(step)] = e['variants']
        if base.sha256(p) != sha: raise RuntimeError('checkpoint changed during load')
        sources.append(dict(step=step, path=str(p.resolve()), sha256=sha, state=v['models']['gt']))
        # Retain NO source optimizer or RNG. The epoch2 header constructs the original architecture only.
        if step == 10292:
            header = {k: v[k] for k in ('protocol', 'contract', 'models')}
    if base.sha256(path) != report_sha: raise RuntimeError('source report changed during load')
    return c, sources, controls, header, report_sha


def mean_payload(sources, body, contract):
    if tuple(s['step'] for s in sources) != STEPS: raise RuntimeError('fixed THREE-checkpoint recipe required')
    return dict(protocol=PROTOCOL, checkpoint_role='evaluation_only', resume_allowed=False,
        coefficients=[1/3]*3, source_updates=list(STEPS), buffers='identical_copy',
        arithmetic='named_parameters_float64_equal_mean_then_original_dtype',
        contract=contract, models={'gt': average_named_parameters([s['state'] for s in sources], body)})
