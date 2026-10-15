"""Explicit same-lineage checkpoints only; reuse established parameter-only mean."""
import json
from pathlib import Path
import torch
from tools.joint_occ_experiments import run as base
from tools.dense_valid_ddp.run import PROTOCOL as ORIGINAL
from tools.dense_valid_epoch3.run import PROTOCOL as CONTINUATION, validate_source
from tools.shared_history_experiments.full_gt.model import GTBody
from tools.real_motion.joint_surface_checkpoint_selection import average_named_parameters

STEPS = (10000, 10292, 12000, 14000)
RECIPES = dict(recent=STEPS, wide=(4000, 6000, 8000, 10292))
PROTOCOL = 'dense_shared_semantic_refiner_fixed_mean4_evaluation_only_v1'
ROUTES = ('occ_gt_transport', 'occ_gt_semantic_refine', 'stc_gt_transport', 'stc_gt_semantic_refine')
METRICS = ('mIoU', 'IoU', 'MovingMicro')


def model_from_spec(spec):
    sh, rep = spec['shared_history'], spec['repair']
    body = GTBody(base.config_from_mapping_v17(spec['motion']), feature_width=sh['width'], z_bins=sh['z_bins'],
        memory_width=rep['memory_width'], width=rep['width'], chunk_size=rep['chunk_size'])
    body.columns.epsilon = rep['epsilon']
    return body


def select_directories(epoch3, original, search):
    if epoch3: stage = Path(epoch3).resolve()
    else:
        candidates = [p for p in Path(search).glob('dense_valid_shared_epoch3_*')
                      if (p/'contract.json').is_file() and (p/'update_012000.pt').is_file()
                      and (p/'update_014000.pt').is_file()]
        if not candidates: raise RuntimeError('no epoch3 run with BOTH12000/14000; set DENSE_EPOCH3_DIR')
        stage = max(candidates, key=lambda p: (p/'update_014000.pt').stat().st_mtime_ns).resolve()
    c = json.loads((stage/'contract.json').read_text(encoding='utf-8'))
    if c.get('protocol') != CONTINUATION or c['source_updates'] != 10292:
        raise RuntimeError('expected original dense-valid epoch2 -> epoch3 continuation')
    parent = Path(original).resolve() if original else Path(c['source_checkpoint']).resolve().parent
    return stage, parent, c


def load_sources(stage, parent, stage_contract, *, steps=STEPS):
    if tuple(steps) not in RECIPES.values(): raise ValueError('only the two explicitly fixed recipes supported')
    paths = [(parent/'epoch_0002.pt' if step == 10292 else
              (parent if step < 10292 else stage)/f'update_{step:06d}.pt') for step in steps]
    sources = []
    for step, path in zip(steps, paths):
        sha = base.sha256(path); saved = torch.load(path, map_location='cpu', weights_only=False)
        c = saved['contract']; expected = ORIGINAL if step <= 10292 else CONTINUATION
        if (saved.get('protocol') != expected or c.get('protocol') != expected or saved['updates'] != step
                or set(saved['models']) != {'gt'} or c['training_known'] != 'dense_valid'
                or c['source_policy'] != 'shared_pinned'):
            raise RuntimeError('wrong checkpoint/protocol/step/source policy: '+str(path))
        if step == 10292:
            validate_source(saved)
            if sha != stage_contract['source_sha256']: raise RuntimeError('epoch2 SHA is not continuation initializer')
        if step <= 10292:
            if c != stage_contract['source_contract']: raise RuntimeError('original training lineage differs')
        elif c != stage_contract: raise RuntimeError('continuation contracts differ')
        keys = c['eval_keys']
        if len(keys) != 64 or len(set(map(tuple, keys))) != 64 or keys != stage_contract['eval_keys']:
            raise RuntimeError('exact SAME ordered remaining64 required')
        if c['model'] != stage_contract['model']: raise RuntimeError('model configs differ')
        if base.sha256(path) != sha: raise RuntimeError('checkpoint changed while reading')
        sources.append(dict(step=step, path=str(path), sha256=sha, contract=c, state=saved['models']['gt'],
                            evaluations=saved.get('evaluations', {})))
    return sources


def historical_report(source):
    """A complete per-step JSON takes precedence over a run still marked stopped."""
    folder, step, expected = Path(source['path']).parent, source['step'], source['contract']
    tag = str(step)
    candidates = [(folder/f'eval_update_{step:06d}.json', 'evaluation'), (folder/'results.json', 'evaluations')]
    for path, field in candidates:
        if not path.is_file(): continue
        value = json.loads(path.read_text(encoding='utf-8'))
        if value.get('contract') != expected: raise RuntimeError('historical report contract differs: '+str(path))
        report = value.get(field) if field == 'evaluation' else value.get(field, {}).get(tag)
        if report is None: continue
        return checked_report(report, source, str(path))
    report = source['evaluations'].get(tag)
    return checked_report(report, source, source['path']+'#evaluations') if report else None


def checked_report(report, source, path):
    if (report['windows'] != 64 or report.get('evaluation_keys') != source['contract']['eval_keys']
            or set(report['variants'].get('gt', {})) != set(ROUTES)):
        raise RuntimeError('historical report incomplete or population differs: '+path)
    c = source['contract']
    return dict(path=path, known=dict(occ=c['OCC_eval_known'], stc=c['STC_eval_known']),
                metrics=report['variants']['gt'])


def averaged_payload(sources, body, contract):
    # The old helper only supplies arithmetic; NO old Surface model/renderer is executed.
    state = average_named_parameters([s['state'] for s in sources], body)
    return dict(protocol=PROTOCOL, checkpoint_role='evaluation_only', resume_allowed=False,
        source_updates=[s['step'] for s in sources], coefficients=[.25]*4, buffers='identical_copy',
        arithmetic='equal_named_parameter_mean_float64_then_original_dtype', contract=contract,
        model_configs=body.configs(), models={'gt': state})
