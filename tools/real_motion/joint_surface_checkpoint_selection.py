"""Read-only clean-joint checkpoint audit and evaluation-only weight averaging."""
import copy
import json
import math
import os
from pathlib import Path

import torch

from real_motion.joint_surface_ccr import JointSurfaceCCR, PROTOCOL as TRAIN_PROTOCOL
from real_motion.local_st_world_model_v17 import config_from_mapping_v17
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion.height_field_screen_recovery import validate_cursor
from tools.real_motion.joint_checkpoint_selection import load_cpu_checkpoint, weight_fingerprint
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import sha256
from tools.real_motion.static_evidence_selector_common import finite_json

PROTOCOL = 'p0_f9_joint_surface_ccr_weights_only_v1'
METRICS = ('IoU', 'mIoU', 'MovingMacro', 'MovingMicro')
AVERAGE_EPOCHS = (5, 6, 8, 12, 14)
SINGLE_EPOCHS = (6, 8, 12)
AVERAGE_NAME = 'mean_top5_dev64_mIoU'
TRAIN_FILES = (
    'real_motion/joint_surface_ccr.py', 'real_motion/surface_canonical_repair.py',
    'real_motion/surface_projection_execution.py', 'tools/real_motion/joint_surface_ccr_common.py',
    'tools/real_motion/joint_surface_ccr_recovery.py', 'tools/real_motion/train_p0_f9_joint_surface_ccr.py',
    'tools/real_motion/ccr_screen_common.py', 'tools/real_motion/surface_ccr_screen_common.py',
    'tools/real_motion/joint_column_common.py', 'tools/real_motion/height_field_screen_common.py',
)


def training_implementation(root):
    return stable_json_fingerprint({name: sha256(Path(root)/name) for name in TRAIN_FILES})


def model_from_configs(configs, *, device='cpu', z_bins=16):
    motion = config_from_mapping_v17(configs['motion'])
    model = JointSurfaceCCR(motion, width=int(configs['repair']['width']), z_bins=z_bins).to(device)
    if stable_json_fingerprint(model.configs()) != stable_json_fingerprint(configs):
        raise RuntimeError('checkpoint architecture/configuration mismatch')
    return model


def validate_training_checkpoint(saved, contract, *, epoch=None):
    if (saved.get('protocol') != TRAIN_PROTOCOL or saved.get('initialization') != 'RANDOM'
            or saved.get('transport_frozen') is not False or saved.get('deployable') is not False
            or stable_json_fingerprint(saved.get('contract')) != stable_json_fingerprint(contract)):
        raise RuntimeError('only identical clean-joint training lineage is eligible')
    validate_cursor(contract, *[saved[k] for k in ('epoch', 'batch', 'updates', 'executed')])
    if saved['batch'] != 0 or (epoch is not None and saved['epoch'] != epoch):
        raise RuntimeError('completed epoch boundary required; no partial last checkpoint')
    configs = saved['model_configs']
    if stable_json_fingerprint(configs) != stable_json_fingerprint(contract['model']):
        raise RuntimeError('training model configuration differs from contract')
    state = saved['state_dict']
    if any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in state.values()):
        raise RuntimeError('nonfinite or non-tensor checkpoint state')
    prior = saved.get('reports', {}).get('train_prior')
    if not isinstance(prior, dict) or 'positive_weights' not in prior:
        raise RuntimeError('missing TRAIN-only positive weights')
    expected = torch.as_tensor(prior['positive_weights'], dtype=torch.float32)
    if not torch.equal(state.get('columns.positive_weight', torch.empty(0)).cpu(), expected):
        raise RuntimeError('TRAIN-only positive weight buffer/report mismatch')
    return prior


def average_named_parameters(states, model):
    """Float64 equal mean of PARAMETERS only; every non-parameter buffer must match."""
    if not states:
        raise ValueError('empty checkpoint average')
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise RuntimeError('BatchNorm needs separate TRAIN recalibration; not supported here')
    template = model.state_dict()
    parameters = set(dict(model.named_parameters()))
    for state in states:
        if set(state) != set(template):
            raise RuntimeError('state keys differ; averaging incompatible architectures rejected')
        for key, value in state.items():
            if (not isinstance(value, torch.Tensor) or value.shape != template[key].shape
                    or value.dtype != template[key].dtype or not torch.isfinite(value).all()):
                raise RuntimeError('state shape/dtype/finiteness mismatch: '+key)
    result = {}
    for key in template:
        if key not in parameters:
            if any(not torch.equal(state[key].cpu(), states[0][key].cpu()) for state in states[1:]):
                raise RuntimeError('fixed buffers differ; never average: '+key)
            result[key] = states[0][key].detach().cpu().clone()
        else:
            if not template[key].is_floating_point():
                raise RuntimeError('non-floating learnable parameter: '+key)
            total = torch.zeros_like(template[key], device='cpu', dtype=torch.float64)
            for state in states:
                total.add_(state[key].detach().cpu().to(torch.float64))
            value = (total/len(states)).to(template[key].dtype)
            if not torch.isfinite(value).all():
                raise RuntimeError('nonfinite averaged parameter: '+key)
            result[key] = value
    return result


def evaluation_payload(state, configs, contract, prior, sources, *, average=False):
    return dict(protocol=PROTOCOL, source_training_protocol=TRAIN_PROTOCOL,
        checkpoint_role='evaluation_only', resume_allowed=False, deployable=False,
        state_dict={k: v.detach().cpu().clone() for k, v in state.items()},
        model_configs=copy.deepcopy(configs), training_contract=copy.deepcopy(contract),
        train_prior=copy.deepcopy(prior), sources=copy.deepcopy(sources),
        source_epochs=[row['epoch'] for row in sources],
        averaging=(dict(method='equal_named_parameter_mean_float64',
                        coefficients=[1/len(sources)]*len(sources), buffers='identical_copy')
                   if average else None),
        weight_fingerprint=weight_fingerprint(state))


def load_evaluation_model(path, *, device='cpu', z_bins=16):
    saved = load_cpu_checkpoint(path)
    if (saved.get('protocol') != PROTOCOL or saved.get('checkpoint_role') != 'evaluation_only'
            or saved.get('resume_allowed') is not False or saved.get('deployable') is not False
            or saved.get('source_training_protocol') != TRAIN_PROTOCOL
            or any(k in saved for k in ('optimizer', 'torch_rng', 'sampling_rng', 'epoch', 'batch'))):
        raise RuntimeError('weights-only evaluation artifact required; not a resume checkpoint')
    if weight_fingerprint(saved['state_dict']) != saved['weight_fingerprint']:
        raise RuntimeError('evaluation weights fingerprint mismatch')
    model = model_from_configs(saved['model_configs'], device=device, z_bins=z_bins)
    model.load_state_dict(saved['state_dict'], strict=True)
    if (not all(torch.isfinite(v).all() for v in model.state_dict().values())
            or not torch.equal(model.columns.positive_weight.cpu(),
                               torch.as_tensor(saved['train_prior']['positive_weights'], dtype=torch.float32))):
        raise RuntimeError('invalid evaluation state/TRAIN buffer')
    model.eval().requires_grad_(False)
    return saved, model


def save_new_weights(path, value):
    """Atomic publication without replacing an existing checkpoint."""
    path = Path(path)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    created = False
    try:
        with temporary.open('xb') as handle:
            created = True
            torch.save(value, handle)
            handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path)  # fails closed if path already exists
    finally:
        if created: temporary.unlink(missing_ok=True)


def discover(run_dir, runs_root):
    """Search retained epochs by identical contract AND saved dev64 report, not name/time."""
    run_dir, runs_root = Path(run_dir).resolve(), Path(runs_root).resolve()
    if not run_dir.is_relative_to(runs_root):
        raise RuntimeError('anchor run must be inside the explicitly named runs root')
    training = json.loads((run_dir/'training.json').read_text(encoding='utf-8'))
    contract = training['contract']
    if (training.get('status') != 'complete' or contract.get('protocol') != TRAIN_PROTOCOL
            or contract.get('history_frames') != 4 or contract.get('future_frames') != 6
            or contract.get('thresholds') != [.5, None]):
        raise RuntimeError('completed FOUR-history clean-joint training required')
    anchor_path = run_dir/'last.pt'; anchor_digest = sha256(anchor_path)
    anchor = load_cpu_checkpoint(anchor_path)
    prior = validate_training_checkpoint(anchor, contract, epoch=contract['epochs'])
    if sha256(anchor_path) != anchor_digest:
        raise RuntimeError('anchor checkpoint changed during read')
    if stable_json_fingerprint(finite_json(anchor['reports'])) != stable_json_fingerprint(training['reports']):
        raise RuntimeError('training JSON and final checkpoint reports disagree')
    reports = {int(r['epoch']): r for r in anchor['reports'].get('epochs', [])}
    if len(reports) != contract['epochs']:
        raise RuntimeError('missing completed dev64 monitors')
    for row in reports.values():
        evaluation = row['evaluation']
        if evaluation.get('windows') != 64:
            raise RuntimeError('checkpoint selection must use the frozen dev64 population')
        if any(not isinstance(evaluation['variants']['joint']['metrics'].get(k), (int, float))
               or not math.isfinite(evaluation['variants']['joint']['metrics'][k]) for k in METRICS):
            raise RuntimeError('nonfinite/missing dev64 selection metrics')
    top = sorted(reports, key=lambda e: (-reports[e]['evaluation']['variants']['joint']['metrics']['mIoU'], e))[:5]
    if set(top) != set(AVERAGE_EPOCHS):
        raise RuntimeError(f'predeclared top-five epochs changed: {top}; do not silently change recipe')
    found = {e: [] for e in AVERAGE_EPOCHS}
    excluded = []
    for directory in sorted(runs_root.iterdir()):
        if not directory.is_dir() or not (directory/'training.json').is_file():
            continue
        try:
            info = json.loads((directory/'training.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if stable_json_fingerprint(info.get('contract')) != stable_json_fingerprint(contract):
            excluded.append(str(directory)); continue
        for epoch in AVERAGE_EPOCHS:
            path = directory/f'epoch_{epoch:04d}.pt'
            if not path.is_file():
                continue
            digest = sha256(path); saved = load_cpu_checkpoint(path)
            candidate_prior = validate_training_checkpoint(saved, contract, epoch=epoch)
            if (stable_json_fingerprint(candidate_prior) != stable_json_fingerprint(prior)
                    or saved['updates'] != reports[epoch]['update']
                    or stable_json_fingerprint(next((r for r in saved['reports']['epochs'] if r['epoch']==epoch), None))
                       != stable_json_fingerprint(reports[epoch])):
                raise RuntimeError('retained epoch differs from anchor lineage/report: '+str(path))
            if sha256(path) != digest:
                raise RuntimeError('source checkpoint changed during read: '+str(path))
            found[epoch].append(dict(epoch=epoch, update=saved['updates'], path=str(path),
                sha256=digest, weight_fingerprint=weight_fingerprint(saved['state_dict'])))
    sources = []
    for epoch, choices in found.items():
        if not choices:
            raise RuntimeError(f'missing retained epoch_{epoch:04d}.pt in matching runs; no fallback/substitution')
        if len({v['weight_fingerprint'] for v in choices}) != 1:
            raise RuntimeError(f'conflicting weights for epoch {epoch}; explicit lineage decision required')
        sources.append(choices[0])
    return dict(contract=contract, configs=anchor['model_configs'], prior=prior, sources=sources,
        anchor=dict(path=str(anchor_path), sha256=anchor_digest, epoch=anchor['epoch'],
                    weight_fingerprint=weight_fingerprint(anchor['state_dict'])),
        dev64=[dict(epoch=e, update=r['update'], **{k:r['evaluation']['variants']['joint']['metrics'][k]
                for k in METRICS}) for e,r in sorted(reports.items())],
        final_dev512=finite_json(anchor['reports']['final_dev512']), excluded_runs=excluded)


def build_bundle(audit, out):
    out = Path(out); states = []; by_epoch = {}
    model = model_from_configs(audit['configs'])
    for source in audit['sources']:
        if sha256(source['path']) != source['sha256']:
            raise RuntimeError('source checkpoint changed before averaging')
        saved = load_cpu_checkpoint(source['path'])
        validate_training_checkpoint(saved, audit['contract'], epoch=source['epoch'])
        state = saved['state_dict']
        if weight_fingerprint(state) != source['weight_fingerprint']:
            raise RuntimeError('source weight fingerprint changed')
        if sha256(source['path']) != source['sha256']:
            raise RuntimeError('source checkpoint changed during averaging')
        # Validate every tensor against the actual architecture, including buffers.
        average_named_parameters([state], model)
        states.append(state); by_epoch[source['epoch']] = (state, source)
    averaged = average_named_parameters(states, model)
    candidates = {}
    for epoch in SINGLE_EPOCHS:
        state, source = by_epoch[epoch]; name = f'epoch_{epoch:04d}'; path = out/(name+'.pt')
        value = evaluation_payload(state,audit['configs'],audit['contract'],audit['prior'],[source])
        save_new_weights(path,value)
        candidates[name] = dict(path=str(path.resolve()),sha256=sha256(path),
                               weight_fingerprint=value['weight_fingerprint'],source_epochs=[epoch])
    path = out/(AVERAGE_NAME+'.pt')
    value = evaluation_payload(averaged,audit['configs'],audit['contract'],audit['prior'],audit['sources'],average=True)
    save_new_weights(path,value)
    candidates[AVERAGE_NAME] = dict(path=str(path.resolve()),sha256=sha256(path),
        weight_fingerprint=value['weight_fingerprint'],source_epochs=list(AVERAGE_EPOCHS))
    return dict(protocol=PROTOCOL,candidates=candidates,audit=audit,
        selection_rule='dev64 joint mIoU top5; equal weights; no coefficient/threshold search',
        resume_allowed=False,original_checkpoints_unchanged=True)
