"""Strict full model/optimizer/cursor/RNG recovery, not a weights-only transfer."""
import copy
import random
import numpy as np
import torch
from real_motion.joint_surface_ccr import PROTOCOL
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion.height_field_screen_recovery import validate_cursor


def payload(joint, optimizer, rng, contract, *, epoch, batch, updates, executed, reports):
    validate_cursor(contract, epoch, batch, updates, executed)
    return dict(protocol=PROTOCOL, contract=copy.deepcopy(contract), state_dict=joint.state_dict(),
                model_configs=joint.configs(), optimizer=optimizer.state_dict(),
                python_rng=random.getstate(), numpy_global_rng=np.random.get_state(),
                sampling_rng=copy.deepcopy(rng.bit_generator.state), torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                epoch=epoch, batch=batch, updates=updates, executed=executed,
                reports=copy.deepcopy(reports), initialization='RANDOM', transport_frozen=False,
                checkpoint_role='resume_last', deployable=False)


def restore(saved, joint, optimizer, rng, contract):
    if (saved.get('protocol') != PROTOCOL or saved.get('transport_frozen') is not False
            or saved.get('initialization') != 'RANDOM' or saved.get('deployable') is not False
            or stable_json_fingerprint(saved.get('contract')) != stable_json_fingerprint(contract)):
        raise RuntimeError('identical clean-joint contract required; frozen/legacy runs and changed budgets rejected')
    validate_cursor(contract, *[saved[k] for k in ('epoch', 'batch', 'updates', 'executed')])
    reports = saved.get('reports')
    if not isinstance(reports, dict) or 'train_prior' not in reports:
        raise RuntimeError('missing completed TRAIN-only prior')
    joint.load_state_dict(saved['state_dict'], strict=True); optimizer.load_state_dict(saved['optimizer'])
    if not all(torch.isfinite(v).all() for v in joint.state_dict().values()):
        raise RuntimeError('nonfinite clean joint state')
    if not np.array_equal(joint.columns.positive_weight.detach().cpu().numpy(),
                          np.asarray(reports['train_prior']['positive_weights'], np.float32)):
        raise RuntimeError('checkpoint TRAIN weights disagree with model buffers')
    random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_global_rng'])
    rng.bit_generator.state = saved['sampling_rng']; torch.set_rng_state(saved['torch_rng'].cpu())
    if saved['cuda_rng']:
        if not torch.cuda.is_available(): raise RuntimeError('cannot restore CUDA training on CPU')
        torch.cuda.set_rng_state_all([r.cpu() for r in saved['cuda_rng']])
    return tuple(saved[k] for k in ('epoch', 'batch', 'updates', 'executed')), copy.deepcopy(reports)
