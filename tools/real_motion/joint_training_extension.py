"""Explicit completed-run continuation; ordinary resume remains recipe-exact."""
import copy
import math

from real_motion.joint_causal_columns import (
    FULL_PROTOCOL, FULL4_PROTOCOL, FULL_EXT_PROTOCOL, FULL4_EXT_PROTOCOL,
    FULL_EXT_CONTRACT, FULL4_EXT_CONTRACT,
)
from real_motion.v21_source_induction import stable_json_fingerprint


EXT_PROTOCOLS = (FULL_EXT_PROTOCOL, FULL4_EXT_PROTOCOL)


def extension_identity(checkpoint, identity, plans, parent_sha):
    """Only a fully completed original last.pt can start ONE explicit extension."""
    ck = checkpoint
    mutable = {'protocol', 'training_contract', 'epochs', 'target_updates', 'schedule_steps'}
    if ck.get('checkpoint_role') != 'resume_last' or ck.get('protocol') not in (FULL_PROTOCOL, FULL4_PROTOCOL):
        raise RuntimeError('extension requires completed original full last.pt, not candidate/epoch/already extended')
    for key, value in identity.items():
        if key not in mutable and stable_json_fingerprint(ck.get(key)) != stable_json_fingerprint(value):
            raise RuntimeError('extension recipe/population mismatch at '+key)
    old_epochs = ck['epochs']; old_steps = sum(map(len, plans[:old_epochs]))
    if (not 0 < old_epochs < identity['epochs'] or ck.get('cursor_epoch') != old_epochs or ck.get('cursor_batch') != 0
            or ck.get('attempted_updates') != old_steps or ck.get('target_updates') != old_steps
            or ck.get('schedule_steps') != old_steps or ck.get('executed_windows') != old_epochs*len(identity['train_keys'])
            or not ck.get('prior_completed', True) or 'optimizer' not in ck):
        raise RuntimeError('extension requires fully completed original epoch/optimizer/prior/counters')
    lrs = [float(g['lr']) for g in ck['optimizer']['param_groups']]
    if len(lrs) != 2 or any(not math.isfinite(v) or v <= 0 for v in lrs):
        raise RuntimeError('invalid extension endpoint learning rates')
    result = copy.deepcopy(identity)
    result.update(protocol=FULL4_EXT_PROTOCOL if ck['protocol'] == FULL4_PROTOCOL else FULL_EXT_PROTOCOL,
                  training_contract=FULL4_EXT_CONTRACT if ck['protocol'] == FULL4_PROTOCOL else FULL_EXT_CONTRACT,
                  continuation={'version': 1, 'parent_checkpoint_sha256': parent_sha,
                      'original_epochs': old_epochs, 'original_updates': old_steps,
                      'extension_updates': identity['target_updates']-old_steps,
                      'start_learning_rates': lrs, 'floor_ratio': .1,
                      'schedule': 'endpoint_lr_cosine_no_reset_original_initial_lr'})
    return result


def validate_extension(identity, plans):
    c = identity.get('continuation')
    if identity['protocol'] not in EXT_PROTOCOLS:
        if c is not None: raise RuntimeError('continuation metadata on original protocol')
        return
    if not isinstance(c, dict) or c.get('version') != 1: raise RuntimeError('missing extension metadata')
    e = c.get('original_epochs'); rates = c.get('start_learning_rates', [])
    if (type(e) is not int or not 0 < e < identity['epochs']
            or c.get('original_updates') != sum(map(len, plans[:e]))
            or c.get('extension_updates') != identity['target_updates']-c['original_updates']
            or c.get('floor_ratio') != .1 or len(rates) != 2
            or any(not math.isfinite(v) or v <= 0 for v in rates)
            or c.get('schedule') != 'endpoint_lr_cosine_no_reset_original_initial_lr'
            or len(c.get('parent_checkpoint_sha256', '')) != 64):
        raise RuntimeError('invalid extension schedule/identity')


def set_extension_lr(optimizer, completed_update, continuation):
    offset = completed_update-continuation['original_updates']
    steps = continuation['extension_updates']
    if not 0 <= offset < steps: raise RuntimeError('extension update outside declared schedule')
    # First added update is exactly the last published LR; final added update
    # reaches the declared floor. No optimizer/initial_lr or Adam moments reset.
    phase = offset/max(steps-1, 1)
    factor = .1+.9*.5*(1+math.cos(math.pi*phase))
    rates = continuation['start_learning_rates']
    if len(optimizer.param_groups) != len(rates): raise RuntimeError('extension optimizer group mismatch')
    for group, start in zip(optimizer.param_groups, rates): group['lr'] = start*factor
