"""Read-only Local lineage audit. Directory names never identify an experiment."""
from pathlib import Path
import hashlib
import json
import math
import torch

from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion.manage_p0_f9_joint_training import model_directory
from tools.real_motion.static_evidence_selector_common import finite_json
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import sha256

PROTOCOL = 'p0_f9_local_checkpoint_selection_v1'
IDENTITY_KEYS = ('reference_checkpoint_sha256', 'runtime_config_fingerprint', 'model_configs',
    'train_keys', 'dev_keys', 'dev_manifest_fingerprint', 'seed', 'window_batch_size', 'source_budget',
    'source_link', 'info_fingerprints', 'cache_fingerprints')


def identity(value):
    return stable_json_fingerprint({k: value.get(k) for k in IDENTITY_KEYS})


def load_cpu_checkpoint(path):
    # Trusted experiment artifacts only. mmap avoids materializing Adam tensors.
    try: return torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    except RuntimeError as error:
        if 'mmap' not in str(error): raise
        return torch.load(path, map_location='cpu', weights_only=False)


def weight_fingerprint(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        if not isinstance(value, torch.Tensor): raise RuntimeError('non-tensor model state')
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def lineage(run, runs_root):
    root = Path(runs_root).resolve(); current = model_directory(run); seen = set(); rows = []
    while current is not None:
        if not current.is_relative_to(root): raise RuntimeError('resume ancestor outside explicit runs root')
        if current in seen: raise RuntimeError('cyclic resume lineage')
        seen.add(current)
        contract = json.loads((current/'execution_contract.json').read_text(encoding='utf-8'))
        if rows and identity(contract) != identity(rows[0]['contract']):
            raise RuntimeError('ancestor scientific identity/history budget changed')
        rows.append(dict(directory=str(current), contract=contract))
        parent = contract.get('arguments', {}).get('resume')
        if not parent: break
        parent = Path(parent)
        if not parent.is_absolute():
            if 'launch_cwd' not in contract: raise RuntimeError('relative resume ancestor without launch cwd')
            parent = Path(contract['launch_cwd'])/parent
        current = model_directory(parent.resolve().parent)
    return rows


def score_row(row):
    values = row['dev64_fixed_gate']['variants']['joint']['metrics']
    return {k: values[k] for k in ('mIoU', 'IoU', 'MovingMicro', 'MovingMacro')}


def shortlist(rows, count=8):
    if count < 4: raise ValueError('at least four shortlist slots required')
    candidates = [r for r in rows if r.get('checkpoint') and all(
        isinstance(r['dev64'].get(k), (float, int)) and math.isfinite(r['dev64'][k])
        for k in ('mIoU', 'MovingMicro'))]
    if not candidates: raise RuntimeError('no available finite-scored epoch snapshots')
    chosen = []
    def add(row, reason):
        if row in chosen: row['selection_reasons'].append(reason); return
        if len(chosen) < count:
            row['selection_reasons'] = [reason]; chosen.append(row)
    final = max(candidates, key=lambda r: r['epoch']); add(final, 'latest_available_epoch')
    for row in candidates:
        if row['epoch'] == 15: add(row, 'original15_reference')
    overall = sorted(candidates, key=lambda r: (-r['dev64']['mIoU'], -r['dev64']['MovingMicro'], r['epoch']))
    moving = sorted(candidates, key=lambda r: (-r['dev64']['MovingMicro'], -r['dev64']['mIoU'], r['epoch']))
    for row in overall[:max(1, count-4)]: add(row, 'high_dev64_joint_mIoU')
    for row in moving[:2]: add(row, 'high_dev64_MovingMicro')
    for row in overall: add(row, 'fill_by_dev64_joint_mIoU')
    return sorted(chosen, key=lambda r: r['epoch'])


def audit_runs(run, runs_root, count=8):
    chain = lineage(run, runs_root); anchor = chain[0]['contract']; scientific = identity(anchor)
    frames = anchor['model_configs']['motion']['history_frames']
    if frames != 4: raise RuntimeError('this selection workflow requires strict four-history Local')
    history = {}
    for item in reversed(chain):
        path = Path(item['directory'])/'epoch_history.json'
        for row in json.loads(path.read_text(encoding='utf-8')) if path.is_file() else []:
            epoch = int(row['epoch']); scores = finite_json(score_row(row))
            if epoch in history and (row['attempted_updates'], scores) != (history[epoch]['update'], history[epoch]['dev64']):
                raise RuntimeError('conflicting inherited dev64 record at epoch '+str(epoch))
            history[epoch] = dict(epoch=epoch, update=row['attempted_updates'], dev64=scores,
                score_source=str(path), checkpoint=None, selection_reasons=[])
    alternatives = {}; excluded = []
    for item in chain:
        directory = Path(item['directory'])
        files = list(sorted(directory.glob('epoch_*.pt')))+[directory/'candidate.pt', directory/'last.pt']
        for path in files:
            if not path.is_file(): continue
            ck = load_cpu_checkpoint(path)
            if identity(ck) != scientific: raise RuntimeError('checkpoint/lineage identity mismatch: '+str(path))
            epoch = ck.get('cursor_epoch'); role = ck.get('checkpoint_role')
            if (epoch not in history or ck.get('cursor_batch') != 0 or not ck.get('prior_completed', True)
                    or ck.get('attempted_updates') != history[epoch]['update']
                    or role not in ('epoch_snapshot', 'calibrated_candidate', 'resume_last')):
                excluded.append(dict(path=str(path), reason='not_a_matching_complete_epoch')); continue
            fingerprint = weight_fingerprint(ck['state_dict']); file_sha = sha256(path)
            entry = dict(path=str(path.resolve()), sha256=file_sha, weight_fingerprint=fingerprint, role=role)
            if epoch in alternatives and alternatives[epoch][0]['weight_fingerprint'] != fingerprint:
                raise RuntimeError('different weights for the same epoch/update: '+str(epoch))
            alternatives.setdefault(epoch, []).append(entry)
    for epoch, row in history.items():
        available = alternatives.get(epoch, [])
        available.sort(key=lambda a: ({'epoch_snapshot': 0, 'calibrated_candidate': 1, 'resume_last': 2}[a['role']], a['path']))
        if available: row['checkpoint'] = available[0]
        row['aliases'] = available
    rows = sorted(history.values(), key=lambda r: r['epoch']); selected = shortlist(rows, count)
    directories = []; chain_dirs = {r['directory'] for r in chain}
    for path in sorted(Path(runs_root).resolve().glob('full*/model/execution_contract.json')):
        contract = json.loads(path.read_text(encoding='utf-8'))
        directories.append(dict(directory=str(path.parent), in_lineage=str(path.parent) in chain_dirs,
            history_frames=contract.get('model_configs', {}).get('motion', {}).get('history_frames')))
    result = dict(protocol=PROTOCOL, run_directory=str(Path(run).resolve()), runs_root=str(Path(runs_root).resolve()),
        scientific_identity=scientific, history_frames=frames, thresholds=[.5, .5, None],
        lineage=[r['directory'] for r in chain], directories=directories, epochs=rows, selected=selected,
        missing_weight_epochs=[r['epoch'] for r in rows if not r['checkpoint']], excluded=excluded,
        note='dev64 shortlist of RETAINED weights, not best among missing epochs; no deletion or promotion')
    result['selection_fingerprint'] = stable_json_fingerprint(result)
    return result


def audit_text(audit):
    lines = ['===== LOCAL CHECKPOINT LINEAGE / DEV64 AUDIT =====',
        f"history={audit['history_frames']} -> 6; lineage_directories={len(audit['lineage'])}",
        'epoch update    joint_mIoU MovingMicro weight_available selected']
    selected = {r['epoch'] for r in audit['selected']}
    def number(value): return 'NA' if value is None else f'{value:.6f}'
    for r in audit['epochs']:
        lines.append(f"{r['epoch']:5} {r['update']:8} {number(r['dev64']['mIoU']):>11} "
            f"{number(r['dev64']['MovingMicro']):>11} {bool(r['checkpoint'])!s:>16} {r['epoch'] in selected}")
    lines += ['missing_weight_epochs='+str(audit['missing_weight_epochs']),
        'selected_epochs='+str(sorted(selected)), 'All directories/checkpoints are READ ONLY.']
    return '\n'.join(lines)+'\n'
