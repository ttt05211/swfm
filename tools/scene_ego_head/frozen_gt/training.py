"""Only SceneEgoHead Adam updates; full TRAIN fixed final-epoch geometry loss."""
import math
from pathlib import Path
import numpy as np
import torch
from tools.joint_occ_experiments import run as legacy
from tools.joint_occ_experiments.model import ego_geometry_loss
from tools.real_motion.joint_training_recovery import save_resume_checkpoint
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint, trajectory_report
from .bank import pack

PROTOCOL = 'full_gt_frozen_scene_ego_head_r10_v1'


@torch.no_grad()
def report(head, bank, batch_size):
    head.eval(); pred, prior = [], []
    device = next(head.parameters()).device
    for start in range(0, len(bank['target']), batch_size):
        e, commands, scene, _ = pack(bank, range(start, min(start+batch_size, len(bank['target']))), device)
        out = head(e, commands, scene)
        pred.append(out['se2'].cpu().numpy()); prior.append(out['prior'].cpu().numpy())
    target, commands = bank['target'].numpy(), bank['commands'].numpy()
    return {n: trajectory_report(np.concatenate(v), target, commands)
            for n, v in (('learned', pred), ('prior', prior))}


def train(head, bank, directory, contract, *, stop_event=None, resume=False):
    directory = Path(directory); schedule = contract['schedule']
    epochs, batch_size, seed, lr = (schedule[k] for k in ('epochs', 'batch_size', 'seed', 'lr'))
    device = next(head.parameters()).device; count = len(bank['target'])
    per_epoch = math.ceil(count/batch_size); total = epochs*per_epoch
    save_epochs = contract.get('save_epochs', [epochs])
    if (not save_epochs or len(set(save_epochs)) != len(save_epochs)
            or any(type(n) is not int or not 1 <= n <= epochs for n in save_epochs)):
        raise ValueError('fixed checkpoint epochs must lie inside configured cycle')
    adam = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=.01)
    initial = tensor_state_fingerprint(head.state_dict())
    epoch = cursor = updates = 0; history = []; epoch_loss = 0.; epoch_windows = 0
    path = directory/'head_last.pt'
    if resume and path.exists():
        old = torch.load(path, map_location='cpu', weights_only=False)
        if old['protocol'] != PROTOCOL or old['contract'] != contract or old['initializer'] != initial:
            raise RuntimeError('head checkpoint source/schedule/initializer changed')
        epoch, cursor, updates = (old[k] for k in ('epoch', 'cursor', 'updates'))
        history, epoch_loss, epoch_windows = (old[k] for k in ('history', 'epoch_loss', 'epoch_windows'))
        if (any(type(v) != int for v in (epoch, cursor, updates, epoch_windows)) or not 0 <= epoch <= epochs
                or not 0 <= cursor < per_epoch or epoch == epochs and cursor != 0
                or updates != epoch*per_epoch+cursor or len(history) != epoch
                or epoch_windows != min(cursor*batch_size, count) or not math.isfinite(epoch_loss)):
            raise RuntimeError('invalid completed head-update cursor')
        head.load_state_dict(old['head'], strict=True); adam.load_state_dict(old['adam']); legacy.restore_rng(old['rng'])
    def persist(status):
        save_resume_checkpoint(path, dict(protocol=PROTOCOL, contract=contract, initializer=initial,
            head={k: v.detach().cpu().clone() for k, v in head.state_dict().items()}, adam=adam.state_dict(),
            epoch=epoch, cursor=cursor, updates=updates, rng=legacy.rng_state(), history=history,
            epoch_loss=epoch_loss, epoch_windows=epoch_windows, status=status))
    def export_epoch():
        if not epoch or epoch not in save_epochs and epoch != epochs: return
        export = directory/f'head_epoch{epoch}.pt'
        payload = dict(protocol=PROTOCOL, contract=contract, head=head.state_dict(), epochs=epoch,
            updates=updates, final_epoch_only=epoch == epochs, state_sha256=tensor_state_fingerprint(head.state_dict()))
        if export.exists():
            old = torch.load(export, map_location='cpu', weights_only=False)
            if (old['contract'] != contract or old['state_sha256'] != payload['state_sha256'] or
                    tensor_state_fingerprint(old['head']) != old['state_sha256']):
                raise RuntimeError('head epoch export changed; no overwrite')
        else: save_resume_checkpoint(export, payload)
    # Recover a boundary snapshot if interruption happened after head_last publication.
    if cursor == 0: export_epoch()
    while epoch < epochs:
        order = np.random.default_rng(seed+epoch).permutation(count)
        while cursor < per_epoch:
            if stop_event is not None and stop_event.is_set():
                persist('stopped'); raise InterruptedError('head complete update saved')
            ids = order[cursor*batch_size:(cursor+1)*batch_size]
            e, commands, scene, target = pack(bank, ids, device)
            rate = lr*(.01+.99*.5*(1+math.cos(math.pi*updates/max(1, total-1))))
            for group in adam.param_groups: group['lr'] = rate
            head.train(); adam.zero_grad(set_to_none=True)
            loss = ego_geometry_loss(head(e, commands, scene)['se2'], target)
            if not bool(torch.isfinite(loss)): raise RuntimeError('nonfinite head loss; no retry')
            loss.backward(); norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
            adam.step(); updates += 1; cursor += 1
            epoch_loss += float(loss.detach())*len(ids); epoch_windows += len(ids)
            if updates % 32 == 0:
                item = dict(event='frozen_ego_train', epoch=epoch+1, update=updates, lr=rate,
                            loss=float(loss.detach()), grad_norm=float(norm))
                with (directory/'head_progress.jsonl').open('a', encoding='utf-8') as f:
                    f.write(legacy.json.dumps(item)+'\n')
                print('FROZEN_EGO_TRAIN '+legacy.json.dumps(item), flush=True)
                if cursor < per_epoch: persist('running')
        epoch += 1; history.append(dict(epoch=epoch, update=updates, train_mean_R10=epoch_loss/epoch_windows))
        cursor = 0; epoch_loss = 0.; epoch_windows = 0
        persist('complete' if epoch == epochs else 'running')
        export_epoch()
        print(f'FROZEN_EGO_EPOCH {epoch}/{epochs} train_R10={history[-1]["train_mean_R10"]:.6f}', flush=True)
    export = directory/f'head_epoch{epochs}.pt'
    result = dict(status='complete', complete_epochs=epochs, updates=updates, train_windows=count,
                  report=report(head, bank, batch_size), history=history, head_checkpoint=str(export),
                  train_report_in_sample=True, WM_STWM_Refiner_updated=False)
    legacy.write_json(directory/'head_training.json', result)
    return result
