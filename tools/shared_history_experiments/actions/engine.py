"""GT full-six objective and matching continuous dual-action inference."""
import time
import numpy as np
import torch
from tools.shared_history_experiments.model import encode_window
from tools.shared_history_experiments.speed.engine import train_arm_grouped
from tools.joint_occ_experiments.data import prepare_projection
from tools.joint_occ_experiments.soft_geometry import source_targets, effective_yaw, continuous_context
from tools.joint_occ_experiments.soft_inference.fast import predict
from tools.joint_occ_experiments.soft_inference.engine import soft_predictions
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import Metrics
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.shared_evidence_pilot_common import moving_support_masks
from real_motion.nuscenes_adapter import gt_moving_support_sequence

EVAL_PROTOCOL = 'shared_history_soft_ADD_REMOVE_eval_v1'
ROUTES = ('transport', 'add_only', 'add_remove')


def forward_window(model, provider, row):
    record, raw, ego_inputs, history = row
    e, commands, target = ego_inputs
    device = provider.device
    cast = lambda x: torch.as_tensor(np.asarray(x), device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        field, reads, times = encode_window(model, raw, history, e, device)
        output = model.transport(field, reads, times, cast(record['source_class_id']).long(),
                                 cast(record['kta_displacement_xy_m']).float())
        ego = model.ego(cast(e[None]), cast(commands[None]))['se2'][0]
        encoded = model.columns.encode_history(field, reads, times, history[1])
    poses = cast(raw['future_poses']).double()
    prep, evidence, plan = prepare_projection(provider, record, raw, output,
        poses.detach().cpu().numpy(), history, use_gt_cache=True)
    targets = source_targets(output, record, prep.state['current_pose'], prep.state['source_z_t0'])
    yaws = effective_yaw(output, [int(c['class_id']) for c in prep.state['current']])
    mapped, context = continuous_context(evidence, plan, prep.state, targets, yaws, poses, provider.pcfg.grid)
    actor = cast(evidence.actor).long()
    actions = []
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        for first in range(0, len(evidence), 8192):
            at = slice(first, first + 8192)
            actions.append(model.columns(encoded[at], output, actor[at], context[at],
                cast(plan.base[at]), cast(plan.legal[at]), fallback=cast(plan.fallback[at])))
    actions = torch.cat(actions) if actions else field.new_empty((0, 6, 2), dtype=torch.float32)
    return dict(output=output, ego=ego, ego_target=target, poses=poses, targets=targets, yaws=yaws,
        prep=prep, evidence=evidence, plan=plan, mapped=mapped, actions=actions, field=field)


def train_arm(model, optimizer, provider, rows, mode, **kwargs):
    # Start with the established sequential H2 schedule, not an unmeasured
    # packed backend. Full six horizons are still supervised every update.
    return train_arm_grouped(model, optimizer, provider, rows, mode,
        horizon_group=2, forward_fn=forward_window, **kwargs)


@torch.no_grad()
def evaluate(models, provider, source, batches, *, remove_threshold=.95, stop_event=None,
             progress=None, saved=None, save=None):
    if set(models) != {'gt'}:
        raise ValueError('GT-only dual-action pilot')
    model = models['gt']; model.eval()
    metrics = {'soft_' + route: Metrics() for route in ROUTES}
    windows, errors, verified = 0, [], False
    if saved:
        if saved.get('protocol') != EVAL_PROTOCOL or set(saved['metrics']) != set(metrics):
            raise RuntimeError('dual-action evaluation protocol/readouts differ')
        windows, errors, verified = int(saved['windows']), list(saved['errors']), saved['verified']
        if windows < 0 or len(errors) != windows or windows and verified is not True:
            raise RuntimeError('invalid completed evaluation cursor')
        for name, fields in saved['metrics'].items():
            if set(fields) != set(vars(metrics[name])):
                raise RuntimeError('metric fields differ')
            for key, value in fields.items():
                array = np.asarray(value, np.int64)
                if array.shape != getattr(metrics[name], key).shape or np.any(array < 0) or not np.array_equal(array, value):
                    raise RuntimeError('invalid integer evaluation counts')
                setattr(metrics[name], key, array)

    def snapshot():
        return dict(protocol=EVAL_PROTOCOL, windows=windows, errors=list(errors), verified=verified,
            metrics={n: {k: v.tolist() for k, v in vars(m).items()} for n, m in metrics.items()})

    begun, offset = time.perf_counter(), 0
    for rows in batches:
        for row in rows:
            offset += 1
            if offset <= windows:
                continue
            if stop_event is not None and stop_event.is_set():
                if save: save(snapshot())
                raise InterruptedError('saved after complete dual-action evaluation window')
            fw = forward_window(model, provider, row)
            predictions = {name: predict(provider, fw, name, range(6), group_size=3) for name in ROUTES}
            if not verified:
                expected = soft_predictions(provider, fw, ROUTES, range(6))
                if any(not np.array_equal(a, b) for name in ROUTES for a, b in zip(predictions[name], expected[name])):
                    raise RuntimeError('dual-action fast/reference six-frame mismatch')
            # All three complete predictions precede future labels/support.
            raw = row[1]; window = window_from_record(row[0])
            support = gt_moving_support_sequence(source.nusc, window.t0_token, window.future_tokens,
                tuple(.5 * (h + 1) for h in range(6)), grid=provider.pcfg.grid, workers=provider.workers)
            moving = moving_support_masks(support, raw['future_gt_occ'].shape[1:])
            for name in ROUTES:
                for i, h in enumerate((1, 3, 5)):
                    metrics['soft_' + name].update(i, predictions[name][h], raw['future_gt_occ'][h], moving[h])
            error = fw['ego'].float().cpu().numpy() - np.asarray(fw['ego_target'])
            errors.append(dict(xy=np.linalg.norm(error[:, :2], axis=-1).tolist(),
                yaw=(np.abs(np.arctan2(np.sin(error[:, 2]), np.cos(error[:, 2]))) * 180 / np.pi).tolist()))
            windows += 1; verified = True
            if windows % 8 == 0 and save: save(snapshot())
            if windows % 16 == 0:
                if progress: progress(dict(event='shared_actions_eval', windows=windows))
                print(f'SHARED_ACTIONS_EVAL {windows} ' + str({n: m.compute()['mIoU'] for n, m in metrics.items()}), flush=True)
    if save: save(snapshot())
    return dict(windows=windows, variants={'gt': {n: m.compute() for n, m in metrics.items()}},
        trajectory={'gt': dict(xy_mean_m=np.mean([e['xy'] for e in errors], 0).tolist(),
            yaw_mean_deg=np.mean([e['yaw'] for e in errors], 0).tolist())} if errors else {},
        seconds=time.perf_counter() - begun, fast_six_frame_byte_parity=verified,
        main_readout='soft_add_remove', scoring='full grid / GT future ego / NO metric mask')
