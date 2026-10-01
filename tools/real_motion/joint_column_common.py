"""Live geometries, bounded online supervision and fair random-init control."""
from collections import defaultdict
import math
import numpy as np
import torch
from real_motion.causal_column_completion import GENERATE, REFINE, KEEP, ADD, REMOVE, action_targets, sample_queries
from real_motion.causal_column_model import column_loss
from real_motion.local_st_world_model_v18_se2 import periodic_yaw_loss, soft_se2_transport_overlap_loss
from tools.real_motion.causal_column_common import FrozenColumns, candidate_plan, sample_column_features, FEATURE_KEYS, render_column_layers, pose_motion
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
from real_motion.causal_column_sampling import ColumnFeatureSampler


class JointColumnProvider(FrozenColumns):
    def __init__(self, checkpoint, expected_sha, pcfg, device, workers, joint, control):
        super().__init__(checkpoint, expected_sha, pcfg, device, workers)
        self.reference = self.model
        self.joint, self.control = joint, control
        self.model = joint.transport
        self.latents_checked = True
        self.reference_enabled = False

    def encode_record(self, record):
        with torch.no_grad(): return self.joint.motion(record, self.device)

    def reference_predictions(self, prep, record):
        if not self.reference_enabled: return {}
        values = runtime._gpu_inputs(record, self.device)
        result = {}
        for name, model in (('frozen_E14', self.reference), ('paired_scratch_V18_only', self.control)):
            if model is None: continue
            output = runtime._model_forward(model, values, self.device)
            result[name] = render_column_layers(prep.state, record, output, self.pcfg.grid)[0]
        return result


def weights_from_counts(counts):
    gen, ref = counts['generation'], counts['refine']
    present = ref[ref > 0]
    return {'generation_pos_weight': float(np.clip(gen[0]/max(gen[1], 1), 1, 20)),
        'refine_class_weights': np.clip((np.median(present) if len(present) else 1)/np.maximum(ref, 1), .2, 20).tolist(),
        'full_unsampled_generation_counts': gen.tolist(), 'full_unsampled_refine_action_counts': ref.tolist(),
        'population': 'initial_random_transport_TRAIN_only_proposals', 'query_importance_restores_sampling': True}


def count_proposals(prep, grid, config, counts):
    for h in range(6):
        plan = candidate_plan(prep, h, grid, config); target = action_targets(plan, prep.raw['future_gt_occ'][h])
        gen = (plan.kind == GENERATE)[:, None]&plan.legal[..., ADD]
        ref = (plan.kind == REFINE)[:, None]&(plan.legal[..., ADD]|plan.legal[..., REMOVE])
        counts['generation'] += np.bincount((target[gen] == ADD).astype(int), minlength=2)
        counts['refine'] += np.bincount(target[ref], minlength=3)


def build_online_column_candidates(prep, config, grid):
    """CPU-only plans/labels for one current prediction; no sampling/RNG."""
    return [(h, plan, action_targets(plan, prep.raw['future_gt_occ'][h]))
            for h in range(6) for plan in (candidate_plan(prep, h, grid, config),)]


def select_online_columns(prep, config, grid, rng, candidates=None):
    """Draw in original horizon/window order on the caller's sole RNG thread."""
    selected = []
    # 2*sum(budgets) = 256; 128 GEN + 128 REF when all strata are available.
    # Missing strata are not filled with duplicate/replacement queries.
    budgets = (24, 24, 20, 20, 20, 20)
    if candidates is None: candidates = build_online_column_candidates(prep, config, grid)
    for h, plan, labels in candidates:
        budget = budgets[h]
        ids, weight = sample_queries(plan, labels, budget, rng)
        if not len(ids): continue
        small = plan.subset(ids)
        selected.append((h, small, labels[ids], weight))
    return selected


def sample_online_column(prep, selected, grid, config):
    """Pure CPU/NumPy: no RNG, Torch, CUDA or learned latent access."""
    h, small, labels, weight = selected
    features = ColumnFeatureSampler(prep, h, small, grid, config, pose_motion,
        workers=1).sample(small, sample_column_features)
    return {**features, 'legal': small.legal, 'target': labels, 'weight': weight}


def assemble_online_columns(prep, model, selected, arrays, device):
    """Latent gathering stays on the autograd-owning caller, never a worker."""
    parts = defaultdict(list)
    for (h, small, _, _), features in zip(selected, arrays):
        for k, v in features.items(): parts[k].append(v)
        parts['source_features'].append(model.source_features_for(prep, h, small, device))
    if not parts: return None
    batch = {k: torch.as_tensor(np.concatenate(v), device=device) for k, v in parts.items() if k != 'source_features'}
    batch['source_features'] = torch.cat(parts['source_features'])
    if len(batch['kind']) > 256: raise RuntimeError('online query budget exceeded')
    return batch


def online_columns(prep, model, grid, rng, device):
    selected = select_online_columns(prep, model.config, grid, rng)
    arrays = [sample_online_column(prep, row, grid, model.config) for row in selected]
    return assemble_online_columns(prep, model, selected, arrays, device)


def motion_loss(output, record, device, patch_resolution=.8):
    """Original V18 objective; future labels enter this function, NOT motion()."""
    pred = output['residual_xy_m'].float()
    if not len(pred):
        # Empty motion output is not connected to parameters; caller still trains columns.
        return pred.new_zeros(()), {'translation_smooth_l1': 0., 'existence_bce': 0., 'yaw_periodic_loss': 0., 'se2_shape_loss': 0.}
    get = lambda k: torch.as_tensor(record[k], device=device)
    sup = get('supervised_source').bool()
    valid = get('se2_target_valid').bool() & sup[:, None]
    target = get('target_source_residual_xy_m').float()
    trans = torch.nn.functional.smooth_l1_loss(pred[valid], target[valid], beta=1.) if valid.any() else pred.sum()*0.
    logits = output['existence_logits'].float()
    exist = torch.nn.functional.binary_cross_entropy_with_logits(logits[sup], get('existence').float()[sup]) if sup.any() else logits.sum()*0.
    yaw, _ = periodic_yaw_loss(output['yaw_delta_rad'].float(), get('target_yaw_rad').float(),
        get('yaw_enabled'), get('yaw_label_valid').bool() & valid, materialize_stats=False)
    shape, _ = soft_se2_transport_overlap_loss(get('kta_displacement_xy_m').float()+pred,
        get('target_source_displacement_xy_m').float(), output['yaw_delta_rad'].float(), get('target_yaw_rad').float(),
        get('target_source_mask_tube')[:, -1].float(), valid, get('yaw_enabled'), get('yaw_label_valid'),
        patch_resolution_m=patch_resolution, materialize_stats=False)
    values = {'translation_smooth_l1': trans, 'existence_bce': exist, 'yaw_periodic_loss': yaw, 'se2_shape_loss': shape}
    return trans+exist+19.*yaw+.25*shape, {k: float(v.detach()) for k, v in values.items()}


def set_lr(optimizer, update, target):
    factor = .1+.9*.5*(1+math.cos(math.pi*min(update/max(target, 1), 1)))
    for group in optimizer.param_groups: group['lr'] = group['initial_lr']*factor


def train_window(joint, control, optimizer, control_optimizer, provider, source, record, raw, rng,
                 update, target, *, probe=False, patch_resolution=.8):
    joint.train(); control.train(); optimizer.zero_grad(set_to_none=True); control_optimizer.zero_grad(set_to_none=True)
    set_lr(optimizer, update-1, target); set_lr(control_optimizer, update-1, target)
    output = joint.motion(record, provider.device)
    prep = provider.prepare_columns(source, record, include_gt=True, raw_window=raw, outputs=output)
    batch = online_columns(prep, joint.columns, provider.pcfg.grid, rng, provider.device)
    lm, stats = motion_loss(output, record, provider.device, patch_resolution)
    link_grad = 0.; lc = lm.new_zeros(()); cs = {}; sampled = 0
    if batch is not None:
        with torch.autocast(device_type=provider.device.type, dtype=torch.bfloat16, enabled=provider.device.type == 'cuda'):
            g, r = joint.columns(**{k: batch[k] for k in (*FEATURE_KEYS, 'source_features')})
        lc, cs = column_loss(joint.columns, g, r, batch['kind'], batch['legal'], batch['target'], batch['weight'])
        sampled = len(batch['kind'])
        if probe and output['future_transport_queries'].requires_grad and len(output['future_transport_queries']):
            grad = torch.autograd.grad(lc, output['future_transport_queries'], retain_graph=True, allow_unused=True)[0]
            link_grad = float(grad.float().norm()) if grad is not None else 0.
    loss = lm+lc
    if not loss.requires_grad:
        # A valid stationary/empty window need not contain any supervised edit.
        # Do not decay parameters or count it as a successful update.
        return {'loss': 0., 'motion_loss': 0., 'column_loss': 0., 'paired_control_motion_loss': 0.,
            'grad_norm': 0., 'column_grad_norm': 0., 'source_query_gradient_norm': 0. if probe else None,
            'gradient_probe': probe, 'sampled_columns': 0, 'sources': 0, 'optimizer_updated': False, **stats}
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(joint.transport.parameters(), 5., error_if_nonfinite=True)
    column_norm = torch.nn.utils.clip_grad_norm_(joint.columns.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    # Same architecture/input/objective/order/LR; independent parameters and optimizer.
    keys = runtime._gpu_inputs(record, provider.device)
    with torch.autocast(device_type=provider.device.type, dtype=torch.bfloat16, enabled=provider.device.type == 'cuda'):
        co = control(keys['features'], keys['tube'], keys['kta'], keys['frame_motion'], keys['source_mask'])
    control_loss, _ = motion_loss(co, record, provider.device, patch_resolution)
    if control_loss.requires_grad:
        control_loss.backward(); torch.nn.utils.clip_grad_norm_(control.parameters(), 5., error_if_nonfinite=True); control_optimizer.step()
    return {'loss': float(loss.detach()), 'motion_loss': float(lm.detach()), 'column_loss': float(lc.detach()),
        'paired_control_motion_loss': float(control_loss.detach()), 'grad_norm': float(norm), 'column_grad_norm': float(column_norm),
        'source_query_gradient_norm': link_grad if probe else None, 'gradient_probe': probe,
        'sampled_columns': sampled, 'sources': len(record['features']), 'optimizer_updated': True, **stats, **cs}
