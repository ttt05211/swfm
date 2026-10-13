"""Standalone Pred forward adapter, not registered with ANY existing trainer.

Hard Strong/CLEAR background is rebuilt with detached predicted poses. Its
discrete choices do not backpropagate. Continuous prior/memory warps DO carry
occupancy gradients into the predicted ego and shared history encoder.
"""
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
import torch
from torch import nn

from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.runtime_fastpath import compose_component_replacements_fast_exact
from real_motion.strong_majority_execution import strong_majority_execution
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
from tools.joint_occ_experiments.data import PURE_STATE
from tools.joint_occ_experiments.model import torch_poses
from tools.joint_occ_experiments.soft_geometry import source_targets, effective_yaw
from tools.shared_history_experiments.model import SharedTransport, encode_window
from tools.shared_history_experiments.refine.model import SemanticRefiner
from tools.shared_history_experiments.refine.geometry import TransportPrior, aligned_memory
from .model import SceneEgoHead, SceneEgoConfig, prepare_scene


class SceneEgoRefiner(nn.Module):
    """Opt-in new structure; never replaces ResidualJoint or its old optimizer."""
    def __init__(self, motion_config, *, feature_width=64, z_bins=16, memory_width=16,
                 width=32, chunk_size=65536, ego_width=128, ego_heads=4, pool_side=8):
        super().__init__()
        self.transport = SharedTransport(motion_config, feature_width, z_bins)
        self.columns = SemanticRefiner(motion_config.d_model, feature_width, z_bins,
                                       memory_width, width, chunk_size)
        self.ego = SceneEgoHead(SceneEgoConfig(feature_width, ego_width, ego_heads, pool_side))

    def configs(self):
        return dict(motion=asdict(self.transport.config), ego=self.ego.specification(),
                    shared_history=dict(width=self.transport.history_encoder.width,
                                        z_bins=self.columns.z_bins, stride=2),
                    repair=dict(type='full_grid_semantic_residual', width=self.columns.width,
                                memory_width=self.columns.memory_width, z_bins=self.columns.z_bins,
                                chunk_size=self.columns.chunk_size, epsilon=self.columns.epsilon),
                    protocol='isolated_scene_ego_refiner_pred_forward_v1')


def rebuild_pred_backgrounds(historical, state, poses, provider):
    """No GT future cache/owner/pose access; hard execution detached explicitly."""
    predicted = poses.detach().cpu().numpy()
    previous = runtime.extract_instances_cropped_exact(historical['history_occ'][-2],
        historical['history_poses'][-2], grid=provider.pcfg.grid, cfg=provider.strong)
    velocities = runtime.match_instances(previous, state['current'], float(provider.pcfg.frame_dt_s),
                                         max_speed_mps=provider.strong.max_match_speed_mps)
    majority = getattr(provider, 'strong_majority', None)
    with strong_majority_execution(majority) if majority is not None else nullcontext():
        anchors, components = runtime._strong_all_horizons(historical['history_occ'][-1],
            state['current_pose'], predicted, state['current'], velocities, state['source_world_points'],
            frame_dt_s=float(provider.pcfg.frame_dt_s), grid=provider.pcfg.grid,
            cfg=provider.strong, runtime_device=provider.device,
            majority_backend='native' if majority is not None else 'sparse_cuda')
    return [compose_component_replacements_fast_exact(a, components_h, [],
        dynamic_class_ids=DYNAMIC_CLASS_IDS, free_label=17, grid=provider.pcfg.grid,
        precomputed_clear_flat_indices=runtime.baseline_clear_flat_indices(components_h, grid=provider.pcfg.grid))
        for a, components_h in zip(anchors, components)]


def forward_pred_window(model, provider, row, *, background_builder=None):
    """Consumes existing PURE history reads, ignores labels and future raw fields.

    This is a fusion interface/gradient proof, NOT a training/evaluation runner.
    Independent losses must receive GT only AFTER this forward returns.
    """
    record, raw, ego_inputs, history = row
    historical = {key: raw[key] for key in ('history_occ', 'history_observed', 'history_poses')}
    causal = {key: history[0][key] for key in ('shared_reads', 'memory_reads', 'registrations')}
    state = {key: history[0]['state'][key] for key in PURE_STATE}
    causal['state'] = state
    clean_history = causal, None
    e, commands = ego_inputs[:2]  # deliberately DO NOT read ego target
    device = provider.device
    cast = lambda x: torch.as_tensor(np.asarray(x), device=device)
    classes = cast(record['source_class_id']).long()
    if (len(state['current']) != len(classes) or
            not np.array_equal(np.asarray(record['source_class_id']), [int(s['class_id']) for s in state['current']])):
        raise RuntimeError('source identity/order differs')
    current = np.asarray(state['current_pose'])
    if not np.array_equal(current, np.asarray(historical['history_poses'][-1])):
        raise RuntimeError('pure historical state does not end at current ego pose')
    xyz = np.asarray([s['centroid_world'] for s in state['current']]).reshape(-1, 3)
    native = (xyz-current[:3, 3]) @ current[:3, :3]
    if not np.allclose(native[:, :2], record['source_centroid_xy_t0_m'], atol=2e-4, rtol=0):
        raise RuntimeError('source centroid/order differs')
    center = np.asarray(record['source_centroid_xy_t0_m'])[:, None]
    if not np.allclose(record['anchors_xy_t0_m'], center+np.asarray(record['kta_displacement_xy_m']), atol=2e-4, rtol=0):
        raise RuntimeError('KTA anchors must be expressed in fixed t0 coordinates')
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        field, reads, times = encode_window(model, historical, clean_history, e, device)
        known = (np.asarray(historical['history_observed'], bool) &
                 (np.asarray(historical['history_occ']) >= 0) & (np.asarray(historical['history_occ']) < 18))
        scene = prepare_scene(field, observed=known, history_poses=historical['history_poses'],
                              ego_history=e, grid=provider.pcfg.grid, pool_side=model.ego.config.pool_side)
        ego = model.ego(cast(np.asarray(e)[None]), cast(np.asarray(commands)[None]), scene)
        output = model.transport(field, reads, times, classes, cast(record['kta_displacement_xy_m']).float())
        lifted = model.columns.lift_history(field)
    poses = torch_poses(current, ego['se2'][0])
    return _forecast(model, provider, historical, causal, state, field, output, lifted,
                     poses, record, ego, background_builder)


def forecast_conditioned(model, provider, row, prepared, poses, *, background_builder=None):
    """Explicit evaluation-only pose intervention on the SAME historical forward.

    External planner/GT reference poses must be supplied by the caller. This is
    NOT called by the internal Pred training route; backgrounds are rebuilt.
    """
    record, raw, _, history = row
    historical = {key: raw[key] for key in ('history_occ', 'history_observed', 'history_poses')}
    causal = {key: history[0][key] for key in ('shared_reads', 'memory_reads', 'registrations')}
    state = {key: history[0]['state'][key] for key in PURE_STATE}
    causal['state'] = state
    poses = torch.as_tensor(np.array(poses, copy=True), device=provider.device, dtype=torch.float64)
    return _forecast(model, provider, historical, causal, state, prepared['field'],
                     prepared['output'], prepared['lifted'], poses, record, None, background_builder)


def _forecast(model, provider, historical, causal, state, field, output, lifted,
              poses, record, ego, background_builder):
    from time import perf_counter
    if poses.shape != (6, 4, 4): raise ValueError('six future poses required')
    if not bool(torch.isfinite(poses).all()):
        raise RuntimeError('nonfinite predicted ego poses; do not run hard geometry')
    tick = perf_counter()
    builder = rebuild_pred_backgrounds if background_builder is None else background_builder
    backgrounds = builder(historical, state, poses, provider)
    if len(backgrounds) != 6 or any(np.asarray(b).shape != np.asarray(historical['history_occ']).shape[1:]
                                   for b in backgrounds):
        raise ValueError('background builder must return six full grids')
    state = dict(state, column_backgrounds=backgrounds)
    background_seconds = perf_counter()-tick
    current = np.asarray(state['current_pose'])
    targets = source_targets(output, record, current, state['source_z_t0'])
    yaws = effective_yaw(output, [int(s['class_id']) for s in state['current']])
    tick = perf_counter()
    prior_renderer = TransportPrior(historical, state, provider.pcfg.grid, provider.device)
    prior = prior_renderer.render(poses, targets, yaws)
    prior_seconds = perf_counter()-tick; tick = perf_counter()
    memory = aligned_memory(model.columns, lifted, output, (causal, None), historical,
                            targets, yaws, poses, provider.pcfg.grid)
    memory_seconds = perf_counter()-tick; tick = perf_counter()
    logits = model.columns(memory, prior, range(6), prior_renderer.shape)
    return dict(ego=ego, poses=poses, field=field, output=output, targets=targets,
                yaws=yaws, prior=prior, memory=memory, logits=logits,
                shape=prior_renderer.shape, state=state, lifted=lifted,
                stages=dict(pred_background=background_seconds, transport_prior=prior_seconds,
                    aligned_feature_memory=memory_seconds, semantic_refiner=perf_counter()-tick))
