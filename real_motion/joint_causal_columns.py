"""One-stage source transport + linked refinement. Hard geometry has no gradient."""
from dataclasses import asdict
import torch
from torch import nn
from .causal_column_model import CausalColumnModel
from .causal_column_completion import ColumnConfig
from .local_st_world_model_v18_se2 import LocalSpatialTemporalWorldModelV18SE2
from .local_st_world_model_v17 import LocalSTWMV17Config

PROTOCOL = 'p0_f9_joint_causal_columns_screen_v1'
LINK_PROTOCOL = 'dynamic_source_future_query_to_refine_no_detach_v1'
CONTRACT = {'initialization': 'random_both_transport_and_columns_no_E14_weights',
    'geometry': 'original_hard_SE2_source_renderer_stop_gradient', 'source_link': LINK_PROTOCOL,
    'supervision': 'original_V18_SE2_plus_importance_corrected_column_loss',
    'candidates_labels': 'online_current_prediction_every_update',
    'calibration': 'final_held_out_TRAIN64_fixed_grid_no_dev_selection',
    'control': 'identical_initial_transport_window_order_motion_loss_and_schedule',
    'schedule': 'fixed_updates_cosine_floor0p1', 'selection': 'final_only_no_dev_best',
    'report_horizons': [1., 2., 3.], 'generation_classes': [11, 13],
    'query_budget': '256_max_per_window_all_six_horizons_no_training_duplication',
    'dynamic_context': 'future_query_from_history_not_GT_motion',
    'motion_lr': 5e-4, 'column_lr': 3e-4, 'motion_weight_decay': 1e-4,
    'column_weight_decay': .01, 'yaw_weight': 19., 'shape_weight': .25,
    'motion_clip_norm': 5., 'column_clip_norm': 1.}

FULL_PROTOCOL = 'p0_f9_joint_causal_columns_full_train_v1'
FULL_CONTRACT = {**CONTRACT, 'control': 'optional_paired_control_default_off_E14_reference_only',
    'schedule': 'whole_configured_epochs_cosine_floor0p1_no_tail',
    'calibration': 'final_TRAIN64_in_sample_fixed_grid_no_dev_selection',
    'prior_population': 'deterministic_TRAIN1024_unsampled_proposals_once',
    'training_population': 'ALL_20430_unique_windows_every_epoch',
    'batch_objective': 'all_sources_motion_mean_and_concatenated_columns_type_mean',
    'selection': 'final_epoch_only_with_fixed_gate_dev64_each_epoch',
    'prefetch': 'one_next_window_batch_causal_CPU_only_no_model_dependent_cache'}


class LinkedColumns(CausalColumnModel):
    extra_input_keys = ('source_features',)
    def __init__(self, config, source_dim):
        super().__init__(config)
        self.source_dim = int(source_dim)
        self.source_projection = nn.Linear(source_dim, config.width, bias=False)
        nn.init.normal_(self.source_projection.weight, std=1e-3)

    def source_features_for(self, prepared, h, plan, device):
        if prepared.outputs is None: raise RuntimeError('current live transport latents required')
        q = prepared.outputs['future_transport_queries']
        if not isinstance(q, torch.Tensor) or q.shape[1:] != (6, self.source_dim):
            raise RuntimeError('source feature protocol mismatch')
        actor = torch.as_tensor(plan.actor, device=device)
        active = actor >= 0
        if active.any() and int(actor[active].max()) >= len(q): raise RuntimeError('actor/source order mismatch')
        # index_copy is differentiable; static/frontier get exactly zero (no bias).
        result = q.new_zeros((len(plan), self.source_dim))
        if active.any(): result = result.index_copy(0, torch.nonzero(active).flatten(), q[actor[active].long(), h])
        return result

    def forward(self, history, flags, base, fallback, context, kind, classes, *, source_features):
        if source_features.shape != (len(kind), self.source_dim) or not torch.isfinite(source_features).all():
            raise RuntimeError('invalid continuous source features')
        return super().forward(history, flags, base, fallback, context, kind, classes,
            query_extra=self.source_projection(source_features))


class JointCausalColumns(nn.Module):
    def __init__(self, motion_config=LocalSTWMV17Config(), column_config=ColumnConfig(), context_config=None):
        super().__init__()
        self.transport = LocalSpatialTemporalWorldModelV18SE2(motion_config)
        if context_config is None:
            self.columns = LinkedColumns(column_config, motion_config.d_model)
        else:
            from .adaptive_column_context import AdaptiveLinkedColumns
            self.columns = AdaptiveLinkedColumns(column_config, motion_config.d_model, context_config)

    def motion(self, record, device):
        keys = ('features', 'local_semantic_tube', 'kta_displacement_xy_m',
                'frame_motion_features', 'target_source_mask_tube')
        values = [torch.as_tensor(record[k], device=device) for k in keys]
        values[0], values[2], values[3] = values[0].float(), values[2].float(), values[3].float()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            return self.transport(*values, return_latents=True)

    def configs(self):
        result = {'motion': asdict(self.transport.config), 'columns': asdict(self.columns.config)}
        if hasattr(self.columns, 'context_config'): result['adaptive_context'] = asdict(self.columns.context_config)
        return result
