"""Optional real server-export replay check on local actual CUDA, no nuScenes.

SWFM_HEIGHT_FIELD_REPLAY=<user-owned replay.zip> enables this integration test.
No local replay file path is built into CI/server and no original weights edited.
"""
import copy
from contextlib import closing
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

pytestmark = pytest.mark.integration


def test_real_replay_training_recovery_and_six_complete_predictions(tmp_path, monkeypatch):
    path = os.environ.get('SWFM_HEIGHT_FIELD_REPLAY')
    if not path:
        pytest.skip('optional user-provided replay required')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip('actual CUDA/BF16 required')
    from real_motion.local_replay_bundle import ReplayBundle, file_digest
    from real_motion.runtime_config import validate_runtime_config, make_prepare_config
    from real_motion.v21_source_induction import stable_json_fingerprint
    from real_motion.height_causal_field import HeightCausalField
    from real_motion.causal_column_completion import actions_from_probabilities, compose_dense
    from tools.real_motion.run_p0_f9_shared_evidence_pilot import PilotProvider
    from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint
    from tools.real_motion.joint_column_full_common import build_fixed_geometry
    from tools.real_motion.height_field_screen_common import train_step, rows_for, probabilities, six_frame_speed, evaluate
    from tools.real_motion import height_field_screen_common as common
    from tools.real_motion.height_field_screen_recovery import payload, restore
    from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256
    device = torch.device('cuda'); torch.set_num_threads(1)
    with closing(ReplayBundle(Path(path))) as bundle:
        for member in ('runtime.yaml', 'checkpoints/epoch_0019.pt', 'checkpoints/clean_e14.pt'):
            bundle.copy_member(member, tmp_path/Path(member).name)
        cfg = yaml.safe_load((tmp_path/'runtime.yaml').read_text(encoding='utf-8'))
        assert stable_json_fingerprint(cfg) == bundle.manifest['config_fingerprint']
        validate_runtime_config(cfg)
        _, teacher = load_joint(tmp_path/'epoch_0019.pt', device, reference_sha=CLEAN_SHA256,
                                config_sha=bundle.manifest['config_fingerprint'], allow_diagnostic=True)
        teacher.eval().requires_grad_(False)
        provider = PilotProvider(tmp_path/'clean_e14.pt', CLEAN_SHA256, make_prepare_config(cfg), device, 2, teacher, None)
        idx = next(i for i, m in enumerate(bundle.manifest['windows']) if m['split'] == 'train' and m['stratum'] == 'representative')
        record, raw, labels = bundle.window(idx, labels=True)
        raw['_column_causal_preparation'] = build_fixed_geometry(raw, record, provider.pcfg, provider.strong, 2, teacher.columns.config)
        head = HeightCausalField('shared_field', z_bins=teacher.columns.config.z_bins,
                                 source_dim=teacher.columns.source_dim).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=.002)
        rng = np.random.default_rng(900)
        before = copy.deepcopy(teacher.state_dict())
        stat = train_step(provider, [(record, raw)], teacher, head, optimizer, rng)
        assert stat['sampled_columns'] > 0 and stat['optimizer_updated'] and stat['generation_bce'] > 0
        contract = dict(epoch_batches=[2], epoch_batch_sizes=[[1, 1]], schedule_steps=2)
        saved = copy.deepcopy(payload(head, optimizer, rng, contract, epoch=0, batch=1, updates=1, executed=1,
                                      reports={'train_prior': {'population': 'fixture TRAIN'}}))
        train_step(provider, [(record, raw)], teacher, head, optimizer, rng)
        other = copy.deepcopy(head); other_optimizer = torch.optim.AdamW(other.parameters(), lr=1.)
        other_rng = np.random.default_rng(2)
        restore(saved, other, other_optimizer, other_rng, contract)
        train_step(provider, [(record, raw)], teacher, other, other_optimizer, other_rng)
        for k, v in head.state_dict().items():
            assert torch.equal(v, other.state_dict()[k]), k
        assert all(torch.equal(v, teacher.state_dict()[k]) for k, v in before.items())
        with torch.no_grad():
            output = teacher.motion(record, device)
            prep = provider.prepare_columns(None, record, include_gt=True, raw_window=raw, outputs=output)
            rows = rows_for(prep, provider.pcfg.grid, teacher.columns.config)
            prep.raw = {**prep.raw, 'future_gt_occ': 'POISON', 'future_annotations': 'POISON'}
            p = probabilities(head.eval(), prep, output, rows, provider.pcfg.grid, teacher.columns.config, device)
            dense = [compose_dense(prep.baseline[h], plan, actions_from_probabilities(plan, probability, (.5, .5, .95)))
                     for (h, plan, _, _), probability in zip(rows, p)]
        assert len(dense) == 6 and all(d.shape == tuple(provider.pcfg.grid.shape_hwd) for d in dense)
        # Real fresh Strong/KTA, resident original source tensors, old graph and
        # new shared path: verify the server FPS entrypoint, not cached logits.
        provider.raw_prefetch_workers = provider.raw_prefetch_depth = 1
        provider.load_raw_columns = lambda source, row, *, include_gt: {
            **raw, 'future_gt_occ': labels['future_gt_occ'].numpy() if include_gt else None}
        monkeypatch.setattr(common, 'gt_moving_support_sequence', lambda *args, **kwargs:
                            [(m, [], {}) for m in labels['moving_support'].numpy()])
        report = evaluate(provider, SimpleNamespace(nusc=None), [record], teacher, head, include_old=True)
        assert report['windows'] == 1 and report['variants']['old_joint']['metrics']['mIoU'] > 0
        assert report['variants']['joint']['quality']['changed'] >= 0
        speed = six_frame_speed(provider, SimpleNamespace(), [record], teacher, head, repeats=1)
        assert speed['all_six_dense_repeat_exact'] and speed['not_raw_sensor_end_to_end']
        for mode, seconds in speed['six_frame_mean_seconds'].items():
            assert seconds > 0 and speed['six_frame_amortized_FPS'][mode] == pytest.approx(6/seconds)
        for member in ('runtime.yaml', 'checkpoints/epoch_0019.pt', 'checkpoints/clean_e14.pt'):
            assert file_digest(tmp_path/Path(member).name) == bundle.manifest['members'][member]['sha256']
