"""Actual head/phase/six compositions; synthetic external IO is NOT formal FPS."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from real_motion.canonical_causal_repair import build_canonical_evidence, map_canonical_evidence, compose_canonical
from real_motion.surface_canonical_repair import SurfaceCanonicalRepairHead
from tools.real_motion import surface_ccr_validation_common as common
from tools.real_motion import benchmark_p0_f9_joint_surface_mean_fps as cli
from test_canonical_causal_repair import scene


def setup(monkeypatch, device='cpu', *, incomplete=False):
    torch.manual_seed(1024)
    grid, prep = scene(); plain = build_canonical_evidence(prep, grid)
    provider = SimpleNamespace(device=torch.device(device), pcfg=SimpleNamespace(grid=grid), workers=2)
    teacher = SimpleNamespace(transport=torch.nn.Identity())
    head = SurfaceCanonicalRepairHead(8).to(device).eval().requires_grad_(False)
    output = {'history_source_context': torch.randn(1, 8, device=device),
              'future_transport_queries': torch.randn(1, 6, 8, device=device)}
    records = [dict(scene_name='scene-'+str(i % 18), t0_token=str(i), features=torch.zeros(1+i % 3, 8))
               for i in range(64)]
    def history(*args, **kwargs):
        return SimpleNamespace(canonical_evidence=plain, current_pose=np.eye(4), future_poses=[np.eye(4)]*6)
    def forecast(h, provider, motion, model, probability_fn, **kwargs):
        plan = map_canonical_evidence(h.canonical_evidence, prep, grid)
        probability = probability_fn(model, h.canonical_evidence, plan, output, provider.device)
        dense = compose_canonical(prep.baseline, h.canonical_evidence, plan, probability[..., 0], probability[..., 1])
        return dict(probability=probability, dense=dense[:5] if incomplete else dense,
                    stages_seconds={'shared_encode_six_readouts': .001})
    monkeypatch.setattr(common, 'prepare_history', history)
    monkeypatch.setattr(common, 'forecast_six', forecast)
    return provider, records, teacher, head


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_more_windows_mean_latency_and_frozen_joint_head(monkeypatch, device):
    if device == 'cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        pytest.skip('CUDA BF16 required')
    provider, records, teacher, head = setup(monkeypatch, device)
    before = {k: v.clone() for k, v in head.state_dict().items()}; logged = []
    report = common.paired_speed(provider, None, records, teacher, head, None,
        windows=24, repeats=2, stress_windows=0, surface_only=True, progress=logged.append)
    assert report['fps_windows'] == 24 and report['repeats'] == 2
    assert len(report['trials']) == len(logged) == 96
    assert report['probability_and_six_dense_parity_windows'] == 24
    assert len({row['key'][0] for row in report['population']}) == 18
    assert report['dynamic_byte_parity_windows'] is None  # not comparing to frozen E19 dynamic weights
    assert report['graph_execution']['capture_full_chunks_only'] is True
    assert set(report['six_frame_mean_seconds']) == {'surface_fused_eager', 'surface_fused_graph'}
    for mode, fps in report['six_frame_amortized_FPS'].items():
        seconds = [row['seconds'] for row in logged if row['mode'] == mode]
        assert fps == pytest.approx(6/np.mean(seconds))
    assert all(torch.equal(value, head.state_dict()[key]) for key, value in before.items())
    report['scenes'] = 18
    text = cli.summary(report)
    assert 'FPS=6/mean_six_frame_latency' in text and 'NOT raw-input E2E' in text
    assert 'no latency selection' in report['selection_basis']


def test_missing_dense_frame_rejected(monkeypatch):
    provider, records, teacher, head = setup(monkeypatch, incomplete=True)
    with pytest.raises(RuntimeError, match='SIX dense'):
        common.paired_speed(provider, None, records, teacher, head, None,
            windows=1, repeats=1, stress_windows=0, surface_only=True)


def test_capture_inside_timing_rejected(monkeypatch):
    provider, records, teacher, head = setup(monkeypatch)
    original = common.SurfaceExecution
    class MisbehavingGraph(original):
        def __call__(self, *args, **kwargs):
            value = super().__call__(*args, **kwargs)
            self.counts['captures_verified'] += 1
            return value
    monkeypatch.setattr(common, 'SurfaceExecution', MisbehavingGraph)
    with pytest.raises(RuntimeError, match='inside FPS timer'):
        common.paired_speed(provider, None, records, teacher, head, None,
            windows=1, repeats=1, stress_windows=0, surface_only=True)
