from copy import deepcopy
from threading import Event

import numpy as np
import pytest
import torch

from real_motion.geometry import OccupancyGrid
from real_motion.prepared import PrepareConfig
from real_motion.local_st_world_model_v17 import LocalSTWMV17Config
from real_motion.joint_surface_ccr import JointSurfaceCCR
from real_motion.stc_camera_protocol import SETTINGS, PROTOCOL
from tools.real_motion.stc_shared_execution import Predictor, check_and_time, migrate_state, bytes_equal
from tools.real_motion.stc_camera_evaluation import evaluate, restore
from test_stc_camera_protocol import fixture, fake_predict


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_real_shared_pair_motion_probability_all_six_bytes_and_integer_scores(tmp_path, monkeypatch, device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('actual CUDA unavailable')
    from real_motion.waymo_native_execution import prepare_waymo_native
    prepare_waymo_native(tmp_path/'native')
    source = fixture(tmp_path, shape=(32, 32, 4)); source.preflight(source.windows[:2])
    joint = JointSurfaceCCR(LocalSTWMV17Config(history_frames=4, d_model=16, semantic_dim=4,
        blocks=1, decoder_blocks=1), width=16, z_bins=4).to(device).eval().requires_grad_(False)
    before = {k: v.clone() for k, v in joint.state_dict().items()}
    cfg = PrepareConfig(grid=OccupancyGrid(-6.4, -6.4, -1., (.4,)*3, source.shape))
    threads = torch.get_num_threads(); torch.set_num_threads(1)
    original = Predictor(joint, cfg, device, workers=2, shared=False, graphs=device == 'cuda')
    shared = Predictor(joint, cfg, device, workers=2, graphs=device == 'cuda', geometry_mib=8)
    try:
        # Timing/check must never read targets. Distinct history visibility for GT/STC
        # and distinct Pred/GT future transforms are included in this real fixture.
        targets = source.metric_targets
        monkeypatch.setattr(source, 'metric_targets', lambda *_: pytest.fail('GT read during byte/speed check'))
        result = check_and_time(source, source.windows[:2], original, shared, repeats=1)
        assert result['probability_bytes_exact'] and result['six_dense_bytes_exact']
        assert shared.provider.reused_histories >= 4
        monkeypatch.setattr(source, 'metric_targets', targets)
        contract = dict(windows=2)
        a = evaluate(source, source.windows[:2], original, contract)
        b = evaluate(source, source.windows[:2], shared, contract)
        assert a['reports'] == b['reports']
        for key, value in joint.state_dict().items(): assert torch.equal(value, before[key])
        # Same tokens with changed visibility/occupancy must not hit pair evidence.
        rec, raw = source.prediction_inputs(source.windows[0], 'stc_gt')
        shared.full(rec, raw); key = shared.provider.pair_key; plain = shared.provider.plain_evidence
        _, raw = source.prediction_inputs(source.windows[0], 'stc_pred')
        raw['history_occ'] = raw['history_occ'].copy(); raw['history_occ'][0, 0, 0, 0] = 17
        shared.full(rec, raw)
        assert shared.provider.pair_key != key and shared.provider.plain_evidence is not plain
    finally:
        original.close(); shared.close(); torch.set_num_threads(threads)


def test_migration_keeps_atomic_prefix_and_rejects_scientific_or_original_code_change(tmp_path):
    source = fixture(tmp_path); source.preflight(source.windows)
    old = dict(protocol=PROTOCOL, windows=len(source.windows), implementation={'old.py': 'fixed'},
        checkpoint_sha256='weights', thresholds=[.5, None])
    stop = Event(); saved = {}
    evaluate(source, source.windows, fake_predict, old, stop_event=stop,
        progress=lambda r: stop.set(), save=lambda s: saved.update(s))
    new = old | dict(implementation={'old.py': 'fixed', 'fast.py': 'added'}, fast_execution={'v': 1})
    unchanged = deepcopy(saved)
    migrated = migrate_state(old, saved, new, shape=source.shape)
    assert saved == unchanged and migrated['counts'] == saved['counts'] and migrated['completed_windows'] == 1
    restore(migrated, new, int(np.prod(source.shape)))
    resumed = evaluate(source, source.windows, fake_predict, new, saved=migrated)
    full = evaluate(source, source.windows, fake_predict, new)
    assert resumed['reports'] == full['reports']
    for changed in (new | {'checkpoint_sha256': 'other'}, new | {'thresholds': [.4, None]},
                    new | {'implementation': {'old.py': 'changed'}}, new | {'windows': 2}):
        with pytest.raises(RuntimeError): migrate_state(old, saved, changed, shape=source.shape)


def test_byte_comparison_not_tolerance():
    with pytest.raises(RuntimeError): bytes_equal(np.array([1.], np.float32), np.array([1.+1e-7], np.float32), 'x')


def test_cli_readonly_migration_and_strict_shared_resume(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from real_motion.waymo_i2world import file_sha256, fingerprint
    from tools.real_motion import eval_p0_f9_joint_surface_stc_shared as cli
    source = fixture(tmp_path); source.preflight(source.windows)
    cfg = PrepareConfig(grid=OccupancyGrid(-40, -40, -1., (.4,)*3, source.shape))
    (tmp_path/'weights').mkdir()
    checkpoint = tmp_path/'weights'/'mean.pt'; checkpoint.write_bytes(b'fixture-model')
    config = tmp_path/'config.yaml'; config.write_text('fixture-config')
    monkeypatch.setattr(cli.original.STCFourSettingSource, 'from_files', lambda *a, **kw: source)
    monkeypatch.setattr(cli.original, 'load_runtime_config', lambda *_: None)
    monkeypatch.setattr(cli.original, 'make_prepare_config', lambda *_: cfg)
    monkeypatch.setattr(cli.original, 'SHAPE', source.shape)
    monkeypatch.setattr(cli.torch.cuda, 'is_available', lambda: True)
    model = SimpleNamespace(transport=SimpleNamespace(config=SimpleNamespace(history_frames=4)))
    monkeypatch.setattr(cli.original, 'load_evaluation_model', lambda *a, **kw:
        (dict(source_epochs=list(cli.original.AVERAGE_EPOCHS), averaging=True), model))
    class FakePredictor:
        def __init__(self, *a, **kw): self.provider = SimpleNamespace(reused_histories=2, clear_pair=lambda: None)
        def __call__(self, *a, verify): return fake_predict(*a, verify)
        def close(self): pass
    monkeypatch.setattr(cli, 'Predictor', FakePredictor)
    monkeypatch.setattr(cli, 'check_and_time', lambda *a, **kw: dict(speedup=2., six_dense_bytes_exact=True))
    common = ['--dataroot', str(source.root), '--stc-root', str(source.stc_root), '--plan-cache', str(source.plan_cache),
              '--population', 'all', '--config', str(config), '--checkpoint', str(checkpoint)]
    first = tmp_path/'first'
    assert cli.main(argv=common+['--out-dir', str(first)]) == 0
    generated = json.loads((first/'contract.json').read_text())
    old = {k: v for k, v in generated.items() if k not in ('fast_execution', 'execution_migration')}
    old['implementation'] = {k: v for k, v in old['implementation'].items() if k in cli.original.IMPLEMENTATION_FILES}
    original = tmp_path/'original'; original.mkdir()
    (original/'contract.json').write_text(json.dumps(old))
    stop = Event()
    evaluate(source, source.windows, fake_predict, old, stop_event=stop,
        progress=lambda _: stop.set(), save=lambda v: (original/'state.json').write_text(json.dumps(v)))
    sha = [file_sha256(original/p) for p in ('contract.json','state.json')]
    fast = tmp_path/'continued'
    assert cli.main(argv=common+['--out-dir', str(fast), '--continue-from-dir', str(original)]) == 0
    assert [file_sha256(original/p) for p in ('contract.json','state.json')] == sha
    state = json.loads((fast/'state.json').read_text())
    assert state['completed_windows'] == len(source.windows)
    assert json.loads((fast/'evaluation.json').read_text())['reports'] == json.loads((first/'evaluation.json').read_text())['reports']
    assert cli.main(argv=common+['--out-dir', str(fast), '--resume']) == 0
    with pytest.raises(RuntimeError, match='contract'):
        cli.main(argv=common+['--out-dir', str(fast), '--resume', '--geometry-cache-mib', '256'])


def test_active_source_lease_blocks_migration_and_receipt_detects_tampering(tmp_path):
    import json
    from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
    from tools.real_motion.eval_p0_f9_joint_surface_stc_shared import read_receipt, assert_receipt
    (tmp_path/'contract.json').write_text('{}')
    (tmp_path/'state.json').write_text(json.dumps(dict(completed_windows=2)))
    with evaluation_lock(tmp_path):
        with pytest.raises(RuntimeError, match='owns'):
            with evaluation_lock(tmp_path): pass
        _, _, receipt = read_receipt(tmp_path)
        assert_receipt(receipt)
        (tmp_path/'state.json').write_text(json.dumps(dict(completed_windows=3)))
        with pytest.raises(RuntimeError, match='changed'): assert_receipt(receipt)


def test_new_terminal_launcher_restores_frozen_swfm_without_shell_eval(tmp_path, monkeypatch):
    import json
    import os
    import sys
    from pathlib import Path
    script = (Path(__file__).resolve().parents[1]/'tools/real_motion/run_p0_f9_joint_surface_stc_shared.sh').read_text(encoding='utf-8')
    launcher = script.split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
    contract = tmp_path/'contract.json'
    contract.write_text(json.dumps(dict(runtime_environment={'SWFM_COLUMN_CPU_BACKEND':'native', 'SWFM_TEST':'literal $()'})))
    monkeypatch.setenv('SWFM_STALE', 'do-not-inherit')
    monkeypatch.setenv('KEEP_OTHER_ENV', 'yes')
    monkeypatch.setattr(sys, 'argv', ['-', str(contract), '/repo', '--out-dir', '/new'])
    captured = {}
    monkeypatch.setattr(os, 'execve', lambda executable, args, env: captured.update(executable=executable,args=args,env=env))
    exec(compile(launcher, '<wrapper-launcher>', 'exec'), {})
    assert captured['env']['SWFM_TEST'] == 'literal $()' and 'SWFM_STALE' not in captured['env']
    assert captured['env']['KEEP_OTHER_ENV'] == 'yes'
    assert captured['args'][-2:] == ['--out-dir','/new']
    contract.write_text(json.dumps(dict(runtime_environment={'NOT_SWFM':'bad'})))
    with pytest.raises(RuntimeError, match='environment'): exec(compile(launcher, '<wrapper-launcher>', 'exec'), {})
