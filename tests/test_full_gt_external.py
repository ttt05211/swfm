import copy
import json
import threading
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from test_full_gt_training import fixture
from test_scene_ego_head import causal_background
from tools.scene_ego_head import integration
from tools.shared_history_experiments.full_gt_external import engine, run
from tools.joint_occ_experiments import run as legacy


def fake_planner(row, keys):
    return SimpleNamespace(metadata=dict(requested=len(keys)+1, covered=len(keys), selected_keys=keys,
        missing_keys=[['MISSING', 'none']]), poses=lambda *a: np.stack(row[1]['future_poses']))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_external_only_no_gt_forward_and_atomic_resume(monkeypatch, device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    model, _, p, row = fixture(); model.to(device); p.device = torch.device(device)
    monkeypatch.setattr(integration, 'rebuild_pred_backgrounds', causal_background)
    before = copy.deepcopy(model.state_dict())
    poses = np.stack(row[1]['future_poses'])
    fw = engine.external_forward(model, p, row, poses)
    raw = copy.deepcopy(row[1]); raw['future_poses'] = object(); raw['future_gt_occ'] = object()
    raw['_column_causal_preparation'] = object()
    poisoned = (row[0], raw, (row[2][0], object(), object()), row[3])
    actual = engine.external_forward(model, p, poisoned, poses)
    for k in ('prior', 'memory', 'logits'): torch.testing.assert_close(fw[k], actual[k], rtol=0, atol=0)
    keys = [[row[0]['scene_name'], row[0]['t0_token']], [row[0]['scene_name'], 'next']]
    second = ({**row[0], 't0_token': 'next'}, *row[1:]); planner = fake_planner(row, keys)
    states = []; event = threading.Event(); original = engine.external_forward
    def stopping(*a, **kw):
        value = original(*a, **kw); event.set(); return value
    monkeypatch.setattr(engine, 'external_forward', stopping)
    with pytest.raises(InterruptedError):
        engine.evaluate({'gt': model}, p, None, [[row], [second]], planner=planner,
            stop_event=event, save=states.append)
    assert states[-1]['keys'] == keys[:1]
    monkeypatch.setattr(engine, 'external_forward', original); event.clear()
    resumed, full = [], []
    result = engine.evaluate({'gt': model}, p, None, [[row], [second]], planner=planner,
        saved=states[-1], save=resumed.append)
    engine.evaluate({'gt': model}, p, None, [[row], [second]], planner=planner, save=full.append)
    assert resumed[-1] == full[-1] and result['windows'] == 2
    assert set(result['variants']) == {'external'} and not result['Moving_evaluated']
    bad = copy.deepcopy(states[-1]); bad['keys'][0][1] = 'WRONG'
    with pytest.raises(RuntimeError, match='prefix'):
        engine.evaluate({'gt': model}, p, None, [[row]], planner=planner, saved=bad)
    for k, v in before.items(): torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)


def test_real_cli_full_val_intersection_readonly_resume(tmp_path, monkeypatch):
    from test_full_gt_training import test_real_cli_full_population_epoch_snapshots_resume_and_eval_readonly
    # Real tiny GT training creates an exact full-TRAIN-contract epoch1 snapshot.
    test_real_cli_full_population_epoch_snapshots_resume_and_eval_readonly(tmp_path, monkeypatch)
    source = tmp_path/'pilot/epoch_0001.pt'; out = tmp_path/'external'
    receipts = {p: p.read_bytes() for p in (source, tmp_path/'pilot/last.pt', tmp_path/'pilot/summary.txt')}
    monkeypatch.setattr(integration, 'rebuild_pred_backgrounds', causal_background)
    class Planner:
        def __init__(self, path): pass
        def prepare(self, requested):
            assert len(requested) == 4369
            self.metadata = dict(requested=4369, covered=2,
                selected_keys=[list(requested[i]) for i in (513, 1000)],
                missing_keys=[list(k) for i, k in enumerate(requested) if i not in (513, 1000)])
        def poses(self, record, current): return np.stack([current]*6)
    monkeypatch.setattr(run, 'PlannerCache', Planner)
    args = [v for name, path in (
        ('config', tmp_path/'config.yaml'), ('base-checkpoint', tmp_path/'base.pt'),
        ('train-cache', tmp_path/'train-cache'), ('dev-cache', tmp_path/'dev-cache'),
        ('population-manifest', tmp_path/'population-manifest'), ('dataroot', tmp_path),
        ('train-info', tmp_path/'train-info'), ('dev-info', tmp_path/'dev-info'),
        ('ccr-history-cache', tmp_path), ('ccr-val-history-cache', tmp_path),
        ('out-dir', out), ('resume', source), ('plan-cache', tmp_path)) for v in ('--'+name, str(path))]
    originals = (legacy.prefetched_batches, legacy.load_cache, legacy.summarize)
    assert run.main(argv=args) == 0
    result = json.loads((out/'results.json').read_text())
    assert result['actual_eval_population'] == 'all_val_planner_covered'
    assert result['evaluation']['windows'] == 2 and result['evaluation']['population']['requested'] == 4369
    assert result['evaluation']['population']['selected_keys'] == [['DEV', 'dev513'], ['DEV', 'dev1000']]
    assert set(result['evaluation']['variants']) == {'external'} and not (out/'last.pt').exists()
    state = (out/'eval_state_update2.json').read_bytes()
    assert run.main(argv=args+['--resume-evaluation']) == 0
    assert (out/'eval_state_update2.json').read_bytes() == state
    assert (legacy.prefetched_batches, legacy.load_cache, legacy.summarize) == originals
    for p, receipt in receipts.items(): assert p.read_bytes() == receipt
