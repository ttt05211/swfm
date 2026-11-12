import json
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.paper_ablation_eval import run, workflow
from tools.paper_ablation_eval.population import select_jobs
from tools.paper_ablation_train import control, model
from tools.paper_ablation_train.geometry import forward_window
from test_paper_ablation_train import tiny


def test_all_eight_launch_before_poll(tmp_path, monkeypatch):
    launches = []

    class Child:
        def __init__(self, command, env, **kwargs):
            launches.append((command, env['CUDA_VISIBLE_DEVICES']))
            self.pid = len(launches)

        def poll(self):
            assert len(launches) == 8
            return 0

        def wait(self):
            return 0

    monkeypatch.setattr(workflow.subprocess, 'Popen', Child)
    args = SimpleNamespace(config='config.yaml', cpu_workers=2)
    checkpoints = {arm: f'{arm}.pt' for arm in workflow.TABLE_ORDER[2:]}
    codes = workflow.launch_all(args, ('2', '7'), tmp_path, checkpoints, Event())
    assert codes == dict.fromkeys(workflow.TABLE_ORDER, 0)
    assert [gpu for _, gpu in launches] == ['2']*4 + ['7']*4
    for command, _ in launches:
        arm = command[command.index('--arm') + 1]
        if arm in checkpoints:
            assert command[command.index('--checkpoint') + 1] == checkpoints[arm]
        else:
            assert '--checkpoint' not in command


def test_pred_uses_only_covered_keys_in_val_order():
    jobs = [['scene', ['a', 'b', 'c', key], key, list('123456')] for key in ('t1', 't2', 't3')]
    population = dict(jobs=jobs, predicted={'t3': None, 't1': None})
    assert [job.t0_token for job in select_jobs(population, False)] == ['t1', 't2', 't3']
    assert [job.t0_token for job in select_jobs(population, True)] == ['t1', 't3']


def test_kta_geometry_matches_d_transport_and_static_keeps_original_track():
    body, predictor, old, row = tiny('d_kta_refiner')
    try:
        with torch.no_grad():
            poses = np.asarray(row[1]['future_poses'])
            reference = forward_window(body.eval(), predictor.provider, row, poses=poses)
            prior, shape = run.geometry_prediction(predictor.provider, row, poses, stationary=False)
            torch.testing.assert_close(prior, reference['prior'], rtol=0, atol=0)
            before = row[0]['anchors_xy_t0_m'].copy()
            static, static_shape = run.geometry_prediction(predictor.provider, row, poses, stationary=True)
            assert shape == static_shape
            assert torch.isfinite(static).all()
            np.testing.assert_array_equal(before, row[0]['anchors_xy_t0_m'])
            replay, _ = run.geometry_prediction(predictor.provider, row, poses, stationary=False)
            torch.testing.assert_close(prior, replay, rtol=0, atol=0)
    finally:
        predictor.close()
        old.close()


def test_frozen_worker_loads_trained_branch_weights(tmp_path):
    arm = 'c_transport'
    body, predictor, old, _ = tiny(arm)
    try:
        keys = [['train', str(i)] for i in range(20430)]
        budget = dict(train_windows=20430, stop_updates=5145, schedule_reference_updates=10292)
        contract = dict(protocol=control.PROTOCOL, arm=arm, recipe=model.ARMS[arm], model=body.configs(),
                        world_size=2, known='TRAIN_dense_valid', budget=budget, train_keys=keys,
                        data={'config': 'sha'}, overrides=[])
        path = tmp_path/'epoch_0001.pt'
        torch.save(dict(protocol=control.PROTOCOL, training_only=True, epoch=1, batch=0, updates=5145,
                        contract=contract, models={'gt': body.state_dict()}), path)
        population = dict(files={'config': 'sha'}, overrides=[])
        restored, train_keys = run.load_body(arm, path, population, device='cpu')
        assert train_keys == keys
        assert not restored.training and all(not parameter.requires_grad for parameter in restored.parameters())
        torch.testing.assert_close(restored.state_dict(), body.state_dict(), rtol=0, atol=0)
        with pytest.raises(RuntimeError, match='matching one-epoch'):
            run.load_body('d_kta_refiner', path, population, device='cpu')
    finally:
        predictor.close()
        old.close()


def test_integer_counts_resume_and_pred_pose_routing(tmp_path, monkeypatch):
    stop = Event()
    order = []
    labels = np.array([[[0, 17]]], dtype=np.uint8)
    poses = np.arange(6*4*4).reshape(6, 4, 4)

    class Inputs:
        def history(self, job, kind):
            order.append(('history', job, kind))
            return job, {}, np.arange(4)

        def future_poses(self, job, predicted):
            assert predicted
            return poses

        def targets(self, job, indices):
            assert order[-1] == ('prediction', job)
            assert indices == (1, 3, 5)
            return [labels]*3

    def predict(arm, predictor, job, raw, times, supplied):
        assert arm == 'pred_ego' and supplied is poses
        order.append(('prediction', job))
        if job == 'first':
            stop.set()
        return np.stack([labels]*6)

    monkeypatch.setattr(run, 'predict', predict)
    predictor = SimpleNamespace(provider=SimpleNamespace(pcfg=SimpleNamespace(
        grid=SimpleNamespace(shape_hwd=labels.shape))))
    path = tmp_path/'occ_state.json'
    contract = dict(arm='pred_ego', input='occ')
    jobs = ['first', 'second']
    with pytest.raises(InterruptedError):
        run.evaluate('pred_ego', predictor, Inputs(), jobs, 'occ', path, contract, stop)
    assert json.loads(path.read_text())['completed_windows'] == 1
    stop.clear()
    result = run.evaluate('pred_ego', predictor, Inputs(), jobs, 'occ', path, contract, stop)
    assert result['average'] == dict(mIoU=100., IoU=100.)
    assert order == [('history', 'first', 'occ'), ('prediction', 'first'),
                     ('history', 'second', 'occ'), ('prediction', 'second')]
    with pytest.raises(ValueError, match='incompatible'):
        run.read_counts(path, dict(contract, input='stc'), 2, 2)
