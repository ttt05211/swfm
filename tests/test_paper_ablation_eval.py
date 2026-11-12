import json
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.paper_ablation_eval import report, run, workflow
from tools.paper_ablation_eval.population import select_jobs, validate_resume
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
        def history(self, job):
            order.append(('history', job))
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
    contract = dict(arm='pred_ego', input='occ', protocol=run.PROTOCOL,
                    implementation={**dict.fromkeys(run.LEGACY_OCC_CODE, 'new'), 'metrics.py': 'same'})
    jobs = ['first', 'second']
    with pytest.raises(InterruptedError):
        run.evaluate('pred_ego', predictor, Inputs(), jobs, path, contract, stop)
    state = json.loads(path.read_text())
    assert state['completed_windows'] == 1
    state['contract'] = dict(contract, protocol='paper_ablation_VAL4369_eval_v1',
                             implementation={**contract['implementation'], **run.LEGACY_OCC_CODE})
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match='incompatible'):
        run.read_counts(path, dict(contract, input='stc'), 2, 2)
    stop.clear()
    result = run.evaluate('pred_ego', predictor, Inputs(), jobs, path, contract, stop)
    assert result['average'] == dict(mIoU=100., IoU=100.)
    assert order == [('history', 'first'), ('prediction', 'first'),
                     ('history', 'second'), ('prediction', 'second')]
    assert json.loads(path.read_text())['contract'] == contract


def test_resume_ignores_old_stc_inventory(tmp_path):
    args = SimpleNamespace(dataroot=tmp_path, plan_cache=tmp_path, config=tmp_path/'config',
                           dev_info=tmp_path/'info', dev_cache=tmp_path/'cache')
    files = {}
    for name in ('config', 'dev_info', 'dev_cache'):
        path = getattr(args, name)
        path.write_bytes(b'metadata')
        files[name] = run.base.sha256(path)
    manifest = args.plan_cache/'manifest.json'
    manifest.write_text('{}')
    history = ['a', 'b', 'c', 't0']
    future = list('123456')
    inventory = {str(tmp_path/'missing_stc'/'labels.npz'): [1, 1]}
    for token in history + future:
        path = tmp_path/'gts'/'scene'/token/'labels.npz'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'x')
        stat = path.stat()
        inventory[str(path)] = [stat.st_size, stat.st_mtime_ns]
    population = dict(dataroot=str(tmp_path), files=files, poses={}, times={}, predicted={},
                      jobs=[['scene', history, 't0', future]], inventory=inventory,
                      planner=dict(root=str(tmp_path), manifest_sha256=run.base.sha256(manifest)))
    validate_resume(args, population)
    path.write_bytes(b'changed OCC file')
    with pytest.raises(RuntimeError, match='label file changed'):
        validate_resume(args, population)


def test_report_reads_finished_occ_counts_without_worker_summary(tmp_path):
    path = tmp_path/'occ_state.json'
    confusion = np.zeros((3, 18, 18), dtype=np.int64)
    confusion[:, 0, 0] = 2 * int(np.prod(report.SHAPE))
    contract = dict(arm='pred_ego', input='occ', population_sha256='population', windows=2)
    state = dict(contract=contract, completed_windows=2, confusion=confusion.tolist())
    path.write_text(json.dumps(state))
    assert report.read_row(path, 'pred_ego', 'population', 2) == (2, 100., 100.)
    state['completed_windows'] = 1
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match='only 1/2 OCC windows'):
        report.read_row(path, 'pred_ego', 'population', 2)
