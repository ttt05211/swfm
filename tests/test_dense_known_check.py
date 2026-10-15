import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.dense_known_check import run
from tools.history_validity.data import policy_raw
from tools.shared_source_track.adapter import SharedTrackPredictor
from test_shared_source_track import fixture


def test_frozen_loading_update_and_source_unchanged(tmp_path):
    p, rec, raw = fixture()
    try:
        saved = dict(protocol=run.SOURCE_PROTOCOL, updates=6000, models={'gt': copy.deepcopy(p.body.state_dict())},
            contract=dict(protocol=run.SOURCE_PROTOCOL, training_known='dense_valid', model=p.body.configs(),
                          eval_keys=[['s', str(i)] for i in range(64)]))
        path = tmp_path/'frozen.pt'; torch.save(saved, path); original = path.read_bytes()
        _, body = run.load_body(path, 6000, 'cpu')
        assert not body.training and not any(p.requires_grad for p in body.parameters())
        with pytest.raises(RuntimeError, match='matched'): run.load_body(path, 4000, 'cpu')
        assert path.read_bytes() == original
    finally: p.close()


def test_both_encoder_known_and_memory_rebuilt_no_old_read_reuse():
    p, rec, raw = fixture()
    try:
        raw['history_observed'][:] = False
        poses = np.array(raw['history_poses'], copy=True)
        inputs = SimpleNamespace(history=lambda *a: (rec, raw, np.arange(4)*.5))
        known = run.KnownInputs(inputs, 'lidar_known').history(None)[1]
        dense = run.KnownInputs(inputs, 'dense_valid').history(None)[1]
        assert not known['history_observed'].any() and dense['history_observed'].all()
        a = p.prepare(rec, known, times=np.arange(4)*.5)
        b = p.prepare(rec, dense, times=np.arange(4)*.5)
        assert a[3][0]['memory_reads']['coverage'].size == 0
        assert b[3][0]['memory_reads']['coverage'].size > 0
        assert a[3][0]['memory_reads']['static_valid'].sum() < b[3][0]['memory_reads']['static_valid'].sum()
        np.testing.assert_array_equal(raw['history_poses'], poses)
        assert not raw['history_observed'].any()
    finally: p.close()
