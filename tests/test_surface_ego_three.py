from copy import deepcopy
import json
from threading import Event

import numpy as np
import pytest
import torch

from test_surface_ego_partial import fixture, one_thread
from test_surface_ego_full import data, contract
from test_surface_ego_head import config
from test_surface_ego_ablation import assert_nested_equal
from real_motion.ego_trajectory_head import HistoryEgoTrajectoryHead
from tools.ego_experiments import train_surface_ego_three as three
from tools.ego_experiments import screen_surface_ego_partial as pilot
from tools.real_motion.ego_trajectory_common import digest_file


def test_three_epochs_final_export_and_exact_resume(tmp_path,monkeypatch):
    torch.manual_seed(21);initial = HistoryEgoTrajectoryHead(config())
    rows = data(13);c = contract(rows);c['schedule']['epochs'] = 3
    whole,paused = [tmp_path/k for k in ('whole','paused')]
    whole.mkdir();paused.mkdir();kw = dict(batch_size=2,checkpoint_every=1)
    result = three.train(deepcopy(initial),rows,c,whole,**kw)
    assert result['updates'] == 3*int(np.ceil(len(c['split']['train_indices'])/2))
    assert result['completed_epochs'] == result['evaluated_epoch'] == 3
    assert not (whole/'head_epoch1.pt').exists()
    assert torch.load(whole/'head_epoch3.pt',weights_only=False)['selected_epoch'] == 3
    stop = Event();step = torch.optim.AdamW.step;calls = []
    def interrupted(opt,*a,**kw):
        value = step(opt,*a,**kw);calls.append(1)
        if len(calls) == int(np.ceil(len(c['split']['train_indices'])/2))+1:stop.set()
        return value
    with monkeypatch.context() as m:
        m.setattr(torch.optim.AdamW,'step',interrupted)
        assert three.train(deepcopy(initial),rows,c,paused,stop_event=stop,**kw)['status'] == 'stopped'
    three.train(deepcopy(initial),rows,c,paused,resume=True,**kw)
    a = torch.load(whole/'last.pt',weights_only=False);b = torch.load(paused/'last.pt',weights_only=False)
    assert a['optimizer']['param_groups'][0]['lr'] == pytest.approx(c['schedule']['min_lr'])
    for k in ('state_dict','optimizer','updates','epoch','cursor','order','sampling_rng','history'):
        assert_nested_equal(a[k],b[k])
    sha = digest_file(paused/'head_epoch3.pt')
    three.train(deepcopy(initial),rows,c,paused,resume=True,**kw)
    assert digest_file(paused/'head_epoch3.pt') == sha


def test_existing_pilot_fit_reuses_exact_population_no_extraction_and_read_only(tmp_path,monkeypatch):
    parent,_,_,root,_ = fixture(tmp_path,monkeypatch)
    for f in three.IMPLEMENTATION:
        p = root/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('new three-epoch implementation')
    source = tmp_path/'pilot'
    assert pilot.fit(parent,source,max_windows=10,min_windows=4,device='cpu') == 0
    before = {p:digest_file(p) for directory in (parent,source) for p in directory.rglob('*')
        if p.is_file() and p.name != '.evaluation.lock'}
    monkeypatch.setattr(three,'ROOT',root)
    initial = HistoryEgoTrajectoryHead(config())
    monkeypatch.setattr(three,'recover_original_initialization',lambda *a:deepcopy(initial))
    out = tmp_path/'three'
    assert three.fit(source,out,device='cpu') == 0
    saved = torch.load(out/'head_epoch3.pt',weights_only=False)
    assert three.validate_selected(saved,root=root) == saved['contract']
    bad = deepcopy(saved);bad['selected_epoch'] = 1
    with pytest.raises(RuntimeError,match='final-epoch3'):three.validate_selected(bad,root=root)
    original = json.loads((source/'training.json').read_text())
    assert saved['contract']['keys'] == original['keys']
    assert saved['contract']['split'] == original['split']
    assert saved['contract']['schedule']['epochs'] == 3
    assert before == {p:digest_file(p) for p in before}
    sha = digest_file(out/'head_epoch3.pt')
    assert three.fit(source,out,device='cpu',resume=True) == 0
    assert digest_file(out/'head_epoch3.pt') == sha
    with pytest.raises(ValueError,match='independent'):three.fit(source,source/'overwrite',device='cpu')


def test_dev512_population_and_final_epoch_gate():
    # Same selector as real evaluation: dev512 uses all parent keys, not dev64 subset.
    from real_motion.stc_camera_protocol import STCFourSettingSource
    from types import SimpleNamespace
    keys = [(f'scene-{i//32:04d}',str(i)) for i in range(512)]
    source = object.__new__(STCFourSettingSource)
    source.by_key = {k:SimpleNamespace(scene=k[0],t0=k[1]) for k in keys}
    selected,audit = source.select('dev512',keys)
    assert len(selected) == audit['windows'] == 512
    from tools.ego_experiments import eval_surface_ego_three as dev
    assert 'dev512' in dev.main.__code__.co_consts
