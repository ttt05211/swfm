from copy import deepcopy
import json
from threading import Event

import pytest
import torch

from test_surface_ego_ablation import source_artifacts
from test_surface_ego_full import data, contract
from test_surface_ego_head import config
from real_motion.ego_trajectory_head import HistoryEgoTrajectoryHead
from tools.ego_experiments import screen_surface_ego_partial as pilot
from tools.ego_experiments import train_surface_ego_full as full
from tools.ego_experiments import eval_surface_ego_full as dev
from tools.real_motion.ego_trajectory_common import fingerprint, digest_file, atomic_save, implementation_fingerprint
from tools.real_motion.surface_ego_ablation_common import load_source_bank
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock


@pytest.fixture(autouse=True)
def one_thread():
    n = torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def fixture(tmp_path, monkeypatch, *, completed=8):
    root, origin, _ = source_artifacts(tmp_path)
    for f in (*full.IMPLEMENTATION,*pilot.FILES):
        p = root/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('immutable implementation')
    _, _, audit = load_source_bank(origin,root)
    rows = data(13);c = contract(rows)
    c.update(protocol=full.PROTOCOL, source=audit,
        feature_geometry_implementation=implementation_fingerprint(root),
        implementation={p:digest_file(root/p) for p in full.IMPLEMENTATION},
        frozen_checkpoint=audit['frozen_checkpoint'],frozen_sha256=audit['frozen_sha256'],
        execution=dict(device='cpu',torch_version=str(torch.__version__),cuda_version=torch.version.cuda,WM_precision='fp32'),
        runtime_environment={k:v for k,v in sorted(pilot.os.environ.items()) if k.startswith('SWFM_')})
    parent = tmp_path/'full_parent';parent.mkdir();(parent/'training.json').write_text(json.dumps(c))
    records = [dict(scene_name=r['key'][0],t0_token=r['key'][1]) for r in rows]
    full.build_bank(parent,records[:completed],c,rows,None,lambda *a:pytest.fail('new feature extraction'))
    monkeypatch.setattr(full,'ROOT',root)
    initial = HistoryEgoTrajectoryHead(config())
    monkeypatch.setattr(pilot,'recover_original_initialization',lambda *a:deepcopy(initial))
    return parent,c,rows,root,records


def test_existing_prefix_count_gap_and_exact_scene_membership(tmp_path,monkeypatch):
    parent,c,rows,_,_ = fixture(tmp_path,monkeypatch)
    assert pilot.prefix_count(parent,13) == 8
    split = pilot.prefix_split(c,8)
    assert split['train_indices'] == [i for i in c['split']['train_indices'] if i < 8]
    assert split['holdout_indices'] == [i for i in c['split']['holdout_indices'] if i < 8]
    assert not set(split['train_scenes']) & set(split['holdout_scenes'])
    assert c['keys'] == [r['key'] for r in rows]
    with pytest.raises(ValueError):pilot.prefix_split(c,1)
    assert pilot.prefix_count(parent,4) == 4


def test_partial_reader_never_uses_stale_inventory_and_refuses_corruption(tmp_path,monkeypatch):
    parent,c,_,_,_ = fixture(tmp_path,monkeypatch)
    (parent/'bank_inventory.json').write_text('stale/unfinished inventory ignored')
    rows,receipt = pilot.read_prefix(parent,c,8)
    assert len(rows) == len(receipt) == 8
    assert pilot.read_prefix(parent,c,8,expected=receipt)[1] == receipt
    with pytest.raises(RuntimeError,match='memory'):pilot.read_prefix(parent,c,8,max_mib=.0001)
    with pytest.raises(RuntimeError,match='receipt'):pilot.read_prefix(parent,c,8,expected=receipt[:-1])
    path = parent/'bank/000000.pt';bad = torch.load(path,weights_only=False)
    bad['features']['ego_history'][0,0] += 1;atomic_save(path,bad)
    with pytest.raises(RuntimeError,match='content'):pilot.read_prefix(parent,c,8)


def test_parent_contract_and_full_code_remain_exact_for_original_resume(tmp_path,monkeypatch):
    parent,c,_,root,_ = fixture(tmp_path,monkeypatch)
    sha = digest_file(parent/'training.json')
    pilot.verify_parent(parent,c,sha,root=root)
    # Adding the independent partial tool must not alter old feature fingerprint.
    old = implementation_fingerprint(root)
    (root/pilot.FILES[0]).write_text('another partial entrypoint version')
    assert implementation_fingerprint(root) == old
    pilot.verify_parent(parent,c,sha,root=root)
    (root/full.IMPLEMENTATION[0]).write_text('changed original full code')
    with pytest.raises(RuntimeError,match='full training implementation'):
        pilot.verify_parent(parent,c,sha,root=root)


def test_pilot_refuses_active_parent_or_too_small_prefix_without_creating_output(tmp_path,monkeypatch):
    parent,_,_,_,_ = fixture(tmp_path,monkeypatch)
    out = tmp_path/'pilot'
    with evaluation_lock(parent):
        with pytest.raises(RuntimeError,match='another evaluator'):
            pilot.fit(parent,out,max_windows=10,min_windows=4,device='cpu')
    assert not out.exists()
    with pytest.raises(ValueError,match='enough completed'):
        pilot.fit(parent,out,max_windows=12,min_windows=9,device='cpu')
    assert not out.exists()
    with pytest.raises(ValueError,match='independent'):
        pilot.fit(parent,parent/'pilot',device='cpu')


def test_one_epoch_pilot_no_extraction_parent_bytes_unchanged_and_export_compatible(tmp_path,monkeypatch):
    parent,c,_,root,_ = fixture(tmp_path,monkeypatch)
    before = {p:digest_file(p) for p in parent.rglob('*') if p.is_file()}
    source_before = {p:digest_file(p) for p in (tmp_path/'source').rglob('*') if p.is_file()}
    def forbidden(*a,**kw):pytest.fail('pilot must not rebuild features or read raw histories')
    monkeypatch.setattr(full,'build_bank',forbidden);monkeypatch.setattr(full,'extract_history_features',forbidden)
    monkeypatch.setattr(full,'HistoryReader',forbidden)
    out = tmp_path/'pilot';assert pilot.fit(parent,out,max_windows=10,min_windows=4,device='cpu') == 0
    saved = torch.load(out/'head_epoch1.pt',weights_only=False)
    assert saved['selected_epoch'] == 1 and len(saved['contract']['keys']) == 8
    assert saved['contract']['partial_screen']['feature_extraction'] is False
    assert saved['contract']['split'] == pilot.prefix_split(c,8)
    assert saved['contract']['partial_screen']['windows'] == 8
    assert dev.validate_selected(saved,root=root) == saved['contract']
    assert before == {p:digest_file(p) for p in before}
    assert source_before == {p:digest_file(p) for p in source_before}
    result = json.loads((out/'training_summary.json').read_text())
    assert result['partial_population']['no_new_feature_extraction']
    assert 'NOT a representative full TRAIN run' in (out/'summary.txt').read_text()


def test_stop_training_and_resume_keeps_snapshot_even_when_parent_bank_grows(tmp_path,monkeypatch):
    parent,c,rows,_,records = fixture(tmp_path,monkeypatch)
    out = tmp_path/'pilot';stop = Event();step = torch.optim.AdamW.step
    def interrupted(opt,*a,**kw):
        r = step(opt,*a,**kw);stop.set();return r
    with monkeypatch.context() as m:
        m.setattr(torch.optim.AdamW,'step',interrupted)
        assert pilot.fit(parent,out,max_windows=10,min_windows=4,device='cpu',stop_event=stop) == 130
    c_before = json.loads((out/'training.json').read_text());assert len(c_before['keys']) == 8
    # Only MORE immutable shards are added. Frozen pilot population never grows.
    full.build_bank(parent,records,c,rows,None,lambda *a:pytest.fail('no extraction'))
    assert pilot.prefix_count(parent,13) == 13
    assert pilot.fit(parent,out,max_windows=10,min_windows=4,device='cpu',resume=True) == 0
    export_sha = digest_file(out/'head_epoch1.pt');last_sha = digest_file(out/'last.pt')
    assert pilot.fit(parent,out,max_windows=10,min_windows=4,device='cpu',resume=True) == 0
    assert digest_file(out/'head_epoch1.pt') == export_sha and digest_file(out/'last.pt') == last_sha
    assert json.loads((out/'training.json').read_text()) == c_before
    with pytest.raises(RuntimeError,match='budget/code changed'):
        pilot.fit(parent,out,max_windows=11,min_windows=4,device='cpu',resume=True)


def test_changed_prefix_code_and_source_refused_on_resume(tmp_path,monkeypatch):
    parent,c,_,root,_ = fixture(tmp_path,monkeypatch)
    out = tmp_path/'pilot';kw = dict(max_windows=10,min_windows=4,device='cpu')
    assert pilot.fit(parent,out,**kw) == 0
    (root/pilot.FILES[0]).write_text('changed pilot code')
    with pytest.raises(RuntimeError,match='budget/code changed'):pilot.fit(parent,out,resume=True,**kw)
    (root/pilot.FILES[0]).write_text('immutable implementation')
    c['keys'][0][1] = 'different source'
    (parent/'training.json').write_text(json.dumps(c))
    with pytest.raises(RuntimeError,match='identity/content'):pilot.fit(parent,out,resume=True,**kw)


def test_sigint_during_existing_bank_read_never_creates_partial_training_contract(tmp_path,monkeypatch):
    parent,_,_,_,_ = fixture(tmp_path,monkeypatch)
    out = tmp_path/'pilot';stop = Event();stop.set()
    assert pilot.fit(parent,out,max_windows=10,min_windows=4,device='cpu',stop_event=stop) == 130
    assert not out.exists()
