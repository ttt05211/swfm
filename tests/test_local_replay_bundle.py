"""Portable/safe export tests. Synthetic fixtures are NOT real nuScenes timing."""
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import torch

from real_motion.local_replay_bundle import (BundleWriter, ReplayBundle, pack_window, unpack_window,
    validate_window, select_records, fingerprint, file_digest, LABEL_KEYS, MOTION_KEYS, safe_member)


def fixture(scene='train', token='t0', sources=1, shape=(8,8,4)):
    record = dict(sample_id=scene+'/'+token, scene_name=scene, t0_token=token,
        history_tokens=[token+str(i) for i in range(3)]+[token], future_tokens=[token+'f'+str(i) for i in range(6)],
        features=torch.zeros(sources,64), local_semantic_tube=torch.zeros(sources,6,4,4,4,dtype=torch.uint8),
        kta_displacement_xy_m=torch.zeros(sources,6,2), frame_motion_features=torch.zeros(sources,6,10),
        target_source_mask_tube=torch.ones(sources,6,20,20,dtype=torch.uint8),
        anchors_xy_t0_m=torch.zeros(sources,6,2), source_class_id=torch.full((sources,),4),
        source_centroid_xy_t0_m=torch.zeros(sources,2))
    for key in LABEL_KEYS:
        record[key] = torch.zeros((sources,6,2) if 'xy_m' in key else (sources,) if key in ('supervised_source','yaw_enabled') else (sources,6))
    occ = np.full((4,*shape),17,np.uint8); occ[:,2,2,1]=4; occ[:,5,5,0]=11
    raw = dict(history_occ=occ, history_observed=np.ones_like(occ,bool),
        history_poses=np.tile(np.eye(4),(4,1,1)), future_poses=np.tile(np.eye(4),(6,1,1)),
        trajectory=np.zeros((12,2),np.float32), future_gt_occ=np.tile(occ[-1][None],(6,1,1,1)))
    moving = np.zeros((6,*shape),bool)
    return record,raw,moving


def make_bundle(path, *, shape=(8,8,4)):
    r,raw,moving = fixture(shape=shape); inp,labels = pack_window(r,raw,moving)
    writer=BundleWriter(path); writer.add_tensors('windows/000/inputs.pt',inp); writer.add_tensors('windows/000/labels.pt',labels)
    meta=writer.finish(dict(active_history_frames=4,future_frames=6,grid_shape=list(shape),
        windows=[dict(key=['train','t0'],inputs='windows/000/inputs.pt',labels='windows/000/labels.pt')]))
    return meta


def test_tensor_roundtrip_separate_labels_and_full_shape(tmp_path):
    path=tmp_path/'replay.zip';make_bundle(path,shape=(200,200,16))
    bundle=ReplayBundle(path)
    try:
        r,raw,labels=bundle.window(0)
        assert raw['history_occ'].shape==(4,200,200,16) and raw['future_gt_occ'] is None and labels is None
        assert not (set(LABEL_KEYS)&set(r))
        r,raw,labels=bundle.window(0,labels=True)
        assert raw['future_gt_occ'].shape==(6,200,200,16)
        assert raw['history_occ'].dtype==np.uint8 and raw['history_observed'].dtype==bool
    finally:bundle.close()


def test_pack_whitelist_rejects_future_feature_cache():
    r,raw,moving=fixture();r['future_gt_features']='POISON';raw['_column_causal_preparation']='POISON'
    inputs,labels=pack_window(r,raw,moving)
    validate_window(inputs,labels,(8,8,4))
    record,causal=unpack_window(inputs)
    assert 'future_gt_features' not in record and '_column_causal_preparation' not in causal
    for group in (inputs['raw'],inputs['record']): assert not any(k in group for k in ('future_gt_occ','target_yaw_rad'))
    inputs['record']['target_yaw_rad']=labels['record']['target_yaw_rad']
    with pytest.raises(ValueError,match='leak'):validate_window(inputs,labels,(8,8,4))


@pytest.mark.parametrize('kind',('six_histories','short_future','nonfinite_pose','nonrigid_pose','anchor','order','source_count','wrong_visibility'))
def test_invalid_population_or_geometry_fails_closed(kind):
    r,raw,moving=fixture();inputs,labels=pack_window(r,raw,moving)
    if kind=='six_histories':inputs['raw']['history_occ']=torch.zeros(6,8,8,4,dtype=torch.uint8)
    if kind=='short_future':labels['future_gt_occ']=labels['future_gt_occ'][:5]
    if kind=='nonfinite_pose':inputs['raw']['history_poses'][0,0,0]=float('nan')
    if kind=='nonrigid_pose':inputs['raw']['history_poses'][0,0,0]=2
    if kind=='anchor':inputs['record']['anchors_xy_t0_m'][0,0,0]=1
    if kind=='order':inputs['identity']['history_tokens'][-1]='wrong'
    if kind=='source_count':inputs['record']['features']=torch.zeros(2,64)
    if kind=='wrong_visibility':inputs['raw']['history_observed']=inputs['raw']['history_observed'].byte()
    with pytest.raises(ValueError):validate_window(inputs,labels,(8,8,4))


def test_selection_deterministic_unique_and_gt_independent():
    records=[fixture('scene'+str(i%5),'token'+str(i),i%9)[0] for i in range(50)]
    chosen=select_records(records,16,4)
    assert len(chosen)==16 and len(set(r['t0_token'] for r,_ in chosen))==16
    assert [len(r['features']) for r,s in chosen if s=='high_source_stress']==[8]*4
    for r in records:r['existence']='POISON'
    assert [(r['t0_token'],s) for r,s in chosen]==[(r['t0_token'],s) for r,s in select_records(records,16,4)]
    with pytest.raises(ValueError,match='duplicate'):select_records(records+[records[0]],16,4)


@pytest.mark.parametrize('member',('../outside','/absolute','C:/outside','a\\b','a/../b'))
def test_path_traversal_rejected_without_extraction(tmp_path,member):
    path=tmp_path/'bad.zip'
    with pytest.raises(ValueError,match='unsafe'):safe_member(member)
    with zipfile.ZipFile(path,'w') as z:z.writestr(member,b'x')
    # Windows ZipFile normalizes backslashes on write; still not a valid bundle.
    with pytest.raises(ValueError):ReplayBundle(path)
    assert not (tmp_path/'outside').exists()


def rewrite(path, change):
    with zipfile.ZipFile(path) as z: rows={x.filename:z.read(x.filename) for x in z.infolist()}
    change(rows)
    with zipfile.ZipFile(path,'w') as z:
        for k,v in rows.items():z.writestr(k,v)


def test_archive_tamper_missing_member_oversize_and_manifest(tmp_path):
    path=tmp_path/'replay.zip';make_bundle(path)
    with pytest.raises(ValueError,match='oversized'):ReplayBundle(path,max_bytes=16)
    rewrite(path,lambda rows:rows.update({'windows/000/inputs.pt':b'corrupt'}))
    with pytest.raises(ValueError,match='fingerprint'):ReplayBundle(path)
    path=tmp_path/'missing.zip';make_bundle(path)
    rewrite(path,lambda rows:rows.pop('windows/000/labels.pt'))
    with pytest.raises(ValueError,match='missing'):ReplayBundle(path)
    path=tmp_path/'manifest.zip';make_bundle(path)
    rewrite(path,lambda rows:rows.update({'manifest.json':rows['manifest.json'].replace(b'"active_history_frames": 4',b'"active_history_frames": 6')}))
    with pytest.raises(ValueError,match='fingerprint'):ReplayBundle(path)


def test_new_output_only_and_archive_cap(tmp_path):
    path=tmp_path/'new.zip';make_bundle(path)
    with pytest.raises(FileExistsError):BundleWriter(path)
    writer=BundleWriter(tmp_path/'small.zip',max_bytes=4)
    try:
        with pytest.raises(ValueError,match='limit'):writer.add_stream('inputs',io.BytesIO(b'toolarge'))
    finally:writer.close()
    bundle=ReplayBundle(path)
    dest=tmp_path/'old.pt';dest.write_bytes(b'user data')
    try:
        with pytest.raises(FileExistsError):bundle.copy_member('windows/000/inputs.pt',dest)
    finally:bundle.close()
    assert dest.read_bytes()==b'user data'


def test_manifest_key_order_checked_even_with_valid_hashes(tmp_path):
    path=tmp_path/'replay.zip';make_bundle(path)
    def change(rows):
        m=json.loads(rows['manifest.json']);m['windows'][0]['key']=['wrong','t0'];m.pop('manifest_fingerprint')
        m['manifest_fingerprint']=fingerprint(m);rows['manifest.json']=json.dumps(m).encode()
    rewrite(path,change);bundle=ReplayBundle(path)
    try:
        with pytest.raises(ValueError,match='identity'):bundle.window(0)
    finally:bundle.close()


def test_export_end_to_end_fake_source_does_not_modify_inputs(tmp_path,monkeypatch):
    from tools.real_motion import export_p0_f9_local_replay as e
    r,raw,moving=fixture();train=[fixture('train','t'+str(i),i+1)[0] for i in range(4)]
    dev=[fixture('dev','d'+str(i),i+1)[0] for i in range(4)]
    paths={}
    for name in ('checkpoint','base_checkpoint','train_cache','dev_cache','population_manifest','train_info','dev_info','config'):
        paths[name]=tmp_path/(name+'.pt');paths[name].write_bytes(name.encode())
    before={str(p):file_digest(p) for p in paths.values()}
    train_keys=[(r['scene_name'],r['t0_token']) for r in train];dev_keys=[(r['scene_name'],r['t0_token']) for r in dev]
    ck=dict(cursor_epoch=19,model_configs={},train_keys=train_keys,dev_keys=dev_keys,dev_manifest_fingerprint='manifest',
        cache_fingerprints={s:file_digest(paths[s+'_cache']) for s in ('train','dev')},
        info_fingerprints={s:file_digest(paths[s+'_info']) for s in ('train','dev')})
    monkeypatch.setattr(e,'CLEAN_SHA256',file_digest(paths['base_checkpoint']))
    monkeypatch.setattr(e,'load_joint',lambda *args,**kw:(ck,SimpleNamespace(transport=SimpleNamespace(config=SimpleNamespace(history_frames=4)),columns=SimpleNamespace(source_dim=128))))
    monkeypatch.setattr(e,'load_runtime_config',lambda *args:{'fake_config':True})
    monkeypatch.setattr(e,'make_prepare_config',lambda cfg:SimpleNamespace(grid=SimpleNamespace(shape_hwd=(8,8,4))))
    monkeypatch.setattr(e,'load_manifest',lambda path:({'manifest_fingerprint':'manifest','parent_keys':dev_keys},dev_keys,None))
    monkeypatch.setattr(e,'load_cache',lambda path:({},train if 'train_cache' in str(path) else dev))
    monkeypatch.setattr(e,'select_population',lambda *args,**kw:(train_keys,None))
    monkeypatch.setattr(e,'NuScenesWindowSource',lambda *args,**kw:SimpleNamespace(nusc=None))
    monkeypatch.setattr(e,'load_nuscenes_window_raw',lambda *args,**kw:copy.deepcopy(raw))
    monkeypatch.setattr(e,'gt_moving_support_sequence',lambda *args,**kw:[(mask,[],{}) for mask in moving])
    monkeypatch.setattr(e,'git_state',lambda path:{'commit':'synthetic','status':''})
    args=['export']
    for name,path in paths.items():args+=['--'+name.replace('_','-'),str(path)]
    args+=['--dataroot',str(tmp_path),'--out-dir',str(tmp_path/'result'),'--train-windows','2','--dev-windows','2','--stress-per-split','1','--max-package-mib','128']
    monkeypatch.setattr('sys.argv',args);e.main()
    assert before=={str(p):file_digest(p) for p in paths.values()}
    bundle=ReplayBundle(tmp_path/'result/replay.zip')
    try:
        assert len(bundle.manifest['windows'])==4 and not bundle.manifest['cached_geometry']
        assert {r['split'] for r in bundle.manifest['windows']}=={'train','dev'}
        assert bundle.manifest['student'] is None
        for i in range(4):bundle.window(i)
    finally:bundle.close()


@pytest.mark.parametrize('device_name',['cpu','cuda'])
def test_raw_to_evidence_six_frames_and_actual_head_backward(monkeypatch,device_name):
    if device_name=='cuda' and not torch.cuda.is_available():pytest.skip('actual CUDA required')
    from tools.real_motion import replay_p0_f9_local as replay
    from real_motion.geometry import OccupancyGrid
    from real_motion.sparse_evidence_repair import SparseRepairHead
    grid=OccupancyGrid(x_min=0,y_min=0,z_min=0,voxel_size=(1.,1.,1.),shape_hwd=(8,8,4))
    _,raw,_=fixture()
    raw['history_occ'][0,3,2,1]=4
    registrations=[[(np.eye(4),np.argwhere(o==4)) for o in raw['history_occ']]]
    prep=SimpleNamespace(raw=raw,registrations=registrations,
        state=dict(current=[dict(class_id=4,centroid_world=np.array([2.5,2.5,1.5]))],
                   current_pose=np.eye(4),world_to_future=np.tile(np.eye(4),(6,1,1))),
        baseline=raw['future_gt_occ'],targets=np.tile([[2.5,2.5,1.5]],(6,1,1)),yaws=np.zeros((6,1)))
    device=torch.device(device_name)
    output=dict(history_source_context=torch.randn(1,128,device=device),future_transport_queries=torch.randn(1,6,128,device=device))
    teacher=SimpleNamespace(columns=SimpleNamespace(config=None),motion=lambda *args:output)
    def prepare(*args,**kwargs):
        assert kwargs['include_gt'] is False and kwargs['raw_window']['future_gt_occ'] is None
        return prep
    provider=SimpleNamespace(device=device,pcfg=SimpleNamespace(grid=grid),strong=None,workers=1,prepare_columns=prepare)
    monkeypatch.setattr(replay,'build_fixed_geometry',lambda *args:dict(raw_only=True))
    raw={**prep.raw,'future_gt_occ':None};record=dict(features=torch.zeros(1,64))
    head=SparseRepairHead('local_consensus').to(device)
    before=head.score.bias.detach().clone()
    result=replay.replay_one(provider,teacher,head,record,raw,dict(future_gt_occ=torch.tensor(prep.baseline)),train_step=True,chunk=16)
    assert result['all_six_frames'] and result['original_occupied_unchanged']
    assert np.isfinite(result['isolated_GT_only_head_probe_loss'])
    assert not torch.equal(before,head.score.bias)
    assert 'full_neighbor_features_CPU' in result['stages']
