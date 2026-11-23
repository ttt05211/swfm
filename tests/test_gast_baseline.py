"""CPU bootstrap, official input parity, causal rollout, resume and same renderer."""
import copy
import io
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import threading
from types import SimpleNamespace,ModuleType
import zipfile
import numpy as np
import pytest
import torch
from tools.gast_baseline import common,data,download,model,export,render


def frame(i):
    angle=.01*i;q=[np.cos(angle/2),0.,0.,np.sin(angle/2)];t=[.8*i,.1*i,0.]
    return dict(token=str(i),pose=data.pose_matrix(t,q).tolist(),translation=t,rotation=q,lcf=[i*.1,.2,.03],
        command=[0.,0.,1.],timestamp_s=i*.5,semantics=str(i))


def test_real_official_getitem_input_parity_and_no_future_labels(tmp_path,monkeypatch):
    """Execute the actual official dataset function against controlled files."""
    import types
    source=Path(__file__).resolve().parents[1]/'tmp/gast_official_c468725/dataloader/dataset_nus_occ3d.py'
    if not source.is_file():pytest.skip('optional unmodified official source fixture not downloaded')
    q=ModuleType('pyquaternion');q.Quaternion=lambda x:x
    ng=ModuleType('nuscenes.utils.geometry_utils');ng.transform_matrix=data.pose_matrix
    monkeypatch.setitem(sys.modules,'pyquaternion',q);monkeypatch.setitem(sys.modules,'nuscenes',ModuleType('nuscenes'))
    monkeypatch.setitem(sys.modules,'nuscenes.utils',ModuleType('nuscenes.utils'));monkeypatch.setitem(sys.modules,'nuscenes.utils.geometry_utils',ng)
    module=types.ModuleType('official_dataset');exec(compile(source.read_text(),str(source),'exec'),module.__dict__)
    rows=[];frames=[frame(i) for i in range(10)];vol=[]
    for i,f in enumerate(frames):
        folder=tmp_path/str(i);folder.mkdir();sem=np.full((4,4,2),i,np.uint8);np.savez_compressed(folder/'labels.npz',semantics=sem);vol.append(sem)
        feat=[f['lcf'][0],f['lcf'][1],0,0,f['lcf'][2]]
        rows.append(dict(occ_path=str(folder),ego2global_translation=f['translation'],ego2global_rotation=f['rotation'],
            gt_ego_fut_cmd=f['command'],gt_ego_lcf_feat=feat))
    official=module.NuscenesOcc3D.__new__(module.NuscenesOcc3D)
    official.valid_list=[0];official.return_len=10;official.start_frame=0;official.mid_frame=4;official.end_frame=10
    official.data_infos=rows;official.label_path='unused'
    reference=official[0];ours=model.input_dict(np.stack(vol[:4]),frames,'cpu')
    assert 'fut_occ' not in ours and 'fut_ego_lcf_feat' not in ours
    for key,value in ours.items():torch.testing.assert_close(value[0],reference[key],rtol=0,atol=0)
    assert torch.count_nonzero(ours['his_e2g_relative_rotation'][:,-1])==0


def test_rollout_second_history_is_predicted_and_horizons_are_exact():
    frames=[frame(i) for i in range(16)];history=np.full((4,2,2,2),17,np.uint8);calls=[]
    class Forecast(torch.nn.Module):
        def forward(self,d):
            assert 'fut_occ' not in d
            calls.append({k:v.clone() for k,v in d.items()});logits=torch.zeros(1,10,18,2,2,2)
            for i in range(10):logits[:,i,(i+3*len(calls))%17]=1
            return logits,None,None
    predicted=model.rollout(Forecast(),history,frames,'cpu')
    assert predicted.shape==(12,2,2,2) and predicted.dtype==np.uint8 and len(calls)==2
    np.testing.assert_array_equal(calls[1]['his_occ'][0].numpy(),predicted[2:6])
    torch.testing.assert_close(calls[1]['his_ego_pose'][0],torch.tensor([f['pose'] for f in frames[6:10]],dtype=torch.float32))
    assert predicted[list(common.LONG),0,0,0].tolist()==[8,10,12,11,13,15]


def make_assets(tmp_path,count=4):
    shape=common.GRID['shape_hwd'];keys=['short01','turn01','short02','turn02'][:count]
    cases={};infos=[];tables={'sample':{},'sample_data':{},'ego_pose':{},'scene':{}};arrays={}
    for ci,key in enumerate(keys):
        short=key.startswith('short');frames=[frame(i) for i in range(10 if short else 16)]
        scene='scene-'+str(ci);hist=[];future=[]
        tables['scene'][scene]=dict(name=scene)
        for i,f in enumerate(frames):
            token=f'{ci}_{i}';(hist if i<4 else future).append(token)
            path=tmp_path/'dataset/gts'/scene/token;path.mkdir(parents=True)
            v=np.full(shape,17,np.uint8);v[1,1,0]=11
            np.savez_compressed(path/'labels.npz',semantics=v)
            tables['sample'][token]=dict(token=token,prev=f'{ci}_{i-1}' if i else '',next=f'{ci}_{i+1}' if i<len(frames)-1 else '',
                scene_token=scene,timestamp=int(i*5e5),data={'LIDAR_TOP':token})
            tables['sample_data'][token]=dict(ego_pose_token=token)
            tables['ego_pose'][token]=dict(translation=f['translation'],rotation=f['rotation'])
            infos.append(dict(token=token,ego2global_translation=f['translation'],ego2global_rotation=f['rotation'],
                gt_ego_lcf_feat=[f['lcf'][0],f['lcf'][1],0,0,f['lcf'][2]],gt_ego_fut_cmd=f['command']))
        n=3 if short else 6
        cases[key]=dict(scene=scene,t0_token=hist[-1],history_tokens=hist,future_tokens=future,
            future_ego_poses=[f['pose'] for f in frames[4:]],arrays=['t0','gt','c','e'] if short else ['e'])
        arrays[key]={'e':np.stack([v]*n)}
        if short:arrays[key].update(t0=v,gt=np.stack([v]*3),c=np.stack([v]*3))
    assets=tmp_path/'paper_qualitative_test.zip'
    manifest=dict(format='carryworld_qualitative_v1',preview=False,source_sha256=common.SOURCE_SHA,grid=common.GRID,cases=cases)
    with zipfile.ZipFile(assets,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('manifest.json',json.dumps(manifest))
        for key,values in arrays.items():
            for name,value in values.items():
                b=io.BytesIO();np.save(b,value,allow_pickle=False);z.writestr(key+'/'+name+'.npy',b.getvalue())
    info=tmp_path/'info.pkl'
    with info.open('wb') as f:pickle.dump(dict(infos={'validation':infos}),f)
    source=SimpleNamespace(get=lambda table,token:tables[table][token])
    args=SimpleNamespace(work=tmp_path/'work',assets=assets,info=info,dataroot=tmp_path/'dataset')
    return args,source


def test_CPU_data_genuine_fields_flat_or_scene_and_missing_rejected(tmp_path,monkeypatch):
    a,source=make_assets(tmp_path,2);before=common.sha(a.assets)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:(_ for _ in ()).throw(AssertionError('CPU step queried CUDA')))
    prepared=data.prepare(a,source=source)
    assert len(prepared['cases'])==2 and before==common.sha(a.assets)
    assert common.read_json(a.work/'prepared.json')==prepared
    assert prepared==data.prepare(a,source=source)
    rows=data.info_index(a.info);m=common.checked_archive(a.assets)
    with a.info.open('wb') as f:pickle.dump(dict(infos=list(rows.values())),f)
    assert not data.probe_info(a.info,m)[1]
    rows['0_0'].pop('gt_ego_lcf_feat')
    with a.info.open('wb') as f:pickle.dump(dict(infos=list(rows.values())),f)
    with pytest.raises(ValueError,match='genuine ego'):data.prepare(a,source=source)
    assert a.assets.stat().st_size<2**20


@pytest.mark.parametrize('skip_irregular_time',[False,True])
def test_data_rejects_cross_scene_and_mismatched_original_future_pose(tmp_path,skip_irregular_time):
    a,source=make_assets(tmp_path,2)
    a.skip_irregular_time=skip_irregular_time
    original=source.get
    def wrong(table,token):
        r=copy.deepcopy(original(table,token))
        if table=='sample' and token=='0_1':r['scene_token']='wrong'
        return r
    with pytest.raises(ValueError,match='cross-scene'):data.prepare(a,source=SimpleNamespace(get=wrong))
    def wrong_pose(table,token):
        r=copy.deepcopy(original(table,token))
        if table=='ego_pose' and token=='0_4':r['translation'][0]+=1
        return r
    with pytest.raises(ValueError,match='ego2global'):data.prepare(a,source=SimpleNamespace(get=wrong_pose))


def test_time_audit_strict_default_and_explicit_same_window_exclusion(tmp_path):
    a,source=make_assets(tmp_path);before=common.sha(a.assets)
    source.get('sample','1_7')['timestamp']+=300000
    with pytest.raises(ValueError,match='expected actual 2Hz keyframes: turn01'):
        data.prepare(a,source=source)
    assert not (a.work/'prepared.json').exists()
    report=common.read_json(a.work/'timing_audit.json')
    assert (report['candidates'],report['eligible'],report['excluded'])==(4,3,1)
    bad=report['cases']['turn01']['bad_intervals']
    assert [r['seconds'] for r in bad]==[.8,.2]
    assert bad[0]['from_token']=='1_6' and bad[0]['to_token']=='1_7'
    assert report['cases']['turn01']['future_offset_seconds'][3]==2.3
    a.skip_irregular_time=True;prepared=data.prepare(a,source=source)
    assert list(prepared['cases'])==['short01','short02','turn02']
    assert list(prepared['timing_exclusions']['cases'])==['turn01']
    assert prepared['timing_exclusions']['candidate_cases']==4
    original=common.checked_archive(a.assets)
    for key,case in prepared['cases'].items():assert case['original']==original['cases'][key]
    assert not any('/scene-1/' in path.replace('\\','/') for path,_,_ in prepared['inventory'])
    saved=common.sha(a.work/'prepared.json')
    assert prepared==data.prepare(a,source=source)
    assert saved==common.sha(a.work/'prepared.json') and before==common.sha(a.assets)


def test_timing_diagnostic_needs_no_info_pickle_and_no_occupancy_reads(tmp_path):
    a,source=make_assets(tmp_path,2);a.info.unlink()
    report=data.audit_timing(a,source=source)
    assert report['eligible']==2 and report['excluded']==0
    assert report['cases']['turn01']['future_offset_seconds']==[.5*i for i in range(1,13)]
    assert not (a.work/'prepared.json').exists()


def test_timing_exclusion_cannot_remove_all_long_cases(tmp_path):
    a,source=make_assets(tmp_path,2);a.skip_irregular_time=True
    source.get('sample','1_7')['timestamp']+=300000
    with pytest.raises(ValueError,match='retain at least one short and one long'):
        data.prepare(a,source=source)
    assert not (a.work/'prepared.json').exists()


def test_excluded_timing_case_stays_excluded_for_all_models_and_packet_resume(tmp_path):
    a,source=make_assets(tmp_path);a.skip_irregular_time=True
    source.get('sample','1_7')['timestamp']+=300000
    prepared=data.prepare(a,source=source);seen=[]
    args=SimpleNamespace(action='infer',work=a.work,gast_root=tmp_path,weight=tmp_path/'unused',device='cpu')
    def compute(net,case,archive,key,device):
        assert key!='turn01';seen.append(key)
        def predict(net,history,frames,device):return np.repeat(history[-1:],len(frames)-4,axis=0)
        return export.compute_case(net,case,archive,key,device,predict=predict)
    export.run(args,threading.Event(),factory=lambda *args:(object(),{'test':'no released weights'}),compute=compute)
    assert seen==list(prepared['cases'])
    assert not (a.work/'predictions/turn01.npz').exists()
    progress=common.read_json(a.work/'progress.json');assert progress['windows']==progress['total']==3
    packet=a.work/'comparison_assets.zip';before=common.sha(packet)
    with zipfile.ZipFile(packet) as z:
        manifest=json.loads(z.read('manifest.json'))
        assert manifest['timing_exclusions']==prepared['timing_exclusions']
        assert list(manifest['cases'])==list(prepared['cases'])
        assert 'cases/turn01.npz' not in z.namelist()
        for key in manifest['cases']:
            with np.load(io.BytesIO(z.read('cases/'+key+'.npz')),allow_pickle=False) as values:
                np.testing.assert_array_equal(values['gt'],values['g'])
                np.testing.assert_array_equal(values['gt'],values['e'])
    export.run(args,threading.Event(),factory=lambda *args:(object(),{'test':'no released weights'}),
        compute=lambda *args:pytest.fail('completed matching subset must be reused'))
    assert before==common.sha(packet)


def test_actual_streamed_metadata_matches_SDK_reverse_lookup_without_GPU(tmp_path,monkeypatch):
    a,sdk=make_assets(tmp_path,2);m=common.checked_archive(a.assets)
    tokens={t for c in m['cases'].values() for t in c['history_tokens']+c['future_tokens']}
    folder=a.dataroot/'v1.0-trainval';folder.mkdir()
    rows={name:[] for name in ('sample','sample_data','ego_pose','scene','sensor','calibrated_sensor')}
    rows['sensor']=[dict(token='lidar_sensor',channel='LIDAR_TOP'),dict(token='camera_sensor',channel='CAM_FRONT')]
    rows['calibrated_sensor']=[dict(token='lidar',sensor_token='lidar_sensor'),dict(token='camera',sensor_token='camera_sensor')]
    for token in sorted(tokens):
        sample=copy.deepcopy(sdk.get('sample',token));sample.pop('data');rows['sample'].append(sample)
        rows['sample_data'].append(dict(token=token,sample_token=token,is_key_frame=True,calibrated_sensor_token='lidar',ego_pose_token=token))
        rows['sample_data'].append(dict(token=token+'camera',sample_token=token,is_key_frame=True,calibrated_sensor_token='camera',ego_pose_token=token))
        rows['ego_pose'].append(dict(token=token,**sdk.get('ego_pose',token)))
    for scene in {c['scene'] for c in m['cases'].values()}:rows['scene'].append(dict(token=scene,name=scene))
    for name,values in rows.items():(folder/(name+'.json')).write_text(json.dumps(values))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:pytest.fail('CPU streaming must not query GPU'))
    streamed=data.Metadata(a.dataroot,tokens)
    for token in tokens:
        assert streamed.get('sample',token)['data']==sdk.get('sample',token)['data']
        np.testing.assert_allclose(streamed.get('ego_pose',token)['rotation'],sdk.get('ego_pose',token)['rotation'])
    prepared=data.prepare(a)
    assert len(prepared['cases'])==2 and len(prepared['metadata'])==6
    assert all('camera' not in str(f['pose']) for c in prepared['cases'].values() for f in c['frames'])


def test_VAL_download_selection_never_chooses_train_or_ambiguous_files():
    files=[SimpleNamespace(id='t',path='world-nuscenes_infos_train.pkl'),SimpleNamespace(id='v',path='world-nuscenes_infos_val.pkl')]
    assert download.choose_val(files).id=='v'
    with pytest.raises(ValueError):download.choose_val(files+files)
    with pytest.raises(ValueError):download.choose_val(files[:1])


def test_weight_architecture_guard_and_DDP_collision(tmp_path):
    good=dict(temp_embed=torch.zeros(1,10,128),**{'plan_translation_output.2.weight':torch.zeros(108,64)})
    path=tmp_path/'gt.pth';torch.save(good,path)
    assert download.inspect_weights(path)['tensors']==2
    bad=dict(good,temp_embed=torch.zeros(1,16,128));torch.save(bad,path)
    with pytest.raises(ValueError,match='4-history'):download.inspect_weights(path)
    bad=dict(good,**{'module.temp_embed':good['temp_embed']});torch.save(bad,path)
    with pytest.raises(ValueError,match='duplicate'):download.inspect_weights(path)
    torch.save({'model':good},path)
    with pytest.raises(ValueError,match='raw state_dict'):download.inspect_weights(path)


def test_CPU_strict_load_does_not_query_CUDA_and_rejects_missing_keys(tmp_path,monkeypatch):
    import yaml
    root=tmp_path/'official';(root/'models').mkdir(parents=True);(root/'config').mkdir()
    config={'model_params':dict(num_his_frames=4,num_fut_frames=6,use_gt=True),'common_params':{},'infer_params':{}}
    (root/'config/gast_nuscenes.yaml').write_text(yaml.safe_dump(config))
    module=ModuleType('models.gast');module.__file__=str(root/'models/gast.py')
    class Tiny(torch.nn.Module):
        def __init__(self,config):super().__init__();self.layer=torch.nn.Linear(2,2)
    module.GAST=Tiny;monkeypatch.setattr(model,'check_code',lambda _:dict(verified=True))
    monkeypatch.setattr(model.importlib,'import_module',lambda _:module)
    monkeypatch.setitem(sys.modules,'models',ModuleType('models'));sys.modules['models'].__path__=[str(root/'models')]
    monkeypatch.setitem(sys.modules,'models.gast',module)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:(_ for _ in ()).throw(AssertionError('CPU setup queried GPU')))
    net=Tiny(config);state={'module.'+k:v for k,v in net.state_dict().items()};path=tmp_path/'weights.pt';torch.save(state,path)
    loaded,_=model.load_model(root,path,tmp_path/'work','cpu')
    for k,v in net.state_dict().items():torch.testing.assert_close(v,loaded.state_dict()[k],rtol=0,atol=0)
    state.pop('module.layer.bias');torch.save(state,path)
    with pytest.raises(RuntimeError,match='Missing key'):model.load_model(root,path,tmp_path/'work','cpu')


def test_upstream_fingerprints_reject_modified_source(monkeypatch):
    root=Path(__file__).resolve().parents[1]/'tmp/gast_official_c468725'
    if not root.exists():pytest.skip('optional official source fixture not downloaded')
    monkeypatch.setattr(model.subprocess,'check_output',lambda *a,**k:common.COMMIT+'\n')
    assert len(model.check_code(root))==6
    monkeypatch.setattr(model.subprocess,'check_output',lambda *a,**k:'wrong\n')
    with pytest.raises(ValueError,match='pinned'):model.check_code(root)


def test_export_resume_unchanged_controls_six_before_target_reads_and_packet(tmp_path):
    a,source=make_assets(tmp_path);prepared=data.prepare(a,source=source);stop=threading.Event();calls=[]
    args=SimpleNamespace(action='infer',work=a.work,gast_root=tmp_path,weight=tmp_path/'weight',device='cpu')
    before=common.sha(a.assets);history_reads=[];target_reads=[]
    def factory(*_):return object(),{'frozen':'synthetic test; not released weights'}
    def compute(net,case,archive,key,device):
        frames=case['frames'];hist={f['semantics'] for f in frames[:4]};finished=[]
        def reader(path,shape):
            if path in hist:history_reads.append(path)
            else:
                assert finished==[True];target_reads.append(path)
            return common.labels(path,shape)
        def predict(net,history,values,device):
            assert len(history)==4 and len(values) in (10,16);finished.append(True)
            return np.repeat(history[-1:],len(values)-4,axis=0)
        result=export.compute_case(net,case,archive,key,device,predict=predict,reader=reader);calls.append(key)
        if len(calls)==1:stop.set()
        return result
    with pytest.raises(InterruptedError):export.run(args,stop,factory=factory,compute=compute)
    assert common.read_json(a.work/'progress.json')['windows']==1
    digest=common.sha(a.work/'predictions/short01.npz');stop.clear()
    export.run(args,stop,factory=factory,compute=compute)
    assert len(calls)==4 and digest==common.sha(a.work/'predictions/short01.npz') and before==common.sha(a.assets)
    assert len(history_reads)==16 and len(target_reads)==18
    with zipfile.ZipFile(a.work/'comparison_assets.zip') as z:
        m=json.loads(z.read('manifest.json'));assert m['format']==common.FORMAT and len(m['cases'])==4
        assert m['not_official_long_term_table'] is True and all(n.endswith(('.npz','.json')) for n in z.namelist())
    archive_hash=common.sha(a.work/'comparison_assets.zip')
    export.run(args,stop,factory=factory,compute=lambda *args:pytest.fail('must reuse frozen completed predictions'))
    assert common.sha(a.work/'comparison_assets.zip')==archive_hash
    path=a.work/'predictions/short01.npz';path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='output changed'):export.run(args,stop,factory=factory,compute=compute)


def test_render_same_1s_reference_for_GT_GAST_ours_and_real_native_backend(tmp_path):
    from tools.paper_assets import vtk_render
    if __import__('importlib').util.find_spec('pyvista') is None:pytest.skip('local VTK dependencies unavailable')
    shape=(4,4,2);grid=dict(x_min=-.8,y_min=-.8,z_min=-.4,shape_hwd=list(shape),voxel_size=[.4]*3)
    sem=np.full(shape,17,np.uint8);sem[1:3,1:3,0]=11
    arrays={n:np.stack([sem]*6) for n in ('gt','g','e')};arrays['t0']=sem
    poses=[frame(i)['pose'] for i in range(12)]
    rows,columns,settings=render.rows_and_settings('turn01',arrays,poses)
    assert [n for n,_ in rows]==['GT','GAST','CarryWorld'] and columns==['1 s','2 s','3 s','4 s','5 s','6 s']
    reference=np.linalg.inv(poses[1])@np.asarray(poses)[1::2]
    for _,panels in rows:
        for i,panel in enumerate(panels):np.testing.assert_allclose(settings[id(panel)]['display_transform'],reference[i],atol=1e-12)
    from tools.paper_assets.qualitative_render import render_grid
    picture,audit=render_grid([('GT',[sem]),('GAST',[sem]),('CarryWorld',[sem])],['1 s'],grid,(320,180),'come_reference',preview=True)
    assert audit['mayavi_executed'] is False and audit['rows']==['GT','GAST','CarryWorld']
    assert audit['camera_parameters']['automatic_camera_fit'] is False
    picture.save(tmp_path/'actual_renderer.png')


def test_packet_gallery_labels_and_resume_without_repainting(tmp_path):
    from PIL import Image
    a,source=make_assets(tmp_path,2);prepared=data.prepare(a,source=source)
    args=SimpleNamespace(action='infer',work=a.work,gast_root=tmp_path,weight=tmp_path/'weight',device='cpu')
    def compute(net,case,archive,key,device):
        frames=case['frames'];n=len(case['report_indices']);t0=common.labels(frames[3]['semantics'])
        return dict(t0=t0,gt=np.stack([t0]*n),g=np.stack([t0]*n),e=np.stack([t0]*n))
    export.run(args,threading.Event(),factory=lambda *a:(object(),{}),compute=compute)
    calls=[]
    def draw(rows,columns,grid,size,camera,**options):
        assert [r for r,_ in rows]==['GT','GAST','CarryWorld'];calls.append((columns,options))
        return Image.new('RGB',(800,350),'white'),dict(rows=['GT','GAST','CarryWorld'],backend='test')
    out=tmp_path/'images';render.render(a.work/'comparison_assets.zip',out,renderer=draw)
    assert len(calls)==2 and len(calls[0][0])==4 and len(calls[1][0])==6
    assert len(calls[1][1]['panel_settings'])==18
    render.render(a.work/'comparison_assets.zip',out,renderer=lambda *a,**k:pytest.fail('should reuse completed images'))
    assert 'GAST' in (out/'index.html').read_text()
    with pytest.raises(ValueError,match='options'):render.render(a.work/'comparison_assets.zip',out,camera='road_high',renderer=draw)


def test_CPU_shell_steps_do_not_require_GPU_or_nvcc(tmp_path):
    bash=Path('D:/Program Files/Git/bin/bash.exe')
    if not bash.is_file():pytest.skip('Git Bash not available')
    script=Path(__file__).resolve().parents[1]/'tools/gast_baseline/step.sh'
    subprocess.run([str(bash),'-n',str(script)],check=True)
    log=tmp_path/'calls';fake=tmp_path/'fakeconda'
    fake.write_text('#!/usr/bin/env bash\n[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] || exit 99\nprintf "%s\\n" "$*" >> "$GAST_TEST_LOG"\n',encoding='utf-8')
    env=dict(os.environ,CONDA_EXE=fake.as_posix(),GAST_TEST_LOG=log.as_posix())
    for phase in ('env','weights','infos','timing','data','cpucheck','status','render-check','render'):
        subprocess.run([str(bash),str(script),phase],env=env,check=True,capture_output=True)
    subprocess.run([str(bash),str(script),'data','--skip-irregular-time'],env=env,check=True,capture_output=True)
    text=log.read_text();assert 'nvcc' not in text and 'cuda.is_available' not in text
    assert '--no-cache-dir' in text and '--only-binary' in text and '-n gast' in text
    assert 'tools.gast_baseline.data timing' in text and '--skip-irregular-time' in text


@pytest.mark.parametrize('phase', ['adopt-weights', 'cpucheck', 'render', 'check', 'infer'])
def test_shell_isolates_foreign_torch_libraries_even_after_conda_activation(tmp_path, phase):
    bash=Path('D:/Program Files/Git/bin/bash.exe')
    if not bash.is_file():pytest.skip('Git Bash not available')
    script=Path(__file__).resolve().parents[1]/'tools/gast_baseline/step.sh'
    probe=tmp_path/'probe.sh';fake=tmp_path/'fakeconda';log=tmp_path/'probe.log'
    probe.write_text('''#!/usr/bin/env bash
set -euo pipefail
[[ -z "${LD_PRELOAD+x}" && -z "${PYTHONHOME+x}" ]]
[[ "${PYTHONNOUSERSITE:-}" == 1 ]]
[[ "$PYTHONPATH" == "$(cd "$GAST_TEST_ROOT" && pwd)" ]]
[[ "${LD_LIBRARY_PATH:-}" == "$GAST_TEST_LIBS" ]]
[[ "${CUDA_VISIBLE_DEVICES:-}" == "$GAST_TEST_GPU" ]]
printf '%s\\n' "$*" >> "$GAST_TEST_LOG"
''',encoding='utf-8')
    fake.write_text('''#!/usr/bin/env bash
set -euo pipefail
# No real Conda or Python is launched: simulate activation hooks then inspect
# the environment of the exact command that would otherwise launch Python.
while [[ $# -gt 0 && "$1" != env ]]; do shift; done
[[ $# -gt 0 ]] || exit 98
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib
# MSYS aborts before shell startup for nonexistent preload DLLs; keep the
# variable present but empty to verify that the Python launch unsets it.
export LD_PRELOAD='' PYTHONHOME=/foreign/python
export PYTHONNOUSERSITE=0 PYTHONPATH=/foreign/site
launch=()
while [[ $# -gt 0 && "$1" != python ]]; do launch+=("$1"); shift; done
[[ $# -gt 0 ]] || exit 97
shift
"${launch[@]}" bash "$GAST_TEST_PROBE" "$@"
''',encoding='utf-8')
    gpu=phase in ('check','infer')
    drivers='/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64'
    env={k:os.environ[k] for k in ('PATH','SystemRoot','SYSTEMROOT','WINDIR','TEMP','TMP','USERPROFILE') if k in os.environ}
    env.update(CONDA_EXE=fake.as_posix(),GAST_TEST_PROBE=probe.as_posix(),
        GAST_TEST_LOG=log.as_posix(),GAST_TEST_ROOT=script.parents[2].as_posix(),
        GAST_TEST_LIBS=drivers if gpu else '',GAST_TEST_GPU='3' if gpu else '',GAST_GPU='3',
        CUDA_VISIBLE_DEVICES='6,7',LD_PRELOAD='',PYTHONHOME='/foreign/python',
        PYTHONPATH='/foreign/site',PYTHONNOUSERSITE='0',
        LD_LIBRARY_PATH='/usr/local/lib/python3.12/dist-packages/torch/lib:'+drivers+':/foreign/site-packages/torch/lib:')
    subprocess.run([str(bash),str(script),phase],env=env,check=True,capture_output=True)
    assert log.read_text().count('\n')==(2 if phase=='render' else 1)
