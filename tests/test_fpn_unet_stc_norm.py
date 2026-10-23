"""Small CPU/CUDA checks, not scientific quality claims or extra model arms."""
import copy
import json
import threading
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.fpn_unet.model import UNetRefiner
from tools.fpn_unet_stc_norm import core, run


def example(device='cpu'):
    torch.manual_seed(21)
    body = torch.nn.Module(); body.columns = UNetRefiner(memory_width=16,z_bins=4)
    with torch.no_grad(): body.columns.head.weight.normal_(std=.02)
    body.to(device).eval().requires_grad_(False)
    # Other model components must remain frozen too, not only the convolutions.
    body.encoder = torch.nn.Linear(3,3).to(device).requires_grad_(False)
    memory = torch.randn(6,16,4,4,4,device=device)
    prior = torch.rand(6,18,8*8*4,device=device)
    labels = torch.full((8*8*4,),11,device=device)
    fw = dict(memory=memory,prior=prior,shape=(8,8,4))
    return body,core.cache_example(fw,labels),fw


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_only_GN320_updates_full_volume_and_restores(device):
    if device == 'cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()): pytest.skip('BF16 CUDA')
    body,entry,fw = example(device)
    original = {k:v.clone() for k,v in body.state_dict().items()}
    original_norm = core.norm_state(body)
    result = core.fit(body,[entry],steps=3)
    assert result['update'] == 3 and len(result['norm']) == 20
    assert sum(v.numel() for v in result['norm'].values()) == 320
    assert any(not torch.equal(v,original_norm[k]) for k,v in result['norm'].items())
    for name,value in original.items(): torch.testing.assert_close(body.state_dict()[name],value,rtol=0,atol=0)
    assert not any(p.requires_grad or p.grad is not None for p in body.parameters())
    with torch.no_grad(), core.using_norm(body,result['norm']):
        predictions = body.columns(fw['memory'],fw['prior'],range(6),fw['shape'])
    assert predictions.shape == (6,18,8*8*4) and torch.isfinite(predictions).all()
    for name,value in original.items(): torch.testing.assert_close(body.state_dict()[name],value,rtol=0,atol=0)
    assert entry['prior'].untyped_storage().nbytes() == fw['prior'][:1].numel()*4
    assert entry['memory'].untyped_storage().nbytes() == fw['memory'][:1].numel()*4


def test_stop_resume_fixed_schedule_and_exception_restoration():
    body,entry,_ = example()
    full = core.fit(body,[entry],steps=40)
    event = threading.Event()
    partial = core.fit(body,[entry],steps=40,stopped=event.is_set,publish=lambda _:event.set())
    assert partial['update'] == 20
    continued = core.fit(body,[entry],steps=40,resumed=copy.deepcopy(partial))
    assert continued['losses'] == full['losses']
    for k,v in full['norm'].items(): torch.testing.assert_close(continued['norm'][k],v,rtol=0,atol=0)
    original = core.norm_state(body)
    with pytest.raises(RuntimeError):
        with core.using_norm(body,full['norm']): raise RuntimeError('injected')
    for k,v in original.items(): torch.testing.assert_close(core.norm_state(body)[k],v,rtol=0,atol=0)
    def bad_publish(_): raise RuntimeError('save failure')
    with pytest.raises(RuntimeError): core.fit(body,[entry],steps=1,publish=bad_publish)
    for k,v in original.items(): torch.testing.assert_close(core.norm_state(body)[k],v,rtol=0,atol=0)
    assert not any(p.requires_grad or p.grad is not None for p in body.parameters())


def test_history_only_consensus_dynamic_abstention_free_votes_and_outside():
    from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
    # One independent cell for each rule; all transforms identity initially.
    sem = np.full((4,4,4,4),11,np.uint8); observed = np.ones_like(sem,bool)
    sem[:,0,0,0] = int(next(iter(DYNAMIC_CLASS_IDS)))
    sem[:,0,0,1] = 17; observed[0,0,0,1] = False
    sem[:,0,0,2] = 17
    observed[0,0,0,3] = False
    observed[:2,0,1,0] = False
    raw = dict(history_occ=sem,history_observed=observed,history_poses=np.tile(np.eye(4),(4,1,1)),
        future_gt_occ=object(),mask_lidar=object(),future_annotations=object())
    grid = SimpleNamespace(x_min=0.,y_min=0.,z_min=0.,voxel_size=(1.,1.,1.),shape_hwd=(4,4,4))
    labels = core.consensus(raw,np.eye(4),grid,'cpu').reshape(4,4,4)
    assert labels[0,0].tolist() == [255,255,17,11]
    assert labels[0,1,0] == 255
    outside = np.eye(4); outside[0,3] = 100
    assert (core.consensus(raw,outside,grid,'cpu') == 255).all()
    many = torch.tensor([11]*1000+[17]*1000+[255]*12)
    ids = core.balanced_ids(many)
    assert len(ids) == 1024 and (many[ids] == 11).sum() == 512
    with pytest.raises(ValueError): core.balanced_ids(torch.tensor([255]))


def test_scene_split_and_negative_metrics_remain_visible():
    keys = [[f'scene-{i:04}',str(i)] for i in range(64)]
    fit,held = core.split_keys(keys)
    assert len(fit) == len(held) == 32 and not {k[0] for k in fit}&{k[0] for k in held}
    with pytest.raises(ValueError): core.split_keys(keys[:-1]+[keys[0]])
    counts = {n:run.metrics.empty(3) for n in run.NAMES}
    for name in run.NAMES: counts[name]['confusion'][:,11,11] = 8
    counts[run.NAMES[-1]]['confusion'][:,11,11] = 4
    counts[run.NAMES[-1]]['confusion'][:,11,17] = 4
    saved = dict(keys=held[:1],counts={n:{k:v.tolist() for k,v in f.items()} for n,f in counts.items()})
    restored,completed = run.restore_counts(saved,held,8)
    result = dict(status='complete',update=200,keys=completed,
        reports={n:run.metrics.report(v,(1.,2.,3.)) for n,v in restored.items()})
    text = run.summary(result)
    assert '-50.0' in text and 'MovingMicro' in text and 'NOT zero-adaptation' in text
    with pytest.raises(RuntimeError): run.restore_counts(saved,held[1:],8)
    saved['counts'][run.NAMES[0]]['confusion'][0][11][11] = 7
    with pytest.raises(RuntimeError): run.restore_counts(saved,held,8)


def test_entrypoint_fit_has_no_GT_reads_and_held_window_resume(tmp_path,monkeypatch):
    body,entry,fw = example()
    original = {k:v.clone() for k,v in body.state_dict().items()}
    (tmp_path/'training').mkdir()
    source = tmp_path/'training/source.pt'; source.write_bytes(b'test-source-not-scientific-weights')
    files = {n:tmp_path/n for n in ('config','dev_cache','dev_info')}
    for path in files.values(): path.write_bytes(b'tiny-metadata')
    keys = [[f'scene-{i:04}',str(i)] for i in range(64)]
    records = [dict(scene_name=k[0],t0_token=k[1],history_tokens=('a','b','c','d'),
        future_tokens=tuple(str(h) for h in range(6))) for k in keys]
    source_contract = dict(eval_keys=keys,data={n:run.base.sha256(p) for n,p in files.items()},
        implementation={},STC_inventory=[])
    grid = SimpleNamespace(x_min=0.,y_min=0.,z_min=0.,voxel_size=(1.,1.,1.),shape_hwd=entry['shape'])
    pcfg = SimpleNamespace(grid=SimpleNamespace(shape_hwd=entry['shape']))
    raw = dict(history_occ=np.full((4,*entry['shape']),11,np.uint8),
        history_observed=np.ones((4,*entry['shape']),bool),history_poses=np.tile(np.eye(4),(4,1,1)))
    event = threading.Event(); target_scenes = []
    class Inputs:
        def __init__(self,*args,**kwargs): pass
        def preflight(self,*args): return []
        def history(self,job,kind): return {},raw,np.arange(4)*.5
        def poses(self,job): return np.tile(np.eye(4),(6,1,1))
        def targets(self,job,indices):
            patch = torch.load(tmp_path/'output/norm_last.pt',weights_only=False)
            assert patch['update'] == 2  # no targets BEFORE fitting completes
            assert job.scene_name not in {k[0] for k in keys[:32]}
            target_scenes.append(job.scene_name)
            if len(target_scenes) == 3: event.set()
            return [raw['history_occ'][0]]*3
        def moving(self,*args): return [np.zeros(entry['shape'],bool)]*3
    class Predictor:
        def __init__(self,*args,**kwargs): self.provider=SimpleNamespace(workers=1)
        def prepare(self,*args,**kwargs): return None
        def forward(self,*args,**kwargs):
            with torch.no_grad(): logits=body.columns(fw['memory'],fw['prior'],range(6),fw['shape'])
            return dict(fw,logits=logits,poses=torch.eye(4)[None].expand(6,-1,-1))
        def close(self): pass
    monkeypatch.setattr(run,'select_source',lambda *args:source)
    monkeypatch.setattr(run,'load_frozen',lambda *args:(source_contract,body))
    monkeypatch.setattr(run.base,'load_cache',lambda *args:(None,records))
    monkeypatch.setattr(run.base,'load_runtime_config',lambda *args:None)
    monkeypatch.setattr(run.base,'make_prepare_config',lambda *args:pcfg)
    monkeypatch.setattr(run,'DenseInputs',Inputs); monkeypatch.setattr(run,'FPNPredictor',Predictor)
    real_consensus=core.consensus
    monkeypatch.setattr(core,'consensus',lambda raw,pose,_,device:real_consensus(raw,pose,grid,device))
    monkeypatch.setattr(core,'STEPS',2)
    argv=['--out-dir',str(tmp_path/'output'),'--device','cpu','--dataroot',str(tmp_path),'--stc-root',str(tmp_path)]
    for name,path in files.items(): argv.extend(['--'+name.replace('_','-'),str(path)])
    assert run.main(argv,event=event) == 130
    first=json.loads((tmp_path/'output/evaluation.json').read_text())
    assert len(first['keys']) == 3 and first['update'] == 2
    assert run.main(argv+['--resume'],event=threading.Event()) == 0
    final=json.loads((tmp_path/'output/evaluation.json').read_text())
    assert final['status'] == 'complete' and len(final['keys']) == 32
    assert len(target_scenes) == len(set(target_scenes)) == 32
    assert set(final['reports']) == set(run.NAMES)
    assert source.read_bytes() == b'test-source-not-scientific-weights'
    for name,value in original.items(): torch.testing.assert_close(body.state_dict()[name],value,rtol=0,atol=0)
