"""Frozen selection, integer population merge, exact speed entrance and resume."""
import copy
import json
from threading import Event
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.fpn_unet_eval import common, speed, run
from tools.final_semantic_evaluation import metrics


def records():
    return [dict(scene_name=f'VAL{i%150:03d}', t0_token=f't{i}', history_tokens=['h0','h1','h2',f't{i}'],
        future_tokens=[f'f{h}' for h in range(6)]) for i in range(4369)]


def fixture(device='cpu'):
    from test_fpn_unet import fixture as previous
    body, old, row = previous(device); pcfg = old.provider.pcfg; old.close()
    predictor = common.FPNPredictor(body.eval().requires_grad_(False), pcfg, device, workers=1)
    raw = {k: row[1][k] for k in ('history_occ', 'history_observed', 'history_poses')}
    return predictor, row, raw


def test_population_is_unique_64plus448_deterministic_and_never_planner_trimmed():
    data = records(); original = [[r['scene_name'],r['t0_token']] for r in data[1000:1064]]
    jobs, keys = common.select_jobs(data, original, [['TRAIN', 'token']])
    assert len(jobs) == len(keys) == len(set(map(tuple, keys))) == 512 and keys[:64] == original
    assert common.select_jobs(list(reversed(data)), original, [['TRAIN', 'token']])[1] == keys
    with pytest.raises(RuntimeError): common.select_jobs(data[:-1], original, [])
    with pytest.raises(RuntimeError): common.select_jobs(data, original, [['VAL001', 'TRAIN']])


def test_semantics_reader_does_not_access_visibility_masks(tmp_path):
    inputs = common.DenseInputs.__new__(common.DenseInputs)
    from collections import OrderedDict
    inputs.source = SimpleNamespace(dataroot=str(tmp_path)); inputs.stc = tmp_path/'stc'
    inputs.cache = OrderedDict(); inputs.bytes = 0; inputs.limit = 1024; inputs.inventory = {}
    # There are deliberately NO mask fields in this OCC file.
    folder = tmp_path/'gts'/'scene'/'token'; folder.mkdir(parents=True)
    sem = np.full((200,200,16), 4, np.uint8); np.savez(folder/'labels.npz', semantics=sem)
    labels, observed = inputs.frame('occ','scene','token')
    np.testing.assert_array_equal(labels,sem); assert observed.all()


def test_integer_merge_not_average_of_group_scores_and_duplicate_rejected():
    parts = []
    keys = [['VAL', str(i)] for i in range(512)]
    for lo, hi, correct in ((0,64,True),(64,512,False)):
        counts = metrics.empty(3)
        counts['confusion'][:,4,4 if correct else 17] = (hi-lo)*8
        counts['moving'][:] = counts['confusion']
        parts.append(dict(keys=keys[lo:hi], counts={name:{k:v.tolist() for k,v in counts.items()}
            for name in ('gt/transport','gt/semantic_refine')}))
    merged = common.merge_counts(parts, keys, 8)
    assert merged['gt/transport']['average']['mIoU'] == pytest.approx(12.5)
    bad = copy.deepcopy(parts); bad[1]['keys'][0] = bad[0]['keys'][0]
    with pytest.raises(RuntimeError): common.merge_counts(bad, keys, 8)


@pytest.mark.parametrize('device', ['cpu','cuda'])
def test_real_fresh_speed_matches_frozen_quality_and_recomputes_Strong(device,monkeypatch):
    if device=='cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()): pytest.skip('BF16 CUDA')
    p, oldrow, raw = fixture(device)
    try:
        prepared = p.prepare(oldrow[0], raw, times=np.arange(4)*.5)
        poses = oldrow[1]['future_poses']; calls = []; actual = speed.backgrounds
        monkeypatch.setattr(speed,'backgrounds',lambda *a,**k:(calls.append(1),actual(*a,**k))[1])
        with torch.no_grad():
            reference = p.dense(p.forward(prepared,poses=poses))['semantic_refine']
            ready = speed.prepare(prepared,poses,p.provider.device)
            first = speed.fresh(p.body,p.provider,ready); second = speed.fresh(p.body,p.provider,ready)
        assert calls==[1,1]
        # CUDA scatter drift, if present, must be reported by the measurement,
        # not hidden behind a relaxed numerical tolerance here.
        if device=='cpu':
            np.testing.assert_array_equal(first,reference); np.testing.assert_array_equal(second,reference)
        else:
            assert first.shape==reference.shape and first.dtype==np.uint8
        assert p.body.transport.ego_future_proj[-1].weight.requires_grad is False
        assert not any(v.grad is not None for v in p.body.parameters())
    finally:p.close()


def test_speed_aggregation_retains_byte_failures_and_correct_six_frame_units():
    samples=[]
    for kind in ('occ','stc'):
        for r in range(2): samples.append(dict(modality=kind,index=0,repeat=r,seconds=.2,changed_voxels=r,
            reference_repeat_changed_voxels=0,peak_allocated_mib=10.,peak_reserved_mib=20.,profile={}))
    report=speed.report(samples,1,2)
    assert report['occ']['output_frame_fps']==pytest.approx(30.) and not report['occ']['strict_byte_parity']
    with pytest.raises(RuntimeError):speed.report(samples+samples[:1],1,2)


def bundle(root,p):
    paths={k:root/k for k in ('config','dev_cache','dev_info')}
    for path in paths.values():path.write_text('fixture')
    keys=[['VAL',str(i)] for i in range(64)]
    c=dict(protocol=common.TRAIN_PROTOCOL,epochs=1,total_updates=1031,model=p.body.configs(),
        known='TRAIN_OCC_STC_dense_valid',source_policy='shared_pinned',registration='none',
        objective=dict(CE=1.,Lovasz=1.,motion=0.,existence=0.,ego=0.),parameters=sum(v.numel() for v in p.body.parameters()),
        train_keys=[['TRAIN',str(i)] for i in range(4086)],eval_keys=keys,STC_inventory=[],implementation={},
        data={k:run.base.sha256(v) for k,v in paths.items()},overrides=[])
    folder=root/'source';folder.mkdir()
    torch.save(dict(protocol=common.TRAIN_PROTOCOL,contract=c,epoch=1,batch=0,updates=1031,
        models={'gt':p.body.state_dict()}),folder/'epoch_0001.pt')
    (folder/'results.json').write_text(json.dumps(dict(status='complete',epoch=1,updates=1031,contract=c)))
    return folder,paths


def test_checkpoint_frozen_and_only_completed_source(tmp_path):
    p,_,_=fixture()
    try:
        folder,_=bundle(tmp_path,p)
        path=common.select_source(folder,tmp_path);before=path.read_bytes()
        c,model=common.load_frozen(path,'cpu')
        assert not model.training and not any(v.requires_grad for v in model.parameters()) and path.read_bytes()==before
        value=torch.load(path,weights_only=False);value['epoch']=0
        with pytest.raises(RuntimeError):common.validate_source(value)
        assert common.select_source(folder,tmp_path)==path
    finally:p.close()


def test_worker_real_metric_engine_resume_no_duplicate_quality(tmp_path,monkeypatch):
    p,record,raw=fixture();event=Event();calls=[]
    try:
        folder,paths=bundle(tmp_path,p);source=folder/'epoch_0001.pt';before=source.read_bytes()
        # Keep real metric accumulation/recovery, use only 1+1 jobs for speed.
        keys=[['VAL',str(i)] for i in range(65)]
        jobs=[SimpleNamespace(scene_name=k[0],t0_token=k[1]) for k in keys]
        monkeypatch.setattr(run,'select_jobs',lambda *a:(jobs,keys))
        monkeypatch.setattr(run.base,'load_cache',lambda *a:({},[]))
        monkeypatch.setattr(run.base,'load_runtime_config',lambda *a:None)
        monkeypatch.setattr(run.base,'make_prepare_config',lambda *a:p.provider.pcfg)
        class Inputs:
            def __init__(self,*a,**k):pass
            def preflight(self,*a):return []
            def history(self,job,kind):
                calls.append((kind,job.t0_token))
                if len(calls)==66:event.set()
                return record[0],raw,np.arange(4)*.5
            def poses(self,job):return np.tile(np.eye(4),(6,1,1))
            def targets(self,job,indices):return [np.full(p.provider.pcfg.grid.shape_hwd,4,np.uint8)]*3
            def moving(self,*a):return [np.ones(p.provider.pcfg.grid.shape_hwd,bool)]*3
        class Predictor:
            def __init__(self,body,pcfg,device,workers):self.provider=SimpleNamespace(pcfg=pcfg,workers=workers)
            def prepare(self,*a,**k):return None
            def forward(self,*a,**k):return None
            def dense(self,*a):return {n:np.full((6,*p.provider.pcfg.grid.shape_hwd),4,np.uint8) for n in ('transport','semantic_refine')}
            def close(self):pass
        monkeypatch.setattr(run,'DenseInputs',Inputs);monkeypatch.setattr(run,'FPNPredictor',Predictor)
        monkeypatch.setattr(run.speed,'measure',lambda *a,**k:{})
        out=tmp_path/'new';argv=['--source-dir',str(folder),'--out-dir',str(out),'--device','cpu',
            '--dataroot',str(tmp_path),'--stc-root',str(tmp_path/'stc')]
        for k,path in paths.items():argv+=['--'+k.replace('_','-'),str(path)]
        assert run.main(argv,event)==130 and len(calls)==66
        event.clear();assert run.main(argv+['--resume'],event)==0 and len(calls)==130
        assert run.main(argv+['--resume'],event)==0 and len(calls)==130
        result=json.loads((out/'evaluation.json').read_text())
        assert result['status']=='complete' and len(result['reports']['all512'])==4 and source.read_bytes()==before
    finally:p.close()
