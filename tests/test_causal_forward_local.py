"""Real current-model roundtrip, small export, source provenance and Windows LF."""
import copy
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import zipfile
import pytest
import torch
from real_motion.prepared import PrepareConfig
from tools.causal_forward_local import export,portable
from test_causal_forward_average import body_and_contract


def source(tmp_path):
    body,c=body_and_contract()
    name='tools/causal_forward_full/model.py'
    c['implementation']={name:export.base.sha256(Path(__file__).resolve().parents[1]/name)}
    folder=tmp_path/'adaln_forward_full_2ep_fixture';folder.mkdir();path=folder/'epoch_0002.pt'
    torch.save(dict(protocol=c['protocol'],contract=c,epoch=2,batch=0,updates=10292,
        models={'gt':body.state_dict()},optimizers={'gt':'POISON Adam'},rank_rng=['POISON RNG']),path)
    score=dict(windows=64,keys=c['eval_keys'],variants={r:dict(mIoU=18.,IoU=28.,MovingMicro=19.)
        for r in ('occ_gt_transport','occ_gt_semantic_refine','stc_gt_transport','stc_gt_semantic_refine')})
    return body,c,path,score


def test_actual_current_model_weights_only_roundtrip_and_source_unchanged(tmp_path,monkeypatch):
    body,c,path,scores=source(tmp_path);before=export.base.sha256(path)
    monkeypatch.setattr(torch.optim,'AdamW',lambda *a,**k:pytest.fail('optimizer created'))
    result=export.export(path,tmp_path/'small',asdict(PrepareConfig()),scores,expected_contract=c)
    assert result['source_checkpoint_sha256']==before and export.base.sha256(path)==before
    assert result['no_optimizer'] and result['no_RNG'] and result['no_dataset_copy']
    with zipfile.ZipFile(result['archive']) as z:
        assert set(z.namelist())=={'epoch_0002.pt','manifest.json','source_dev64.json'} and z.testzip() is None
        packed=torch.load(io.BytesIO(z.read('epoch_0002.pt')),weights_only=False)
        assert set(packed)=={'protocol','contract','epoch','batch','updates','models'}
    loaded,contract,info=portable.load_weights(result['archive'])
    assert contract==c and loaded.configs()==body.configs() and sum(p.numel() for p in loaded.parameters())==2816122
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())
    for k,v in body.state_dict().items():torch.testing.assert_close(loaded.state_dict()[k],v,rtol=0,atol=0)
    assert info['prepare_config']==PrepareConfig() and info['source_dev64']==scores
    assert info['manifest']['source_checkpoint_sha256']==before
    with pytest.raises(FileExistsError):export.export(path,tmp_path/'small',asdict(PrepareConfig()),scores)


def test_export_rejects_old_or_mean_source_mismatch_and_source_output_directory(tmp_path):
    _,c,path,scores=source(tmp_path)
    with pytest.raises(RuntimeError,match='outside training'):export.export(path,path.parent/'new',{},scores)
    bad=copy.deepcopy(scores);bad['keys']=bad['keys'][::-1]
    with pytest.raises(RuntimeError,match='reference64'):export.export(path,tmp_path/'bad_keys',{},bad)
    with pytest.raises(RuntimeError,match='contracts differ'):export.export(path,tmp_path/'bad_contract',{},scores,dict(c,world_size=1))
    saved=torch.load(path,weights_only=False);saved['protocol']='old_or_mean';torch.save(saved,path)
    with pytest.raises(RuntimeError,match='epoch2'):export.export(path,tmp_path/'old',{},scores)
    assert not any((tmp_path/n).exists() for n in ('bad_keys','bad_contract','old'))


def test_archive_tamper_and_implementation_change_rejected(tmp_path):
    _,_,path,scores=source(tmp_path)
    receipt=export.export(path,tmp_path/'small',asdict(PrepareConfig()),scores)
    with zipfile.ZipFile(receipt['archive']) as z:members={n:z.read(n) for n in z.namelist()}
    members['epoch_0002.pt']+=b'tampered'
    changed=tmp_path/'tampered.zip'
    with zipfile.ZipFile(changed,'w') as z:
        for name,value in members.items():z.writestr(name,value)
    with pytest.raises(RuntimeError,match='verified weights-only'):portable.load_weights(changed)
    text=b'unchanged_line\n';file=tmp_path/'implementation.py';file.write_bytes(text.replace(b'\n',b'\r\n'))
    manifest={'implementation.py':hashlib.sha256(text).hexdigest()}
    assert portable.verify_implementation(tmp_path,[manifest])['implementation.py']['source_sha256']==manifest['implementation.py']
    file.write_bytes(b'modified_line\r\n')
    with pytest.raises(RuntimeError,match='implementation changed'):portable.verify_implementation(tmp_path,[manifest])
    with pytest.raises(RuntimeError,match='outside repository'):portable.verify_implementation(tmp_path,[{'../outside':'unused'}])


def test_cli_uses_completed_current_epoch2_without_data_or_optimizer(tmp_path,monkeypatch):
    _,c,path,scores=source(tmp_path);config=tmp_path/'config.yaml';config.write_text('fixture')
    saved=torch.load(path,weights_only=False);saved['contract']['data']={'config':export.base.sha256(config)}
    torch.save(saved,path);c=saved['contract']
    (path.parent/'results.json').write_text(json.dumps(dict(status='complete',epoch=2,updates=10292,
        contract=c,evaluations={'10292':scores})))
    monkeypatch.setattr(export.base,'load_runtime_config',lambda *a:None)
    monkeypatch.setattr(export.base,'make_prepare_config',lambda *a:PrepareConfig())
    monkeypatch.setattr(export.base,'load_cache',lambda *a:pytest.fail('dataset loaded'))
    args=['--source-dir',str(path.parent),'--config',str(config),'--out-dir',str(tmp_path/'small')]
    receipt=export.main(args)
    assert Path(receipt['archive']).is_file() and receipt['parameters']==2816122
