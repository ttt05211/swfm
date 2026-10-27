"""Strict latest-source loader, admitting ONLY byte-equivalent CRLF/LF text."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
import torch
from real_motion.local_st_world_model_v17 import config_from_mapping_v17
from tools.source_scene_stage1.model import ReaderConfig
from tools.causal_forward_train.model import ForwardBody
from tools.causal_forward_all.common import validate_source
from tools.stc_overnight.bundle import prepare_config
from .export import PROTOCOL


def verify_implementation(root,manifests):
    root=Path(root).resolve();code={}
    for manifest in manifests:
        for name,expected in manifest.items():
            file=(root/name).resolve()
            if root not in file.parents:raise RuntimeError('source implementation path outside repository')
            raw=file.read_bytes();actual=hashlib.sha256(raw).hexdigest();lf=hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()
            if expected not in (actual,lf):raise RuntimeError('original source implementation changed: '+name)
            code[name]=dict(source_sha256=expected,local_sha256=actual,LF_sha256=lf)
    return code


def load_weights(path,device='cpu'):
    path=Path(path).resolve();container_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    with zipfile.ZipFile(path) as z:
        if len(z.namelist())!=len(set(z.namelist())):raise RuntimeError('duplicate archive member')
        m=json.loads(z.read('manifest.json'));data=z.read('epoch_0002.pt');scores=json.loads(z.read('source_dev64.json'))
    if (m.get('protocol')!=PROTOCOL or m.get('optimizer_included') is not False or m.get('RNG_included') is not False
            or m.get('raw_data_included') is not False or m.get('evaluation_only') is not True
            or len(data)!=m['checkpoint_payload_bytes'] or hashlib.sha256(data).hexdigest()!=m['checkpoint_payload_sha256']):
        raise RuntimeError('not the latest verified weights-only portable export')
    saved=torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
    if set(saved)!={'protocol','contract','epoch','batch','updates','models'}:raise RuntimeError('only weights/config headers allowed; no optimizer or RNG')
    c=validate_source(saved);s=c['model'];root=Path(__file__).resolve().parents[2]
    if m['source_protocol']!=saved['protocol'] or m['source_update']!=saved['updates'] or m['parameters']!=c['parameters']:
        raise RuntimeError('portable source metadata differs')
    code=verify_implementation(root,(c['implementation'],(s.get('experiment') or {}).get('candidate_implementation',{})))
    # Same exact ForwardBody constructor as the original full loader, after
    # validating all source hashes above. Never rewrite/bypass the source contract.
    body=ForwardBody(config_from_mapping_v17(s['motion']),feature_width=s['shared_history']['width'],
        z_bins=s['shared_history']['z_bins'],memory_width=s['repair']['memory_width'],
        reader_config=ReaderConfig(**s['reader']),experiment=s.get('experiment'))
    if body.configs()!=s or sum(p.numel() for p in body.parameters())!=m['parameters']:raise RuntimeError('original architecture differs')
    template=body.state_dict();state=saved['models']['gt']
    if set(state)!=set(template):raise RuntimeError('weight keys differ')
    for name,value in state.items():
        if value.shape!=template[name].shape or value.dtype!=template[name].dtype or not torch.isfinite(value).all():
            raise RuntimeError('weight shape/dtype/finite differs: '+name)
    body.load_state_dict(state,strict=True)
    if m['reference64_keys']!=c['eval_keys'] or scores['keys']!=c['eval_keys'] or scores['windows']!=64:
        raise RuntimeError('source reference64 differs')
    return body.to(device).eval().requires_grad_(False),c,dict(manifest=m,source_dev64=scores,
        archive_sha256=container_sha,code_receipts=code,prepare_config=prepare_config(m['prepare_config']))
