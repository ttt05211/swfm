from __future__ import annotations
import hashlib,random
from pathlib import Path
import numpy as np,torch
from real_motion.msp import FEATURE_DIM,MSPProbeHead

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def msp_checkpoint_provenance(path,ck=None):
    ck=torch.load(path,map_location='cpu',weights_only=False) if ck is None else ck;tm=ck.get('train_metadata') or {};vm=ck.get('val_metadata') or {}
    return {'path':str(Path(path).resolve()),'sha256':sha256_file(path),'feature_dim':ck.get('feature_dim'),'hidden_dim':ck.get('hidden_dim'),'num_heads':ck.get('num_heads'),'num_modes':ck.get('num_modes'),'future_frames':ck.get('future_frames'),'train_scene_names':list(tm.get('scene_names',[])),'val_scene_names':list(vm.get('scene_names',[]))}
def load_frozen_msp(path,device):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if int(ck.get('feature_dim',-1))!=int(FEATURE_DIM):raise RuntimeError('MSP feature_dim mismatch')
    if int(ck.get('future_frames',-1))!=6:raise RuntimeError('MT-V1 requires six-horizon MSP checkpoint')
    model=MSPProbeHead(feature_dim=FEATURE_DIM,hidden_dim=int(ck['hidden_dim']),num_heads=int(ck['num_heads']),num_modes=int(ck['num_modes']),future_frames=int(ck['future_frames']));model.load_state_dict(ck['state_dict'],strict=True);model.to(device).eval()
    for p in model.parameters():p.requires_grad_(False)
    return ck,model
@torch.no_grad()
def infer_source_scores(candidates,sources,msp,device):
    if not candidates:
        for s in sources:s.msp_score=0.
        return {int(s.source_id):0. for s in sources}
    feats=torch.as_tensor(np.stack([np.asarray(c.feature,np.float32) for c in candidates]),device=device).unsqueeze(0);mask=torch.ones((1,len(candidates)),device=device,dtype=torch.bool);out=msp(feats,mask)
    logits=out['activation_logits'] if isinstance(out,dict) else out.activation_logits;prob=torch.sigmoid(logits[0]).detach().float().cpu().numpy();by={int(c.candidate_id):i for i,c in enumerate(candidates)};scores={}
    for s in sources:
        if not s.msp_candidate_indices:score=0.
        else:
            rows=np.stack([prob[by[int(cid)]] for cid in s.msp_candidate_indices]);w=np.asarray(s.overlap_weights,float);ph=(rows*w[:,None]).sum(0) if len(w)==len(rows) and w.sum()>0 else rows.mean(0);score=float(ph.max())
        s.msp_score=score;scores[int(s.source_id)]=score
    return scores
def _eligible(sources):return [s for s in sources if s.crop_eligible]
def route_sources(sources,budget,strategy='msp',scores=None,seed=3407,gt_moving_source_ids=None):
    ss=_eligible(sources)
    if budget in (0,'0'):return ()
    if budget=='all':return tuple(int(s.source_id) for s in ss)
    q=max(0,int(budget));q=min(q,len(ss))
    if strategy=='msp':order=sorted(ss,key=lambda s:(-float((scores or {}).get(int(s.source_id),s.msp_score)),int(s.source_id)))
    elif strategy=='speed':order=sorted(ss,key=lambda s:(-float(np.linalg.norm(s.velocity_world[:2])),int(s.source_id)))
    elif strategy=='uniform':
        ids=[int(s.source_id) for s in ss];rng=random.Random(int(seed));rng.shuffle(ids);return tuple(sorted(ids[:q]))
    elif strategy in ('gt_moving','gt_moving_diagnostic'):
        allowed=set(map(int,gt_moving_source_ids or ()));order=[s for s in sorted(ss,key=lambda s:int(s.source_id)) if int(s.source_id) in allowed]
    else:raise ValueError(f'unknown routing strategy {strategy}')
    return tuple(int(s.source_id) for s in order[:q])
def _u01(seed,key):return int.from_bytes(hashlib.sha256(f'{seed}:{key}'.encode()).digest()[:8],'big')/2**64
def training_budget_selection(sources,progress,*,seed=3407,sample_id='',warmup_fraction=.2,later_all_probability=.5,sparse_budget=16):
    eligible=_eligible(sources)
    if float(progress)<float(warmup_fraction):return tuple(int(s.source_id) for s in eligible),'all_warmup'
    if _u01(seed,sample_id+':budget')<float(later_all_probability):return tuple(int(s.source_id) for s in eligible),'all_later'
    sel=route_sources(eligible,sparse_budget,strategy='uniform',seed=int.from_bytes(hashlib.sha256(f'{seed}:{sample_id}:subset'.encode()).digest()[:8],'big'));return sel,'uniform_q16'
def scene_mirror_flag(*,seed=3407,sample_id='',probability=.5):return _u01(seed,sample_id+':mirror')<float(probability)
