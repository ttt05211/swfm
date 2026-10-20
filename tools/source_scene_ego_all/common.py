"""Latest FULL-TRAIN ego-conditioned epoch2 ONLY; no legacy model fallback."""
import json
from pathlib import Path
import torch
from tools.source_scene_ego_full.run import PROTOCOL as TRAIN_PROTOCOL
from tools.source_scene_ego_full.model import checkpoint_model
from tools.source_scene_ego_conditioned.model import VARIANT
from tools.source_scene_stage2.adapter import DirectPredictor
from tools.source_scene_ego_conditioned.adapter import forward_window
from tools.history_validity.data import policy_raw


def validate_source(saved):
    c=saved['contract']; s=c['model']
    if (saved.get('protocol')!=TRAIN_PROTOCOL or c.get('protocol')!=TRAIN_PROTOCOL or
            saved.get('epoch')!=2 or saved.get('batch')!=0 or saved.get('updates')!=10292 or
            c.get('epochs')!=2 or c.get('total_updates')!=10292 or c.get('epochs_updates')!=[5145,5147] or
            c.get('arm')!='ego_query_pure_memory_cross_first_full' or c.get('registration')!='none' or
            c.get('known')!='TRAIN_OCC_STC_dense_valid' or c.get('source_policy')!='shared_pinned' or
            c.get('objective')!=dict(CE=1.,Lovasz=1.,motion=0.,existence=0.,ego=0.) or
            s.get('variant')!=VARIANT or s.get('kta_conditioning') is not True or
            s.get('memory',{}).get('future_query_content') is not False or
            s.get('future_query',{}).get('decoder_order')!='history_cross_future_self_FFN' or
            set(saved.get('models',{}))!={'gt'} or len(c['train_keys'])!=20430 or len(c['eval_keys'])!=64 or
            len(set(map(tuple,c['train_keys'])))!=20430 or len(set(map(tuple,c['eval_keys'])))!=64):
        raise RuntimeError('fixed completed NEW ego-conditioned FULL TRAIN epoch2/update10292 required; no pilot/old/mean fallback')
    return c


def select_source(explicit,search):
    if explicit:
        p=Path(explicit).resolve()
        if not p.is_dir():raise RuntimeError('--source-dir must be the original full training DIRECTORY')
        return p/'epoch_0002.pt'
    candidates=[]
    for p in Path(search).glob('scene_ego_query_full_2ep_*/results.json'):
        v=json.loads(p.read_text(encoding='utf-8'))
        if v.get('status')=='complete' and v.get('contract',{}).get('protocol')==TRAIN_PROTOCOL and (p.parent/'epoch_0002.pt').is_file():
            candidates.append((p.parent/'epoch_0002.pt').resolve())
    if len(candidates)!=1:raise RuntimeError('set EGO_FULL_DIR to the unique completed NEW full run; candidates='+str(candidates))
    return candidates[0]


def load_frozen(path,device):
    saved=torch.load(path,map_location='cpu',weights_only=False);c=validate_source(saved)
    return c,checkpoint_model(saved,device)


class EgoPredictor(DirectPredictor):
    @torch.no_grad()
    def forward(self,row,*,poses):
        # SAME explicit future poses condition Query AND ALL future geometry; no inherited old forward.
        return forward_window(self.body,self.provider,row,poses=poses,fresh=True)


class DenseModalInputs:
    def __init__(self,source,kind):self.source,self.kind=source,kind
    def history(self,job,_kind='occ'):
        rec,raw,times=self.source.history(job,self.kind)
        return rec,policy_raw(raw,'dense_valid'),times
    def poses(self,job):return self.source.poses(job)
    def targets(self,job,indices):return self.source.targets(job,indices)
    def moving(self,*args):return self.source.moving(*args)
