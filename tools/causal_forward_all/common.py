"""Only current full epoch2, dense input and validated forward-history inference."""
import torch
from tools.causal_forward_average.weights import validate_contract,select_source as source_directory
from tools.causal_forward_full.run import PROTOCOL as TRAIN_PROTOCOL
from tools.causal_forward_full.model import checkpoint_model
from tools.causal_forward_recovery.adapter import RecoveryPredictor
from tools.fpn_unet_eval.common import DenseInputs,ModalInputs


def select_source(explicit,root):return source_directory(explicit,root)/'epoch_0002.pt'


def validate_source(saved):
    c=saved['contract'];validate_contract(c)
    if (saved.get('protocol')!=TRAIN_PROTOCOL or saved.get('epoch')!=2 or saved.get('batch')!=0
            or saved.get('updates')!=10292 or set(saved.get('models',{}))!={'gt'}):
        raise RuntimeError('only completed current forward AdaLN epoch2/update10292; no mean/pilot/old checkpoint')
    return c


def load_frozen(path,device):
    saved=torch.load(path,map_location='cpu',weights_only=False);c=validate_source(saved)
    return c,checkpoint_model(saved,device)


class Epoch2Predictor(RecoveryPredictor):
    @torch.no_grad()
    def forward(self,row,*,poses):return self.infer(row,poses)
