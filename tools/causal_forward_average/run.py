"""Current forward AdaLN mean(4000,10000,10292) / single GPU / original same64."""
import importlib.util
import json
from pathlib import Path
import signal
import sys
import threading
import torch
from tools.source_scene_integrated_average import run as original
from tools.causal_forward_full.model import checkpoint_model
from tools.causal_forward_recovery.adapter import RecoveryPredictor
from tools.fpn_unet_eval.common import DenseInputs,ModalInputs
from .weights import PROTOCOL,STEPS,METRICS,select_source,load_sources,mean_payload

spec=importlib.util.spec_from_file_location('tools.source_scene_integrated_average._forward_mean3',original.__file__)
runner=importlib.util.module_from_spec(spec);runner.__file__=__file__
source=Path(original.__file__).read_text(encoding='utf-8')
old="execution_files = sorted(Path(__file__).parent.glob('*.py'))+[repo/n for n in ("
extra=('tools/source_scene_integrated_average/run.py','tools/causal_forward_average/run.sh',
       'tools/causal_forward_full/model.py','tools/causal_forward_recovery/adapter.py',
       'tools/causal_forward_train/data.py','tools/fpn_unet_eval/common.py')
new="execution_files = sorted(Path(__file__).parent.glob('*.py'))+[repo/n for n in "+repr(extra)+"]+[repo/n for n in ("
if source.count(old)!=1:raise RuntimeError('original frozen mean workflow provenance hook changed')
exec(compile(source.replace(old,new),original.__file__,'exec'),runner.__dict__)


class ForwardMeanPredictor(RecoveryPredictor):
    @torch.no_grad()
    def forward(self,row,*,poses):return self.infer(row,poses)


def summary(v):
    lines=['===== CURRENT FORWARD AdaLN / STC-mIoU-SELECTED MEAN3 / SAME64 =====','status='+v['status'],
        'Equal 1/3: 4000,10000,10292(epoch2); ONE GPU / one model process / no training.',
        'Three source steps chosen by STC semantic_refine mIoU in the already-inspected DEV64 report.',
        'Current forward identities/KTA query+skip / AdaLN-Zero / FPN / U-Net / no ICP; no old network route.',
        'OCC AND STC dense-valid; GT future ego; full-grid/Moving/1,2,3s mean; FREE bias and fixed age OFF.',
        'Source controls READ from this run reports; ONLY averaged weights freshly evaluated.',
        'PARAMETERS only, float64 mean -> original dtype; identical buffers copied; no Adam/RNG averaging.',
        'model    route                         mIoU       IoU MovingMicro']
    for name,scores in v['comparison'].items():
        for route,m in scores.items():lines.append(f'{name:8s} {route:28s} '+' '.join(f'{m[k]:.6f}' for k in METRICS))
    for step in map(str,STEPS):
        for route,m in v['comparison'].get('mean3',{}).items():
            old=v['comparison'][step][route]
            lines.append(f'mean3_minus_{step} {route}='+json.dumps({k:m[k]-old[k] for k in METRICS}))
    return '\n'.join(lines+['Original checkpoints/Adam/RNG/reports/caches unchanged; mean3.pt is EVAL ONLY.',
        'DEV-selected recipe and repeatedly inspected development64; NOT independent/full-VAL proof or FPS.'])+'\n'


runner.__doc__=__doc__
runner.PROTOCOL,runner.STEPS,runner.METRICS=PROTOCOL,STEPS,METRICS
runner.select_source,runner.load_sources,runner.mean_payload=select_source,load_sources,mean_payload
runner.checkpoint_model,runner.QueryPredictor=checkpoint_model,ForwardMeanPredictor
runner.NuscInputs,runner.DenseInputs=DenseInputs,ModalInputs
runner.summary=summary


def main(argv=None,event=None):return runner.main(argv,event)


if __name__=='__main__':
    event=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:event.set())
    sys.exit(main(event=event) or 0)
