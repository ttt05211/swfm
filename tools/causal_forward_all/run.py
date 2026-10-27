"""Current forward AdaLN epoch2 / OCC+STC / ALL4369, optionally one-GPU shards."""
import importlib.util
from pathlib import Path
import signal
import sys
import threading
from tools.source_scene_ego_all import run as original
from .common import select_source,load_frozen,Epoch2Predictor,DenseInputs,ModalInputs

PROTOCOL='frozen_current_forward_AdaLN_epoch2_raw_ALL4369_v1'
spec=importlib.util.spec_from_file_location('tools.source_scene_ego_all._forward_all_worker',original.__file__)
runner=importlib.util.module_from_spec(spec);runner.__file__=__file__
source=Path(original.__file__).read_text(encoding='utf-8')
old="files = [*sorted(Path(__file__).parent.glob('*.py')),"
extra=('tools/source_scene_ego_all/run.py','tools/source_scene_ego_all/parallel.py','tools/causal_forward_all/run.sh',
       'tools/causal_forward_average/weights.py','tools/causal_forward_full/model.py',
       'tools/causal_forward_recovery/adapter.py','tools/causal_forward_train/data.py','tools/fpn_unet_eval/common.py')
new=old+" *[repo/n for n in "+repr(extra)+"],"
if source.count(old)!=1:raise RuntimeError('original ALL worker provenance hook changed')
exec(compile(source.replace(old,new).replace('EGO_ALL','FORWARD_ALL'),original.__file__,'exec'),runner.__dict__)


def summary(value):
    c=value['contract'];lines=['===== CURRENT FORWARD AdaLN / EPOCH2 / ALL VAL4369 =====','status='+value['status'],
        'Fixed source epoch2/update10292; forward identities / AdaLN-Zero / FPN / U-Net / KTA query+skip / no ICP.',
        'FOUR histories -> SIX futures; GT future ego in Decoder AND renderer; dense-valid OCC+STC.',
        'FREE bias and fixed age OFF; no mean, internal head, training, calibration or backend switch.',
        f'windows={c["windows"]}; parent=ALL4369/150; full-grid/no metric masks/alignment/planner trimming. NOT FPS.',
        'route                         horizon       IoU      mIoU MovingMicro']
    for name,report in value['reports'].items():
        for h,row in [*report['per_horizon'].items(),('Avg',report['average'])]:
            numbers=['NA' if row.get(k) is None else f'{row[k]:10.6f}' for k in ('IoU','mIoU','MovingMicro')]
            lines.append(f'{name:29s} {h:>5s} '+' '.join(numbers))
    return '\n'.join(lines+['Source checkpoints/Adam/RNG/reports/history caches read only; no automatic selection/promotion.'])+'\n'


runner.__doc__=__doc__;runner.PROTOCOL=PROTOCOL
runner.select_source,runner.load_frozen=select_source,load_frozen
runner.EgoPredictor,runner.NuscInputs,runner.DenseModalInputs=Epoch2Predictor,DenseInputs,ModalInputs
runner.summary=summary
select_jobs=runner.select_jobs


def main(event=None,argv=None):return runner.main(event,argv)


if __name__=='__main__':
    event=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:event.set())
    sys.exit(main(event) or 0)
