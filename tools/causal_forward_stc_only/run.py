"""Frozen shared checkpoint: original OCC route, fixed STC-only adaptation."""
import importlib.util
from pathlib import Path
import signal
import sys
import threading
from tools.source_scene_ego_all import run as original
from .common import select_source,load_frozen,InputPolicyPredictor,DenseInputs,ModalInputs

PROTOCOL='frozen_current_epoch2_OCC_original_STC_state_range_ALL4369_v1'
spec=importlib.util.spec_from_file_location('tools.source_scene_ego_all._stc_only_worker',original.__file__)
runner=importlib.util.module_from_spec(spec);runner.__file__=__file__
source=Path(original.__file__).read_text(encoding='utf-8')
old="files = [*sorted(Path(__file__).parent.glob('*.py')),"
extra=('tools/source_scene_ego_all/run.py','tools/source_scene_ego_all/parallel.py',
       'tools/causal_forward_stc_only/run.sh','tools/causal_forward_all/common.py',
       'tools/causal_forward_stc_candidate/common.py','tools/causal_forward_stc_candidate/method.py',
       'tools/causal_forward_stc_candidate/recipe.py','tools/causal_forward_average/weights.py',
       'tools/causal_forward_full/model.py','tools/causal_forward_recovery/adapter.py',
       'tools/causal_forward_train/data.py','tools/fpn_unet_eval/common.py')
if source.count(old)!=1:raise RuntimeError('original ALL worker provenance hook changed')
source=source.replace(old,old+" *[repo/n for n in "+repr(extra)+"],").replace('EGO_ALL','STC_ONLY')
exec(compile(source,original.__file__,'exec'),runner.__dict__)


def summary(value):
    c=value['contract'];lines=['===== ONE CHECKPOINT / OCC ORIGINAL / STC ADAPTED / ALL VAL =====',
        'status='+value['status'],
        'SAME exact current forward AdaLN/FPN/U-Net epoch2/update10292 weights; no network training.',
        'Input provenance selects policy BEFORE prediction; no labels or per-window score selection.',
        'OCC: original source geometry and original logits, ALL adaptation OFF.',
        'STC: fixed center/velocity half-blend plus FREE += -2 + min(t0 range/40m,1).',
        'SAME weights, DIFFERENT input-specific inference policies, explicitly declared.',
        'STC SUPERVISED OUTPUT CALIBRATION: existing two coefficients used FIRST32 VAL scenes GT future labels.',
        'No new fitting; correct GT future ego in Decoder/renderer; dense-valid/full-grid/no metric masks.',
        f'windows={c["windows"]}; shard={c.get("shard",[0,1])}; parent=ALL4369/150. NOT FPS.',
        'route                         horizon       IoU      mIoU MovingMicro']
    for name,report in value['reports'].items():
        for h,row in [*report['per_horizon'].items(),('Avg',report['average'])]:
            values=['NA' if row.get(k) is None else f'{row[k]:10.6f}' for k in ('IoU','mIoU','MovingMicro')]
            lines.append(f'{name:29s} {h:>5s} '+' '.join(values))
    return '\n'.join(lines+['Original unified-candidate and baseline entries/results unchanged. No automatic promotion.'])+'\n'


runner.__doc__=__doc__;runner.PROTOCOL=PROTOCOL
runner.select_source,runner.load_frozen=select_source,load_frozen
runner.EgoPredictor,runner.NuscInputs,runner.DenseModalInputs=InputPolicyPredictor,DenseInputs,ModalInputs
runner.summary=summary
select_jobs=runner.select_jobs


def main(event=None,argv=None):return runner.main(event,argv)


if __name__=='__main__':
    event=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:event.set())
    sys.exit(main(event) or 0)
