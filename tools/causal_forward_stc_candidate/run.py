"""FROZEN latest source-state + fixed range calibration; complete VAL only."""
import importlib.util
from pathlib import Path
import signal
import sys
import threading
from tools.source_scene_ego_all import run as original
from .common import select_source,load_frozen,CandidatePredictor,DenseInputs,ModalInputs

PROTOCOL='frozen_current_forward_CVstate_range_calibrated_epoch2_ALL4369_v1'
spec=importlib.util.spec_from_file_location('tools.source_scene_ego_all._stc_state_range_worker',original.__file__)
runner=importlib.util.module_from_spec(spec);runner.__file__=__file__
source=Path(original.__file__).read_text(encoding='utf-8')
old="files = [*sorted(Path(__file__).parent.glob('*.py')),"
extra=('tools/source_scene_ego_all/run.py','tools/source_scene_ego_all/parallel.py',
       'tools/causal_forward_stc_candidate/run.sh','tools/causal_forward_all/common.py',
       'tools/causal_forward_average/weights.py','tools/causal_forward_full/model.py',
       'tools/causal_forward_recovery/adapter.py','tools/causal_forward_train/data.py','tools/fpn_unet_eval/common.py')
new=old+" *[repo/n for n in "+repr(extra)+"],"
if source.count(old)!=1:raise RuntimeError('original ALL worker provenance hook changed')
exec(compile(source.replace(old,new).replace('EGO_ALL','STC_CANDIDATE'),original.__file__,'exec'),runner.__dict__)


def summary(value):
    c=value['contract'];lines=['===== CURRENT FROZEN BODY / CV STATE + RANGE FREE / ALL VAL =====','status='+value['status'],
        'Exact original forward AdaLN/FPN/U-Net epoch2/update10292 weights; no new network or optimizer steps.',
        'Original forward identities/no ICP; source center+velocity half-blend with four-history line fit.',
        'Observed history feature points stay at actual positions; future source/Strong/direct Memory geometry are consistent.',
        'FREE logit += -2 + min(range in LAST observed ego /40m,1); SAME model/formula for GT and STC.',
        'SUPERVISED OUTPUT CALIBRATION: two coefficients fitted using FIRST32 DEV scenes GT future occupancy.',
        'GT future ego conditions Decoder/rendering/calibration; dense-valid inputs; full-grid/no metric masks or pose alignment.',
        f'windows={c["windows"]}; shard={c.get("shard",[0,1])}; parent=ALL4369/150; no planner trimming. NOT FPS.',
        'Coefficients frozen BEFORE ALL4369; no fitting or coefficient selection during this evaluation.',
        'route                         horizon       IoU      mIoU MovingMicro']
    for name,report in value['reports'].items():
        for h,row in [*report['per_horizon'].items(),('Avg',report['average'])]:
            numbers=['NA' if row.get(k) is None else f'{row[k]:10.6f}' for k in ('IoU','mIoU','MovingMicro')]
            lines.append(f'{name:29s} {h:>5s} '+' '.join(numbers))
    stc=value['reports'].get('stc_gt/semantic_refine',{}).get('average',{})
    goal='pending'
    if value['status']=='complete' and c['windows']==4369 and stc:
        goal=str(stc['mIoU']>=19 and stc['IoU']>=29)
    return '\n'.join(lines+[f'Fixed target STC mIoU>=19 and IoU>=29: met={goal}',
        'Source checkpoint/Adam/RNG/history caches unchanged. Independent CANDIDATE, old defaults/results retained.'])+'\n'


runner.__doc__=__doc__;runner.PROTOCOL=PROTOCOL
runner.select_source,runner.load_frozen=select_source,load_frozen
runner.EgoPredictor,runner.NuscInputs,runner.DenseModalInputs=CandidatePredictor,DenseInputs,ModalInputs
runner.summary=summary
select_jobs=runner.select_jobs


def main(event=None,argv=None):return runner.main(event,argv)


if __name__=='__main__':
    event=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:event.set())
    sys.exit(main(event) or 0)
