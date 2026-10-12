"""Single matched RANDOM 500-update CE+Lovasz+ego control; no direct source aux."""
from pathlib import Path
import json
import signal
import sys
import threading
from tools.joint_occ_experiments import run as legacy
from tools.shared_history_experiments.refine import run as refiner
from tools.shared_history_experiments.loss_controls.run import capped_groups

PROTOCOL='shared_semantic_refiner_loss500_ce_lovasz_ego_v1'


def train(original,*args,**kwargs):
    auxiliary=refiner.engine.source_auxiliary
    def zero(output,*_):
        value=output['residual_xy_m'].sum()*0
        return value,value  # no existence graph/AdamW decay; no direct label computation
    try:
        refiner.engine.source_auxiliary=zero
        result=original(*args,**kwargs)
    finally:refiner.engine.source_auxiliary=auxiliary
    result.update(source_auxiliary_computed=False,motion_weight=0.,existence_weight=0.)
    return result


def main(event=None,argv=None):
    a=legacy.parser().parse_args(argv)
    if (a.epochs!=1 or a.mode!='gt' or a.eval_population!='dev512' or a.max_updates or a.evaluate_only
            or (a.object_weight,a.ego_weight,a.lovasz_weight)!=(1.,1.,1.)):
        raise ValueError('single 500-update GT cycle; final DEV512; CE/Lovasz/ego retained')
    saved=(refiner.PROTOCOL,refiner.ResidualJoint,refiner.engine.train_arm,legacy.epoch_groups,
           legacy.stable_json_fingerprint,legacy.summarize)
    class MinimalJoint(saved[1]):
        def configs(self):return {**super().configs(),'loss_control':dict(motion_weight=0.,existence_weight=0.,
            fixed_updates=500,complete_dataset_epoch=False,auxiliary_computed=False)}
    repo=Path(__file__).resolve().parents[3]
    dependencies=list(Path(__file__).parent.glob('*.py'))+[repo/'tools/shared_history_experiments/loss_controls/run.py']
    def fingerprint(value):
        base=saved[4](value)
        if isinstance(value,dict) and 'tools/joint_occ_experiments/run.py' in value:
            return saved[4](dict(base=base,minimal_loss={str(p.relative_to(repo)):legacy.sha256(p) for p in dependencies}))
        return base
    def summary(result):
        text=saved[5](result)
        text=text.replace('Final full-grid CE+Lovasz + direct source geometry/existence + direct R10 ego geometry.',
            'ONLY full6 CE+Lovasz+ego R10; direct source motion/existence omitted, not computed.')
        return text.replace('completed_epochs=','completed_500_update_cycles=')+'\nPartial 500-update pass, NOT a complete dataset epoch.\n'
    try:
        refiner.PROTOCOL,refiner.ResidualJoint=PROTOCOL,MinimalJoint
        refiner.engine.train_arm=lambda *a,**kw:train(saved[2],*a,**kw)
        legacy.epoch_groups=lambda *a:capped_groups(saved[3],*a)
        legacy.stable_json_fingerprint,legacy.summarize=fingerprint,summary
        print('CE+Lovasz+ego ONLY: independent same RANDOM/seed/order/500-cycle cosine. Old three arms NOT rerun.',flush=True)
        return refiner.main(event,argv)
    finally:
        (refiner.PROTOCOL,refiner.ResidualJoint,refiner.engine.train_arm,legacy.epoch_groups,
         legacy.stable_json_fingerprint,legacy.summarize)=saved
        path=Path(a.out_dir)/'summary.txt';receipt=path.with_name('results.json')
        result=json.loads(receipt.read_text(encoding='utf-8')) if receipt.is_file() else {}
        if result.get('contract',{}).get('protocol')==PROTOCOL and path.is_file():
            result['loss_control']=dict(motion_weight=0.,existence_weight=0.,fixed_updates=500,complete_dataset_epochs=0)
            legacy.write_json(receipt,result)
            text=path.read_text(encoding='utf-8').replace('Fixed TRAIN20% x1 RANDOM pilot;', 'Fixed 500-update partial-pass pilot;')
            text=text.replace('Full six-horizon CE/Lovasz + original source geometry/existence + ego R10.',
                'Only full six-horizon CE/Lovasz+ego R10; no direct motion or existence labels/loss.')
            path.write_text(text,encoding='utf-8')


if __name__=='__main__':
    event=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:event.set())
    sys.exit(main(event) or 0)
