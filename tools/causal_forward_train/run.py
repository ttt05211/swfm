"""ONE new RANDOM AdaLN TRAIN4086x1 arm; unchanged old optimizer/DDP loop."""
import importlib.util
import json
from pathlib import Path
import signal
import sys
import threading
from tools.fpn_unet import run as original, evaluation as original_evaluation
from tools.fpn_adaln_zero.run import check_settings
from .control import reference,sha
from .model import make_model,SOURCE_POLICY
from .data import ForwardProvider,ForwardPredictor,batches,forward_window

PROTOCOL = 'matched_AdaLN_forward_association_RANDOM_TRAIN4086x1_H6_v1'
spec = importlib.util.spec_from_file_location('tools.fpn_unet._forward_training',original.__file__)
runner = importlib.util.module_from_spec(spec)
source = Path(original.__file__).read_text(encoding='utf-8')
# A single checked METADATA literal is the only source adaptation. The loop,
# optimizer, reduction, scheduling, recovery and evaluation cadence are unchanged.
# Avoid recording a false shared_pinned policy in the new top-level contract.
old_literal = "source_policy='shared_pinned'"
if source.count(old_literal) != 1:raise RuntimeError('original trainer metadata hook changed')
exec(compile(source.replace(old_literal,"source_policy="+repr(SOURCE_POLICY)),original.__file__,'exec'),runner.__dict__)
_parser,_launch,_plan = runner.parser,runner.launch,runner.plan
espec = importlib.util.spec_from_file_location('tools.fpn_unet._forward_evaluation',original_evaluation.__file__)
evaluation = importlib.util.module_from_spec(espec);espec.loader.exec_module(evaluation)
evaluation.DirectPredictor,evaluation.forward_window = ForwardPredictor,forward_window
evaluation.PROTOCOL = 'AdaLN_forward_association_same64_dense_GTego_H6_v1'
runner.DirectProvider,runner.batches,runner.forward_window,runner.evaluate = (
    ForwardProvider,batches,forward_window,evaluation.evaluate)


def parser():
    p = _parser();p.description = __doc__
    p.add_argument('--control-dir',required=True,help='completed AdaLN original-association pilot; report ONLY')
    return p


def compare(result,control):
    if result['status'] != 'complete':return ''
    fields = ('train_keys','eval_keys','epochs','total_updates','seed','objective','learning_rates','schedule',
        'known','precision','world_size','data','overrides','STC_inventory','execution','parameters')
    for name in fields:
        if result['contract'][name] != control['contract'][name]:raise RuntimeError('unmatched C/A: '+name)
    lines = ['===== A ORIGINAL TRAIN/ORIGINAL EVAL vs C FORWARD TRAIN/FORWARD EVAL =====',
        'SAME RANDOM initialization/order/globalB4/source128/H6/CE+Lovasz/dense-valid/same64.',
        'Architecture/parameters/optimizer/budget unchanged; whole association policy is the variable.',
        'model route mIoU IoU MovingMicro']
    for label,value in (('A_recorded',control),('C_forward',result)):
        for route,m in value['evaluations']['1']['variants'].items():
            lines.append(f'{label:12s} {route:28s} {m["mIoU"]:.6f} {m["IoU"]:.6f} {m["MovingMicro"]:.6f}')
    for route,m in result['evaluations']['1']['variants'].items():
        old = control['evaluations']['1']['variants'][route]
        lines.append(route+' C_minus_A_recorded='+json.dumps({k:m[k]-old[k] for k in ('mIoU','IoU','MovingMicro')}))
    lines.append('Fixed final epoch; no selection, age/calibration/noise/extra loss or auto expansion. NOT FPS.')
    return '\n'.join(lines)+'\n'


def summary(result):
    c = result['contract'];ex = c['model'].get('experiment')
    text = ('===== AdaLN-Zero / FORWARD ASSOCIATION / TRAIN20pct x1 =====\n'
        f'status={result["status"]}; updates={result["updates"]}/{c["total_updates"]}; epoch={result["epoch"]}; parameters={c["parameters"]}\n'
        'RANDOM, NO trained weights/Adam imported. Only history identity policy changed.\n'
        'TRAIN and OCC/STC eval rebuild KTA/anchors/history reads/Memory/fresh Strong consistently.\n'
        'Only full6 CE+Lovasz, GT future ego; same dense-valid/same64; no ICP/age/bias/calibration/head.\n')
    if ex:
        path = Path(ex['control_results'])
        if sha(path) != ex['control_results_sha256']:raise RuntimeError('control report changed')
        text += compare(result,json.loads(path.read_text(encoding='utf-8')))
    return text


def launch(a,event,ctx):
    path,control = reference(a.control_dir);check_settings(a,ctx,control)
    root = Path(original.__file__).resolve().parents[2]
    for name,digest in control['contract']['model']['experiment']['candidate_implementation'].items():
        p = (root/name).resolve()
        if root not in p.parents or sha(p) != digest:raise RuntimeError('AdaLN control source changed: '+name)
    files = {p for folder in ('tools/causal_forward_train','tools/causal_forward_track')
        for p in (root/folder).glob('*') if p.suffix in ('.py','.sh')}
    files.add(root/'tools/fpn_adaln_zero/model.py')
    experiment = dict(control_results=str(path),control_results_sha256=sha(path),
        candidate_implementation={p.relative_to(root).as_posix():sha(p) for p in sorted(files)})
    def checked_plan(*args):
        keys,evaluation,groups = _plan(*args);c = control['contract']
        values = dict(train_keys=[list(k) for k in keys],
            eval_keys=[list(k) for k in original.base.record_keys(evaluation)],total_updates=len(groups[0]))
        for name,value in values.items():
            if value != c[name]:raise RuntimeError('ordered plan differs from AdaLN control: '+name)
        inventory = set()
        for r in evaluation:
            for token in r['history_tokens'][-4:]:
                p = (Path(a.stc_root)/r['scene_name']/str(token)/'labels.npz').resolve();s = p.stat()
                inventory.add((str(p),s.st_size,s.st_mtime_ns))
        if [list(v) for v in sorted(inventory)] != c['STC_inventory']:raise RuntimeError('STC inventory differs')
        return keys,evaluation,groups
    def factory(cfg):return make_model(cfg,expected_control_spec=control['contract']['model'],experiment=experiment)
    old_factory,old_plan = runner.make_model,runner.plan
    runner.make_model,runner.plan = factory,checked_plan
    try:
        print('FORWARD_TRAIN: C RANDOM TRAIN4086x1, forward identities in TRAIN AND EVAL; old A READ ONLY.',flush=True)
        return _launch(a,event,ctx)
    finally:runner.make_model,runner.plan = old_factory,old_plan


runner.PROTOCOL = PROTOCOL
runner.parser,runner.summary,runner.launch = parser,summary,launch


def main(argv=None,event=None):return runner.main(argv,event)


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:event.set())
    sys.exit(main(event=event) or 0)
