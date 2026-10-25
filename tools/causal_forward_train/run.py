"""ONE new RANDOM AdaLN TRAIN4086x1 arm; unchanged old optimizer/DDP loop."""
import importlib.util
import json
from pathlib import Path
import signal
import sys
import threading
from tools.fpn_unet import run as original
from .control import reference,sha
from .model import make_model,SOURCE_POLICY
from .data import ForwardProvider,batches,forward_window
from . import evaluation
from .calibration import CALIBRATION

PROTOCOL = 'AdaLN_forward_RANDOM_TRAIN4086x1_singleGPU_raw_FREEreadout_v2'
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
runner.DirectProvider,runner.batches,runner.forward_window,runner.evaluate = (
    ForwardProvider,batches,forward_window,evaluation.evaluate)


def parser():
    p = _parser();p.description = __doc__
    p.add_argument('--control-dir',required=True,help='completed AdaLN original-association pilot; report ONLY')
    return p


def check_settings(a,ctx,control):
    c = control['contract']
    # User explicitly changed NEW training to ONE GPU. Preserve the actual old
    # control world-size, globalB4 and all other controls; do not fake metadata.
    if ctx.world_size != 1 or c['world_size'] not in (1,2):
        raise RuntimeError('new training must use ONE GPU; completed control may use one or two')
    expected = dict(seed=a.seed,learning_rates=[a.motion_lr,a.repair_lr],overrides=a.override,
        schedule='whole_1_epoch_cosine_floor0.01',precision='BF16_network_FP64_geometry_full6',
        execution=dict(torch=str(original.torch.__version__),cuda=original.torch.version.cuda,
            cpu_workers=a.cpu_workers,prefetch_workers=a.prefetch_workers))
    for name,value in expected.items():
        if c.get(name) != value:raise RuntimeError('unmatched AdaLN control: '+name)
    for name,digest in c['data'].items():
        if sha(getattr(a,name)) != digest:raise RuntimeError('unmatched control data: '+name)
    root = Path(original.__file__).resolve().parents[2]
    for name,digest in c['implementation'].items():
        if sha(root/name) != digest:raise RuntimeError('original control implementation changed: '+name)


def compare(result,control):
    if result['status'] != 'complete':return ''
    fields = ('train_keys','eval_keys','epochs','total_updates','seed','objective','learning_rates','schedule',
        'known','precision','data','overrides','STC_inventory','execution','parameters')
    for name in fields:
        if result['contract'][name] != control['contract'][name]:raise RuntimeError('unmatched C/A: '+name)
    lines = ['===== A ORIGINAL TRAIN/ORIGINAL EVAL vs C FORWARD TRAIN/FORWARD EVAL =====',
        'SAME RANDOM initialization/order/globalB4/source128/H6/CE+Lovasz/dense-valid/same64.',
        'Architecture/parameters/optimizer/globalB4 unchanged. Training world-size change is declared below.',
        f'Training GPUs: A={control["contract"]["world_size"]}, C={result["contract"]["world_size"]}; not bit-identical execution.',
        'model route mIoU IoU MovingMicro']
    for label,value in (('A_recorded',control),('C_forward',result)):
        for route,m in value['evaluations']['1']['variants'].items():
            lines.append(f'{label:12s} {route:28s} {m["mIoU"]:.6f} {m["IoU"]:.6f} {m["MovingMicro"]:.6f}')
    for route in control['evaluations']['1']['variants']:
        m = result['evaluations']['1']['variants'][route]
        old = control['evaluations']['1']['variants'][route]
        lines.append(route+' C_minus_A_recorded='+json.dumps({k:m[k]-old[k] for k in ('mIoU','IoU','MovingMicro')}))
    for kind,value in result['evaluations']['1'].get('free_minus1_minus_raw',{}).items():
        lines.append(kind+' FIXED_FREE_MINUS1_minus_raw='+json.dumps(value))
    lines.append('Fixed final epoch/coefficient; inherited labeled output calibration only, no fitting/age/noise/extra loss/promotion. NOT FPS.')
    return '\n'.join(lines)+'\n'


def summary(result):
    c = result['contract'];ex = c['model'].get('experiment')
    text = ('===== AdaLN-Zero / FORWARD ASSOCIATION / TRAIN20pct x1 =====\n'
        f'status={result["status"]}; updates={result["updates"]}/{c["total_updates"]}; epoch={result["epoch"]}; parameters={c["parameters"]}\n'
        'RANDOM, NO trained weights/Adam imported. Only history identity policy changed.\n'
        'TRAIN and OCC/STC eval rebuild KTA/anchors/history reads/Memory/fresh Strong consistently.\n'
        'Only full6 CE+Lovasz, GT future ego; same dense-valid/same64; no ICP/age/head.\n'
        'ONE GPU training/evaluation; raw + fixed FREE-1 share each forward; bias NEVER enters training loss.\n')
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
    files.add(root/'tools/fpn_stc_bias448/run.py')
    experiment = dict(control_results=str(path),control_results_sha256=sha(path),
        output_calibration=CALIBRATION,training_execution=dict(control_world_size=control['contract']['world_size'],
            candidate_world_size=ctx.world_size,global_window_batch=4,world_size_change_explicitly_requested=True),
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
