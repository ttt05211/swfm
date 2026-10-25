"""RANDOM FULL TRAIN20430 x2, original AdaLN-Zero; reuse the full FPN trainer."""
import importlib.util
import json
from pathlib import Path
import signal
import sys
import threading
from tools.fpn_unet_full import run as original
from tools.fpn_unet_full.model import VARIANT as CONTROL_VARIANT
from tools.fpn_unet.evaluation import NAMES
from .model import make_model

PROTOCOL = 'matched_FPN_UNet_AdaLN_Zero_RANDOM_fullTRAIN20430x2_GT_ego_H6_v1'
spec = importlib.util.spec_from_file_location('tools.fpn_unet_full._adaln_zero_training', original.__file__)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
_parser, _launch, _plan = runner.parser, runner.launch, runner.plan


def reference(folder):
    path = Path(folder).resolve()/'results.json'
    value = json.loads(path.read_text(encoding='utf-8')); c = value['contract']
    if (value.get('status') != 'complete' or value.get('epoch') != 2
            or c.get('protocol') != original.PROTOCOL or c.get('epochs') != 2
            or c.get('model', {}).get('variant') != CONTROL_VARIANT
            or value.get('updates') != c.get('total_updates')
            or c.get('known') != 'TRAIN_OCC_STC_dense_valid' or c.get('registration') != 'none'
            or c.get('source_policy') != 'shared_pinned'
            or c.get('objective') != dict(CE=1., Lovasz=1., motion=0., existence=0., ego=0.)
            or len(c.get('train_keys', [])) != 20430 or len(set(map(tuple, c['train_keys']))) != 20430
            or len(c.get('eval_keys', [])) != 64 or len(set(map(tuple, c['eval_keys']))) != 64
            or len(c.get('epochs_updates', [])) != 2 or sum(c['epochs_updates']) != c['total_updates']):
        raise RuntimeError('completed original additive FPN/U-Net RANDOM full TRAIN20430x2 control required')
    expected_updates = (c['epochs_updates'][0], c['total_updates'])
    if set(value.get('evaluations', {})) != set(map(str, expected_updates)):
        raise RuntimeError('both fixed epoch-end control evaluations required')
    for e in value['evaluations'].values():
        if e.get('windows') != 64 or e.get('keys') != c['eval_keys'] or set(e.get('variants', {})) != set(NAMES):
            raise RuntimeError('control evaluations must use the same ordered64 and four routes')
    return path, value


def parser():
    p = _parser(); p.description = __doc__
    p.add_argument('--control-dir', required=True, help='completed additive full TRAIN20430x2; metrics ONLY')
    return p


def check_settings(a, ctx, control):
    c = control['contract']
    expected = dict(seed=a.seed, world_size=ctx.world_size, learning_rates=[a.motion_lr, a.repair_lr],
        overrides=a.override, schedule='whole_2_epoch_cosine_floor0.01',
        precision='BF16_network_FP64_geometry_full6', initialization='RANDOM_no_pilot_weights',
        checkpoint_every=a.checkpoint_every, evaluation='epoch_end_only_same64',
        execution=dict(torch=str(original.torch.__version__), cuda=original.torch.version.cuda,
            cpu_workers=a.cpu_workers, prefetch_workers=a.prefetch_workers))
    for key, actual in expected.items():
        if actual != c.get(key): raise RuntimeError('unmatched additive full control: '+key)
    for name, digest in c['data'].items():
        if original.base.sha256(getattr(a, name)) != digest: raise RuntimeError('unmatched control data: '+name)
    root = Path(original.__file__).resolve().parents[2]
    for name, digest in c['implementation'].items():
        if original.base.sha256(root/name) != digest: raise RuntimeError('original control implementation changed: '+name)


def compare(result, control):
    fields = ('train_keys','eval_keys','epochs','total_updates','epochs_updates','seed','objective',
        'learning_rates','schedule','known','precision','world_size','data','overrides','STC_inventory',
        'execution','initialization','checkpoint_every','evaluation')
    for name in fields:
        if result['contract'][name] != control['contract'][name]:
            raise RuntimeError('unmatched full comparison: '+name)
    lines = ['===== MATCHED ADDITIVE EGO / AdaLN-Zero / FULL TRAIN x2 =====',
        'Control reports reused, NOT trained weights. SAME full TRAIN/order/budget/remaining64.',
        'Conditioning includes zero residual gates and affine-free query norms; NOT equal parameters.']
    for update, evaluation in sorted(result['evaluations'].items(), key=lambda x:int(x[0])):
        prior = control['evaluations'][update]
        lines += [f'===== update={update}; windows=64 =====', 'model input_route mIoU IoU MovingMicro']
        for label, report in (('additive',prior),('AdaLN_Zero',evaluation)):
            for route, m in report['variants'].items():
                lines.append(f'{label:12s} {route:28s} {m["mIoU"]:.6f} {m["IoU"]:.6f} {m["MovingMicro"]:.6f}')
        for route, m in evaluation['variants'].items():
            lines.append(route+' candidate_minus_control='+json.dumps({
                k:m[k]-prior['variants'][route][k] for k in ('mIoU','IoU','MovingMicro')}))
    lines.append('parameters='+json.dumps(dict(additive=control['contract']['parameters'],
        AdaLN_Zero=result['contract']['parameters'])))
    lines.append('Fixed epoch2, no best selection, age decay/calibration/Pred/noise/automatic promotion. NOT FPS.')
    return '\n'.join(lines)+'\n'


def summary(result):
    c = result['contract']
    text = ('===== FPN + U-NET / EGO AdaLN-Zero / RANDOM FULL TRAIN x2 =====\n'
        f'status={result["status"]}; updates={result["updates"]}/{c["total_updates"]}; epoch={result["epoch"]}; parameters={c["parameters"]}\n'
        'Same successful TRAIN20pct pilot architecture, RANDOM initialization; NO pilot/trained control weights.\n'
        'TRAIN20430 x2; globalB4/source128/H6 CE+Lovasz/dense-valid; whole TWO-epoch cosine.\n'
        'KTA query+skip, GT ego via three AdaLN-Zero branches; no additive ego/ICP/head/navigation.\n'
        'History-only memory/FPN/U-Net unchanged; SAME GT future poses in Decoder and ALL renderers.\n'
        'Save every2000 GLOBAL updates and epoch ends; SAME64 OCC/STC on rank0 at each epoch end ONLY.\n')
    experiment = c['model'].get('experiment')
    if experiment:
        path = Path(experiment['control_results'])
        if original.base.sha256(path) != experiment['control_results_sha256']:
            raise RuntimeError('read-only control report changed')
        text += compare(result, json.loads(path.read_text(encoding='utf-8')))
    return text


def launch(a, event, ctx):
    path, control = reference(a.control_dir); check_settings(a, ctx, control)
    root = Path(original.__file__).resolve().parents[2]
    files = sorted([*Path(__file__).parent.glob('*.py'),Path(__file__).parent/'run.sh',
        root/'tools/fpn_adaln_zero/model.py'])
    experiment = dict(control_results=str(path), control_results_sha256=original.base.sha256(path),
        candidate_implementation={p.relative_to(root).as_posix():original.base.sha256(p) for p in files})
    def checked_plan(*args):
        keys, evaluation, groups = _plan(*args); c = control['contract']
        actual = dict(train_keys=[list(k) for k in keys],
            eval_keys=[list(k) for k in original.base.record_keys(evaluation)],
            epochs_updates=list(map(len, groups)), total_updates=sum(map(len, groups)))
        for name, value in actual.items():
            if value != c[name]: raise RuntimeError('ordered full plan differs from additive control: '+name)
        inventory = set()
        for record in evaluation:
            for token in record['history_tokens'][-4:]:
                p = (Path(a.stc_root)/record['scene_name']/str(token)/'labels.npz').resolve()
                st = p.stat(); inventory.add((str(p),st.st_size,st.st_mtime_ns))
        if [list(v) for v in sorted(inventory)] != c['STC_inventory']:
            raise RuntimeError('STC inventory differs from additive full control')
        return keys,evaluation,groups
    def factory(cfg):
        return make_model(cfg, expected_control_spec=control['contract']['model'], experiment=experiment)
    previous_factory, previous_plan = runner.make_model, runner.plan
    runner.make_model, runner.plan = factory, checked_plan
    try:
        print('AdaLN_ZERO_FULL: independent RANDOM TRAIN20430x2; same pilot architecture; control READ ONLY.',flush=True)
        return _launch(a,event,ctx)
    finally:
        runner.make_model, runner.plan = previous_factory, previous_plan


runner.PROTOCOL = PROTOCOL
runner.parser, runner.summary, runner.launch = parser, summary, launch


def main(argv=None,event=None):
    return runner.main(argv,event)


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda *_:event.set())
    sys.exit(main(event=event) or 0)
