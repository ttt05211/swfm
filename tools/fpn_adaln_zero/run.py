"""Reuse the unchanged FPN trainer in a PRIVATE namespace, not a copied loop."""
import importlib.util
import json
from pathlib import Path
import signal
import sys
import threading
from tools.fpn_unet import run as original
from tools.fpn_unet.model import VARIANT as CONTROL_VARIANT
from tools.fpn_unet.evaluation import NAMES
from .model import make_model

PROTOCOL = 'matched_FPN_UNet_AdaLN_Zero_RANDOM_TRAIN20x1_GT_ego_H6_v1'
spec = importlib.util.spec_from_file_location('tools.fpn_unet._adaln_zero_training', original.__file__)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
_parser, _launch, _plan = runner.parser, runner.launch, runner.plan


def reference(folder):
    path = Path(folder).resolve()/'results.json'
    v = json.loads(path.read_text(encoding='utf-8')); c = v['contract']
    e = v.get('evaluations', {}).get('1', {})
    if (v.get('status') != 'complete' or c.get('protocol') != original.PROTOCOL or
            c.get('model', {}).get('variant') != CONTROL_VARIANT or c.get('epochs') != 1 or
            v.get('epoch') != 1 or v.get('updates') != c.get('total_updates') or
            c.get('known') != 'TRAIN_OCC_STC_dense_valid' or c.get('registration') != 'none' or
            c.get('source_policy') != 'shared_pinned' or
            c.get('objective') != dict(CE=1., Lovasz=1., motion=0., existence=0., ego=0.) or
            len(c.get('train_keys', [])) != 4086 or len(set(map(tuple, c['train_keys']))) != 4086 or
            len(c.get('eval_keys', [])) != 64 or len(set(map(tuple, c['eval_keys']))) != 64 or
            e.get('windows') != 64 or e.get('keys') != c['eval_keys'] or
            set(e.get('variants', {})) != set(NAMES)):
        raise RuntimeError('completed original FPN/U-Net RANDOM TRAIN4086x1 / same64 control required')
    return path, v


def parser():
    p = _parser(); p.description = __doc__
    p.add_argument('--control-dir', required=True, help='completed additive FPN TRAIN20pct x1; metrics ONLY')
    return p


def check_settings(a, ctx, control):
    c = control['contract']
    expected = dict(seed=a.seed, world_size=ctx.world_size, learning_rates=[a.motion_lr, a.repair_lr],
        overrides=a.override, schedule='whole_1_epoch_cosine_floor0.01',
        precision='BF16_network_FP64_geometry_full6')
    for key, value in expected.items():
        if c.get(key) != value: raise RuntimeError('unmatched additive control: '+key)
    execution = dict(torch=str(original.torch.__version__),cuda=original.torch.version.cuda,
        cpu_workers=a.cpu_workers,prefetch_workers=a.prefetch_workers)
    if c.get('execution') != execution: raise RuntimeError('unmatched additive control: execution')
    for name, digest in c['data'].items():
        if original.base.sha256(getattr(a, name)) != digest:
            raise RuntimeError('unmatched control data: '+name)
    # No silent comparison after changing an old source implementation.
    root = Path(original.__file__).resolve().parents[2]
    for name, digest in c['implementation'].items():
        if original.base.sha256(root/name) != digest:
            raise RuntimeError('original control implementation changed: '+name)


def compare(result, control):
    if result['status'] != 'complete': return ''
    fields = ('train_keys','eval_keys','epochs','total_updates','seed','objective','learning_rates',
        'schedule','known','precision','world_size','data','overrides','STC_inventory','execution')
    for name in fields:
        if result['contract'][name] != control['contract'][name]:
            raise RuntimeError('unmatched final comparison: '+name)
    lines = ['===== MATCHED ADDITIVE EGO / AdaLN-Zero =====',
        'Control metrics reused; RANDOM common fresh weights, NOT control trained weights.',
        'SAME TRAIN4086x1/order/B4-source128/H6/dense-valid/CE+Lovasz/GT ego/remaining64.',
        'Conditioning strategy includes zero residual gates and affine-free query norms; NOT equal parameters.',
        'model input_route mIoU IoU MovingMicro']
    for label, v in (('additive',control),('AdaLN_Zero',result)):
        for route, m in v['evaluations']['1']['variants'].items():
            lines.append(f'{label:12s} {route:28s} {m["mIoU"]:.6f} {m["IoU"]:.6f} {m["MovingMicro"]:.6f}')
    for route,m in result['evaluations']['1']['variants'].items():
        old = control['evaluations']['1']['variants'][route]
        lines.append(route+' candidate_minus_control='+json.dumps({k:m[k]-old[k] for k in ('mIoU','IoU','MovingMicro')}))
    lines.append('parameters='+json.dumps(dict(additive=control['contract']['parameters'],AdaLN_Zero=result['contract']['parameters'])))
    lines.append('Fixed final epoch; no best selection, retries, Pred run, expansion or automatic promotion. NOT FPS.')
    return '\n'.join(lines)+'\n'


def summary(result):
    c = result['contract']
    text = ('===== FPN + U-NET / EGO AdaLN-Zero / TRAIN20% x1 =====\n'
        f'status={result["status"]}; updates={result["updates"]}/{c["total_updates"]}; epoch={result["epoch"]}; parameters={c["parameters"]}\n'
        'RANDOM; no trained body imported. FPN/Scene-Temporal2/Decoder2/history-only memory/U-Net unchanged.\n'
        'Ego enters THREE query-side AdaLN-Zero branches per Decoder block; NO additive ego path.\n'
        'SAME GT future poses in Decoder and renderer; no historical K/V or Memory-content conditioning.\n'
        'Only full6 CE+Lovasz; dense-valid OCC/STC; no age decay/calibration/norm adaptation/ego head.\n')
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
    files = sorted([*Path(__file__).parent.glob('*.py'),Path(__file__).parent/'run.sh'])
    experiment = dict(control_results=str(path), control_results_sha256=original.base.sha256(path),
        candidate_implementation={p.relative_to(root).as_posix():original.base.sha256(p) for p in files})
    def checked_plan(*args):
        keys, evaluation, groups = _plan(*args); c = control['contract']
        # record_keys returns tuple[tuple], while the saved JSON contract uses
        # list[list]. Normalize containers ONLY; never sort or change identities.
        ordered = dict(train_keys=[list(k) for k in keys],
            eval_keys=[list(k) for k in original.base.record_keys(evaluation)])
        for name, actual in ordered.items():
            expected = c[name]
            if actual != expected:
                first = next((i for i, pair in enumerate(zip(actual, expected))
                              if pair[0] != pair[1]), min(len(actual), len(expected)))
                raise RuntimeError(f'{name} differ from additive control: '
                    f'actual_count={len(actual)}, expected_count={len(expected)}, first_difference={first}')
        if len(groups[0]) != c['total_updates']:
            raise RuntimeError(f'total_updates differ from additive control: '
                f'actual={len(groups[0])}, expected={c["total_updates"]}')
        inventory = set()
        for record in evaluation:
            for token in record['history_tokens'][-4:]:
                p = (Path(a.stc_root)/record['scene_name']/str(token)/'labels.npz').resolve()
                stat = p.stat(); inventory.add((str(p),stat.st_size,stat.st_mtime_ns))
        if [list(v) for v in sorted(inventory)] != c['STC_inventory']:
            raise RuntimeError('STC inventory differs from additive control')
        return keys,evaluation,groups
    def factory(cfg):
        return make_model(cfg, expected_control_spec=control['contract']['model'], experiment=experiment)
    # These bindings belong ONLY to this private module. Original run/model/eval
    # globals and files are untouched; the existing loop and evaluator are reused.
    previous_factory, previous_plan = runner.make_model, runner.plan
    runner.make_model, runner.plan = factory, checked_plan
    try:
        print('AdaLN_ZERO: independent RANDOM matched one-epoch candidate; original control READ ONLY.',flush=True)
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
