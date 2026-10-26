"""RANDOM FULL TRAIN20430 x2 / forward AdaLN / dual-card training / one-card eval."""
import importlib.util
import json
from pathlib import Path
import signal
import sys
import threading
from tools.fpn_unet_full import run as original
from tools.causal_forward_train.run import PROTOCOL as PILOT_PROTOCOL
from tools.causal_forward_train.control import sha
from tools.causal_forward_train.data import ForwardProvider,batches,forward_window
from .model import make_model,SOURCE_POLICY
from .evaluation import evaluate,NAMES

PROTOCOL = 'forward_AdaLN_RANDOM_fullTRAIN20430x2_raw_every2000_v1'
spec = importlib.util.spec_from_file_location('tools.fpn_unet_full._forward_full_training',original.__file__)
runner = importlib.util.module_from_spec(spec)
source = Path(original.__file__).read_text(encoding='utf-8')
# Preserve the existing full optimizer/loop/Adam-RNG recovery. Record actual
# policy/cadence in its contract and log; no changes to original source files.
for old,new in (("source_policy='shared_pinned'","source_policy="+repr(SOURCE_POLICY)),
                ("evaluation='epoch_end_only_same64'","evaluation='every2000_and_epoch_end_same64'"),
                ('world={ctx.world_size}; epoch-end DEV64; NEW=',
                 'world={ctx.world_size}; every2000+epoch-end DEV64; NEW=')):
    if source.count(old)!=1:raise RuntimeError('original full trainer metadata hook changed: '+old)
    source = source.replace(old,new)
exec(compile(source,original.__file__,'exec'),runner.__dict__)
_parser,_launch,_plan = runner.parser,runner.launch,runner.plan
runner.DirectProvider,runner.batches,runner.forward_window,runner.evaluate = ForwardProvider,batches,forward_window,evaluate


def reference(path):
    path = Path(path).resolve();c = json.loads(path.read_text(encoding='utf-8'))
    if (c.get('protocol')!=PILOT_PROTOCOL or c.get('source_policy')!=SOURCE_POLICY or c.get('world_size')!=1
            or c.get('epochs')!=1 or len(c.get('train_keys',[]))!=4086 or len(set(map(tuple,c['train_keys'])))!=4086
            or len(c.get('eval_keys',[]))!=64 or len(set(map(tuple,c['eval_keys'])))!=64
            or c.get('known')!='TRAIN_OCC_STC_dense_valid' or c.get('registration')!='none'
            or c.get('objective')!=dict(CE=1.,Lovasz=1.,motion=0.,existence=0.,ego=0.)
            or c.get('model',{}).get('source_policy')!=SOURCE_POLICY or c.get('parameters')!=2816122):
        raise RuntimeError('original selected single-GPU forward AdaLN pilot CONTRACT required; no weights imported')
    return path,c


def parser():
    p = _parser();p.description=__doc__;p.add_argument('--pilot-contract',required=True,
        help='selected C_train/contract.json: architecture/data/keys ONLY; NEVER trained weights')
    return p


def evaluation_due(update,batch):return update>0 and (update%2000==0 or batch==0)


def check_settings(a,ctx,pilot):
    if ctx.world_size not in (1,2):raise RuntimeError('ONE or TWO training GPUs required')
    expected=dict(seed=a.seed,learning_rates=[a.motion_lr,a.repair_lr],overrides=a.override,
        precision='BF16_network_FP64_geometry_full6',execution=dict(torch=str(original.torch.__version__),
            cuda=original.torch.version.cuda,cpu_workers=a.cpu_workers,prefetch_workers=a.prefetch_workers))
    for name,value in expected.items():
        if pilot.get(name)!=value:raise RuntimeError('selected pilot settings differ: '+name)
    for name,digest in pilot['data'].items():
        if sha(getattr(a,name))!=digest:raise RuntimeError('selected pilot input differs: '+name)
    root=Path(original.__file__).resolve().parents[2]
    for name,digest in pilot['implementation'].items():
        if sha(root/name)!=digest:raise RuntimeError('selected pilot source changed: '+name)


def summary(result):
    c = result['contract']
    lines = ['===== FORWARD AdaLN-Zero / RANDOM FULL TRAIN x2 =====','status='+result['status'],
        f'TRAIN20430 x2; updates={result["updates"]}/{c["total_updates"]}; completed_epochs={result["epoch"]}; parameters={c["parameters"]}',
        'SAME validated forward AdaLN architecture; RANDOM initialization, NO pilot/old trained weights or optimizer import.',
        'FPN + SceneTemporal2 + Cross-first FutureDecoder2/AdaLN-Zero + plain3D U-Net; KTA query+skip.',
        'Forward identities/KTA in TRAIN AND OCC/STC eval; track-valid, no ICP; pure historical content/learned history yaw/direct Memory.',
        'GT future ego in Decoder AND renderer; dense-valid TRAIN/OCC/STC; full6 CE+Lovasz ONLY.',
        'FREE bias OFF; fixed age decay OFF; no head/navigation/noise/calibration/norm adaptation.',
        f'Training GPUs={c["world_size"]}; globalB4/source128, unchanged global loss means; evaluation rank0 ONE GPU ONLY.',
        'Original LR/AdamW; whole TWO-epoch cosine floor0.01, Adam/per-rank RNG/cursor preserved on SAME-world resume.',
        'Save+eval every2000 GLOBAL updates AND epoch ends, SAME64/full grid/1,2,3s/Moving. Released STC is EVAL ONLY. NOT FPS.']
    for update,value in sorted(result['evaluations'].items(),key=lambda item:int(item[0])):
        lines.append(f'===== update={update}; windows={value["windows"]} =====')
        for name,m in value['variants'].items():
            lines.append(f'{name:28s} mIoU={m["mIoU"]:.6f} IoU={m["IoU"]:.6f} MovingMicro={m["MovingMicro"]:.6f}')
    return '\n'.join(lines)+'\n'


def launch(a,event,ctx):
    path,pilot = reference(a.pilot_contract)
    check_settings(a,ctx,pilot)
    if a.epochs!=2 or a.train_fraction!=1.:raise RuntimeError('full TRAIN x2 required')
    root = Path(original.__file__).resolve().parents[2]
    for name,digest in pilot['model']['experiment']['candidate_implementation'].items():
        file = (root/name).resolve()
        if root not in file.parents or sha(file)!=digest:raise RuntimeError('pilot source changed: '+name)
    files = [*Path(__file__).parent.glob('*.py'),Path(__file__).parent/'run.sh',
        root/'tools/causal_forward_train/model.py',root/'tools/causal_forward_train/data.py',
        root/'tools/causal_forward_train/run.py',root/'tools/causal_forward_track/track.py',
        root/'tools/causal_forward_recovery/adapter.py',root/'tools/fpn_adaln_zero/model.py']
    experiment = dict(pilot_contract=str(path),pilot_contract_sha256=sha(path),FREE_bias=0.,history_age=False,
        training_execution=dict(pilot_world_size=pilot['world_size'],full_world_size=ctx.world_size,
            global_window_batch=4,world_size_change_explicitly_requested=True),
        candidate_implementation={p.relative_to(root).as_posix():sha(p) for p in files})
    def checked_plan(*args):
        keys,evaluation,groups = _plan(*args)
        if not set(map(tuple,pilot['train_keys']))<=set(map(tuple,keys)):raise RuntimeError('pilot keys absent from full TRAIN')
        if [list(k) for k in original.base.record_keys(evaluation)]!=pilot['eval_keys']:raise RuntimeError('same64 eval order differs')
        inventory = set()
        for record in evaluation:
            for token in record['history_tokens'][-4:]:
                p = (Path(a.stc_root)/record['scene_name']/str(token)/'labels.npz').resolve();s=p.stat()
                inventory.add((str(p),s.st_size,s.st_mtime_ns))
        if [list(row) for row in sorted(inventory)]!=pilot['STC_inventory']:raise RuntimeError('same64 STC inventory differs')
        return keys,evaluation,groups
    def factory(cfg):return make_model(cfg,expected_pilot_spec=pilot['model'],experiment=experiment)
    old_factory,old_plan = runner.make_model,runner.plan
    runner.make_model,runner.plan = factory,checked_plan
    try:
        print('FORWARD_FULL: RANDOM TRAIN20430x2; pilot CONTRACT only; NO trained weights. FREE/age OFF.',flush=True)
        return _launch(a,event,ctx)
    finally:runner.make_model,runner.plan = old_factory,old_plan


runner.PROTOCOL,runner.evaluation_due = PROTOCOL,evaluation_due
runner.parser,runner.summary,runner.launch = parser,summary,launch


def main(argv=None,event=None):
    # Reject accidental torchrun before the old distributed helper initializes.
    import os
    if int(os.environ.get('WORLD_SIZE','1')) not in (1,2):raise RuntimeError('ONE or TWO training GPUs required')
    return runner.main(argv,event)


if __name__=='__main__':
    event = threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:event.set())
    sys.exit(main(event=event) or 0)
