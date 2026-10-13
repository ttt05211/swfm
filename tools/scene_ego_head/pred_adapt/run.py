"""Frozen Encoder/STWM, GT epoch2 Refiner + RANDOM ego, 500 Pred updates."""
import hashlib
from pathlib import Path
import signal
import sys
import threading
import torch
from tools.joint_occ_experiments import run as legacy
from tools.scene_ego_head.integration import SceneEgoRefiner
from tools.scene_ego_head.model import SceneEgoHead, SceneEgoConfig
from tools.scene_ego_head.frozen_gt.run import validate_completed_source
from tools.scene_ego_head.experiment.internal_only.run import evaluate, summarize as base_summary
from tools.shared_history_experiments.refine.history import ResidualHistoryInputs
from tools.shared_history_experiments.loss_controls.run import capped_groups
from tools.shared_history_experiments.run import NoLegacyHeadGraph
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from . import engine

PROTOCOL = 'GTepoch2_RANDOMscene_ego_refiner_joint_adapt500_v1'


class AdaptModel(SceneEgoRefiner):
    def __init__(self, config, source, source_sha256, *, pool_side=8):
        c = source['contract']['model']; shared, repair = c['shared_history'], c['repair']
        super().__init__(config, feature_width=shared['width'], z_bins=shared['z_bins'],
            memory_width=repair['memory_width'], width=repair['width'], chunk_size=repair['chunk_size'],
            pool_side=pool_side)
        weights = source['models']['gt']
        if any(not k.startswith(('transport.', 'columns.')) for k in weights):
            raise RuntimeError('source contains unexpected non-body weights')
        self.transport.load_state_dict({k[10:]: v for k, v in weights.items() if k.startswith('transport.')}, strict=True)
        self.columns.load_state_dict({k[8:]: v for k, v in weights.items() if k.startswith('columns.')}, strict=True)
        self.columns.epsilon = repair['epsilon']
        # Independent seed: neither GT checkpoint RNG nor any previously trained head.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(21)
            self.ego = SceneEgoHead(SceneEgoConfig(shared['width'], pool_side=pool_side))
        self.source_sha256 = source_sha256
        self.transport.eval().requires_grad_(False)

    def train(self, mode=True):
        super().train(mode); self.transport.eval(); return self

    def requires_grad_(self, value=True):
        super().requires_grad_(value); self.transport.requires_grad_(False); return self

    def configs(self):
        return {**super().configs(), 'experiment': dict(protocol=PROTOCOL, fixed_updates=500,
            initialization='GT_epoch2_body_RANDOM_ego_seed21', source_sha256=self.source_sha256,
            frozen=['history_encoder', 'STWM'], trainable=['semantic_refiner', 'scene_ego'],
            objective=dict(ce=1., lovasz=1., ego=1., motion=0., existence=0.),
            planner_used=False, new_feature_bank=False)}


def summarize(result):
    text = base_summary(result)
    text = text.replace('===== SCENE EGO / RANDOM JOINT500 / INTERNAL ONLY =====',
                        '===== GT EPOCH2 / RANDOM EGO / REFINER JOINT ADAPT500 =====')
    text = text.replace('Entire Shared Encoder/STWM/Refiner/scene ego RANDOM jointly optimized; original H6.',
        'GT epoch2 body initialized; RANDOM ego seed21. Encoder/STWM frozen; Refiner+ego optimized.')
    return text+'Refiner LR=1e-5; ego LR=1e-4; fresh Adam/RNG cycle, no old head/optimizer import.\n'


def main(event=None, argv=None):
    names = ('parser', 'train_arm', 'evaluate', 'JointOccEgo', 'HistoryInputs', 'BothActionsExecution',
             'PROTOCOL', 'summarize', 'stable_json_fingerprint', 'epoch_groups', 'write_json', 'restore_checkpoint')
    old = {n: getattr(legacy, n) for n in names}
    def parser():
        p = old['parser'](); p.description = __doc__
        p.set_defaults(mode='pred', epochs=1, train_fraction=.2, motion_lr=1e-5, repair_lr=1e-5, ego_lr=1e-4)
        p.add_argument('--gt-checkpoint', required=True)
        return p
    a = parser().parse_args(argv)
    if (a.mode != 'pred' or a.epochs != 1 or a.train_fraction != .2 or a.window_batch_size != 4
            or a.source_budget != 128 or a.seed != 20261011 or a.training_backend != 'fused'
            or a.eval_population != 'dev512' or a.max_updates > 500
            or (a.repair_lr, a.ego_lr) != (1e-5, 1e-4)
            or (a.object_weight, a.ego_weight, a.lovasz_weight) != (1., 1., 1.)):
        raise ValueError('fixed GT epoch2 + RANDOM ego / Refiner+ego / full6 / 500 / DEV512')
    origin, out = Path(a.gt_checkpoint).resolve(), Path(a.out_dir).resolve()
    if out == origin.parent or origin.parent in out.parents: raise ValueError('new output outside GT source')
    if a.resume and Path(a.resume).resolve().parent != out/'train': raise ValueError('resume THIS adaptation train/last.pt only')
    if a.evaluate_only: raise ValueError('this workflow trains then evaluates; resume same output when stopped')
    lock = out.parent/'.pred_adapt_locks'/hashlib.sha256(str(out).encode()).hexdigest(); lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(lock):
        if not a.resume:
            out.mkdir(parents=True, exist_ok=False); (out/'frozen').mkdir()
            snapshot_checkpoint(origin, out/'frozen/GTepoch2.pt')
        frozen = out/'frozen/GTepoch2.pt'; digest = legacy.sha256(frozen)
        if legacy.sha256(origin) != digest: raise RuntimeError('GT source checkpoint changed')
        saved = torch.load(frozen, map_location='cpu', weights_only=False); validate_completed_source(saved)
        for k in ('train_cache', 'dev_cache', 'train_info', 'dev_info', 'base_checkpoint'):
            if legacy.sha256(Path(getattr(a, k))) != saved['contract']['data'][k]:
                raise RuntimeError('GT source data/architecture metadata differs: '+k)
        frozen_digest = []
        def construct(config):
            m = AdaptModel(config, saved, digest)
            frozen_digest[:] = [m, tensor_state_fingerprint(m.transport.state_dict())]
            legacy.write_json(out/'initialization_audit.json', dict(source_sha256=digest,
                Encoder_STWM_sha256=frozen_digest[1], refiner_sha256=tensor_state_fingerprint(m.columns.state_dict()),
                random_ego_sha256=tensor_state_fingerprint(m.ego.state_dict()), ego_seed=21,
                imported_head=False, imported_Adam=False, imported_RNG=False))
            return m
        root = Path(__file__).resolve().parents[3]
        files = list(Path(__file__).parent.glob('*.py'))+[root/'tools/scene_ego_head/integration.py',
            root/'tools/scene_ego_head/model.py', root/'tools/scene_ego_head/experiment/internal_only/run.py']
        def fingerprint(value):
            base = old['stable_json_fingerprint'](value)
            if isinstance(value, dict) and 'tools/joint_occ_experiments/run.py' in value:
                return old['stable_json_fingerprint'](dict(base=base, adaptation={str(p.relative_to(root)): legacy.sha256(p) for p in files}))
            return base
        def checked_eval(models, *args, **kwargs):
            if tensor_state_fingerprint(models['pred'].transport.state_dict()) != frozen_digest[1]:
                raise RuntimeError('frozen Encoder/STWM weights changed')
            return evaluate(models, *args, **kwargs)
        def mark_initialization(contract):
            contract['initialization'] = 'GT_epoch2_body_RANDOM_ego_seed21_fresh_Adam'
            contract['source_GT_sha256'] = digest
        def write_json(path, value):
            if Path(path).name == 'contract.json': mark_initialization(value)
            return old['write_json'](path, value)
        def restore_checkpoint(saved, models, optimizers, contract):
            mark_initialization(contract)
            return old['restore_checkpoint'](saved, models, optimizers, contract)
        args = sys.argv[1:] if argv is None else list(argv)
        at = args.index('--gt-checkpoint'); del args[at:at+2]
        args[args.index('--out-dir')+1] = str(out/'train')
        try:
            legacy.parser, legacy.PROTOCOL, legacy.JointOccEgo = lambda: old['parser'](), PROTOCOL, construct
            # Original parser receives the already fixed defaults explicitly.
            defaults = dict(mode='pred', epochs=1, train_fraction=.2, motion_lr=1e-5, repair_lr=1e-5, ego_lr=1e-4)
            def inner_parser():
                p = old['parser'](); p.set_defaults(**defaults); return p
            legacy.parser = inner_parser
            legacy.HistoryInputs, legacy.BothActionsExecution = ResidualHistoryInputs, NoLegacyHeadGraph
            legacy.train_arm, legacy.evaluate, legacy.summarize = engine.train_arm, checked_eval, summarize
            legacy.epoch_groups = lambda *a: capped_groups(old['epoch_groups'], *a)
            legacy.stable_json_fingerprint = fingerprint
            legacy.write_json, legacy.restore_checkpoint = write_json, restore_checkpoint
            print('GT epoch2 -> frozen Encoder/STWM; RANDOM ego21; only Refiner+ego full6 CE/Lovasz/R10 x500.', flush=True)
            status = legacy.main(event, args)
            result = legacy.json.loads((out/'train/results.json').read_text(encoding='utf-8'))
            old['write_json'](out/'results.json', result)
            (out/'summary.txt').write_text(summarize(result), encoding='utf-8')
            return status
        finally:
            for n, v in old.items(): setattr(legacy, n, v)
            if frozen_digest:
                m, before = frozen_digest
                unchanged = tensor_state_fingerprint(m.transport.state_dict()) == before
                legacy.write_json(out/'freeze_audit.json', dict(Encoder_STWM_unchanged=unchanged,
                    frozen_parameter_gradients=sum(p.grad is not None for p in m.transport.parameters())))
                if not unchanged: raise RuntimeError('frozen transport mutated')


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
